"""
HiFi-GAN Model Implementation.

This module contains the core architecture for the Generator and Discriminators ruquired for HiFi-GAN training.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import utils


class SampleRateEmbedding(nn.Module):
    def __init__(self, embedding_dim, sample_rates):
        super().__init__()
        if embedding_dim % 2 != 0: raise ValueError("Embedding dim must be even.")
        self.embedding_dim = embedding_dim
        self.sr_to_idx = {sr: i for i, sr in enumerate(sample_rates)}
        
        num_embeddings = len(sample_rates)
        position = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        embedding_table = torch.zeros(num_embeddings, embedding_dim)
        embedding_table[:, 0::2] = torch.sin(position * div_term)
        embedding_table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('embedding_table', embedding_table)

    def forward(self, sr_values):
        indices = torch.tensor([self.sr_to_idx[sr.item()] for sr in sr_values], device=sr_values.device)

        return self.embedding_table[indices]


class _FilmLayer(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.film_generator = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, features, condition):
        gamma_beta = self.film_generator(condition)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        while len(gamma.shape) < len(features.shape):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return gamma * features + beta


class _FilmResBlock(nn.Module):
    def __init__(self, channels, sr_emb_dim, kernel_size=3, dilation=(1, 3, 5)):
        super(_FilmResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=((kernel_size - 1) * d) // 2))
            for d in dilation
        ])
        self.convs1.apply(utils.init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2))
            for _ in dilation
        ])
        self.convs2.apply(utils.init_weights)

        self.film_layers = nn.ModuleList([_FilmLayer(sr_emb_dim, channels) for _ in dilation])

    def forward(self, x, sr_embedding):
        for c1, c2, film in zip(self.convs1, self.convs2, self.film_layers):
            xt = F.leaky_relu(x, 0.1)
            xt = film(xt, sr_embedding)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.num_kernels = len([3, 7, 11])
        self.num_upsamples = len([8, 8, 2, 2])
        
        self.conv_pre = weight_norm(nn.Conv1d(config.n_mels, 512, 7, 1, padding=3))
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip([8, 8, 2, 2], [16, 16, 4, 4])):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(512 // (2**i), 512 // (2**(i+1)), k, u, padding=(k-u)//2)
            ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 512 // (2**(i+1))
            for j, (k, d) in enumerate(zip([3, 7, 11], [(1, 3, 5), (1, 3, 5), (1, 3, 5)])):
                self.resblocks.append(_FilmResBlock(ch, config.sr_embedding_dim, k, d))

        self.conv_post = weight_norm(nn.Conv1d(512 // (2**len(self.ups)), 1, 7, 1, padding=3))

        self.ups.apply(utils.init_weights)
        self.conv_post.apply(utils.init_weights)

    def forward(self, x, sr_embedding):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, sr_embedding)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, sr_embedding)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class _DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(_DiscriminatorP, self).__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            _DiscriminatorP(2), _DiscriminatorP(3), _DiscriminatorP(5), _DiscriminatorP(7), _DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class _DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(_DiscriminatorS, self).__init__()
        norm = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            _DiscriminatorS(use_spectral_norm=True),
            _DiscriminatorS(),
            _DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# Loss Functions
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
    return loss

def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        loss += l
    return loss