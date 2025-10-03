import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

import utils

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