import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from .film_resblock import _FilmResBlock
import utils

class Generator(nn.Module):
    """
    The conditional HiFi-GAN Generator model.

    This class defines the generator architecture, which takes a mel-spectrogram
    and a conditioning signal (sample rate embedding) to produce a raw audio waveform.
    It uses a series of transposed convolutions for upsampling and FiLM-conditioned
    residual blocks (FilmResBlock) for feature refinement.
    """
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