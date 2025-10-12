import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm

class _DiscriminatorS(nn.Module):
    """
    A single scale-based discriminator from the HiFi-GAN paper.

    This discriminator is a 1D convolutional network that operates directly on the
    raw audio waveform to evaluate its overall structure and texture.

    Args:
        use_spectral_norm (bool): If True, applies spectral normalization instead
        of weight normalization to the convolutional layers.
    """
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
    """
    The Multi-Scale Discriminator (MSD) from HiFi-GAN.

    This class is a container for multiple `_DiscriminatorS` instances. Each
    sub-discriminator operates on a different-scaled version of the input audio
    (the original, 2x downsampled, and 4x downsampled), allowing the model to
    evaluate the audio's structure at various resolutions.
    """
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