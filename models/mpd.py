import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

class _DiscriminatorP(nn.Module):
    """
    A single period-based discriminator from the HiFi-GAN paper.

    This discriminator is designed to capture periodic patterns in audio. It
    achieves this by reshaping the input 1D waveform into a 2D grid where the
    width is equal to `period`, and then applying 2D convolutions.

    Args:
        period (int): The period to analyze in the audio.
        kernel_size (int): The kernel size for the convolutional layers.
        stride (int): The stride for the convolutional layers.
    """
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
    """
    The Multi-Period Discriminator (MPD) from HiFi-GAN.

    This class is a container for multiple `_DiscriminatorP` instances, each
    configured with a different prime period. This allows the model to critique
    the generated audio on a wide range of periodic structures simultaneously,
    which is crucial for modeling pitch and harmonics correctly.
    """
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