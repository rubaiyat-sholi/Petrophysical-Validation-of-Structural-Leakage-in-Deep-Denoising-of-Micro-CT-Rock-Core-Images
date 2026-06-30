"""
Denoiser model zoo.

Models
------
DnCNN      : 17-layer residual CNN (Zhang et al. 2017).
UNet       : lightweight encoder-decoder with skip connections.
NLMDenoiser: non-local means wrapper (classical baseline, CPU).
GaussDenoiser: Gaussian filter baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import gaussian_filter
import numpy as np


# ── DnCNN ─────────────────────────────────────────────────────────────────────
class DnCNN(nn.Module):
    """Residual CNN — predicts noise; output = input - predicted_noise."""

    def __init__(self, depth: int = 17, n_channels: int = 64):
        super().__init__()
        layers = [nn.Conv2d(1, n_channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(n_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return torch.clamp(x - noise, 0.0, 1.0)


# ── U-Net ─────────────────────────────────────────────────────────────────────
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Lightweight 4-level U-Net for denoising (1→1 channel).
    channels = [32, 64, 128, 256] by default.
    """

    def __init__(self, base: int = 32):
        super().__init__()
        c = [base, base*2, base*4, base*8]

        # encoder
        self.enc1 = _DoubleConv(1,    c[0])
        self.enc2 = _DoubleConv(c[0], c[1])
        self.enc3 = _DoubleConv(c[1], c[2])
        self.enc4 = _DoubleConv(c[2], c[3])

        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = _DoubleConv(c[3], c[3]*2)

        # decoder
        self.up4 = nn.ConvTranspose2d(c[3]*2, c[3], 2, stride=2)
        self.dec4 = _DoubleConv(c[3]*2, c[3])

        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = _DoubleConv(c[2]*2, c[2])

        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = _DoubleConv(c[1]*2, c[1])

        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = _DoubleConv(c[0]*2, c[0])

        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.clamp(self.out(d1), 0.0, 1.0)


# ── UNet + leakage-regularised variant ────────────────────────────────────────
class UNetLeakage(UNet):
    """Same architecture; the structural-leakage penalty is applied in losses.py."""
    pass


# ── Classical baselines (numpy, CPU) ─────────────────────────────────────────
class GaussDenoiser:
    """Simple Gaussian filter baseline."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return gaussian_filter(img, sigma=self.sigma)


class NLMDenoiser:
    """Non-local means (skimage) baseline."""

    def __init__(self, h_factor: float = 1.15, patch_size: int = 5,
                 patch_distance: int = 6, fast: bool = True):
        self.h_factor       = h_factor
        self.patch_size     = patch_size
        self.patch_distance = patch_distance
        self.fast           = fast

    def __call__(self, img: np.ndarray) -> np.ndarray:
        sigma_est = np.mean(estimate_sigma(img))
        h = self.h_factor * sigma_est
        return denoise_nl_means(
            img, h=h,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            fast_mode=self.fast,
        )
