"""Pyramid generator on the HEALPix NESTED grid: white noise -> map.

A decoder-only network in the spirit of FOSCAT's `SynthHalfUNet2D`, laid
on the NESTED pyramid.  The noise `z` (one value per pixel of the finest
level) is pooled to every level (the 4-children mean, rescaled to unit
variance): `z_k` at `nside_k = nside0 2^k`.  Features start at `nside0`
from `z_0`; at every finer level they are up-sampled to the 4 children,
concatenated with `z_k` and go through two 3 x 3 convolutions.  Every level
adds its own correction to the up-sampled map of the coarser level,

    x_k = up(x_{k-1}) + head_k(h_k),

so the coarse part of a trained generator can be frozen and a finer level
added (progressive growing, section 4.5 of `docs/amortized_synthesis.md`).

Convolutions run on the 12 base-face images with a one-pixel ring taken
from the neighbouring faces (`diffusion/model.py`, `Padding`); channels are
scalars (no re-orientation across face edges).
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Appended, not prepended: diffusion/ has its own train.py.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diffusion"))
import healpix_faces as hf  # noqa: E402
from model import Conv3, Padding  # noqa: E402


def default_channels(levels):
    return [64] * min(levels, 3) + [48, 32, 32, 32, 32][: max(levels - 3, 0)]


class Generator(nn.Module):
    def __init__(self, nside, nside0=4, channels=None, slope=0.2, up="smooth"):
        super().__init__()
        self.nside, self.nside0, self.up_mode = nside, nside0, up
        self.levels = int(round(math.log2(nside // nside0))) + 1
        ch = list(channels or default_channels(self.levels))
        assert len(ch) == self.levels, "one channel count per level"
        self.channels = ch
        self.pad = Padding()
        self.pad.mode = "sphere"
        act = lambda: nn.LeakyReLU(slope)  # noqa: E731
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        for k, c in enumerate(ch):
            cin = 1 if k == 0 else ch[k - 1] + 1
            self.blocks.append(nn.Sequential(Conv3(self.pad, cin, c), act(), Conv3(self.pad, c, c), act()))
            self.heads.append(nn.Conv2d(c, 1, 1))
        fn = hf.face_nest(nside).ravel()
        self.register_buffer("face_of_nest", torch.as_tensor(np.argsort(fn)), persistent=False)
        self.register_buffer("nest_of_face", torch.as_tensor(fn), persistent=False)

    def up(self, x):
        """Children of every pixel.  'nearest' copies the parent; 'smooth' then
        blurs with [1, 2, 1]^2 / 16 across face edges.  After 'nearest' the
        convolutions see a field that depends on the child's position in its
        parent, and the trained generator adds a z-independent pattern,
        periodic on the pixel grid (7.7 % of the variance at nside 64)."""
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.up_mode == "nearest":
            return x
        k = torch.tensor([1.0, 2.0, 1.0], dtype=x.dtype, device=x.device)
        k = (k[:, None] * k[None, :] / 16).expand(x.shape[1], 1, 3, 3)
        return F.conv2d(self.pad(x), k, groups=x.shape[1])

    def noise(self, batch, generator=None, device=None):
        n = self.nside
        return torch.randn(batch * 12, 1, n, n, generator=generator, device=device)

    def to_nested(self, img):
        """Face images `[B 12, 1, n, n]` -> NESTED maps `[B, 12 n^2]`."""
        b = img.shape[0] // 12
        flat = img.reshape(b, 12 * self.nside**2)
        return flat[:, self.face_of_nest]

    def forward(self, z):
        """`z`: `[B 12, 1, nside, nside]` white noise; returns NESTED maps `[B, 12 nside^2]`."""
        zs = [z]
        for _ in range(self.levels - 1):
            zs.append(F.avg_pool2d(zs[-1], 2) * 2.0)
        zs = zs[::-1]
        x = None
        h = None
        for k in range(self.levels):
            if k == 0:
                h = self.blocks[0](zs[0])
            else:
                h = self.blocks[k](torch.cat([self.up(h), zs[k]], 1))
            y = self.heads[k](h)
            x = y if x is None else self.up(x) + y
        return self.to_nested(x)
