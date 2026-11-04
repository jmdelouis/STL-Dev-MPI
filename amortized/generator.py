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
    """`up`: up-sampling of the features ('smooth' or 'nearest'); `map_up`:
    of the map path (`x_k = up(x_{k-1}) + y_k`; 'nearest' keeps the finest
    details sharp: with 'smooth' the blur of every level is never undone and
    the finest level of the maps had 0.4 of the target's detail variance).
    `latent`: every block also sees one constant channel per map, the global
    mean of the map's noise times sqrt(npix) (a N(0, 1) scalar), so that
    global properties such as the variance of a map can vary from map to map.
    `set_transform` stores a monotone map f (knots) applied by `kappa`."""

    def __init__(self, nside, nside0=4, channels=None, slope=0.2, up="smooth", map_up="nearest", latent=False):
        super().__init__()
        self.nside, self.nside0, self.up_mode, self.map_up, self.latent = nside, nside0, up, map_up, latent
        self.levels = int(round(math.log2(nside // nside0))) + 1
        ch = list(channels or default_channels(self.levels))
        assert len(ch) == self.levels, "one channel count per level"
        self.channels = ch
        self.pad = Padding()
        self.pad.mode = "sphere"
        act = lambda: nn.LeakyReLU(slope)  # noqa: E731
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        extra = 1 if latent else 0
        for k, c in enumerate(ch):
            cin = (1 if k == 0 else ch[k - 1] + 1) + extra
            self.blocks.append(nn.Sequential(Conv3(self.pad, cin, c), act(), Conv3(self.pad, c, c), act()))
            self.heads.append(nn.Conv2d(c, 1, 1))
        self.register_buffer("knots_g", torch.zeros(0))
        self.register_buffer("knots_x", torch.zeros(0))
        fn = hf.face_nest(nside).ravel()
        self.register_buffer("face_of_nest", torch.as_tensor(np.argsort(fn)), persistent=False)

    def config(self):
        return {"nside": self.nside, "nside0": self.nside0, "channels": self.channels, "up": self.up_mode,
                "map_up": self.map_up, "latent": self.latent}

    def set_transform(self, g, x):
        """Knots of a monotone map f: `g` increasing, `x` non-decreasing."""
        self.knots_g = torch.as_tensor(g, dtype=torch.float64, device=self.knots_g.device)
        self.knots_x = torch.as_tensor(x, dtype=torch.float64, device=self.knots_g.device)

    def kappa(self, g):
        """f(g), piecewise linear between the knots, linear beyond the ends
        (a constant extrapolation would leave the pixels beyond the last knot
        without gradient, as noted in `stl-cli/src/kappa.rs`)."""
        kg, kx = self.knots_g.to(g.dtype), self.knots_x.to(g.dtype)
        i = torch.searchsorted(kg, g.contiguous()).clamp(1, kg.numel() - 1)
        g0, g1, x0, x1 = kg[i - 1], kg[i], kx[i - 1], kx[i]
        return x0 + (x1 - x0) / (g1 - g0) * (g - g0)

    def up(self, x, mode=None):
        """Children of every pixel.  'nearest' copies the parent; 'smooth' then
        blurs with [1, 2, 1]^2 / 16 across face edges.  After 'nearest' the
        convolutions see a field that depends on the child's position in its
        parent, and the trained generator adds a z-independent pattern,
        periodic on the pixel grid (7.7 % of the variance at nside 64)."""
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if (mode or self.up_mode) == "nearest":
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
        u = None
        if self.latent:
            b = z.shape[0] // 12
            u = z.reshape(b, -1).mean(1) * math.sqrt(12 * self.nside**2)
            u = u.repeat_interleave(12)[:, None, None, None]
        x = None
        h = None
        for k in range(self.levels):
            inp = zs[0] if k == 0 else torch.cat([self.up(h), zs[k]], 1)
            if u is not None:
                inp = torch.cat([inp, u.expand(-1, 1, inp.shape[2], inp.shape[3])], 1)
            h = self.blocks[k](inp)
            y = self.heads[k](h)
            x = y if x is None else self.up(x, self.map_up) + y
        return self.to_nested(x)


def load(path, device):
    """Generator of a `train.py` checkpoint (older checkpoints: map path
    up-sampled like the features)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    gen = Generator(ck["nside"], ck["nside0"], ck["channels"], up=ck.get("up", "nearest"),
                    map_up=ck.get("map_up", ck.get("up", "nearest")), latent=ck.get("latent", False)).to(device)
    st = ck["state"]
    if "knots_g" in st:
        gen.knots_g, gen.knots_x = st["knots_g"].clone(), st["knots_x"].clone()
    gen.load_state_dict(st, strict=False)
    return gen.eval(), ck
