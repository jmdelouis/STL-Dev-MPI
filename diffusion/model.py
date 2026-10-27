"""Conditional denoiser of one pyramid level: a small U-Net on face images
with HEALPix face padding, wrapped in the EDM preconditioning (Karras et
al. 2022).

Input of the U-Net: the noisy whitened Haar details `[B, 3, n, n]` (scaled
by `c_in`) and the whitened coarse map `[B, 1, n, n]` on the same parent
grid; output: the `F` of EDM, `D = c_skip x + c_out F`.

Padding: every 3 x 3 convolution pads its input by one pixel, either with
zeros (`mode = "zeros"`: training crops, and the face tiles of `FaceTiles`,
used for sampling) or with the neighbouring faces (`mode = "healpix"`: the
batch is then `S * 12` face images, the 12 faces of every sample
contiguous and in face order).  The "healpix" mode pads oriented channels
and features across the edges between polar faces without turning them
(the frames of these faces differ by 90 degrees or a flip): it is wrong
there, and kept only to reproduce the first runs.  Normalisations are per
pixel (over channels only), so the network is local and
translation-equivariant: what it learns on crops applies to whole faces.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import healpix_faces as hf


def amp_dtype(device):
    """Autocast dtype: bf16 from Ampere on, fp16 before (the T4s of garoupe)."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        return torch.float16
    return torch.bfloat16


class Padding:
    """Shared padding mode of a network and its index tables per face size."""

    def __init__(self):
        self.mode = "zeros"
        self.tables = {}

    def __call__(self, x):
        if self.mode == "zeros":
            return F.pad(x, (1, 1, 1, 1))
        b12, c, n, _ = x.shape
        key = (n, x.device)
        if key not in self.tables:
            self.tables[key] = torch.as_tensor(hf.pad_flat(n, 1).ravel(), device=x.device)
        idx = self.tables[key]
        b = b12 // 12
        y = x.reshape(b, 12, c, n * n).permute(0, 2, 1, 3).reshape(b, c, 12 * n * n)
        y = y[:, :, idx].reshape(b, c, 12, n + 2, n + 2)
        return y.permute(0, 2, 1, 3, 4).reshape(b12, c, n + 2, n + 2)


class Conv3(nn.Module):
    def __init__(self, pad, cin, cout):
        super().__init__()
        self.pad = pad
        self.conv = nn.Conv2d(cin, cout, 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class PixelNorm(nn.Module):
    """Layer norm over the channels of every pixel."""

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) * torch.rsqrt(var + 1e-5)
        return (x * self.weight[:, None, None] + self.bias[:, None, None]).to(dtype)


class ResBlock(nn.Module):
    def __init__(self, pad, cin, cout, emb, dropout=0.0):
        super().__init__()
        self.norm1 = PixelNorm(cin)
        self.conv1 = Conv3(pad, cin, cout)
        self.film = nn.Linear(emb, 2 * cout)
        self.norm2 = PixelNorm(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = Conv3(pad, cout, cout)
        nn.init.zeros_(self.conv2.conv.weight)
        nn.init.zeros_(self.conv2.conv.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return (self.skip(x) + h) / math.sqrt(2)


class UNet(nn.Module):
    def __init__(self, channels=(64, 128, 128), nres=2, cin=4, cout=3, emb=128, dropout=0.0):
        super().__init__()
        self.pad = Padding()
        self.freqs = nn.Parameter(torch.exp(torch.linspace(0, math.log(1000), emb // 4)), requires_grad=False)
        self.emb = nn.Sequential(nn.Linear(emb // 2, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = Conv3(self.pad, cin, channels[0])
        self.down = nn.ModuleList()
        skips = [channels[0]]
        c = channels[0]
        for k, ck in enumerate(channels):
            if k > 0:
                skips.append(c)  # pooled input of the level
            blocks = nn.ModuleList()
            for _ in range(nres):
                blocks.append(ResBlock(self.pad, c, ck, emb, dropout))
                c = ck
                skips.append(c)
            self.down.append(blocks)
        self.mid = nn.ModuleList([ResBlock(self.pad, c, c, emb, dropout), ResBlock(self.pad, c, c, emb, dropout)])
        self.up = nn.ModuleList()
        for k, ck in reversed(list(enumerate(channels))):
            blocks = nn.ModuleList()
            for _ in range(nres + 1):
                blocks.append(ResBlock(self.pad, c + skips.pop(), ck, emb, dropout))
                c = ck
            self.up.append(blocks)
        self.out_norm = PixelNorm(c)
        self.out = Conv3(self.pad, c, cout)
        nn.init.zeros_(self.out.conv.weight)
        nn.init.zeros_(self.out.conv.bias)
        self.depth = len(channels) - 1

    def set_padding(self, mode):
        assert mode in ("zeros", "healpix")
        self.pad.mode = mode

    def forward(self, x, c_noise):
        f = c_noise[:, None] * self.freqs[None]
        emb = self.emb(torch.cat([torch.sin(f), torch.cos(f)], dim=1))
        h = self.inp(x)
        hs = [h]
        for k, blocks in enumerate(self.down):
            if k > 0:
                h = F.avg_pool2d(h, 2)
                hs.append(h)
            for blk in blocks:
                h = blk(h, emb)
                hs.append(h)
        for blk in self.mid:
            h = blk(h, emb)
        for k, blocks in enumerate(self.up):
            if k > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            for blk in blocks:
                h = blk(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(F.silu(self.out_norm(h)))


class Denoiser(nn.Module):
    """EDM preconditioning around the U-Net, sigma_data = 1 (whitened details)."""

    def __init__(self, **unet):
        super().__init__()
        self.net = UNet(**unet)

    def forward(self, x, sigma, cond):
        s = sigma.reshape(-1, 1, 1, 1)
        c_skip = 1 / (s**2 + 1)
        c_out = s / torch.sqrt(s**2 + 1)
        c_in = 1 / torch.sqrt(s**2 + 1)
        f = self.net(torch.cat([c_in * x, cond], dim=1), torch.log(sigma.reshape(-1)) / 4)
        return c_skip * x + c_out * f.float()

    def loss(self, d, cond, mask=None, p_mean=-1.2, p_std=1.2):
        """EDM loss: lambda(sigma) |D(d + n) - d|^2, ln sigma ~ N(p_mean, p_std)."""
        sigma = torch.exp(p_mean + p_std * torch.randn(d.shape[0], device=d.device))
        s = sigma.reshape(-1, 1, 1, 1)
        x = d + s * torch.randn_like(d)
        weight = (s**2 + 1) / s**2
        err = weight * (self(x, sigma, cond) - d) ** 2
        if mask is not None:
            return (err * mask).sum() / (mask.sum() * d.shape[0] * d.shape[1])
        return err.mean()


class FaceTiles(nn.Module):
    """A denoiser run face by face on full skies `[S * 12, 3, n, n]`: every
    face is padded by `halo` pixels of its neighbours (`pad_flat`), the Haar
    channels of the halo re-oriented to the face's frame (`orient_haar`: the
    frames of neighbouring polar faces are turned by 90 degrees or flipped),
    the network sees the tile with zero padding at the outer edge (like the
    training crops, which were cut from the padded scalar map) and only the
    face is kept.  Unlike the 12-face HEALPix padding of the network's
    features, every layer works in one frame.  Activations of one face
    instead of 12 (nside 2048: ~2 GB instead of ~24 GB for the finest level)."""

    def __init__(self, denoiser, halo):
        super().__init__()
        self.denoiser = denoiser
        self.halo = halo
        self.tables = {}

    def forward(self, x, sigma, cond):
        b12, c, n, _ = x.shape
        s, h = b12 // 12, min(self.halo, n)
        key = (n, x.device)
        if key not in self.tables:
            ids, bits = hf.pad_flat(n, h, with_bits=True)
            self.tables[key] = (torch.as_tensor(ids.reshape(12, -1), device=x.device),
                                torch.as_tensor(bits.reshape(12, -1), device=x.device))
        idx, bits = self.tables[key]

        def flat(t):
            return t.reshape(s, 12, t.shape[1], n * n).permute(0, 2, 1, 3).reshape(s, t.shape[1], 12 * n * n)

        xf, cf = flat(x), flat(cond)
        sig = sigma.reshape(s, 12)[:, 0]
        out = torch.empty((s, 12, c, n, n), dtype=x.dtype, device=x.device)
        self.denoiser.net.set_padding("zeros")
        m = n + 2 * h
        for f in range(12):
            xi = hf.orient_haar(xf[:, :, idx[f]], bits[f]).reshape(s, c, m, m)
            ci = cf[:, :, idx[f]].reshape(s, cond.shape[1], m, m)
            out[:, f] = self.denoiser(xi, sig, ci)[..., h:h + n, h:h + n]
        return out.reshape(b12, c, n, n)


@torch.no_grad()
def heun_sample(denoiser, cond, steps=32, sigma_max=80.0, sigma_min=0.002, rho=7.0, generator=None,
                s_churn=0.0, s_noise=1.0):
    """EDM sampler (Karras et al. 2022, algorithm 2) for the details given `cond`."""
    shape = (cond.shape[0], 3) + cond.shape[2:]
    i = torch.arange(steps, device=cond.device, dtype=torch.float64)
    t = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros(1, device=cond.device, dtype=torch.float64)])
    x = torch.randn(shape, device=cond.device, generator=generator) * t[0].float()
    for k in range(steps):
        tc, tn = t[k], t[k + 1]
        gamma = min(s_churn / steps, math.sqrt(2) - 1) if s_churn > 0 else 0.0
        th = tc * (1 + gamma)
        if gamma > 0:
            x = x + torch.sqrt(th**2 - tc**2).float() * s_noise * torch.randn(shape, device=cond.device,
                                                                                 generator=generator)
        sig = torch.full((shape[0],), float(th), device=cond.device)
        dx = (x - denoiser(x, sig, cond)) / float(th)
        xn = x + float(tn - th) * dx
        if tn > 0:
            sig = torch.full((shape[0],), float(tn), device=cond.device)
            dn = (xn - denoiser(xn, sig, cond)) / float(tn)
            xn = x + float(tn - th) * 0.5 * (dx + dn)
        x = xn
    return x
