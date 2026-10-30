"""Sphere-native convolutions on regions of the HEALPix NESTED grid, with
the stencils of the STL engine, and a U-Net built on them.

Stencil (`stencil`): for every pixel, the `k x k` taps of the STL kernel in
the tangent plane (pixel spacing `sqrt(4 pi / npix)`), rotated to the pixel
and its local north (gauge "cosmo", one orientation), each tap the bilinear
interpolation of its 4 nearest pixels: `stl_torch_geometry`, the torch port
of healpix-analyse used by STL-Dev.  No base faces, no frames: the same
convolution everywhere on the sphere, except the gauge flip at the poles.

Regions (`region_ids`): a central cell of the NESTED quadtree (side `cell`
fine pixels) and a halo of `rings` rings of sub-cells (side `sub` fine
pixels) around it, found with the HEALPix neighbours of the sub-cells
across faces.  A region is a set of complete sub-cells, so its pyramid
(NESTED parents, 4 consecutive sorted ids) is exact down to the sub-cell
size; the U-Net works on it and only the central cell is kept.  Taps that
fall outside the region are dropped and the 4 weights of the tap
renormalised (a tap with none of its pixels in the region reads the pixel
itself), as the STL engine's `bind_support` does on partial grids.

Tensors are flat, `[pixels of all the regions of a batch, channels]`; the
convolution is a sparse gather (`S`, 4 entries per pixel and tap) followed
by a dense matmul with the `[cout, k^2 cin]` kernel.
"""

import math

import healpy as hp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stl_torch_geometry as tg


def stencil(ids, nside, ksz):
    """NESTED ids `[M]` -> (`[M, ksz^2, 4]` NESTED pixels, `[M, ksz^2, 4]`
    bilinear weights) of the STL stencil, gauge cosmo, one orientation."""
    order = int(round(math.log2(nside)))
    ids = torch.as_tensor(ids, dtype=torch.long)
    ix, iy, face = tg.nest2xyf(ids, order)
    theta, phi = tg.xyf2ang(ix, iy, face, nside)
    rot = tg.rotation_matrices(theta.to(torch.float64), phi.to(torch.float64), 1, "cosmo", torch.float64)[:, 0]
    vec = tg.local_kernel_grid(ksz, nside, ids.device, torch.float64)
    taps = torch.einsum("mij,pj->mpi", rot, vec).reshape(-1, 3)
    nb, w = tg.interp_weights_nested(taps, order)
    m, p = ids.numel(), ksz * ksz
    return nb.view(m, p, 4), w.to(torch.float64).view(m, p, 4)


def region_ids(cells, cell_nside, nside, sub=4, rings=1):
    """Fine NESTED ids (sorted) and central masks of the regions of the
    cells `cells` (NESTED ids at `cell_nside`): every cell split in `sub x
    sub` sub-cells, dilated by `rings` rings of sub-cells (HEALPix
    neighbours, across faces), then all the descendants at `nside`."""
    sub_nside = cell_nside * sub
    k = (nside // sub_nside) ** 2
    out = []
    for c in np.atleast_1d(cells):
        core = int(c) * sub * sub + np.arange(sub * sub)
        cur = core
        for _ in range(rings):
            nb = hp.get_all_neighbours(sub_nside, cur, nest=True).ravel()
            cur = np.union1d(cur, nb[nb >= 0])
        fine = (cur[:, None] * k + np.arange(k)[None, :]).ravel()
        central = np.isin(cur, core).repeat(k)
        out.append((fine, central))
    return out


def bind(ids, nside, ksz, device):
    """Stencil of the sorted NESTED ids `[m]` of one region, bound to the
    region: (`[m, ksz^2, 4]` positions in `ids`, weights renormalised over
    the pixels present; a tap with none present reads the pixel itself)."""
    ids_t = torch.as_tensor(ids, device=device)
    nb, w = stencil(ids_t, nside, ksz)
    pos = torch.searchsorted(ids_t, nb.reshape(-1)).view_as(nb).clamp(max=len(ids) - 1)
    present = ids_t[pos] == nb
    w = w * present
    tot = w.sum(-1, keepdim=True)
    empty = tot == 0
    own = torch.arange(len(ids), device=device)[:, None, None].expand_as(pos)
    pos = torch.where(empty, own, torch.where(present, pos, own))
    w = torch.where(empty, torch.full_like(w, 0.25), w / tot.clamp_min(1e-300))
    return pos, w.float()


class RegionGeometry:
    """Fine ids, central mask and the bound stencils of every U-Net level of
    one region (computed once, reused by every batch that holds it)."""

    def __init__(self, ids, central, nside, depth, ksz, device):
        self.central = torch.as_tensor(central, device=device)
        self.ids, self.levels = [], []
        for k in range(depth + 1):
            assert len(ids) % 4 == 0 or k == depth
            self.ids.append(torch.as_tensor(ids, device=device))
            self.levels.append(bind(ids, nside >> k, ksz, device))
            if k < depth:
                assert np.all(ids[::4] % 4 == 0), "region not made of complete sibling groups"
                ids = ids[::4] // 4


class Level:
    """Geometry of one resolution for a batch of regions (concatenated):
    the gather matrix `S` `[n p, n]` and its transpose."""

    def __init__(self, parts, ids, ksz):
        """parts: per region (pos, w) from `bind`; ids: per region ids."""
        device = parts[0][0].device
        self.p = ksz * ksz
        self.sizes = [len(i) for i in ids]
        self.n = sum(self.sizes)
        offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.region = torch.as_tensor(np.repeat(np.arange(len(ids)), self.sizes), device=device)
        self.ids = torch.cat(list(ids))
        pos = torch.cat([p + int(o) for (p, _), o in zip(parts, offsets)])
        w = torch.cat([w for _, w in parts])
        rows = torch.arange(self.n * self.p, device=device)[:, None].expand(-1, 4).reshape(-1)
        coo = torch.sparse_coo_tensor(torch.stack([rows, pos.reshape(-1)]), w.reshape(-1),
                                      (self.n * self.p, self.n)).coalesce()
        self.S = coo.to_sparse_csr()
        self.St = coo.t().coalesce().to_sparse_csr()  # for the backward pass


class Pyramid:
    """Levels of a batch of regions, finest first, for a U-Net of `depth`
    poolings; from region ids (geometry computed here) or from cached
    `RegionGeometry` objects."""

    def __init__(self, regions, nside=None, depth=None, ksz=3, device="cpu"):
        if not isinstance(regions[0], RegionGeometry):
            regions = [RegionGeometry(r, np.ones(len(r), bool), nside, depth, ksz, device) for r in regions]
        self.regions = regions
        self.levels = [Level([r.levels[k] for r in regions], [r.ids[k] for r in regions], ksz)
                       for k in range(len(regions[0].levels))]
        self.central = torch.cat([r.central for r in regions])


class _Gather(torch.autograd.Function):
    """`S @ x` with the transpose precomputed for the backward pass."""

    @staticmethod
    def forward(ctx, x, S, St):
        ctx.St = St
        return torch.sparse.mm(S, x)

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.St, g.contiguous()), None, None


class SphereConv(nn.Module):
    def __init__(self, cin, cout, ksz=3):
        super().__init__()
        p = ksz * ksz
        self.weight = nn.Parameter(torch.empty(cout, p * cin))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.p, self.cin = p, cin

    def forward(self, x, level):
        # The sparse gather in fp32 outside autocast (cuSPARSE has no
        # fp32 x bf16/fp16 product), the dense part in the autocast dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            taps = _Gather.apply(x.float(), level.S, level.St)  # [n * p, cin]
        taps = taps.view(level.n, self.p * self.cin).to(x.dtype)
        return F.linear(taps, self.weight.to(x.dtype), self.bias.to(x.dtype))


class PixelNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = (x - x.mean(1, keepdim=True)) * torch.rsqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        return (x * self.weight + self.bias).to(dtype)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, ksz, dropout=0.0):
        super().__init__()
        self.norm1 = PixelNorm(cin)
        self.conv1 = SphereConv(cin, cout, ksz)
        self.film = nn.Linear(emb, 2 * cout)
        self.norm2 = PixelNorm(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = SphereConv(cout, cout, ksz)
        nn.init.zeros_(self.conv2.weight)
        self.skip = nn.Linear(cin, cout) if cin != cout else nn.Identity()

    def forward(self, x, emb, level):
        h = self.conv1(F.silu(self.norm1(x)), level)
        scale, shift = self.film(emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)), level)
        return (self.skip(x) + h) / math.sqrt(2)


class SphereUNet(nn.Module):
    """U-Net on flat region tensors; pooling is the NESTED parent mean."""

    def __init__(self, channels=(64, 128, 128), nres=2, cin=2, cout=1, emb=128, ksz=3, dropout=0.0):
        super().__init__()
        self.ksz, self.depth = ksz, len(channels) - 1
        self.freqs = nn.Parameter(torch.exp(torch.linspace(0, math.log(1000), emb // 4)), requires_grad=False)
        self.emb = nn.Sequential(nn.Linear(emb // 2, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = SphereConv(cin, channels[0], ksz)
        self.down, skips, c = nn.ModuleList(), [channels[0]], channels[0]
        for k, ck in enumerate(channels):
            if k > 0:
                skips.append(c)
            blocks = nn.ModuleList()
            for _ in range(nres):
                blocks.append(ResBlock(c, ck, emb, ksz, dropout))
                c = ck
                skips.append(c)
            self.down.append(blocks)
        self.mid = nn.ModuleList([ResBlock(c, c, emb, ksz, dropout), ResBlock(c, c, emb, ksz, dropout)])
        self.up = nn.ModuleList()
        for k, ck in reversed(list(enumerate(channels))):
            blocks = nn.ModuleList()
            for _ in range(nres + 1):
                blocks.append(ResBlock(c + skips.pop(), ck, emb, ksz, dropout))
                c = ck
            self.up.append(blocks)
        self.out_norm = PixelNorm(c)
        self.out = SphereConv(c, cout, ksz)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, c_noise, pyr):
        """x `[n0, cin]` at the finest level, c_noise `[regions]`."""
        f = c_noise[:, None] * self.freqs[None]
        emb_r = self.emb(torch.cat([torch.sin(f), torch.cos(f)], dim=1))
        embs = [emb_r[lv.region] for lv in pyr.levels]
        h = self.inp(x, pyr.levels[0])
        hs = [h]
        for k, blocks in enumerate(self.down):
            if k > 0:
                h = h.view(-1, 4, h.shape[1]).mean(1)
                hs.append(h)
            for blk in blocks:
                h = blk(h, embs[k], pyr.levels[k])
                hs.append(h)
        for blk in self.mid:
            h = blk(h, embs[self.depth], pyr.levels[self.depth])
        for j, blocks in enumerate(self.up):
            k = self.depth - j
            if j > 0:
                h = h.repeat_interleave(4, dim=0)
            for blk in blocks:
                h = blk(torch.cat([h, hs.pop()], dim=1), embs[k], pyr.levels[k])
        return self.out(F.silu(self.out_norm(h)), pyr.levels[0])


def project(d):
    """Remove the mean of every group of 4 NESTED siblings: `[n, ...]`, n a multiple of 4, sorted ids."""
    g = d.view(-1, 4, *d.shape[1:])
    return (g - g.mean(1, keepdim=True)).view_as(d)


class SphereDenoiser(nn.Module):
    """EDM denoiser of the whitened detail field `d = x_n - up(x_{n+1})`
    (scalar, fine level, zero mean over every group of siblings) given the
    whitened coarse map seen at the fine level; sigma_data = 1."""

    def __init__(self, **unet):
        super().__init__()
        self.net = SphereUNet(**unet)

    def forward(self, x, sigma, cond, pyr):
        """x, cond `[n0, 1]`; sigma `[regions]`."""
        s = sigma[pyr.levels[0].region][:, None]
        c_skip, c_out, c_in = 1 / (s**2 + 1), s / torch.sqrt(s**2 + 1), 1 / torch.sqrt(s**2 + 1)
        f = self.net(torch.cat([c_in * x, cond], dim=1), torch.log(sigma) / 4, pyr)
        return project(c_skip * x + c_out * f.float())

    def loss(self, d, cond, pyr, mask, p_mean=-1.2, p_std=1.2):
        nreg = len(pyr.levels[0].sizes)
        sigma = torch.exp(p_mean + p_std * torch.randn(nreg, device=d.device))
        s = sigma[pyr.levels[0].region][:, None]
        x = d + s * project(torch.randn_like(d)) * math.sqrt(4 / 3)
        err = (s**2 + 1) / s**2 * (self(x, sigma, cond, pyr) - d) ** 2
        return (err[mask]).mean()
