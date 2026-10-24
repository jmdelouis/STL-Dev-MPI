"""Checks of the sphere-native convolution and U-Net.

    python diffusion/test_sphere_conv.py
"""

import time

import healpy as hp
import numpy as np
import torch

import sphere_conv as sc

if __name__ == "__main__":
    torch.manual_seed(0)
    nside, cell_nside = 64, 4  # cells of 16 x 16 fine pixels
    cells = [0, 100, 191]  # face 0 (pole corner), an equatorial cell, the last cell
    regions = sc.region_ids(cells, cell_nside, nside, sub=4, rings=2)
    for (ids, central), c in zip(regions, cells):
        assert np.all(np.diff(ids) > 0)
        assert central.sum() == (nside // cell_nside) ** 2
        assert np.all(ids[central] // (nside // cell_nside) ** 2 == c)
    print("regions:", [len(r[0]) for r in regions], "pixels; central cells complete")

    pyr = sc.Pyramid([r[0] for r in regions], nside, depth=2, ksz=3, device="cpu")
    for lv in pyr.levels:
        ones = torch.sparse.mm(lv.S, torch.ones(lv.n, 1))
        assert torch.allclose(ones, torch.ones_like(ones), atol=1e-6)
    print("stencil weights sum to 1 on every tap, levels:",
          [lv.n for lv in pyr.levels])

    # A smooth field (low multipoles) seen through the centre tap, which
    # falls on the pixel centre: the interpolation returns the pixel itself.
    npix = 12 * nside * nside
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    field = np.cos(theta) + 0.3 * np.sin(theta) * np.cos(phi)
    lv = pyr.levels[0]
    x = torch.tensor(field[lv.ids.numpy()], dtype=torch.float32)[:, None]
    taps = torch.sparse.mm(lv.S, x).view(lv.n, 9)
    centre = taps[:, 4]
    print("centre tap vs pixel value, max error:", float((centre - x[:, 0]).abs().max()))

    model = sc.SphereDenoiser(channels=(16, 32, 32), nres=1)
    d = sc.project(torch.randn(lv.n, 1))
    cond = torch.randn(lv.n, 1)
    mask = torch.ones(lv.n, dtype=torch.bool)
    loss = model.loss(d, cond, pyr, mask)
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    print(f"loss {float(loss):.3f}, gradient norm sum {g:.3e}")
    out = model(d, torch.ones(len(regions)), cond, pyr)
    assert torch.allclose(out.view(-1, 4).sum(1), torch.zeros(lv.n // 4), atol=1e-5)
    print("denoiser output has zero mean over every sibling group")

    # Geometry cost at nside 256, regions of a 32 x 32 cell plus 2 rings of 8 x 8 sub-cells.
    t0 = time.time()
    regs = sc.region_ids(np.arange(16) * 37, 8, 256, sub=4, rings=2)
    t1 = time.time()
    pyr = sc.Pyramid([r[0] for r in regs], 256, depth=2, ksz=3, device="cpu")
    t2 = time.time()
    print(f"nside 256, 16 regions of {len(regs[0][0])} pixels: regions {t1 - t0:.2f} s, geometry {t2 - t1:.2f} s (CPU)")
