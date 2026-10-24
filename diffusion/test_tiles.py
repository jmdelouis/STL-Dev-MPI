"""Consistency of the face-by-face denoiser (FaceTiles) across face edges,
on a trained model and the target's own coarse map.

    python diffusion/test_tiles.py TARGET.npy MODEL_DIR LEVEL [HALO]

The pixels within `band` of the middle half of a face edge (away from the
face corners, whose tiles differ), seen in the halo of the face's
tile, are denoised there and also as part of their own face; re-oriented
to the same frame, the two must agree up to the limited context of the
tiles.  Reported per kind of edge (equatorial faces, whose neighbours share
their frame, and polar faces, whose polar neighbours are turned), with and
without the re-orientation of the Haar channels.
"""

import sys

import numpy as np
import torch

import healpix_faces as hf
from model import FaceTiles
from sample import load_level


def own_and_halo(model, x, sigma, cond, h, orient):
    """Output of every face's tile on its whole tile, `[12, 3, m, m]`."""
    n = x.shape[-1]
    m = n + 2 * h
    ids, bits = hf.pad_flat(n, h, with_bits=True)
    idx = torch.as_tensor(ids.reshape(12, -1), device=x.device)
    b = torch.as_tensor(bits.reshape(12, -1), device=x.device)
    xf = x.permute(1, 0, 2, 3).reshape(1, 3, -1)
    cf = cond.permute(1, 0, 2, 3).reshape(1, 1, -1)
    model.net.set_padding("zeros")
    out = []
    for f in range(12):
        xi = xf[:, :, idx[f]]
        xi = hf.orient_haar(xi, b[f]) if orient else xi
        out.append(model(xi.reshape(1, 3, m, m), sigma[:1], cf[:, :, idx[f]].reshape(1, 1, m, m))[0])
    return torch.stack(out), idx, b


if __name__ == "__main__":
    target, models, level, *rest = sys.argv[1:]
    level, h = int(level), int(rest[0]) if rest else 48
    band = 6
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_level(models, level, dev)
    x = np.load(target).astype(np.float64)
    while hf.nside_of(x.size) > level // 2:
        x = hf.down(x)
    cond = torch.tensor(hf.to_faces(x) / ck["cond_std"], dtype=torch.float32, device=dev)[:, None]
    n = cond.shape[-1]
    noise = torch.randn((12, 3, n, n), device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    r = np.arange(-h, n + h)
    halo_band = ((r >= -band) & (r < 0)) | ((r >= n) & (r < n + band))
    inside = (r >= n // 4) & (r < 3 * n // 4)  # middle half of every edge, away from the face corners
    sel = (halo_band[:, None] & inside[None, :]) | (inside[:, None] & halo_band[None, :])  # edges, no corners
    sel = torch.as_tensor(sel.ravel(), device=dev)
    with torch.no_grad():
        for s in (0.05, 0.5, 5.0):
            sigma = torch.full((12,), s, device=dev)
            own = FaceTiles(model, h)(s * noise, sigma, cond)  # [12, 3, n, n], re-oriented halos
            ownf = own.permute(1, 0, 2, 3).reshape(3, -1)
            norm = own.pow(2).mean().sqrt()
            for orient in (True, False):
                tiles, idx, b = own_and_halo(model, s * noise, sigma, cond, h, orient)
                err = {"equatorial": [], "polar": []}
                for f in range(12):
                    halo_out = tiles[f].reshape(3, -1)[:, sel]
                    src = hf.orient_haar(ownf[:, idx[f][sel]], b[f][sel])
                    err["equatorial" if 4 <= f < 8 else "polar"].append((halo_out - src).pow(2).mean())
                print(f"sigma {s:4.2f} {'re-oriented' if orient else 'raw halo   '}: "
                      + ", ".join(f"{k} faces {float(torch.stack(v).mean().sqrt() / norm):.2e}" for k, v in err.items()))
