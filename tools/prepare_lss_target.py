"""NESTED HEALPix target from the CosmoGrid weak-lensing map of Mousset et
al. 2024 (A&A, arXiv:2407.07007), as distributed with s2scat in MW
sampling (L = 256, array 256 x 511):

    https://github.com/astro-informatics/s2scat/raw/main/notebooks/data/target_map_lss.npy

    python3 tools/prepare_lss_target.py target_map_lss.npy NSIDE OUT.npy

Bilinear interpolation of the MW grid at the NESTED pixel centres, the
recipe of the HealpixML Demo_Synthesis notebook (which also wraps the
grid in longitude and latitude).  Pure numpy/scipy: the NESTED centres
use the healpix_base formulas (as stl-core's healpix.rs), so that it runs
where no HEALPix library is installed.
"""

import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def compress_bits(v):
    v = v & 0x5555555555555555
    v = (v | (v >> 1)) & 0x3333333333333333
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFF
    return (v | (v >> 16)) & 0x00000000FFFFFFFF


def pix2ang_nest(nside, pix):
    """Colatitude and longitude (radians) of NESTED pixel centres."""
    order = int(np.log2(nside))
    pix = np.asarray(pix, dtype=np.int64)
    face = pix >> (2 * order)
    local = pix & ((1 << (2 * order)) - 1)
    ix, iy = compress_bits(local), compress_bits(local >> 1)
    jr = JRLL[face] * nside - ix - iy - 1
    north, south = jr < nside, jr > 3 * nside
    nr = np.where(north, jr, np.where(south, 4 * nside - jr, nside))
    kshift = np.where(north | south, 0, (jr - nside) & 1)
    tmp = nr.astype(np.float64) ** 2 * (4.0 / (12.0 * nside * nside))
    theta_cap = np.arctan2(np.sqrt(tmp * (2.0 - tmp)), 1.0 - tmp)
    z_eq = (2.0 * nside - jr) * (2.0 / (3.0 * nside))
    theta = np.where(north, theta_cap, np.where(south, np.pi - theta_cap, np.arccos(np.clip(z_eq, -1, 1))))
    jp = (JPLL[face] * nr + ix - iy + 1 + kshift) // 2
    jp = np.where(jp > 4 * nr, jp - 4 * nr, jp)
    jp = np.where(jp < 1, jp + 4 * nr, jp)
    phi = (jp - (kshift + 1) * 0.5) * (0.5 * np.pi / nr)
    return theta, phi


def mw_to_healpix(im, nside):
    xsize, ysize = im.shape
    im = np.vstack([im, im[0:1, :]])
    im = np.hstack([im, np.concatenate([im[:-1, 0:1], im[-2:-1, 0:1]], 0)])
    interp = RegularGridInterpolator((np.arange(im.shape[0]), np.arange(im.shape[1])), im)
    theta, phi = pix2ang_nest(nside, np.arange(12 * nside**2))
    return interp(np.stack([theta / np.pi * xsize, phi / (2 * np.pi) * ysize], 1))


def main():
    src, nside, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    m = mw_to_healpix(np.load(src).astype(np.float64), nside)
    np.save(out, m)
    s = (m - m.mean()) / m.std()
    print(f"{out}: nside {nside} NESTED, {m.size} pixels, mean {m.mean():+.4e} std {m.std():.4e} "
          f"skew {(s**3).mean():+.3f} kurt {(s**4).mean() - 3:+.3f}")


if __name__ == "__main__":
    main()
