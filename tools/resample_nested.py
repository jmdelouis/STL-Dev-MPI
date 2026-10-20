"""Change the nside of a NESTED map by a factor 2.

    python3 tools/resample_nested.py up IN.npy OUT.npy     # nside -> 2 nside
    python3 tools/resample_nested.py down IN.npy OUT.npy   # nside -> nside / 2

up: bilinear interpolation (healpix_geo, 4 nearest centres) of the map at
the centres of the children, so the new finest level carries almost no
power of its own: synthesising at 2 nside then degrading back keeps the
scales the stencil handles badly out of the result.  down: mean of the 4
children (NESTED: consecutive ids).
"""

import sys

import numpy as np


def nside_of(n):
    nside = int(round(np.sqrt(n / 12)))
    assert 12 * nside * nside == n, f"{n} pixels is not a HEALPix map"
    return nside


def up(src, dst):
    from healpix_geo.nested import bilinear_interpolation, healpix_to_lonlat

    x = np.load(src).astype(np.float64)
    nside = nside_of(x.size)
    depth = int(np.log2(nside))
    ids = np.arange(12 * (2 * nside) ** 2, dtype=np.uint64)
    lon, lat = healpix_to_lonlat(ids, depth + 1, ellipsoid="sphere")
    cells, weights = bilinear_interpolation(np.asarray(lon), np.asarray(lat), depth, ellipsoid="sphere")
    c = np.asarray(getattr(cells, "data", cells)).astype(np.int64)
    w = np.asarray(getattr(weights, "data", weights)).astype(np.float64)
    mask = getattr(cells, "mask", None)
    if mask is not None and np.any(mask):
        w = np.where(np.asarray(mask), 0.0, w)
        c = np.where(np.asarray(mask), 0, c)
    w /= w.sum(axis=1, keepdims=True)
    y = (x[c] * w).sum(axis=1)
    np.save(dst, y)
    back = y.reshape(-1, 4).mean(axis=1)
    print(f"up {nside} -> {2 * nside}: std {x.std():.4e} -> {y.std():.4e}; degrade(up(x)) - x: rms {np.sqrt(np.mean((back - x) ** 2)):.3e}")


def down(src, dst):
    x = np.load(src).astype(np.float64)
    nside = nside_of(x.size)
    y = x.reshape(-1, 4).mean(axis=1)
    np.save(dst, y)
    print(f"down {nside} -> {nside // 2}: std {x.std():.4e} -> {y.std():.4e}")


if __name__ == "__main__":
    mode, *a = sys.argv[1:]
    {"up": up, "down": down}[mode](*a)
