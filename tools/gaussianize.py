"""Rank-based Gaussian anamorphosis of a map, and its inverse (port of
FOSCAT scat_cov.to_gaussian / from_gaussian).

    python3 tools/gaussianize.py forward IN.npy OUT_gauss.npy TABLE.npz
    python3 tools/gaussianize.py inverse TABLE.npz IN_gauss.npy OUT.npy
    python3 tools/gaussianize.py match TABLE.npz IN.npy OUT.npy

forward: the pixel of rank r (0-based, N pixels) becomes
norm.ppf((r + 0.5) / N); TABLE.npz keeps the sorted Gaussian values and
the sorted original values.  inverse: values are clipped to the Gaussian
range (shrunk by 1e-7 of its width, as FOSCAT) and mapped back with the
cubic interpolation of (gaussian, original).  The synthesis is run on the
Gaussian map, then inverted to get the non-Gaussian field.
"""

import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


def forward(src, dst, table):
    x = np.load(src).astype(np.float64).ravel()
    idx = np.argsort(x, kind="stable")
    n = idx.shape[0]
    g_sorted = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    g = np.empty_like(x)
    g[idx] = g_sorted
    np.save(dst, g)
    np.savez(table, gaussian=g_sorted, values=x[idx])
    print(f"{src}: {n} pixels, min {x[idx[0]]:.4e} max {x[idx[-1]]:.4e} -> gaussian [{g_sorted[0]:.3f}, {g_sorted[-1]:.3f}]")


def inverse(table, src, dst):
    t = np.load(table)
    g_sorted, v_sorted = t["gaussian"], t["values"]
    lo, hi = g_sorted[0], g_sorted[-1]
    eps = 1e-7 * (hi - lo)
    g = np.load(src).astype(np.float64)
    clipped = np.count_nonzero((g < lo + eps) | (g > hi - eps))
    f = interp1d(g_sorted, v_sorted, kind="cubic")
    x = f(np.clip(g, lo + eps, hi - eps))
    np.save(dst, x)
    print(f"{src}: {g.size} pixels, {clipped} clipped, std {x.std():.4e} max {x.max():.4e}")


def match(table, src, dst):
    """Rank-based histogram matching: the pixel of rank r of IN takes the
    r-th sorted value of the original map in TABLE (same pixel count)."""
    v_sorted = np.load(table)["values"]
    x = np.load(src).astype(np.float64).ravel()
    assert x.size == v_sorted.size, f"{src}: {x.size} pixels, table has {v_sorted.size}"
    y = np.empty_like(x)
    y[np.argsort(x, kind="stable")] = v_sorted
    np.save(dst, y)
    print(f"{src}: matched to the table histogram, std {y.std():.4e} max {y.max():.4e}")


if __name__ == "__main__":
    mode, *a = sys.argv[1:]
    {"forward": forward, "inverse": inverse, "match": match}[mode](*a)
