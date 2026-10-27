"""Functional test of tools/minkowski.py: Minkowski functionals of Gaussian
random fields of known C_l against the analytic expectations (Tomita 1986).

    python3 -m pytest tools/test_minkowski.py      (or: python3 tools/test_minkowski.py)

Fields at nside 256 (lmax 767) with a white spectrum smoothed by a Gaussian
beam of 4 pixels, thresholds nu = -3 ... 3 (31 bins).  Tolerances, relative
to the maximum of each functional, are about 2 times the deviations seen
over the seeds (their source is the finite number of independent patches
and the bins of width 0.2 that estimate delta(x - nu)).
"""

import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minkowski import gaussian_mf, minkowski_alm, minkowski_map  # noqa: E402

NSIDE = 256
LMAX = 3 * NSIDE - 1
NU = np.linspace(-3.0, 3.0, 31)
SEEDS = (0, 1, 2, 3)
# Per realization and for the mean of the realizations: V0, V1, V2.
TOL_ONE = (0.005, 0.025, 0.07)
TOL_MEAN = (0.003, 0.012, 0.03)


def spectrum():
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1.0
    return cl * hp.gauss_beam(4 * hp.nside2resol(NSIDE), lmax=LMAX) ** 2


def theory_tau(cl):
    ell = np.arange(cl.size)
    w = (2 * ell + 1) / (4 * np.pi)
    return 0.5 * np.sum(w * ell * (ell + 1) * cl) / np.sum(w * cl)


def realisation(seed, cl):
    np.random.seed(seed)
    return hp.synalm(cl, lmax=LMAX)


def deviations(r, tau):
    g = gaussian_mf(NU, tau)
    return np.array([(r[f"v{k}"] - g[k]) / np.abs(g[k]).max() for k in range(3)])


def test_gaussian_fields_match_tomita():
    cl = spectrum()
    devs = []
    for seed in SEEDS:
        r = minkowski_alm(realisation(seed, cl), NSIDE, LMAX, NU)
        d = deviations(r, r["tau"])
        for k in range(3):
            assert np.abs(d[k]).max() < TOL_ONE[k], f"seed {seed}: V{k} off by {np.abs(d[k]).max():.4f} of its maximum"
        devs.append(d)
    mean = np.mean(devs, 0)
    for k in range(3):
        assert np.abs(mean[k]).max() < TOL_MEAN[k], f"mean of {len(SEEDS)} fields: V{k} off by {np.abs(mean[k]).max():.4f}"


def test_tau_matches_the_spectrum():
    cl = spectrum()
    taus = [minkowski_alm(realisation(seed, cl), NSIDE, LMAX, NU)["tau"] for seed in SEEDS]
    assert abs(np.mean(taus) / theory_tau(cl) - 1) < 0.01, f"tau {np.mean(taus):.3f} against {theory_tau(cl):.3f}"


def test_map_path_matches_alm_path():
    """minkowski_map on the NESTED pixel map (map2alm with iterations) gives the alm path's functionals."""
    cl = spectrum()
    alm = realisation(0, cl)
    m = hp.reorder(hp.alm2map(alm, NSIDE, lmax=LMAX), r2n=True)
    a = minkowski_alm(alm, NSIDE, LMAX, NU)
    b = minkowski_map(m, nest=True, lmax=LMAX, nu=NU)
    for k in range(3):
        scale = np.abs(a[f"v{k}"]).max()
        assert np.abs(a[f"v{k}"] - b[f"v{k}"]).max() < 0.01 * scale, f"V{k}: map path differs"
    assert abs(b["tau"] / a["tau"] - 1) < 1e-3


def test_non_gaussian_field_is_detected():
    """A lognormal field of the same spectrum shape is far outside the tolerances (negative control)."""
    cl = spectrum()
    alm = realisation(0, cl)
    x = hp.alm2map(alm, NSIDE, lmax=LMAX)
    y = np.exp(x / x.std())
    r = minkowski_map(y, nest=False, lmax=LMAX, nu=NU)
    d = deviations(r, r["tau"])
    assert max(np.abs(d[1]).max() / TOL_ONE[1], np.abs(d[2]).max() / TOL_ONE[2]) > 5


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"{name}: ok")
