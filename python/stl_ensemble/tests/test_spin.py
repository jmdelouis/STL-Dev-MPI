#!/usr/bin/env python3
"""Spin-2 statistics through the native bridge (NativeEngine(..., spin=2));
requires a built libstl_bridge (STL_NATIVE_LIBRARY). The gradients against
finite differences in both pyramid modes, level 0 identical in both, the
local E/B operator against the exact one on a smooth pure-E field (healpy),
the eroded masks; with STL_TEST_GPU=1, the GPU backend against the CPU.
Runs under pytest or as a script."""
from pathlib import Path
import math
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeEngine

try:
    import healpy as hp
except ImportError:  # pragma: no cover
    hp = None

NSIDE = 16
J = 3


def _maps(batch, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch, 2, 12 * NSIDE**2))


def _check_fd(e, x, rng, trials=2):
    v, _ = e.forward(x)
    c = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
    g = e.vjp(c)
    for t in range(trials):
        d = rng.normal(size=x.shape)
        h = 1e-5

        def f(y, gen):
            return float(np.real(np.sum(np.conj(c) * e.forward(y, generation=gen)[0])))

        fd = (f(x + h * d, 2 * t + 1) - f(x - h * d, 2 * t + 2)) / (2 * h)
        an = float(np.sum(g * d))
        assert abs(fd - an) <= 1e-6 * max(abs(an), 1e-6), (fd, an)


def test_vjp():
    rng = np.random.default_rng(1)
    for mode in ("transported", "scalar"):
        e = NativeEngine(NSIDE, J, batch=2, spin=2, spin_pyramid=mode)
        _check_fd(e, _maps(2), rng)


def test_level0_shared():
    x = _maps(1, 3)
    a = NativeEngine(NSIDE, J, spin=2)
    b = NativeEngine(NSIDE, J, spin=2, spin_pyramid="scalar")
    va, vb = a.forward(x)[0], b.forward(x)[0]
    assert va.shape == (1, 2, a.ncoeff)
    lev0 = [i for i, c in enumerate(a.plan) if c["native_dg"] == 0 and c["family"] in ("mean", "var", "S1", "S2")]
    assert np.allclose(va[..., lev0], vb[..., lev0], rtol=1e-12, atol=0)


def test_smooth_pure_e():
    if hp is None:
        print("healpy missing: skipped")
        return
    nside, lmax = 32, 95
    l = np.arange(lmax + 1.0)
    cl = np.zeros(lmax + 1)
    cl[2:] = l[2:] ** -2.4 * np.exp(-(l[2:] / 12.0) ** 2)
    np.random.seed(1)
    alms = hp.synalm([0 * cl, cl, 0 * cl, 0 * cl], lmax=lmax, new=True)
    _, q, u = hp.alm2map(alms, nside, lmax=lmax, pol=True)
    qu = np.stack([hp.reorder(q, r2n=True), hp.reorder(u, r2n=True)])
    e = NativeEngine(nside, 3, spin=2)
    chi = e.spin_fields(qu[None], 0)[1][0]
    f = np.zeros(lmax + 1)
    f[2:] = np.sqrt((l[2:] + 2) * (l[2:] + 1) * l[2:] * (l[2:] - 1)) * 4 * math.pi / (12 * nside**2)
    ref = hp.reorder(hp.alm2map(hp.almxfl(alms[1], f), nside, lmax=lmax), r2n=True)
    err = np.linalg.norm(chi[0] - ref) / np.linalg.norm(ref)
    leak = np.linalg.norm(chi[1]) / np.linalg.norm(chi[0])
    print(f"smooth pure E: chi_E error {err:.2e}, B/E {leak:.2e}")
    assert err < 0.03 and leak < 0.01, (err, leak)


def test_masks_eroded():
    e = NativeEngine(NSIDE, J, spin=2)
    w = np.ones(12 * NSIDE**2)
    w[:200] = 0.0
    e.set_masks(w[None])
    counts = e.native_pixels()
    s1 = [i for i, c in enumerate(e.plan) if c["family"] == "S1" and c["l1"] == 0]
    full = [12 * (NSIDE >> e.plan[i]["j1"])**2 for i in s1]
    assert all(counts[0, i] < n for i, n in zip(s1, full))
    x = _maps(1, 4)
    holed = x * w
    # chi of the pixels kept by the eroded mask does not read the hole.
    for level in range(J):
        _, a = e.spin_fields(x, level)
        _, b = e.spin_fields(holed, level)
        changed = np.abs(a - b).max((0, 1)) > 0
        assert 0 < changed.sum() <= full[level] - counts[0, s1[level]]


def test_gpu_matches_cpu():
    if os.environ.get("STL_TEST_GPU") != "1":
        print("STL_TEST_GPU unset: skipped")
        return
    x = _maps(2, 5)
    for mode in ("transported", "scalar"):
        c = NativeEngine(NSIDE, J, batch=2, spin=2, spin_pyramid=mode)
        g = NativeEngine(NSIDE, J, batch=2, spin=2, spin_pyramid=mode, backend="gpu")
        vc, vg = c.forward(x)[0], g.forward(x)[0]
        assert np.abs(vc - vg).max() <= 1e-12 * np.abs(vc).max(), np.abs(vc - vg).max()
        cot = np.random.default_rng(6).normal(size=vc.shape) + 0j
        gc, gg = c.vjp(cot), g.vjp(cot)
        assert np.abs(gc - gg).max() <= 1e-11 * np.abs(gc).max(), np.abs(gc - gg).max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
