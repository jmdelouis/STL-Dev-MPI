#!/usr/bin/env python3
"""Differentiable anafast through the native bridge (SpectrumEngine);
requires a built libstl_bridge (STL_NATIVE_LIBRARY). The spectra against
healpy.anafast (iter=0) when healpy is installed, the gradients against
finite differences; with STL_TEST_GPU=1, the GPU backend against the CPU.
Runs under pytest or as a script."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import SpectrumEngine

try:
    import healpy as hp
except ImportError:  # pragma: no cover
    hp = None


def _weights(nside):
    """A smooth positive weight and a latitude cut, [2, npix] NESTED."""
    npix = 12 * nside**2
    z = np.cos(hp.pix2ang(nside, np.arange(npix), nest=True)[0]) if hp else None
    if z is None:
        z = np.linspace(-1, 1, npix)
    return np.stack([0.5 + 0.5 * np.tanh(3 * z), (np.abs(z) > 0.3).astype(float)])


def _check_fd(f, x, grad, rng, trials=3, h=1e-4):
    for _ in range(trials):
        d = rng.normal(size=x.shape)
        fd = (f(x + h * d) - f(x - h * d)) / (2 * h)
        an = float(np.sum(grad * d))
        assert abs(fd - an) <= 1e-7 * max(abs(an), 1e-6), (fd, an)


def test_against_healpy():
    if hp is None:
        print("healpy missing: skipped")
        return
    nside, lmax, batch = 32, 63, 2
    npix = 12 * nside**2
    rng = np.random.default_rng(1)
    x = rng.normal(size=(batch, npix))
    y = rng.normal(size=(batch, npix))
    w = _weights(nside)
    with SpectrumEngine(nside, lmax, batch=batch) as auto, SpectrumEngine(nside, lmax, batch=batch, cross=True) as cross:
        cl = auto.forward(x)
        xl = cross.forward_cross(x, y)
        assert cl.shape == xl.shape == (batch, lmax + 1)
        auto.set_masks(w)
        cross.set_masks(w)
        clm = auto.forward(x)
        xlm = cross.forward_cross(x, y)
        assert clm.shape == (batch, 2, lmax + 1)
        np.testing.assert_allclose(auto.fsky, np.mean(w**2, 1))
        for b in range(batch):
            r = lambda m: hp.reorder(m, n2r=True)
            ref = hp.anafast(r(x[b]), lmax=lmax, iter=0)
            np.testing.assert_allclose(cl[b], ref, rtol=1e-11, atol=1e-14 * ref.max())
            ref = hp.anafast(r(x[b]), r(y[b]), lmax=lmax, iter=0)
            np.testing.assert_allclose(xl[b], ref, rtol=1e-11, atol=1e-12 * np.abs(ref).max())
            for k in range(2):
                ref = hp.anafast(r(w[k] * x[b]), lmax=lmax, iter=0)
                np.testing.assert_allclose(clm[b, k], ref, rtol=1e-11, atol=1e-14 * ref.max())
                ref = hp.anafast(r(w[k] * x[b]), r(w[k] * y[b]), lmax=lmax, iter=0)
                np.testing.assert_allclose(xlm[b, k], ref, rtol=1e-11, atol=1e-12 * np.abs(ref).max())


def test_gradients():
    nside, lmax = 8, 23
    rng = np.random.default_rng(2)
    w = _weights(nside)
    for cross in (False, True):
        for masks in (None, w):
            with SpectrumEngine(nside, lmax, batch=2, cross=cross, masks=masks) as e:
                shape = (2, 2, e.npix) if cross else (2, e.npix)
                x = rng.normal(size=shape)
                cl = e.forward(x, generation=1)
                cot = rng.normal(size=cl.shape)
                try:
                    e.vjp(cot, generation=2)
                    raise AssertionError("stale state accepted")
                except RuntimeError:
                    pass
                g = e.vjp(cot, generation=1)
                assert g.shape == shape
                f = lambda y: float(np.sum(cot * e.forward(y, generation=3)))
                _check_fd(f, x, g, rng)


def test_gpu_matches_cpu():
    if os.environ.get("STL_TEST_GPU") != "1":
        print("STL_TEST_GPU unset: skipped")
        return
    nside = int(os.environ.get("STL_TEST_NSIDE", "64"))
    lmax = 2 * nside - 1
    rng = np.random.default_rng(3)
    w = _weights(nside)
    for cross in (False, True):
        shape = (2, 2, 12 * nside**2) if cross else (2, 12 * nside**2)
        x = rng.normal(size=shape)
        out = {}
        for backend in ("cpu", "gpu"):
            with SpectrumEngine(nside, lmax, batch=2, cross=cross, masks=w, backend=backend) as e:
                cl = e.forward(x)
                cot = np.random.default_rng(4).normal(size=cl.shape)
                out[backend] = (cl, e.vjp(cot))
        for a, b in zip(out["gpu"], out["cpu"]):
            err = np.abs(a - b).max() / np.abs(b).max()
            print(f"cross {cross}: GPU vs CPU {err:.2e}")
            assert err <= 1e-12


if __name__ == "__main__":
    test_against_healpy()
    test_gradients()
    test_gpu_matches_cpu()
    print("spectrum tests passed")
