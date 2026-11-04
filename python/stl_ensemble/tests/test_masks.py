#!/usr/bin/env python3
"""Masked averages of NativeEngine (set_masks): shapes, the full-sky mask
against the unmasked engine, ids against indicators, native pixel counts,
the gradient against finite differences, and backend="gpu" against the
CPU (skipped without a GPU build and a device)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from stl_port.native import NativeEngine, PatchEngine

NSIDE, J = 16, 3


def latitude_masks(nside, cuts=(0.0, 20.0, 45.0)):
    """NESTED ids of |b| > cut about the z axis (cut 0: the whole sky)."""
    with PatchEngine(nside, 1) as p:
        z = p.centre_vectors(np.arange(12 * nside**2))[:, 2]
    return [np.arange(z.size) if c == 0 else np.flatnonzero(np.abs(z) > np.sin(np.radians(c))) for c in cuts]


def rel(a, b):
    return np.max(np.abs(np.ravel(a) - np.ravel(b))) / np.max(np.abs(b))


@pytest.mark.parametrize("cross", [False, True])
def test_masked_shapes_full_sky_and_indicators(cross):
    rng = np.random.default_rng(1)
    masks = latitude_masks(NSIDE)
    npix = 12 * NSIDE**2
    shape = (2, 2, npix) if cross else (2, npix)
    x = rng.normal(size=shape)
    with NativeEngine(NSIDE, J, batch=2, cross=cross) as full, NativeEngine(NSIDE, J, batch=2, cross=cross, masks=masks) as eng:
        v0, s0 = full.forward(x, moments=True)
        v, s = eng.forward(x, moments=True)
        assert v.shape == s.shape == (2, 3, eng.ncoeff)
        assert rel(v[:, 0], v0) < 1e-14 and rel(s[:, 0], s0) < 1e-13
        pix = eng.native_pixels()
        assert pix.shape == (3, eng.ncoeff)
        assert np.array_equal(pix[0], full.native_pixels())
        assert all(pix[k, i] == len(masks[k]) for k in range(3) for i, c in enumerate(eng.plan) if c["native_dg"] == 0)
        assert (pix[2] < pix[1]).all() and (pix[1] < pix[0]).all()
        w = np.zeros((3, npix))
        for k, ids in enumerate(masks):
            w[k, ids] = 1.0
        eng.set_masks(w)
        v2, _ = eng.forward(x, moments=True)
        assert np.array_equal(v2, v)
        cot = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
        g = eng.vjp(cot)
        assert g.shape == x.shape
        eng.set_masks(None)
        assert eng.forward(x)[0].shape == (2, eng.ncoeff)


def test_masked_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    masks = latitude_masks(NSIDE)
    x = 1.0 + rng.normal(size=12 * NSIDE**2)
    u = rng.normal(size=x.size)
    with NativeEngine(NSIDE, J, masks=masks) as eng:
        cot = rng.normal(size=(len(masks), eng.ncoeff)) + 1j * rng.normal(size=(len(masks), eng.ncoeff))

        def f(t):
            v, _ = eng.forward(x + t * u)
            return np.sum(np.real(np.conj(cot) * v[0]))

        h = 1e-5
        fd = (f(h) - f(-h)) / (2 * h)
        eng.forward(x)
        an = np.dot(eng.vjp(cot)[0], u)
        assert abs(an - fd) <= 1e-7 * abs(fd)


@pytest.mark.parametrize("cross", [False, True])
def test_masked_gpu_matches_cpu(cross):
    rng = np.random.default_rng(3)
    nside = 32
    masks = latitude_masks(nside)
    npix = 12 * nside**2
    x = rng.normal(size=(2, 2, npix) if cross else (2, npix))
    try:
        gpu = NativeEngine(nside, 4, batch=2, cross=cross, backend="gpu", block_pixels=1024, masks=masks)
    except Exception as e:  # no GPU build or no device
        pytest.skip(f"GPU engine unavailable: {e}")
    with gpu, NativeEngine(nside, 4, batch=2, cross=cross, block_pixels=1024, masks=masks) as cpu:
        vc, sc = cpu.forward(x, moments=True)
        vg, sg = gpu.forward(x, moments=True)
        cot = rng.normal(size=vc.shape) + 1j * rng.normal(size=vc.shape)
        gc, gg = cpu.vjp(cot), gpu.vjp(cot)
        assert np.array_equal(cpu.native_pixels(), gpu.native_pixels())
        print(f"masks gpu/cpu cross {cross}: values {rel(vg, vc):.1e} variances {rel(sg, sc):.1e} gradient {rel(gg, gc):.1e}")
        assert rel(vg, vc) < 1e-13 and rel(sg, sc) < 1e-12 and rel(gg, gc) < 1e-11
