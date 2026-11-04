#!/usr/bin/env python3
"""Disc-patch and cross statistics through the native bridge (PatchEngine,
NativeEngine(cross=True)); requires a built libstl_bridge
(STL_NATIVE_LIBRARY). Runs under pytest or as a script."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeEngine, PatchEngine, delouis_transform, delouis_transform_vjp

NSIDE, J = 8, 3
NPIX = 12 * NSIDE**2


def _objective(values, cot):
    return float(np.real(np.sum(np.conj(cot) * values)))


def _check_fd(f, x, grad, rng, trials=3, h=1e-5):
    for _ in range(trials):
        d = rng.normal(size=x.shape)
        fd = (f(x + h * d) - f(x - h * d)) / (2 * h)
        an = float(np.sum(grad * d))
        assert abs(fd - an) <= 1e-6 * max(abs(an), 1.0), (fd, an)


def test_full_disc_equals_full_sky():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, NPIX))
    with NativeEngine(NSIDE, J, batch=2, block_pixels=256) as full, PatchEngine(NSIDE, J, threads=2) as patch:
        ref, refvar = full.forward(x, moments=True)
        cot = rng.normal(size=ref.shape) + 1j * rng.normal(size=ref.shape)
        gref = full.vjp(cot)
        ids = patch.select(np.array([3, 500]), radius=180.0)
        assert all(np.array_equal(i, np.arange(NPIX)) for i in ids)
        v, var = patch.forward(x)
        np.testing.assert_allclose(v, ref, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(var, refvar, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(patch.vjp(cot), gref, rtol=1e-10, atol=1e-10 * np.abs(gref).max())
        assert patch.plan == full.plan
        np.testing.assert_array_equal(patch.native_pixels[0], [12 * (NSIDE >> c["native_dg"]) ** 2 for c in patch.plan])


def test_disc_gradient_finite_differences():
    rng = np.random.default_rng(2)
    with PatchEngine(NSIDE, J, threads=2) as patch:
        centres = rng.normal(size=(3, 3))
        ids = patch.select(centres, radius=60.0, core=45.0)
        assert len({i.size for i in ids}) >= 1 and all(i.size < NPIX for i in ids)
        assert (patch.native_pixels[:, 0] < [i.size for i in ids]).all()
        maps = [rng.normal(size=i.size) for i in ids]
        v, _ = patch.forward(maps, generation=1)
        cot = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
        g = patch.vjp(cot, generation=1)
        flat = np.concatenate(maps)
        cuts = np.cumsum([i.size for i in ids])[:-1]
        f = lambda y: _objective(patch.forward(np.split(y, cuts), moments=False, generation=2)[0], cot)
        _check_fd(f, flat, np.concatenate(g), rng)


def test_cross_gradient_finite_differences():
    rng = np.random.default_rng(3)
    with PatchEngine(NSIDE, J, cross=True, threads=2) as patch:
        ids = patch.select(np.array([10, 300]), radius=70.0)
        a = [rng.normal(size=i.size) for i in ids]
        b = [rng.normal(size=i.size) for i in ids]
        v, _ = patch.forward_cross(a, b, generation=1)
        cot = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
        ga, gb = patch.vjp_cross(cot, generation=1)
        cuts = np.cumsum([i.size for i in ids])[:-1]
        fa = lambda y: _objective(patch.forward_cross(np.split(y, cuts), b, moments=False, generation=2)[0], cot)
        fb = lambda y: _objective(patch.forward_cross(a, np.split(y, cuts), moments=False, generation=2)[0], cot)
        _check_fd(fa, np.concatenate(a), np.concatenate(ga), rng)
        _check_fd(fb, np.concatenate(b), np.concatenate(gb), rng)
    # Full sky, gradient with respect to a with b fixed.
    with NativeEngine(NSIDE, J, batch=2, block_pixels=256, cross=True) as full:
        a, b = rng.normal(size=(2, NPIX)), rng.normal(size=(2, NPIX))
        v, _ = full.forward_cross(a, b, generation=1)
        cot = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
        ga, _ = full.vjp_cross(cot, generation=1)
        fa = lambda y: _objective(full.forward_cross(y, b, generation=2)[0], cot)
        _check_fd(fa, a, ga, rng)


def _matching(cross_plan, auto_plan):
    key = lambda c: (c["family"], c["native_dg"], c["j1"], c["j2"], c["j3"], c["l1"], c["l2"], c["l3"])
    index = {key(c): i for i, c in enumerate(auto_plan)}
    return np.array([index[key(c)] for c in cross_plan])


def test_cross_with_itself_is_auto():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(1, NPIX))
    with NativeEngine(NSIDE, J, block_pixels=256) as auto, NativeEngine(NSIDE, J, block_pixels=256, cross=True) as cross:
        va, _ = auto.forward(a)
        vc, _ = cross.forward_cross(a, a)
        at = _matching(cross.plan, auto.plan)
        s1 = np.array([c["family"] == "S1" for c in cross.plan])
        assert {c["family"] for c in cross.plan} == {"S1", "S2", "S3", "S4"}
        np.testing.assert_allclose(vc[:, ~s1], va[:, at[~s1]], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(vc[:, s1], va[:, at[s1]], rtol=1e-10, atol=1e-7)
    with PatchEngine(NSIDE, J) as auto, PatchEngine(NSIDE, J, cross=True) as cross:
        ids = auto.select(np.array([77]), radius=50.0, core=35.0)
        cross.select(np.array([77]), radius=50.0, core=35.0)
        p = [a[0, ids[0]]]
        va, _ = auto.forward(p)
        vc, _ = cross.forward_cross(p, p)
        at = _matching(cross.plan, auto.plan)
        np.testing.assert_allclose(vc[:, ~s1], va[:, at[~s1]], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(vc[:, s1], va[:, at[s1]], rtol=1e-10, atol=1e-7)


def test_delouis_transform_vjp():
    rng = np.random.default_rng(5)
    v = rng.normal(size=20) + 1j * rng.normal(size=20)
    cot = rng.normal(size=20) + 1j * rng.normal(size=20)
    g = delouis_transform_vjp(v, cot)
    h = 1e-7
    for k in range(20):
        for e in (1.0, 1j):
            dv = np.zeros(20, complex)
            dv[k] = e * h
            fd = (_objective(delouis_transform(v + dv), cot) - _objective(delouis_transform(v - dv), cot)) / (2 * h)
            an = float(np.real(np.conj(g[k]) * e))
            assert abs(fd - an) <= 1e-5 * max(abs(an), 1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok", flush=True)
