"""Numerical and source-derived checks for population losses, LUT, and CIC."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.special import erfinv

from stl_port.losses import (moments, moments_vjp, sc_loss, power_loss,
                             pdf_loss, mf_loss, equalize_sigma, torch_median)
from stl_port.lut import PooledLUT, probability_grid, source_ppf
from stl_port.pdf import CICPDF

SCRIPT_VERSION = "2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: test_losses.py {SCRIPT_VERSION}", flush=True)


def directional_check(function, x, rng, tolerance=2e-6):
    direction = rng.normal(size=x.shape)
    if np.iscomplexobj(x):
        direction = direction + 1j * rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    loss, cotangent = function(x)
    epsilon = 1e-6
    finite = (function(x + epsilon * direction)[0] - function(x - epsilon * direction)[0]) / (2 * epsilon)
    adjoint = np.vdot(cotangent, direction).real
    assert np.isfinite(loss)
    np.testing.assert_allclose(finite, adjoint, rtol=tolerance, atol=1e-7)


def plan():
    rows = []
    for family in ("mean", "var"):
        rows.append(dict(family=family))
    for family in ("S1", "S2", "S3", "S4"):
        for scale in range(2):
            for angle in range(4):
                rows.append(dict(family=family, j1=scale,
                                 l1=angle, l3=angle))
    return rows


def test_mean_unbiased_complex_variance_and_vjp():
    rng = np.random.default_rng(203)
    x = rng.normal(size=(10, 7)) + 1j * rng.normal(size=(10, 7))
    m, v = moments(x)
    np.testing.assert_allclose(v, x.real.var(0, ddof=1) + x.imag.var(0, ddof=1))
    dm = rng.normal(size=7) + 1j * rng.normal(size=7)
    dv = rng.normal(size=7)
    directional_check(lambda a: (float(np.vdot(dm, moments(a)[0]).real + np.sum(dv * moments(a)[1])),
                                  moments_vjp(a, dm, dv)), x, rng)


@pytest.mark.parametrize("physical", [False, True])
def test_sc_gradients_and_group_reductions(physical):
    rng = np.random.default_rng(87)
    p = plan()
    x = rng.normal(size=(10, len(p))) + 0.3j * rng.normal(size=(10, len(p)))
    t = rng.normal(size=len(p)) + 0.2j * rng.normal(size=len(p))
    variance = rng.uniform(0.5, 2, size=len(p))
    sigma = rng.uniform(0.3, 3, size=len(p))
    directional_check(lambda a: sc_loss(a, t, variance, sigma, p, physical), x, rng)
    mu, v = moments(x)
    sig = equalize_sigma(sigma, p) if physical else sigma
    family = np.array([item["family"] for item in p])
    selected = np.isin(family, ("S1", "S2", "S3", "S4")) if physical else np.ones(len(p), bool)
    if physical:
        mean = sum(np.mean(np.abs((mu[family == f] - t[family == f]) / sig[family == f]) ** 2)
                   for f in ("S1", "S2", "S3", "S4"))
    else:
        mean = np.mean(np.abs((mu - t) / sig) ** 2)
    expected = mean + np.mean((np.log(v[selected]) - np.log(variance[selected])) ** 2)
    np.testing.assert_allclose(sc_loss(x, t, variance, sigma, p, physical)[0], expected)


def test_sc_relative_variance_sensitivity_and_plan_order():
    rng = np.random.default_rng(99)
    p = plan()
    x = 1e-8 * rng.normal(size=(10, len(p)))
    mu, _ = moments(x)
    loss, grad = sc_loss(x, mu, np.full(len(p), 2e-12), np.ones(len(p)), p)
    assert loss > 0
    assert np.isfinite(grad).all() and np.linalg.norm(grad)>0
    x *= 1000
    loss, grad = sc_loss(x, x.mean(0), np.full(len(p), 2e-12), np.ones(len(p)), p)
    assert np.linalg.norm(grad) > 0
    sigma = rng.uniform(0.4, 1.8, len(p))
    permutation = rng.permutation(len(p))
    np.testing.assert_allclose(equalize_sigma(sigma, p)[permutation],
                               equalize_sigma(sigma[permutation], [p[i] for i in permutation]))


def test_power_pdf_mf_gradients_and_division_by_five():
    rng = np.random.default_rng(40)
    x = rng.uniform(0.1, 3, size=(10, 16))
    t = rng.uniform(0.4, 2.5, size=16)
    variance = rng.uniform(0.2, 1, size=16)
    centers = np.geomspace(2, 383, 16)
    directional_check(lambda a: power_loss(a, t, variance, centers), x, rng)
    directional_check(lambda a: pdf_loss(a, t, variance), x, rng)
    y = rng.normal(size=(10, 3, 65))
    tm = rng.normal(size=(3, 65))
    vm = rng.uniform(0.2, 2, size=(3, 65))
    vm[:, :4] = 0  # Saturated tails are absent from both MF moment terms.
    directional_check(lambda a: mf_loss(a, tm, vm), y, rng)
    val, cot = mf_loss(y, tm, vm)
    np.testing.assert_array_equal(cot[:, :, :4], 0)
    # Assembled objectives divide each scalar and its cotangent exactly once.
    directional_check(lambda a: (mf_loss(a, tm, vm)[0] / 5, mf_loss(a, tm, vm)[1] / 5), y, rng)


def test_pdf_informative_masks_and_lower_median():
    assert torch_median([1, 2, 100, 200]) == 2
    x = np.ones((10, 5))
    zero, cot = pdf_loss(x, np.zeros(5), np.zeros(5))
    assert zero == 0
    np.testing.assert_array_equal(cot, 0)
    target = np.array([1., 1., 1e-7, 0, 1.])
    variance = np.array([1., 1., 1., 1., 0])
    _, cot = pdf_loss(x, target, variance)
    np.testing.assert_array_equal(cot[:, 2:4], 0)


def test_cic_partial_boundary_mass_and_vjp():
    rng = np.random.default_rng(55)
    pdf = CICPDF(-2, 2, n_bins=7, chunk_pixels=3)
    x = np.array([-3.1, -2.2, -1.71, -0.11, 0.31, 1.95, 2.22, 3.1])
    h, state = pdf.forward(x)
    q = rng.normal(size=7)
    def objective(a):
        hist, saved = pdf.forward(a)
        return float(hist @ q), pdf.vjp(saved, q)
    directional_check(objective, x, rng)
    # Exact source NumPy formula, including partial contributions outside support.
    pos = (x + 2) / pdf.dx
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    expected = np.zeros(7)
    for i, f in zip(i0, frac):
        if 0 <= i < 7:
            expected[i] += 1 - f
        if 0 <= i + 1 < 7:
            expected[i + 1] += f
    expected /= x.size * pdf.dx
    np.testing.assert_allclose(h, expected)
    assert state.outside_fraction == 0.5
    assert np.sum(h) * pdf.dx < 1  # No normalization of surviving mass.


def test_lut_exact_pool_quantiles_and_memmap_cleanup(tmp_path):
    rng = np.random.default_rng(71)
    maps = rng.lognormal(-4, 0.8, size=(54, 233)).astype(np.float32)
    path = tmp_path / "exact_pool.dat"
    lut = PooledLUT.from_training(maps, work_path=path)
    assert not path.exists()
    p = probability_grid(8192, maps.size)
    expected = np.quantile(maps.ravel().astype(np.float64), p, method="linear").astype(np.float32)
    eps = max(abs(float(expected[-1] - expected[0])) * 1e-7, 1e-12)
    for i in range(1, expected.size):
        if expected[i] <= expected[i - 1]:
            expected[i] = expected[i - 1] + eps
    np.testing.assert_array_equal(lut.physical_knots, expected)
    np.testing.assert_array_equal(lut.gaussian_knots,
        (np.sqrt(2.) * erfinv(2 * p - 1)).astype(np.float32).astype(np.float64))


def test_lut_extrapolation_and_per_member_monopole_vjp():
    rng = np.random.default_rng(69)
    probabilities = np.linspace(.001, .999, 17)
    lut = PooledLUT(probabilities, np.linspace(-.1, .2, 17) ** 3, chunk_pixels=5)
    x = rng.normal(size=(10, 27))
    x[0, :2] = [-12, 13]
    q = rng.normal(size=x.shape)
    def objective(a):
        u, saved = lut.inverse(a)
        return float(np.sum(u * q)), lut.vjp(saved, q)
    directional_check(objective, x, rng)
    u, state = lut.inverse(x)
    np.testing.assert_allclose(u.mean(-1), 0, atol=1e-18)
    np.testing.assert_array_equal(lut.vjp(state, np.ones_like(x)), 0)
    tail, _ = lut.inverse(np.array([-13., -12., 12., 13.]), remove_monopole=False)
    assert tail[0] < tail[1] < tail[2] < tail[3]


def test_reject_wrong_member_grid_and_missing_variance_range():
    a, b = CICPDF(-1, 1), CICPDF(-1, 1)
    _, saved = a.forward(np.linspace(-.9, .9, 20))
    with pytest.raises(ValueError):
        b.vjp(saved, np.ones(100))
    with pytest.raises(ValueError):
        power_loss(np.ones((10, 4)), np.ones(4), np.ones(4), np.arange(4), cutoff=None)
