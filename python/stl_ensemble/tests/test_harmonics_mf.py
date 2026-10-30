"""Numerical checks of the CPU full-sky branch, independent of native SC."""
import os
from pathlib import Path
import ast
import numpy as np
import pytest
import ducc0
from scipy.special import sph_harm_y, ndtr

from stl_port.harmonics import FullSkySHT, lp_windows
from stl_port.minkowski import Minkowski

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: test_harmonics_mf.py {SCRIPT_VERSION}", flush=True)


def _random_alm(s, rng):
    a = rng.normal(size=s.nalm)+1j*rng.normal(size=s.nalm)
    a[s.m == 0] = a[s.m == 0].real
    return a


def _dense_reference(s):
    # Independent scipy associated-Legendre derivatives; used only at nside 2/4.
    theta, phi = ducc0.healpix.Healpix_Base(s.nside, "NEST").pix2ang(
        np.arange(s.npix, dtype=np.int64)).T
    y, dy, ddy = sph_harm_y(s.ell[:, None], s.m[:, None],
                          theta[None, :], phi[None, :], diff_n=2)
    sin = np.sin(theta)
    cot = np.cos(theta)/sin
    fields = (dy[..., 0], dy[..., 1]/sin,
              ddy[..., 0, 0], ddy[..., 1, 1]/sin**2+cot*dy[..., 0],
              ddy[..., 0, 1]/sin-cot*dy[..., 1]/sin)
    # D=S_derivative A, A = area * conj(Y), real inputs. m>0 is doubled.
    return [s.area*np.real((f.T*s.mult)@y.conj()) for f in fields], fields, y


def test_analysis_synthesis_adjoints_and_healpy():
    import healpy as hp
    s = FullSkySHT(8)
    r = np.random.default_rng(138)
    x = r.normal(size=s.npix)
    a = _random_alm(s, r)
    np.testing.assert_allclose(np.vdot(a, s.analysis(x)).real,
                               x@s.analysis_vjp(a), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(s.synthesis(a)@x,
                               np.vdot(a, s.synthesis_vjp(x)).real,
                               rtol=2e-13, atol=2e-13)
    ahp = hp.map2alm(hp.reorder(x, n2r=True), lmax=s.lmax, iter=0)
    np.testing.assert_allclose(s.analysis(x), ahp, rtol=1e-11, atol=2e-14)
    np.testing.assert_allclose(s.power(ahp), hp.alm2cl(ahp), rtol=2e-14, atol=2e-14)


def test_power_vjp_and_stale_states():
    s = FullSkySHT(4)
    r = np.random.default_rng(139)
    x, h = r.normal(size=(2, s.npix))
    c = r.normal(size=s.lmax+1)
    state = s.cache_analysis(x, generation=3, member=4)
    g = s.cached_power_vjp(state, c, generation=3, member=4)
    eps = 1e-6
    fd = c@(s.power(s.analysis(x+eps*h))-s.power(s.analysis(x-eps*h)))/(2*eps)
    np.testing.assert_allclose(g@h, fd, rtol=5e-8, atol=1e-9)
    for kwargs in [dict(generation=4, member=4), dict(generation=3, member=5),
                   dict(generation=3, member=4, objective="gaussian")]:
        with pytest.raises(ValueError):
            s.cached_power_vjp(state, c, **kwargs)
    with pytest.raises(ValueError):
        FullSkySHT(4).cached_power_vjp(state, c, generation=3, member=4)


def test_derivative_conventions_and_adjoint():
    s = FullSkySHT(4)
    r = np.random.default_rng(140)
    a = _random_alm(s, r)
    _, fields, _ = _dense_reference(s)
    got = s.derivative_maps(a)
    for actual, f in zip(got, fields):
        expected = np.real(np.sum((a*s.mult)[:, None]*f, axis=0))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=3e-11)
    x = r.normal(size=s.npix)
    cot = r.normal(size=(5, s.npix))
    d = s.derivative_maps(s.analysis(x))
    np.testing.assert_allclose(sum(c@z for c, z in zip(cot, d)),
                               s.derivatives_vjp(cot)@x, rtol=3e-13, atol=1e-10)


def test_mf_fixed_floor_directional_and_chunk_invariance():
    s = FullSkySHT(4)
    r = np.random.default_rng(141)
    x, h = r.normal(size=(2, s.npix))
    mf = Minkowski(s, pixel_chunk=71, threshold_chunk=7)
    f, st = mf.forward(x)
    whole = Minkowski(s, pixel_chunk=s.npix, threshold_chunk=65)
    fw, sw = whole.forward(x)
    np.testing.assert_allclose(f, fw, rtol=1e-13, atol=1e-13)
    c = r.normal(size=f.shape)
    g = mf.vjp(st, c)
    np.testing.assert_allclose(g, whole.vjp(sw, c), rtol=3e-12, atol=3e-12)
    eps = 2e-6
    fp = mf.forward(x+eps*h, fixed_floor=st.floor)[0]
    fm = mf.forward(x-eps*h, fixed_floor=st.floor)[0]
    fd = np.sum(c*(fp-fm))/(2*eps)
    np.testing.assert_allclose(fd, g@h, rtol=2e-7, atol=1e-8)
    assert abs(g.sum()) < 1e-10  # normalization includes mean derivative


def test_mf_against_independent_source_equations_complex_step():
    s = FullSkySHT(2)
    matrices, _, _ = _dense_reference(s)
    r = np.random.default_rng(142)
    x = r.normal(size=s.npix)
    mf = Minkowski(s)
    f, st = mf.forward(x)
    c = r.normal(size=f.shape)

    def reference(z):
        centered = z-z.mean()
        u = centered/np.sqrt(np.mean(centered**2)+1e-12)
        d1, d2, d11, d22, d12 = [D@u for D in matrices]
        q = d1**2+d2**2
        k = (2*d1*d2*d12-d1**2*d22-d2**2*d11)/(q+st.floor)
        a = (u[None, :]-mf.nu[:, None])/.125
        delta = np.exp(-a*a/2)/(np.sqrt(2*np.pi)*.125)
        return np.stack([ndtr(a).mean(axis=1),
                         (delta*np.sqrt(q+1e-30)).mean(axis=1)/4,
                         (delta*k).mean(axis=1)/(2*np.pi)])

    np.testing.assert_allclose(f, reference(x), rtol=2e-12, atol=2e-12)
    gref = np.empty(s.npix)
    for p in range(s.npix):
        z = x.astype(np.complex128)
        z[p] += 1j*1e-25
        gref[p] = np.sum(c*reference(z)).imag/1e-25
    np.testing.assert_allclose(mf.vjp(st, c), gref, rtol=3e-11, atol=3e-11)


def test_lp_source_contract():
    w, centers = lp_windows(383)
    assert w.shape == (16, 384)
    assert np.all(w[:, :2] == 0)
    delta = np.log(383/2)/15
    x = (np.log(np.arange(384)+.5)[None, :]-np.log(centers[:, None]))/delta
    expected = np.where(np.abs(x) <= 1, np.cos(np.pi*x/2)**2, 0)
    expected[:, :2] = 0
    np.testing.assert_allclose(w, expected, atol=4e-15)
    assert not np.allclose(w.sum(axis=1), 1)  # weighted sums, not averages


def test_original_torch_source_integrand_vjp_when_available():
    """Execute the unchanged reference compute_curves body on exact dense SHTs.

    This checks its detached-floor/autograd convention, not its legacy
    all-Legendre-table numerical calibration. Full source backend reproduction
    remains a separate scientific comparison on the user's environment.
    """
    torch = pytest.importorskip("torch")
    default = Path(__file__).resolve().parents[1]/"reference/08-minkowski_sht_loss.py"
    if not default.is_file():
        default = Path(__file__).resolve().parents[3]/"project_sources/08-minkowski_sht_loss.py"
    source = Path(os.environ.get("STL_REFERENCE_MF", default))
    if not source.is_file():
        pytest.skip("Set STL_REFERENCE_MF to the unchanged supplied MF helper")
    tree = ast.parse(source.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DifferentiableMinkowskiSHT")
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_smooth_heaviside", "_smooth_delta_over_eps"}]
    wanted = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in {"_normalise_in_sorted", "compute_curves"}]
    cls = ast.ClassDef(name="Reference", bases=[], keywords=[], body=wanted, decorator_list=[])
    namespace = dict(torch=torch, np=np, _SQRT2=np.sqrt(2.), _SQRT_2PI=np.sqrt(2*np.pi))
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs+[cls], type_ignores=[])), str(source), "exec"), namespace)
    s = FullSkySHT(2)
    matrices, _, _ = _dense_reference(s)
    ref = namespace["Reference"]()
    ref._device = torch.device("cpu")
    ref._n2s_t = torch.arange(s.npix)
    ref._core_mask_t = ref._taper_t = torch.ones(s.npix)
    ref.n_patch_pixels = s.npix
    ref._nu_t = torch.linspace(-4, 4, 65)
    ref._eps = .125
    ref.grad_floor_frac = 1e-4
    ref.need_V2 = True
    ref.integration_weight = "core"
    operators = [torch.tensor(D, dtype=torch.float32) for D in matrices]
    ref._compute_derivative_maps = lambda z: tuple(z@D.T for D in operators)
    r = np.random.default_rng(144)
    x = r.normal(size=s.npix).astype(np.float32)
    c = r.normal(size=(3, 65)).astype(np.float32)
    t = torch.tensor(x, requires_grad=True)
    values = ref.compute_curves(t)
    out = torch.stack([values[name] for name in ("V0", "V1", "V2")])
    (out*torch.tensor(c)).sum().backward()
    mf = Minkowski(s)
    f, st = mf.forward(x)
    np.testing.assert_allclose(f, out.detach().numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(mf.vjp(st, c), t.grad.numpy(), rtol=8e-5, atol=2e-5)
