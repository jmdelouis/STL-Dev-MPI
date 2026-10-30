"""Minkowski functionals V0, V1, V2 of the excursion sets of a full-sky HEALPix
map as functions of the threshold (Schmalzing & Gorski 1998), and their
expectation for a Gaussian field (Tomita 1986).

    python3 tools/minkowski.py MAP.npy OUT.npz [--ordering nested|ring]
        [--fwhm-arcmin F] [--lmax L] [--nu-min A] [--nu-max B] [--nbins K]

The map (NESTED by default) goes to harmonic space (lmax = 3 nside - 1 by
default), is optionally smoothed by a Gaussian beam of FWHM F arcmin, and
its first and second covariant derivatives on the unit sphere are evaluated
in harmonic space with healpy's alm2map_der1, applied to the map and to its
two first-derivative maps:

    u_t = d_theta u,  u_p = d_phi u / sin(theta),
    u_tt = d_theta u_t,  u_tp = d_theta u_p,  u_pp = d_phi u_p / sin(theta) + cot(theta) u_t.

With x = (u - <u>) / sigma0 the standardised map and delta(x - nu)
estimated by the fraction of pixels in a bin of width dnu around nu,

    V0(nu) = <Theta(x - nu)>
    V1(nu) = <delta(x - nu) |grad x|> / 4
    V2(nu) = <delta(x - nu) (2 x_t x_p x_tp - x_t^2 x_pp - x_p^2 x_tt) / |grad x|^2> / (2 pi)

(V2 is the integral of the geodesic curvature of the contours, so the
Gaussian expectation has no curvature term on the sphere).  For a Gaussian
field with tau = <|grad x|^2> / 2:

    V0 = erfc(nu / sqrt 2) / 2
    V1 = sqrt(tau) exp(-nu^2 / 2) / 8
    V2 = tau nu exp(-nu^2 / 2) / (2 pi)^(3/2)

OUT.npz holds nu, v0, v1, v2, the Gaussian expectations g0, g1, g2 for the
tau measured on the map, sigma0, sigma1 (of u), tau, fwhm_arcmin, lmax.
"""

import argparse

import healpy as hp
import numpy as np
from scipy.special import erfc


def gaussian_mf(nu, tau):
    """Expected V0, V1, V2 of a Gaussian field with tau = <|grad x|^2> / 2."""
    e = np.exp(-0.5 * nu**2)
    return 0.5 * erfc(nu / np.sqrt(2.0)), np.sqrt(tau) * e / 8.0, tau * nu * e / (2.0 * np.pi) ** 1.5


def spectral_moments(alm, lmax):
    """sigma0^2 = <u^2> and sigma1^2 = <|grad u|^2> of the field of alm (monopole excluded)."""
    cl = hp.alm2cl(alm, lmax=lmax)
    ell = np.arange(lmax + 1)
    w = (2 * ell + 1) / (4 * np.pi)
    w[0] = 0.0
    return np.sqrt(np.sum(w * cl)), np.sqrt(np.sum(w * ell * (ell + 1) * cl))


def derivatives(alm, nside, lmax, niter=3):
    """u and its covariant derivatives (RING ordering): u, u_t, u_p, u_tt, u_tp, u_pp."""
    u, ut, up = hp.alm2map_der1(alm, nside, lmax=lmax)
    _, utt, _ = hp.alm2map_der1(hp.map2alm(ut, lmax=lmax, iter=niter), nside, lmax=lmax)
    _, utp, upp = hp.alm2map_der1(hp.map2alm(up, lmax=lmax, iter=niter), nside, lmax=lmax)
    theta, _ = hp.pix2ang(nside, np.arange(u.size))
    upp = upp + ut * np.cos(theta) / np.sin(theta)
    return u, ut, up, utt, utp, upp


def minkowski_alm(alm, nside, lmax, nu):
    """V0, V1, V2 at the thresholds nu (equally spaced) of the field of alm (monopole removed)."""
    alm = alm.copy()
    alm[0] = 0.0
    s0, s1 = spectral_moments(alm, lmax)
    u, ut, up, utt, utp, upp = derivatives(alm, nside, lmax)
    x, xt, xp, xtt, xtp, xpp = (f / s0 for f in (u, ut, up, utt, utp, upp))
    del u, ut, up, utt, utp, upp
    g2 = xt**2 + xp**2
    grad = np.sqrt(g2)
    curv = (2 * xt * xp * xtp - xt**2 * xpp - xp**2 * xtt) / np.maximum(g2, 1e-300)
    dnu = nu[1] - nu[0]
    edges = np.concatenate([nu - dnu / 2, [nu[-1] + dnu / 2]])
    n = x.size
    count = np.histogram(x, edges)[0]
    v0 = np.array([np.count_nonzero(x > t) for t in nu]) / n
    v1 = np.histogram(x, edges, weights=grad)[0] / (n * dnu * 4.0)
    v2 = np.histogram(x, edges, weights=curv)[0] / (n * dnu * 2.0 * np.pi)
    # Second spectral moment of the standardised map.
    tau = 0.5 * (s1 / s0) ** 2
    return dict(nu=nu, v0=v0, v1=v1, v2=v2, count=count, sigma0=s0, sigma1=s1, tau=tau)


def minkowski_map(m, nest=True, lmax=None, fwhm_arcmin=0.0, nu=None, niter=3):
    """Minkowski functionals of a full-sky map; returns the dict of minkowski_alm plus the Gaussian expectations."""
    m = np.asarray(m, dtype=np.float64)
    nside = hp.npix2nside(m.size)
    lmax = 3 * nside - 1 if lmax is None else lmax
    if nest:
        m = hp.reorder(m, n2r=True)
    alm = hp.map2alm(m - m.mean(), lmax=lmax, iter=niter)
    if fwhm_arcmin > 0:
        alm = hp.smoothalm(alm, fwhm=np.radians(fwhm_arcmin / 60.0), inplace=False)
    nu = np.linspace(-3.0, 6.0, 91) if nu is None else np.asarray(nu, dtype=np.float64)
    out = minkowski_alm(alm, nside, lmax, nu)
    out["g0"], out["g1"], out["g2"] = gaussian_mf(nu, out["tau"])
    out["fwhm_arcmin"], out["lmax"] = fwhm_arcmin, lmax
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("map")
    p.add_argument("out")
    p.add_argument("--ordering", choices=("nested", "ring"), default="nested")
    p.add_argument("--fwhm-arcmin", type=float, default=0.0)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--nu-min", type=float, default=-3.0)
    p.add_argument("--nu-max", type=float, default=6.0)
    p.add_argument("--nbins", type=int, default=91)
    a = p.parse_args()
    nu = np.linspace(a.nu_min, a.nu_max, a.nbins)
    r = minkowski_map(np.load(a.map), nest=a.ordering == "nested", lmax=a.lmax, fwhm_arcmin=a.fwhm_arcmin, nu=nu)
    np.savez(a.out, **r)
    print(f"{a.map}: sigma0 {r['sigma0']:.4e} sigma1 {r['sigma1']:.4e} tau {r['tau']:.4e}, "
          f"{len(nu)} thresholds in [{nu[0]:g}, {nu[-1]:g}] -> {a.out}")


if __name__ == "__main__":
    main()
