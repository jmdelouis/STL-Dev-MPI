"""Parts of the Healpix-STL-IA denoising losses that the native U-Net driver
(`train_native_unet.py`) computes on the host side of the trainer, without
torch: galactic masks, the binned half-mission spectrum term `cl`
(`stlia.spectra.BinnedSpectra` and `SpectrumLoss` over the `SpectrumEngine`
of this package) and the conditioning maps of `--cond`.

    cl    C_b(m d1, m d2)  vs  C_b(m u, m u)

for every galactic mask `m` (`|b|` above a cut, apodised): pseudo-spectra of
the masked maps divided by mean(m^2), binned in multipole, compared with
the Gaussian variance of the target bins, `(C11 C22 + C12^2) / modes`, as a
reduced chi^2 (mean over masks and bins). The half-mission noises are
independent: no noise draw and no bias. With `spin=2` the maps are (Q, U)
pairs and the spectra EE and BB (`stlia.spectra.BinnedSpectra(spin=2)`).

`eb_maps` and `qu_maps`: the E and B maps of (Q, U) and back (healpy,
`lmax = 3 nside - 1`), for the joint network on E and B.

healpy is needed for the masks (pixel latitudes, apodisation) and the
E/B transforms.
"""
import math

import numpy as np

from .native import SpectrumEngine


def galactic_mask(nside, bcut, apod=5.0):
    """NESTED mask `[npix]` of `|b| > bcut` degrees, smoothed by a Gaussian of
    `apod` degrees FWHM and clipped to [0, 1]; `bcut` = 0 is the full sky
    (`stlia.spectra.galactic_mask`)."""
    if bcut <= 0:
        return np.ones(12 * nside * nside)
    import healpy as hp
    theta, _ = hp.pix2ang(nside, np.arange(12 * nside * nside))
    m = (np.abs(90.0 - np.degrees(theta)) > bcut).astype(float)
    if apod > 0:
        m = np.clip(hp.smoothing(m, fwhm=math.radians(apod)), 0.0, 1.0)
    return hp.reorder(m, r2n=True)


def stat_masks(nside, cuts):
    """NESTED ids of the unapodised masks `|b| > b0` (`train_denoise.py
    --stat-masks`), one array per cut."""
    return [np.flatnonzero(galactic_mask(nside, b, apod=0) > 0.5) for b in cuts]


def bin_edges(lmin, lmax, width):
    """Multipole bins [lmin, lmin + width), ..., ending at lmax (inclusive)."""
    edges = list(range(lmin, lmax + 1, width))
    if edges[-1] != lmax + 1:
        edges.append(lmax + 1)
    return np.array(edges)


def eb_maps(qu, iter=3):
    """E and B maps `[..., 2, npix]` (NESTED) of NESTED (Q, U) pairs `[...,
    2, npix]`: healpy `map2alm(pol=True, iter)` then `alm2map` of a_E and
    a_B, `lmax = 3 nside - 1`."""
    import healpy as hp
    qu = np.asarray(qu, dtype=np.float64)
    flat = qu.reshape(-1, 2, qu.shape[-1])
    out = np.empty_like(flat)
    nside = hp.npix2nside(qu.shape[-1])
    for i, (q, u) in enumerate(flat):
        q, u = hp.reorder(q, n2r=True), hp.reorder(u, n2r=True)
        _, ae, ab = hp.map2alm([np.zeros_like(q), q, u], pol=True, iter=iter)
        out[i] = [hp.reorder(hp.alm2map(x, nside), r2n=True) for x in (ae, ab)]
    return out.reshape(qu.shape)


def qu_maps(eb, iter=3):
    """(Q, U) `[..., 2, npix]` (NESTED) of NESTED E and B maps: scalar
    `map2alm(iter)` of each, then `alm2map` of (0, a_E, a_B) with
    `pol=True` (the inverse of :func:`eb_maps` within the band limit)."""
    import healpy as hp
    eb = np.asarray(eb, dtype=np.float64)
    flat = eb.reshape(-1, 2, eb.shape[-1])
    out = np.empty_like(flat)
    nside = hp.npix2nside(eb.shape[-1])
    for i, (e, b) in enumerate(flat):
        ae, ab = (hp.map2alm(hp.reorder(x, n2r=True), iter=iter) for x in (e, b))
        _, q, u = hp.alm2map([np.zeros_like(ae), ae, ab], nside, pol=True)
        out[i] = [hp.reorder(q, r2n=True), hp.reorder(u, r2n=True)]
    return out.reshape(eb.shape)


class BinnedSpectra:
    """Binned pseudo-spectra `[K, nb]` of NESTED maps under K apodised galactic
    masks, by `SpectrumEngine` (one map, or a pair with `cross`); `spin=2`:
    `[K, 2, nb]` (EE, BB) of (Q, U) pairs `[2, npix]` (host transforms)."""

    def __init__(self, nside, masks=(0.0, 30.0), lmin=10, lmax=None, width=30, apod=5.0, backend="cpu", device=0,
                 spin=0):
        self.nside, self.lmax, self.spin = nside, lmax or 2 * nside - 1, int(spin)
        assert self.spin in (0, 2), "spin 0 or 2"
        self.cuts = list(masks)
        self.m = np.stack([galactic_mask(nside, b, apod) for b in self.cuts])
        kw = dict(lmax=self.lmax, batch=1, backend=backend, device=device, masks=self.m,
                  **({"spin": 2} if self.spin else {}))
        self.auto = SpectrumEngine(nside, **kw)
        self.pair = SpectrumEngine(nside, cross=True, **kw)
        self.w2 = (self.m**2).mean(1)  # [K]
        self.edges = bin_edges(lmin, self.lmax, width)
        nb = len(self.edges) - 1
        ell = np.arange(self.lmax + 1)
        self.binning = np.zeros((nb, self.lmax + 1))
        for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            self.binning[i, lo:hi] = 1.0 / (hi - lo)
        modes = np.array([(2 * ell[lo:hi] + 1).sum() for lo, hi in zip(self.edges[:-1], self.edges[1:])])
        fsky = self.w2**2 / (self.m**4).mean(1)
        self.modes = fsky[:, None] * modes[None]  # [K, nb]
        self.generation = 0

    def _w2(self):
        return self.w2[:, None, None] if self.spin else self.w2[:, None]

    def __call__(self, a, b=None):
        """Binned spectrum `[K, nb]` (spin 2: `[K, 2, nb]`, EE and BB) of `a`
        (cross spectrum with `b`); the auto spectrum is kept for `grad`."""
        if b is None:
            self.generation += 1
            cl = self.auto.forward(np.asarray(a, dtype=np.float64)[None], generation=self.generation)[0]
        else:
            cl = self.pair.forward_cross(np.asarray(a)[None], np.asarray(b)[None])[0]
        if self.spin:
            cl = cl[:, :2]  # EE, BB (E1E2, B1B2 of a pair)
        return (cl / self._w2()) @ self.binning.T

    def grad(self, cot):
        """Map gradient of `sum(cot * C_b(a))` after the last auto `__call__`."""
        c = (np.asarray(cot, dtype=np.float64) @ self.binning) / self._w2()
        if self.spin:  # no cotangent on EB
            c = np.concatenate([c, np.zeros_like(c[:, :1])], 1)
        return self.auto.vjp(c[None], generation=self.generation)[0]

    def variance(self, c11, c22, c12):
        """Gaussian variance of a binned cross spectrum: (C11 C22 + C12^2) / modes."""
        modes = self.modes[:, None] if self.spin else self.modes
        return (c11 * c22 + c12**2) / modes


class SpectrumLoss:
    """Reduced chi^2 between the binned spectrum of `u` and the half-mission
    cross spectrum of the data (`stlia.spectra.SpectrumLoss`)."""

    def __init__(self, spectra, d1, d2):
        self.spectra = spectra
        self.target = spectra(d1, d2)
        self.var = np.maximum(spectra.variance(spectra(d1), spectra(d2), self.target), 1e-300)

    def __call__(self, u, grad=False):
        """Loss of `u` `[npix]`; with `grad`, (loss, map gradient)."""
        r = self.spectra(u) - self.target
        loss = float(np.mean(r**2 / self.var))
        if not grad:
            return loss
        return loss, self.spectra.grad(2.0 * r / self.var / r.size)


def conditioning(kinds, i857=None, noise_var=None):
    """Conditioning maps `[len(kinds), npix]` (float32) of `--cond`
    (`train_denoise.py`, block "conds"), each standardised (population std):
    log857, the log of the 857 GHz intensity clipped at its 0.1 percentile of
    the positive pixels; sigma, the log of the local noise rms,
    0.5 log(mean over fields and training simulations of n^2), `noise_var`
    the per-field means over the simulations of the full-mission noise
    squared `[npix]`."""
    conds = []
    for k in kinds:
        if k == "log857":
            conds.append(np.log(np.clip(i857, np.percentile(i857[i857 > 0], 0.1), None)))
        elif k == "sigma":
            conds.append(0.5 * np.log(np.mean(noise_var, 0)))
        else:
            raise ValueError(f"unknown conditioning map {k}")
    return np.stack([(x - x.mean()) / x.std() for x in conds]).astype(np.float32)
