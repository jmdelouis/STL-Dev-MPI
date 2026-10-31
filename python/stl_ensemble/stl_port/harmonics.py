"""Standard full-sky HEALPix quadrature and its matching discrete adjoints.

No iterative inverse, mask coupling, pixel-window correction, or beam is used.
Maps are NESTED at every public boundary; DUCC receives RING maps internally.
The packed-alm cotangent convention is Re(vdot(cotangent, delta_alm)), counting
each stored coefficient once. Synthesis includes the real-field m>0 factor 2.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import ducc0

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: harmonics.py {SCRIPT_VERSION}", flush=True)


@dataclass(frozen=True)
class AlmState:
    alm: np.ndarray
    generation: int
    member: int
    objective: str
    nside: int
    lmax: int
    dtype: str
    input_digest: str
    owner: int


def lp_windows(lmax, n_bins=16, ell_min=2, ell_max=None):
    """Source cosine-squared windows, unnormalized sums, ell offset 0.5."""
    ell_max = min(lmax, lmax if ell_max is None else int(ell_max))
    ell_min = max(2, int(ell_min))
    if ell_max <= ell_min or n_bins < 2:
        raise ValueError("LP windows require ell_max>ell_min>=2, n_bins>=2")
    logcenters = np.linspace(np.log(ell_min), np.log(ell_max), n_bins)
    delta = (logcenters[-1] - logcenters[0]) / (n_bins - 1)
    ell = np.arange(lmax + 1)
    x = (np.log(ell + .5)[None, :] - logcenters[:, None]) / delta
    windows = np.where(np.abs(x) <= 1, np.cos(np.pi * x / 2) ** 2, 0.)
    # The source masks only ell<2; centers specify support, not a hard cut.
    windows[:, ell < 2] = 0
    return windows, np.exp(logcenters)


class FullSkySHT:
    """Equal-area HEALPix analysis A=Omega Y*, with m-major packed alms.

    Y is real-field synthesis. The weighted harmonic metric has weights
    mult=1 for m=0 and 2 for m>0. DUCC's adjoint_synthesis is Y*/mult;
    consequently the ordinary packed-coordinate A* is Omega Y(cot/mult).
    This distinction is essential: synthesis alone is not the analysis VJP.
    """
    def __init__(self, nside, lmax=None, nthreads=1):
        self.nside = int(nside)
        if self.nside < 1 or self.nside & (self.nside-1):
            raise ValueError("NESTED HEALPix nside must be a positive power of two")
        self.lmax = 3*self.nside-1 if lmax is None else int(lmax)
        if not 0 <= self.lmax <= 3*self.nside-1:
            raise ValueError("lmax must lie in [0,3*nside-1]")
        self.nthreads = int(nthreads)
        if self.nthreads < 1:
            raise ValueError("Set an explicit positive CPU thread allocation")
        self.npix = 12*self.nside**2
        self.area = 4*np.pi/self.npix
        base = ducc0.healpix.Healpix_Base(self.nside, "RING")
        self.geometry = base.sht_info()
        self.nest_to_ring = base.nest2ring(np.arange(self.npix, dtype=np.int64))
        self.ell = np.concatenate([np.arange(m, self.lmax+1) for m in range(self.lmax+1)])
        self.m = np.concatenate([np.full(self.lmax-m+1, m) for m in range(self.lmax+1)])
        self.mult = np.where(self.m == 0, 1., 2.)
        self.nalm = self.ell.size
        theta_ring = np.repeat(self.geometry["theta"], self.geometry["nphi"].astype(np.int64))
        theta = theta_ring[self.nest_to_ring]
        self.inv_sin = 1/np.sin(theta)
        self.cot_theta = np.cos(theta)*self.inv_sin
        self._kw = dict(self.geometry, lmax=self.lmax, nthreads=self.nthreads)

    def _map(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.npix,):
            raise ValueError(f"Expected one NESTED map ({self.npix},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite full-sky map")
        return x

    def _alm(self, a):
        a = np.asarray(a, dtype=np.complex128)
        if a.shape != (self.nalm,):
            raise ValueError(f"Expected packed alm ({self.nalm},), got {a.shape}")
        return a

    def _ring(self, x):
        x = self._map(x)
        ring = np.empty_like(x)
        ring[self.nest_to_ring] = x
        return ring

    def synthesis(self, alm):
        a = self._alm(alm)
        ring = ducc0.sht.experimental.synthesis(alm=a[None, :], spin=0, **self._kw)[0]
        return ring[self.nest_to_ring]

    def synthesis_vjp(self, cotangent):
        ring = self._ring(cotangent)
        a = ducc0.sht.experimental.adjoint_synthesis(map=ring[None, :], spin=0, **self._kw)[0]
        return a*self.mult

    def analysis(self, map_nest):
        ring = self._ring(map_nest)
        return self.area*ducc0.sht.experimental.adjoint_synthesis(map=ring[None, :], spin=0, **self._kw)[0]

    def analysis_vjp(self, cotangent):
        return self.area*self.synthesis(self._alm(cotangent)/self.mult)

    def power(self, alm):
        a = self._alm(alm)
        return np.bincount(self.ell, weights=self.mult*(a.real*a.real+a.imag*a.imag),
                           minlength=self.lmax+1)/(2*np.arange(self.lmax+1)+1)

    def power_vjp(self, alm, dcl):
        a = self._alm(alm)
        dcl = np.asarray(dcl, dtype=np.float64)
        if dcl.shape != (self.lmax+1,):
            raise ValueError("C_ell cotangent has wrong shape")
        return self.analysis_vjp(2*self.mult*a*dcl[self.ell]/(2*self.ell+1))

    def cache_analysis(self, map_nest, generation, member, objective="physical"):
        x = np.ascontiguousarray(self._map(map_nest))
        a = self.analysis(x)
        a.flags.writeable = False
        return AlmState(a, int(generation), int(member), str(objective), self.nside,
                        self.lmax, "complex128", hashlib.sha256(x.view(np.uint8)).hexdigest(), id(self))

    def validate_alm_state(self, state, generation, member, objective="physical"):
        expected = (int(generation), int(member), str(objective), self.nside, self.lmax, "complex128", id(self))
        got = (state.generation, state.member, state.objective, state.nside, state.lmax, state.dtype, state.owner)
        if got != expected or state.alm.shape != (self.nalm,) or state.alm.dtype != np.complex128:
            raise ValueError("Stale, wrong-member, wrong-objective, or incompatible alm state")

    def cached_power_vjp(self, state, dcl, generation, member, objective="physical"):
        self.validate_alm_state(state, generation, member, objective)
        return self.power_vjp(state.alm, dcl)

    def _deriv1(self, alm):
        ring = ducc0.sht.experimental.synthesis_deriv1(alm=self._alm(alm)[None, :], **self._kw)
        return ring[:, self.nest_to_ring]

    def _deriv1_vjp(self, dtheta, dphi_over_sin):
        ring = np.stack([self._ring(dtheta), self._ring(dphi_over_sin)])
        a = ducc0.sht.experimental.adjoint_synthesis(map=ring, spin=1, mode="DERIV1", **self._kw)[0]
        return a*self.mult

    def derivative_maps(self, alm):
        """Covariant Hessian in the orthonormal (theta,phi) frame.

        Returns d_theta, d_phi/sin(theta), H_theta_theta, H_phi_phi,
        H_theta_phi. The Laplace identity avoids stored Legendre tables.
        """
        a = self._alm(alm)
        d1, d2 = self._deriv1(a)
        d22 = self.synthesis(-(self.m*self.m)*a)*self.inv_sin**2 + self.cot_theta*d1
        d11 = self.synthesis(-self.ell*(self.ell+1)*a)-d22
        mixed = self._deriv1(1j*self.m*a)[0]
        d12 = mixed*self.inv_sin - self.cot_theta*d2
        return d1, d2, d11, d22, d12

    def derivative_alm_vjp(self, cotangents):
        c1, c2, c11, c22, c12 = [self._map(x) for x in cotangents]
        c22effective = c22-c11
        ga = self._deriv1_vjp(c1+self.cot_theta*c22effective,
                             c2-self.cot_theta*c12)
        ga += -(self.m*self.m)*self.synthesis_vjp(c22effective*self.inv_sin**2)
        ga += -self.ell*(self.ell+1)*self.synthesis_vjp(c11)
        ga += -1j*self.m*self._deriv1_vjp(c12*self.inv_sin, np.zeros(self.npix))
        return ga

    def derivatives_vjp(self, cotangents):
        return self.analysis_vjp(self.derivative_alm_vjp(cotangents))
