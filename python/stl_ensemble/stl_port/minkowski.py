"""Full-sky physical V0,V1,V2 with the reference detached curvature floor.

Each realization is standardized independently and differentiated through its
mean and standard deviation. Thresholding uses that raw standardized map;
only spatial derivatives use harmonic analysis/synthesis. No angular beam.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: minkowski.py {SCRIPT_VERSION}", flush=True)


@dataclass(frozen=True)
class MinkowskiState:
    u: np.ndarray
    sigma: float
    derivatives: tuple
    floor: float
    owner: int


class Minkowski:
    def __init__(self, sht, pixel_chunk=262144, threshold_chunk=8):
        self.sht = sht
        self.nu = np.linspace(-4., 4., 65)
        self.epsilon = .125
        self.grad_floor_fraction = 1e-4
        self.pixel_chunk = int(pixel_chunk)
        self.threshold_chunk = int(threshold_chunk)
        if min(self.pixel_chunk, self.threshold_chunk) < 1:
            raise ValueError("Chunk sizes must be positive")

    def _chunks(self):
        for start in range(0, self.sht.npix, self.pixel_chunk):
            yield slice(start, min(start+self.pixel_chunk, self.sht.npix))

    def forward(self, kappa, fixed_floor=None):
        x = self.sht._map(kappa)
        centered = x-x.mean()
        sigma = float(np.sqrt(np.mean(centered*centered)+1e-12))
        u = centered/sigma
        derivatives = tuple(self.sht.derivative_maps(self.sht.analysis(u)))
        if fixed_floor is None:
            # Detached: the VJP deliberately contains no derivative of this.
            qmean = sum(np.sum(derivatives[0][p]**2+derivatives[1][p]**2)
                        for p in self._chunks())/self.sht.npix
            floor = float(self.grad_floor_fraction*qmean+1e-12)
        else:
            floor = float(fixed_floor)
            if not np.isfinite(floor) or floor <= 0:
                raise ValueError("fixed_floor must be finite and positive")
        result = np.zeros((3, self.nu.size))
        for p in self._chunks():
            d1, d2, d11, d22, d12 = [d[p] for d in derivatives]
            q = d1*d1+d2*d2
            mag = np.sqrt(q+1e-30)
            curvature = (2*d1*d2*d12-d1*d1*d22-d2*d2*d11)/(q+floor)
            for k in range(0, self.nu.size, self.threshold_chunk):
                sl = slice(k, k+self.threshold_chunk)
                z = (u[p][None, :]-self.nu[sl, None])/self.epsilon
                result[0, sl] += ndtr(z).sum(axis=1)
                delta = np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*self.epsilon)
                result[1, sl] += .25*(delta@mag)
                result[2, sl] += (delta@curvature)/(2*np.pi)
        result /= self.sht.npix
        return result, MinkowskiState(u, sigma, derivatives, floor, id(self))

    def vjp(self, state, cotangent):
        if state.owner != id(self) or state.u.shape != (self.sht.npix,):
            raise ValueError("MF state belongs to a different evaluator/grid")
        c = np.asarray(cotangent, dtype=np.float64)
        if c.shape != (3, 65) or not np.all(np.isfinite(c)):
            raise ValueError("MF cotangent must be finite with shape (3,65)")
        n = self.sht.npix
        gu = np.zeros(n)
        gd = [np.zeros(n) for _ in range(5)]
        for p in self._chunks():
            d1, d2, d11, d22, d12 = [d[p] for d in state.derivatives]
            q = d1*d1+d2*d2
            mag = np.sqrt(q+1e-30)
            den = q+state.floor
            numerator = 2*d1*d2*d12-d1*d1*d22-d2*d2*d11
            curvature = numerator/den
            a = np.zeros_like(d1)
            b = np.zeros_like(d1)
            local_gu = np.zeros_like(d1)
            for k in range(0, self.nu.size, self.threshold_chunk):
                sl = slice(k, k+self.threshold_chunk)
                z = (state.u[p][None, :]-self.nu[sl, None])/self.epsilon
                delta = np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*self.epsilon)
                delta_prime = -z*delta/self.epsilon
                local_gu += c[0, sl]@delta
                local_gu += (c[1, sl]@delta_prime)*mag/4
                local_gu += (c[2, sl]@delta_prime)*curvature/(2*np.pi)
                a += (c[1, sl]@delta)/4
                b += (c[2, sl]@delta)/(2*np.pi)
            gu[p] = local_gu/n
            a /= n
            b /= n
            gd[0][p] = a*d1/mag+b*((2*d2*d12-2*d1*d22)/den-2*d1*numerator/(den*den))
            gd[1][p] = a*d2/mag+b*((2*d1*d12-2*d2*d11)/den-2*d2*numerator/(den*den))
            gd[2][p] = -b*d2*d2/den
            gd[3][p] = -b*d1*d1/den
            gd[4][p] = b*2*d1*d2/den
        gu += self.sht.derivatives_vjp(gd)
        # Pull back u=(x-mean(x))/sqrt(mean((x-mean(x))^2)+1e-12).
        return (gu-gu.mean()-state.u*np.mean(gu*state.u))/state.sigma


MinkowskiSHT = Minkowski
