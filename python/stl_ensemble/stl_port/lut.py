"""Pooled TRAIN54 QQ LUT, source knot convention and differentiable inverse.

The reference stores physical knots and Gaussian quantile coordinates as
float32. Those exact rounded knots are retained. Native map arithmetic is
float64, avoiding a hidden float32 map cast between the native engine and
its VJP. This precision distinction is recorded, not a different mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.special import erfinv

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: lut.py {SCRIPT_VERSION}", flush=True)
PCLIP = 1e-15


def source_ppf(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), PCLIP, 1.0 - PCLIP)
    # This intentionally uses erfinv(2p-1), as the source, rather than ndtri(p).
    return (np.sqrt(2.0) * erfinv(2.0 * p - 1.0)).astype(np.float32).astype(np.float64)


def probability_grid(nbins, count):
    pmin = max(0.5 / (count + 1.0), PCLIP)
    pmax = 1.0 - pmin
    tail_p = min(max(1e-3, pmin * 10.0), 0.49)
    n_tail = int(max(4, np.floor(0.25 * nbins)))
    n_core = nbins - 2 * n_tail
    if n_core < 8:
        p = np.linspace(pmin, pmax, nbins, dtype=np.float64)
    else:
        left = np.logspace(np.log10(pmin), np.log10(tail_p), n_tail,
                           endpoint=False, dtype=np.float64)
        core = np.linspace(tail_p, 1.0 - tail_p, n_core, endpoint=True,
                           dtype=np.float64)
        p = np.unique(np.concatenate([left, core, 1.0 - left[::-1]]))
        p = p[(p >= pmin) & (p <= pmax)]
        if p.size != nbins:
            p = np.linspace(pmin, pmax, nbins, dtype=np.float64)
    p = np.clip(p, PCLIP, 1.0 - PCLIP)
    for i in range(1, p.size):
        if p[i] <= p[i - 1]:
            p[i] = np.nextafter(p[i - 1], 1.0)
    return p


def _interval(x, knots):
    # The right-sided search matches the source autograd branch at a knot.
    return np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)


def _interpolate(x, knots, values):
    i = _interval(x, knots)
    slope = (values[i + 1] - values[i]) / (knots[i + 1] - knots[i])
    return values[i] + (x - knots[i]) * slope


@dataclass(frozen=True)
class InverseState:
    gaussian_map: np.ndarray
    remove_monopole: bool
    owner: int


class PooledLUT:
    """One fixed shared LUT. No per-map rank match, saturation, or dipole removal."""
    def __init__(self, probabilities, physical_knots, chunk_pixels=262144):
        self.probabilities = np.asarray(probabilities, dtype=np.float64).copy()
        self.physical_knots = np.asarray(physical_knots, dtype=np.float64).copy()
        self.gaussian_knots = source_ppf(self.probabilities)
        self.chunk_pixels = int(chunk_pixels)
        if self.chunk_pixels < 1:
            raise ValueError("chunk_pixels must be positive")
        if self.probabilities.ndim != 1 or self.physical_knots.shape != self.probabilities.shape:
            raise ValueError("LUT knots must be matching 1-D vectors")
        if self.probabilities.size < 2:
            raise ValueError("LUT requires at least two knots")
        for values in (self.probabilities, self.physical_knots, self.gaussian_knots):
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
                raise ValueError("LUT knots must be finite and strictly increasing")

    @classmethod
    def from_training(cls, maps, work_path=None, nbins=8192, chunk_pixels=262144):
        """Exact pooled quantiles, optionally with an explicitly disposable memmap.

        All TRAIN54 pixels enter the quantile. No random subsampling or
        approximate quantile sketch is used. `work_path` is created exclusively
        and removed on success/failure; quantile partition may write all of it.
        An 8192-knot grid is the scientific default; a smaller grid is useful
        solely in numerical unit tests.
        """
        a = np.asanyarray(maps)
        if a.ndim != 2 or a.shape[0] != 54:
            raise ValueError("pooled calibration requires exactly TRAIN54 maps [54,Npix]")
        if nbins < 2:
            raise ValueError("invalid LUT knot count")
        shape = a.shape
        p = probability_grid(nbins, a.size)
        # The source first casts to float32, then computes float64 quantiles.
        if work_path is None and a.size * 8 > 512 * 2**20:
            raise ValueError("large exact pooled LUT requires an explicit work_path for its temporary memmap")
        path = Path(work_path) if work_path is not None else None
        created = False
        pool = None
        try:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as f:
                    f.truncate(a.size * 8)
                created = True
                pool = np.memmap(path, dtype=np.float64, mode="r+", shape=(a.size,))
            else:
                pool = np.empty(a.size, dtype=np.float64)
            for member in range(shape[0]):
                for start in range(0, shape[1], chunk_pixels):
                    end = min(start + chunk_pixels, shape[1])
                    part = np.asarray(a[member, start:end], dtype=np.float32)
                    if not np.all(np.isfinite(part)):
                        raise ValueError("TRAIN54 contains a nonfinite pixel")
                    offset = member * shape[1]
                    pool[offset + start:offset + end] = part
            x = np.quantile(pool, p, method="linear", overwrite_input=True).astype(np.float32)
            eps = max(abs(float(x[-1] - x[0])) * 1e-7, 1e-12)
            for i in range(1, x.size):
                if x[i] <= x[i - 1]:
                    x[i] = x[i - 1] + eps
            return cls(p, x, chunk_pixels=chunk_pixels)
        finally:
            if isinstance(pool, np.memmap):
                pool.flush()
                pool._mmap.close()
            if created:
                path.unlink()

    def gaussianize(self, physical_map):
        x = np.asarray(physical_map)
        out = np.empty(x.shape, dtype=np.float64)
        a, b = x.reshape(-1), out.reshape(-1)
        for start in range(0, a.size, self.chunk_pixels):
            stop = min(start + self.chunk_pixels, a.size)
            p = _interpolate(np.asarray(a[start:stop], dtype=np.float64),
                             self.physical_knots, self.probabilities)
            b[start:stop] = source_ppf(p)
        return out

    def inverse(self, gaussian_map, remove_monopole=True):
        """Return physical map and VJP state. Caller must not mutate the trial map."""
        g = np.asarray(gaussian_map, dtype=np.float64)
        if g.ndim < 1:
            raise ValueError("a map needs a pixel axis")
        u = np.empty_like(g)
        a, b = g.reshape(-1), u.reshape(-1)
        for start in range(0, a.size, self.chunk_pixels):
            stop = min(start + self.chunk_pixels, a.size)
            b[start:stop] = _interpolate(a[start:stop], self.gaussian_knots, self.physical_knots)
        if remove_monopole:
            u -= u.mean(axis=-1, keepdims=True)
        return u, InverseState(g, bool(remove_monopole), id(self))

    def vjp(self, state, physical_cotangent):
        if not isinstance(state, InverseState) or state.owner != id(self):
            raise ValueError("inverse-LUT state belongs to another LUT")
        g = state.gaussian_map
        q = np.asarray(physical_cotangent, dtype=np.float64)
        if q.shape != g.shape:
            raise ValueError("inverse-LUT gradient shape mismatch")
        out = np.empty_like(g)
        # Project each physical cotangent before multiplying by f'(g).
        mean = q.mean(axis=-1, keepdims=True) if state.remove_monopole else np.zeros(q.shape[:-1] + (1,))
        rows_g = g.reshape(-1, g.shape[-1])
        rows_q = q.reshape(rows_g.shape)
        rows_o = out.reshape(rows_g.shape)
        means = mean.reshape(-1)
        for member in range(rows_g.shape[0]):
            for start in range(0, g.shape[-1], self.chunk_pixels):
                stop = min(start + self.chunk_pixels, g.shape[-1])
                indices = _interval(rows_g[member, start:stop], self.gaussian_knots)
                slope = ((self.physical_knots[indices + 1] - self.physical_knots[indices])
                         / (self.gaussian_knots[indices + 1] - self.gaussian_knots[indices]))
                rows_o[member, start:stop] = slope * (rows_q[member, start:stop] - means[member])
        return out
