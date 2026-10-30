"""Bounded-workspace per-member CIC density and its piecewise-exact VJP."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: pdf.py {SCRIPT_VERSION}", flush=True)


@dataclass(frozen=True)
class PDFState:
    physical_map: np.ndarray
    outside_fraction: float
    owner: int


class CICPDF:
    def __init__(self, x_min, x_max, n_bins=100, chunk_pixels=262144):
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.n_bins, self.chunk_pixels = int(n_bins), int(chunk_pixels)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min or n_bins < 3:
            raise ValueError("invalid CIC grid")
        if self.chunk_pixels < 1:
            raise ValueError("chunk_pixels must be positive")
        self.dx = (self.x_max - self.x_min) / (self.n_bins - 1)
        self.centers = np.linspace(self.x_min, self.x_max, self.n_bins)

    @classmethod
    def from_training(cls, maps, **kwargs):
        a = np.asanyarray(maps)
        if a.ndim != 2 or a.shape[0] != 54:
            raise ValueError("CIC support must be calibrated from TRAIN54 only")
        low, high = float(np.min(a)), float(np.max(a))
        span = max(high - low, np.finfo(np.float64).eps)
        return cls(low - 0.02 * span, high + 0.02 * span, **kwargs)

    def forward(self, physical_map):
        x = np.asarray(physical_map, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("CIC forward takes exactly one nonempty member")
        h = np.zeros(self.n_bins, dtype=np.float64)
        outside = 0
        for start in range(0, x.size, self.chunk_pixels):
            part = x[start:start + self.chunk_pixels]
            if not np.all(np.isfinite(part)):
                raise ValueError("nonfinite CIC input")
            pos = (part - self.x_min) / self.dx
            i0 = np.floor(pos).astype(np.int64)
            frac = pos - i0
            valid = (i0 >= 0) & (i0 < self.n_bins)
            h += np.bincount(i0[valid], weights=(1.0 - frac[valid]), minlength=self.n_bins)
            i1 = i0 + 1
            valid = (i1 >= 0) & (i1 < self.n_bins)
            h += np.bincount(i1[valid], weights=frac[valid], minlength=self.n_bins)
            outside += int(np.count_nonzero((part < self.x_min) | (part > self.x_max)))
        h /= x.size * self.dx
        return h, PDFState(x, outside / x.size, id(self))

    def vjp(self, state, cotangent):
        if not isinstance(state, PDFState) or state.owner != id(self):
            raise ValueError("CIC state belongs to another PDF grid")
        q = np.asarray(cotangent, dtype=np.float64)
        if q.shape != (self.n_bins,):
            raise ValueError("CIC cotangent shape mismatch")
        x = state.physical_map
        grad = np.zeros_like(x)
        for start in range(0, x.size, self.chunk_pixels):
            end = min(start + self.chunk_pixels, x.size)
            indices = np.floor((x[start:end] - self.x_min) / self.dx).astype(np.int64)
            valid = (indices >= 0) & (indices < self.n_bins)
            g = grad[start:end]
            g[valid] -= q[indices[valid]]
            indices += 1
            valid = (indices >= 0) & (indices < self.n_bins)
            g[valid] += q[indices[valid]]
        grad /= x.size * self.dx ** 2
        return grad
