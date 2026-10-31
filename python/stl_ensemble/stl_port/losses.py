"""Exact compact-ensemble losses for the approved five-block Step-1 port.

Cotangents use dL = Re(sum(conj(cotangent)*dvalues)). Complex features
therefore count once, and their population variance is Var(Re)+Var(Im).
There is no global division by five here: the joint objective applies it
once to both its scalar and map gradient. All calculations use float64.
"""
from __future__ import annotations

import numpy as np
from .sc_variance import variance_ratio_loss, variance_ratio_summary

SCRIPT_VERSION = "2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: losses.py {SCRIPT_VERSION}", flush=True)

# Keep fallback constants of the source float32 statistics. These are not
# tunable physical floors. Non-SC-variance source floors remain unchanged in v2.
SOURCE_EPS = np.finfo(np.float32).eps
SOURCE_TINY = np.finfo(np.float32).tiny


def torch_median(values):
    """Lower middle order statistic, matching torch.median for even sizes."""
    x = np.asarray(values).ravel()
    if x.size == 0:
        raise ValueError("median requires at least one value")
    k = (x.size - 1) // 2
    return float(np.partition(x, k)[k])


def moments(values):
    """Mean and unbiased ddof=1 variance; the first axis is members."""
    x = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(x) else np.float64
    x = np.asarray(x, dtype=dtype)
    if x.ndim < 2 or x.shape[0] < 2:
        raise ValueError("population losses need at least two members")
    if not np.all(np.isfinite(x)):
        raise ValueError("nonfinite member statistics")
    mu = x.mean(axis=0)
    centered = x - mu
    var = np.sum(np.abs(centered) ** 2, axis=0) / (x.shape[0] - 1)
    return mu, var


def moments_vjp(values, mean_cotangent, variance_cotangent):
    """Exact joint population pullback, not independent member fitting."""
    x = np.asarray(values)
    b = x.shape[0]
    if b < 2:
        raise ValueError("unbiased variance requires B >= 2")
    return (np.asarray(mean_cotangent) / b
            + 2.0 * np.asarray(variance_cotangent) * (x - x.mean(axis=0)) / (b - 1))


def _targets(values, mean, variance):
    x = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(x) else np.float64
    x = np.asarray(x, dtype=dtype)
    t = np.asarray(mean, dtype=dtype)
    v = np.asarray(variance, dtype=np.float64)
    if t.shape != x.shape[1:] or v.shape != t.shape:
        raise ValueError(f"feature/target shape mismatch: {x.shape}, {t.shape}, {v.shape}")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
        raise ValueError("target contains nonfinite values")
    if np.any(v < 0):
        raise ValueError("target population variance cannot be negative")
    return x, t, v


def _family(item):
    return item["family"] if isinstance(item, dict) else item.family


def equalize_sigma(sigma, plan):
    """Source physical sigma equalization on each native final-orientation axis.

    The native plan is flattened and level-sorted. For each family its final
    orientation is l1 for S1/S2 and l3 for S3/S4. Grouped means reproduce an
    explicit (..., L) reshape without assuming contiguous plan ordering.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (len(plan),):
        raise ValueError("sigma must have one real entry per coefficient")
    if np.any(sigma < 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("within-map sigma must be finite and nonnegative")
    out = sigma.copy()
    families = np.array([_family(p) for p in plan])
    for family in ("S1", "S2", "S3", "S4"):
        ii = np.flatnonzero(families == family)
        if ii.size == 0:
            continue
        med = torch_median(sigma[ii])
        s = np.clip(sigma[ii], med * 1e-6, med * 1e6)
        key = "l1" if family in ("S1", "S2") else "l3"
        labels = np.asarray([plan[k][key] if isinstance(plan[k], dict)
                             else getattr(plan[k], key) for k in ii])
        if any(label is None for label in labels):
            raise ValueError(f"{family} plan missing {key}")
        distinct, counts = np.unique(labels, return_counts=True)
        if np.any(counts != counts[0]):
            raise ValueError("physical equalization requires complete orientation tuples")
        scale = np.asarray([s[labels == label].mean() for label in distinct])
        scale /= torch_median(scale) + 1e-12
        for j, label in enumerate(distinct):
            s[labels == label] /= scale[j] + 1e-12
        out[ii] = np.maximum(s, 1e-12)
    return out


def sc_loss(values, target_mean, target_var, sigma, plan, physical=False, *, diagnostics=None):
    """Unchanged native SC mean plus dimensionless log-variance-ratio penalty.

    Sigma is the TRAIN54 mean of within-map estimator sigmas, after the
    externally documented native feature normalization. It must never be
    substituted with the across-map population standard deviation.
    """
    x, t, vref = _targets(values, target_mean, target_var)
    if x.ndim != 2 or x.shape[1] != len(plan):
        raise ValueError("SC expects [B,len(plan)]")
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.shape != t.shape or not np.all(np.isfinite(sig)) or np.any(sig < 0):
        raise ValueError("invalid within-map SC sigma")
    families = np.array([_family(p) for p in plan])
    mask = np.isin(families, ("S1", "S2", "S3", "S4")) if physical else np.ones(t.shape, bool)
    if physical and any(not np.any(families == f) for f in ("S1", "S2", "S3", "S4")):
        raise ValueError("physical native plan must retain all S1-S4 families")
    n = int(mask.sum())
    if n == 0:
        raise ValueError("empty SC plan")
    if physical:
        sig = equalize_sigma(sig, plan)
    sig = np.maximum(sig, 1e-12)
    mu, var = moments(x)
    dm, dv = np.zeros_like(mu), np.zeros_like(var)
    loss = 0.0
    groups = [families == f for f in ("S1", "S2", "S3", "S4")] if physical else [mask]
    for selected in groups:
        count = int(selected.sum())
        residual = mu[selected] - t[selected]
        loss += float(np.sum(np.abs(residual) ** 2 / sig[selected] ** 2) / count)
        dm[selected] = 2.0 * residual / (count * sig[selected] ** 2)
    # Only this population-variance term changes in v2. The mean term above,
    # including physical equalization and its absolute sigma floor, is unchanged.
    variance_loss, dv[mask] = variance_ratio_loss(var[mask], vref[mask])
    if diagnostics is not None:
        diagnostics.update(mean_loss_unchanged_v1=float(loss),
                           variance_loss=float(variance_loss),
                           variance=variance_ratio_summary(var[mask], vref[mask]),
                           by_family={str(f): variance_ratio_summary(var[(families == f) & mask],
                                                                   vref[(families == f) & mask])
                                      for f in np.unique(families[mask])})
    loss += variance_loss
    return loss, moments_vjp(x, dm, dv)


def sc_floor_report(target_var, plan, floor=1e-12):
    """Legacy absolute-floor audit only; NOT the active v2 variance threshold."""
    v = np.asarray(target_var, dtype=np.float64)
    family = np.asarray([_family(p) for p in plan])
    return {str(f): {"total": int(np.sum(family == f)),
                     "below_floor": int(np.sum(v[family == f] < floor)),
                     "at_or_above_floor": int(np.sum(v[family == f] >= floor)),
                     "min": float(v[family == f].min()), "max": float(v[family == f].max())}
            for f in np.unique(family)}


def power_loss(values, target_mean, target_var, centers, cutoff=383):
    x, t, vref = _targets(values, target_mean, target_var)
    if x.ndim != 2 or np.iscomplexobj(x):
        raise ValueError("LP bandpowers must be real [B,nband]")
    centers = np.asarray(centers)
    if centers.shape != t.shape:
        raise ValueError("band centers do not match bandpowers")
    if cutoff is None:
        raise ValueError("the physical PS variance upper cutoff must be resolved explicitly")
    mask = (centers >= 20) & (centers <= cutoff)
    if not np.any(mask):
        raise ValueError("physical PS variance mask retained zero bands")
    mu, var = moments(x)
    epsilon = 1e-12 * max(torch_median(np.abs(t)), 1e-30)
    safe = np.maximum(mu, epsilon)
    d = np.log(safe) - np.log(np.maximum(t, epsilon))
    loss = 10.0 * float(np.mean(d * d))
    dm = np.where(mu >= epsilon, 20.0 * d / (mu.size * safe), 0.0)
    epsb = 0.01 * np.maximum(vref, 1e-30)
    r = np.log((var[mask] + epsb[mask]) / (vref[mask] + epsb[mask]))
    loss += 2.5 * float(np.mean(r * r))
    dv = np.zeros_like(var)
    dv[mask] = 5.0 * r / (int(mask.sum()) * (var[mask] + epsb[mask]))
    return loss, moments_vjp(x, dm, dv)


def pdf_loss(values, target_mean, target_var):
    x, t, vref = _targets(values, target_mean, target_var)
    if x.ndim != 2 or np.iscomplexobj(x):
        raise ValueError("CIC densities must be real [B,nbins]")
    mu, var = moments(x)
    dm, dv = np.zeros_like(mu), np.zeros_like(var)
    mask = np.isfinite(t) & (t > np.max(np.maximum(t, 0.0)) * 5e-6)
    if not np.any(mask):
        return 0.0, np.zeros_like(x)
    std = np.sqrt(vref)
    positive = std[mask & np.isfinite(std) & (std > 0)]
    sfloor = max(0.05 * torch_median(positive) if positive.size else SOURCE_EPS, SOURCE_TINY)
    denom = np.maximum(std, sfloor)
    d = (mu[mask] - t[mask]) / denom[mask]
    loss = float(np.mean(d * d))
    dm[mask] = 2.0 * d / (int(mask.sum()) * denom[mask])
    positive = vref[mask & np.isfinite(vref) & (vref > 0)]
    vfloor = max(1e-4 * torch_median(positive) if positive.size else SOURCE_EPS, SOURCE_TINY)
    vmask = mask & np.isfinite(vref) & (vref > vfloor)
    if np.any(vmask):
        epsilon = max(0.01 * vfloor, SOURCE_TINY)
        r = np.log((var[vmask] + epsilon) / (vref[vmask] + epsilon))
        loss += float(np.mean(r * r))
        dv[vmask] = 2.0 * r / (int(vmask.sum()) * (var[vmask] + epsilon))
    return loss, moments_vjp(x, dm, dv)


def mf_loss(values, target_mean, target_var):
    """Sum V0/V1/V2 mean (weight 1) and variance (weight 10) terms."""
    x, t, vref = _targets(values, target_mean, target_var)
    if x.ndim != 3 or x.shape[1] != 3 or np.iscomplexobj(x):
        raise ValueError("MF values must be real [B,3,nthresholds]")
    mu, var = moments(x)
    dm, dv = np.zeros_like(mu), np.zeros_like(var)
    loss = 0.0
    for component in range(3):
        floor = max(1e-4 * float(np.max(vref[component])), SOURCE_TINY)
        mask = np.isfinite(vref[component]) & (vref[component] > floor)
        count = int(mask.sum())
        if count == 0:
            continue
        d = mu[component, mask] - t[component, mask]
        loss += float(np.mean(d * d / vref[component, mask]))
        dm[component, mask] = 2.0 * d / (count * vref[component, mask])
        epsilon = max(0.01 * floor, SOURCE_TINY)
        r = np.log((var[component, mask] + epsilon) / (vref[component, mask] + epsilon))
        loss += 10.0 * float(np.mean(r * r))
        dv[component, mask] = 20.0 * r / (count * (var[component, mask] + epsilon))
    return loss, moments_vjp(x, dm, dv)
