"""Read-only SC target and effective mean-sigma diagnostics for v2."""
from __future__ import annotations
import numpy as np
from .sc_variance import CONTRACT, RATIO_FLOOR
from .losses import equalize_sigma, torch_median

SCRIPT_VERSION = "2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: sc_reports.py {SCRIPT_VERSION}", flush=True)
SC_FAMILIES = ("S1", "S2", "S3", "S4")


def selected_mask(plan, physical=False):
    f = np.asarray([p["family"] for p in plan])
    return np.isin(f, SC_FAMILIES) if physical else np.ones(len(plan), dtype=bool)


def variance_sensitivity_issues(targets, plan):
    """Reject invalid targets coefficient by coefficient; never silently mask.

    Gaussian mean and recombined S0-second entries are included. Physical S0
    entries are not registered in that block and are reported as such.
    """
    issues = []
    family = np.asarray([p["family"] for p in plan])
    if not len(plan):
        return ["empty native SC plan"]
    for prefix in ("gsc", "psc"):
        v = np.asarray(targets[prefix + "_var"], dtype=np.float64)
        if v.shape != (len(plan),):
            issues.append(f"{prefix}: variance/plan shape mismatch")
            continue
        mask = selected_mask(plan, prefix == "psc")
        for f in SC_FAMILIES:
            if not np.any(family == f):
                issues.append(f"{prefix}/{f}: native coefficient family missing")
        bad = mask & (~np.isfinite(v) | (v <= 0))
        for f in sorted(set(family[bad])):
            ii = np.flatnonzero(bad & (family == f))
            issues.append(f"{prefix}/{f}: {ii.size} selected TRAIN54 variances are "
                          f"nonpositive or nonfinite; plan indices {ii[:16].tolist()}")
    return issues


def _percentiles(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return np.percentile(v, [0, 10, 50, 90, 100]).tolist() if v.size else None


def variance_floor_report(targets, plan):
    """New relative sensitivity with explicitly labeled legacy-floor diagnostics."""
    family = np.asarray([p["family"] for p in plan])
    out = {"contract": dict(CONTRACT), "legacy_absolute_floor_is_not_used": 1e-12}
    for prefix in ("gsc", "psc"):
        mask = selected_mask(plan, prefix == "psc")
        n = max(1, int(mask.sum()))
        out[prefix] = {}
        v = np.asarray(targets[prefix + "_var"], dtype=np.float64)
        for f in sorted(set(family)):
            ii = family == f
            vv = v[ii]
            registered = bool(np.any(ii & mask))
            good = np.isfinite(vv) & (vv > 0)
            positive = vv[good]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                dv = (2*np.log(1.1)/n) / (1.1*positive) if registered else np.empty(0)
            out[prefix][f] = {
                "count": int(ii.sum()), "registered_in_variance_loss": registered,
                "positive_finite_targets": int(good.sum()),
                "zero_targets": int(np.count_nonzero(vv == 0)),
                "negative_targets": int(np.count_nonzero(vv < 0)),
                "nonfinite_targets": int(np.count_nonzero(~np.isfinite(vv))),
                "variance_percentiles_0_10_50_90_100": _percentiles(vv),
                "log10_positive_variance_percentiles_0_10_50_90_100": _percentiles(np.log10(positive)),
                "legacy_at_or_below_absolute_1e_12": int(np.count_nonzero(vv <= 1e-12)),
                "ratio_floor": RATIO_FLOOR,
                "sensitivity_at_syn_equals_1p1_train": {
                    "registered_coefficients": int(vv.size) if registered else 0,
                    "finite_nonzero_derivative_count": int(np.count_nonzero(np.isfinite(dv) & (dv != 0))),
                    "log10_abs_derivative_percentiles": _percentiles(np.log10(np.abs(dv))),
                    "includes_global_division_by_five": False,
                },
            }
    return out


def mean_sigma_report(targets, plan):
    """Report v1 sigma processing, including its FINAL absolute floor.

    The pre-floor computation here mirrors equalize_sigma ONLY to report
    activation; the optimization still calls the unchanged original function.
    An exact check ensures diagnostics did not drift from the executed function.
    """
    family = np.asarray([p["family"] for p in plan])
    out = {"method": "unchanged v1; TRAIN54 average within-map sigma, not population std",
           "final_absolute_sigma_floor": 1e-12,
           "warning": "reporting only; no mean-loss weights changed"}
    for prefix in ("gsc", "psc"):
        raw = np.asarray(targets[prefix + "_sigma"], dtype=np.float64)
        if raw.shape != (len(plan),) or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError(f"{prefix}: invalid within-map sigma")
        pre = raw.copy()
        if prefix == "psc":
            for f in SC_FAMILIES:
                ii = np.flatnonzero(family == f)
                if not ii.size:
                    continue
                med = torch_median(raw[ii])
                vals = np.clip(raw[ii], med*1e-6, med*1e6)
                key = "l1" if f in ("S1", "S2") else "l3"
                labels = np.asarray([plan[k][key] for k in ii])
                distinct = np.unique(labels)
                scale = np.asarray([vals[labels == a].mean() for a in distinct])
                scale /= torch_median(scale) + 1e-12
                for j, a in enumerate(distinct):
                    vals[labels == a] /= scale[j] + 1e-12
                pre[ii] = vals
            effective = np.maximum(equalize_sigma(raw, plan), 1e-12)
        else:
            effective = np.maximum(raw, 1e-12)
        if not np.array_equal(effective, np.maximum(pre, 1e-12)):
            raise RuntimeError("Mean-sigma diagnostic differs from unchanged v1 implementation")
        mask = selected_mask(plan, prefix == "psc")
        out[prefix] = {}
        for f in sorted(set(family)):
            ii = family == f
            weight_ratio = (pre[ii] / effective[ii])**2
            out[prefix][f] = {
                "count": int(ii.sum()), "registered_in_mean_loss": bool(np.any(ii & mask)),
                "raw_sigma_percentiles": _percentiles(raw[ii]),
                "processed_before_final_floor_percentiles": _percentiles(pre[ii]),
                "effective_sigma_percentiles": _percentiles(effective[ii]),
                "strictly_below_final_floor_before_clamp": int(np.count_nonzero(pre[ii] < 1e-12)),
                "at_final_floor_after_clamp": int(np.count_nonzero(effective[ii] == 1e-12)),
                "weight_relative_to_unfloored_processed_sigma_percentiles": _percentiles(weight_ratio),
                "zero_pre_floor_sigma_count": int(np.count_nonzero(pre[ii] == 0)),
            }
    return out
