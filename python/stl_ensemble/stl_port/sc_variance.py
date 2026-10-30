"""Versioned SC variance-ratio penalty; no change to native features or mean weights.

Only SC population-variance terms use this module. Physical power/PDF/MF
regularizers retain their original definitions. Zero/nonfinite selected TRAIN54
variances are errors, not masked coefficients. The hard floor is dimensionless.
"""
from __future__ import annotations
import numpy as np

SCRIPT_VERSION = "2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: sc_variance.py {SCRIPT_VERSION}", flush=True)
RATIO_FLOOR = 1e-12
CONTRACT = {
    "id": "sc-train54-variance-ratio-hard-log-v2",
    "formula": "mean(log(max(V_syn/V_train,delta))**2)",
    "ratio_floor": RATIO_FLOOR,
    "numerics": "log(V_syn)-log(V_train), then clamp at log(delta)",
    "zero_target_policy": "reject every selected nonpositive/nonfinite TRAIN54 variance",
    "zero_synthesis_policy": "finite clamped loss and zero derivative below delta",
    "boundary_derivative": "right derivative at log-ratio equality",
    "population_estimator": "ddof=1; complex variance=Var(Re)+Var(Im)",
    "reduction": "existing coefficient-count mean within each SC block",
    "mean_loss": "unchanged v1 within-map sigma processing and mean reductions",
    "native_features": "unchanged; no feature/map rescaling",
}


def _validated(synthesized, reference):
    v = np.asarray(synthesized, dtype=np.float64)
    t = np.asarray(reference, dtype=np.float64)
    if v.shape != t.shape or v.size == 0:
        raise ValueError("SC variance arrays must have the same nonempty shape")
    if not np.isfinite(t).all() or np.any(t <= 0):
        bad = np.flatnonzero((~np.isfinite(t) | (t <= 0)).ravel())
        raise ValueError("SC ratio loss requires finite positive TRAIN54 variances; "
                         f"bad selected indices={bad[:16].tolist()} (total {bad.size})")
    if not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError("Synthesized SC variances must be finite and nonnegative")
    return v, t


def log_ratios(synthesized, reference):
    """Stable ratios in log space, avoiding overflow of V_syn/V_train.

    Zero synthesized variance returns -inf before the explicit hard clamp.
    No small physical variance is replaced by an absolute numerical floor.
    """
    v, t = _validated(synthesized, reference)
    logv = np.full(v.shape, -np.inf, dtype=np.float64)
    np.log(v, out=logv, where=v > 0)
    return logv - np.log(t)


def variance_ratio_loss(synthesized, reference):
    """Return scalar and dL/dV_syn, averaged over all supplied coefficients.

    Finite differences should not straddle the hard-floor kink. Under the
    real-inner-product convention this real cotangent is pulled through complex
    moments by 2*(S-mean)/(B-1), exactly as in v1.
    """
    v, t = _validated(synthesized, reference)
    lr = log_ratios(v, t)
    floor = float(np.log(RATIO_FLOOR))
    d = np.maximum(lr, floor)
    loss = float(np.mean(d * d))
    active = (lr >= floor) & (v > 0)
    cot = np.zeros_like(v)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            # Divide by N separately: N*v could overflow although v is finite.
            cot[active] = (2.0 / v.size) * d[active] / v[active]
        except FloatingPointError as exc:
            raise FloatingPointError("SC variance cotangent overflow; no silent rescaling applied") from exc
    if not np.isfinite(loss) or not np.isfinite(cot).all():
        raise FloatingPointError("Nonfinite SC variance-ratio loss/cotangent")
    return loss, cot


def variance_ratio_summary(synthesized, reference):
    """JSON-safe sensitivity summary; avoids overflowing raw variance ratios."""
    v, t = _validated(synthesized, reference)
    lr = log_ratios(v, t)
    finite = lr[np.isfinite(lr)]
    log10 = finite / np.log(10.0)
    return {
        "contract": CONTRACT["id"], "ratio_floor": RATIO_FLOOR,
        "coefficients": int(v.size),
        "zero_synthesis_variance": int(np.count_nonzero(v == 0)),
        "below_ratio_floor": int(np.count_nonzero(lr < np.log(RATIO_FLOOR))),
        "at_or_above_ratio_floor": int(np.count_nonzero(lr >= np.log(RATIO_FLOOR))),
        "log10_ratio_positive_synthesis_percentiles_0_10_50_90_100":
            np.percentile(log10, [0, 10, 50, 90, 100]).tolist() if log10.size else None,
        "loss": variance_ratio_loss(v, t)[0],
    }
