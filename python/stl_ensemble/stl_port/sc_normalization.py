"""Source-audited native SC coordinates and within-map estimator sigmas.

There is deliberately no fitted feature rescaling. In the inspected FOSCAT
implementation, norm='self' calls div_norm without assigning its return value:
auto-channel S1--S4 remain in raw coordinates. Native `var` is mean(x*x), so
that slot is recombined with native `mean` to represent the second S0 entry,
the within-map standard error of the map mean. See SC_MAPPING.md.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: sc_normalization.py {SCRIPT_VERSION}", flush=True)

FOSCAT_REVISION = "17baec97672097d68fd4ef1e337e8ec362ef4326"
FOSCAT_SCAT_BLOB = "56f0964ef557ff4236ee737f86f162ae1faa1e17"


def _plan_hash(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


class SCNormalization:
    """No data fitting; the name is retained for the calibration interface.

    All feature arithmetic and cotangents use float64/complex128. Per-feature
    spatial variance is a within-map quantity supplied by the native engine.
    It is not the TRAIN54 across-map sample variance. Complex sigmas use the
    documented native analogue sqrt((Var Re+Var Im)/N), rather than reproducing
    FOSCAT's non-positive-definite complex x*x expression.
    """
    def __init__(self, plan, spec):
        self.plan = [dict(p) for p in plan]
        self.spec = dict(spec)
        if self.spec.get("version") != SCRIPT_VERSION:
            raise ValueError("SC coordinate definition version mismatch")
        if self.spec.get("plan_sha256") != _plan_hash(self.plan):
            raise ValueError("SC normalizer plan mismatch")
        if self.spec.get("feature_coordinates") != "native_raw_with_S0_recombination":
            raise ValueError("Unreviewed native coefficient scaling")
        if self.spec.get("complex_sigma") != "sqrt(total_spatial_variance/native_pixel_count)":
            raise ValueError("Unreviewed complex within-map estimator normalization")
        self.nside = int(self.spec["nside"])
        self.npix = 12*self.nside*self.nside
        self.native_pixels = np.asarray([self.npix/(4**int(p["native_dg"])) for p in self.plan],dtype=np.float64)
        self.mean = [i for i,p in enumerate(plan) if p["family"]=="mean"]
        self.second = [i for i,p in enumerate(plan) if p["family"]=="var"]
        if len(self.mean)!=1 or len(self.second)!=1:
            raise ValueError("This scalar-map port requires exactly one native mean and second moment for S0")
        self.mean, self.second = self.mean[0], self.second[0]
        if any(p.get("c1",0)!=0 or p.get("c2",0)!=0 for p in plan):
            raise ValueError("This audited mapping is for one scalar auto-channel")
        if any(p["family"] not in ("mean","var","S1","S2","S3","S4") for p in plan):
            raise ValueError("Unknown native family")

    @classmethod
    def fit(cls, plan, rawvalues, spatial, nside):
        """Validate calibration inputs and return a fixed source-derived map.

        No calibration-dependent scale, floor, clipping or decorrelation is
        learned here. Native numerical scales are reported and reviewed by
        calibration's variance-floor report, never silently equalized away.
        """
        spec={"version":SCRIPT_VERSION,"nside":int(nside),
              "plan_sha256":_plan_hash(plan),
              "feature_coordinates":"native_raw_with_S0_recombination",
              "S0_mapping":{"mean":"mean(x)","var":"sqrt(max(mean(x^2)-mean(x)^2,0)/N0)"},
              "norm_self_audit":"div_norm returns discarded at auto-channel S1,S3,S4 callsites; S2 divides only for auto",
              "feature_rescaling":"none",
              "complex_sigma":"sqrt(total_spatial_variance/native_pixel_count)",
              "real_sigma":"sqrt(spatial_variance/native_pixel_count)",
              "sigma_average":"arithmetic mean of individual TRAIN54 sigmas",
              "population_variance":"separate ddof=1 across TRAIN54; VarRe+VarIm",
              "source_foscat_revision":FOSCAT_REVISION,
              "source_scat_cov_blob":FOSCAT_SCAT_BLOB,
              "historical_foscat_installation_revision":"not supplied; current inspected source is explicitly recorded",
              "complex_sigma_difference":"native rotation-invariant total complex variance; source uses x*x and real+imag second moment, which is not nonnegative in general"}
        obj=cls(plan,spec)
        obj.sigma(rawvalues,spatial)
        obj.forward(rawvalues)
        return obj

    def _raw(self, raw):
        x=np.asarray(raw,dtype=np.complex128)
        if x.ndim<1 or x.shape[-1]!=len(self.plan):
            raise ValueError("Native SC feature/plan shape mismatch")
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite native SC values")
        return x

    def _s0(self,x):
        m=x[...,self.mean].real
        q=x[...,self.second].real
        v=q-m*m
        # Roundoff-sized cancellation can occur at constant maps. Larger
        # negative variance is a native data/plan problem, not a floor choice.
        tolerance=64*np.finfo(np.float64).eps*np.maximum(np.abs(q)+m*m,np.finfo(float).tiny)
        if np.any(v < -tolerance):
            raise ValueError("Native map second moment implies negative map variance")
        return m,np.sqrt(np.maximum(v,0)/self.npix)

    def forward(self, raw):
        x=self._raw(raw)
        out=x.copy()
        m,s=self._s0(x)
        out[...,self.mean]=m
        out[...,self.second]=s
        return out

    def vjp(self, raw, cotangent):
        x=self._raw(raw)
        cot=np.asarray(cotangent,dtype=np.complex128)
        if cot.shape!=x.shape or not np.all(np.isfinite(cot)):
            raise ValueError("Native SC cotangent shape/value mismatch")
        out=cot.copy()
        m,s=self._s0(x)
        # At exactly constant maps sqrt(var) is nonsmooth. Select zero for
        # the recombined entry; ordinary maps use the complete derivative.
        dq=np.divide(cot[...,self.second].real,2*self.npix*s,
                     out=np.zeros_like(s),where=s>0)
        out[...,self.mean]=cot[...,self.mean].real-2*m*dq
        out[...,self.second]=dq
        return out

    def sigma(self, raw, spatial):
        x=self._raw(raw)
        v=np.asarray(spatial,dtype=np.float64)
        if v.shape!=x.shape or not np.all(np.isfinite(v)) or np.any(v<0):
            raise ValueError("Native within-map spatial variance is missing or invalid")
        sigma=np.sqrt(v/self.native_pixels)
        # Source S0 sigma object repeats sigma(mean) for BOTH S0 entries.
        # It does not use error propagation for the standard-error entry.
        sigma[...,self.second]=sigma[...,self.mean]
        return sigma
