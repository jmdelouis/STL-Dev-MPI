"""Fixtures of P. Campeti's ensemble objective for the Rust port (stl-ensemble).

Runs the Python reference of ``python/stl_ensemble`` (``stl_port``, with
ducc0, healpy and the CPU ``stl-bridge`` library) on a small case and writes
``fixtures/ensemble/<case>/*.npy`` and ``meta.json``:

- ``sht_n16``: analysis, C_l, derivative maps and the adjoints of his
  ``FullSkySHT`` (DUCC) on a random map;
- ``objective_n16``: a calibration from 12 synthetic reference maps (his
  code fixes 54; the same steps are run here on 12, with
  ``build_calibration``'s formulas), the LUT, every target, and the
  five-block loss, its blocks, statistics and gradient for 10 members.

Usage (repository root; library built with ``cargo build --release -p stl-bridge``):

    STL_NATIVE_LIBRARY=target/release/libstl_bridge.so PYTHONPATH=python/stl_ensemble:<ducc0, healpy> \\
        python tools/dump_ensemble_reference.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

from stl_port.calibration import population_targets
from stl_port.config import Config
from stl_port.ensemble import EnsembleObjective
from stl_port.harmonics import FullSkySHT, lp_windows
from stl_port.lut import PooledLUT, probability_grid
from stl_port.minkowski import Minkowski
from stl_port.native import NativeEngine
from stl_port.pdf import CICPDF
from stl_port.sc_normalization import SCNormalization

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ensemble"


def save(case, arrays, meta):
    d = ROOT / case
    d.mkdir(parents=True)
    for k, v in arrays.items():
        v = np.asarray(v)
        if np.iscomplexobj(v):
            np.save(d / f"{k}_re.npy", np.ascontiguousarray(v.real, dtype=np.float64))
            np.save(d / f"{k}_im.npy", np.ascontiguousarray(v.imag, dtype=np.float64))
        elif v.dtype == np.int64:
            np.save(d / f"{k}.npy", v)
        else:
            np.save(d / f"{k}.npy", np.ascontiguousarray(v, dtype=np.float64))
    (d / "meta.json").write_text(json.dumps({"name": case, **meta}, indent=1) + "\n")


def sht_case():
    nside = 16
    sht = FullSkySHT(nside, nthreads=1)
    rng = np.random.default_rng(16)
    x = rng.normal(size=sht.npix)
    alm = sht.analysis(x)
    d = sht.derivative_maps(alm)
    cots = rng.normal(size=(5, sht.npix))
    acot = rng.normal(size=sht.nalm) + 1j * rng.normal(size=sht.nalm)
    acot[sht.m == 0] = acot[sht.m == 0].real
    save("sht_n16", {"x": x, "alm": alm, "cl": sht.power(alm),
                     **{f"deriv{k}": d[k] for k in range(5)},
                     **{f"cot{k}": cots[k] for k in range(5)},
                     "deriv_vjp": sht.derivative_alm_vjp(list(cots)),
                     "acot": acot, "analysis_vjp": sht.analysis_vjp(acot),
                     "synthesis": sht.synthesis(acot)},
         {"nside": nside, "lmax": sht.lmax})


def lut_from(maps, nbins):
    """``PooledLUT.from_training`` without its fixed count of 54 maps."""
    a = np.asarray(maps, dtype=np.float32)
    p = probability_grid(nbins, a.size)
    x = np.quantile(a.astype(np.float64).ravel(), p, method="linear").astype(np.float32)
    eps = max(abs(float(x[-1] - x[0])) * 1e-7, 1e-12)
    for i in range(1, x.size):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return PooledLUT(p, x)


def objective_case():
    nside, nref, members = 16, 12, 10
    config = Config(nside=nside, mode="resident", backend="cpu", threads=1, block_pixels=256,
                    ps_variance_upper=40.0)
    config.validate(scientific=False)
    rng = np.random.default_rng(28102026)
    z = rng.normal(size=(nref, config.npix))
    physical = (0.01 * (z + 0.3 * (z * z - 1))).astype(np.float32)
    lut = lut_from(physical, 8192)
    gauss = lut.gaussianize(physical).astype(np.float32)
    ids = np.random.default_rng(config.bootstrap_seed).choice(np.arange(nref), size=members, replace=True)
    targets, rows = {}, {}
    with NativeEngine(nside, config.scales, batch=1, backend="cpu", block_pixels=256) as engine:
        plan = engine.plan
        for label, maps, prefix in (("gaussian", gauss, "gsc"), ("physical", physical, "psc")):
            v, s = [], []
            for i in range(nref):
                a, b = engine.forward(np.asarray(maps[i:i + 1], dtype=np.float64), generation=1, member=i,
                                      objective="calibration-" + label, moments=True)
                v.append(a[0])
                s.append(b[0])
            v, s = np.stack(v), np.stack(s)
            norm = SCNormalization.fit(plan, v, s, nside=nside)
            values = norm.forward(v)
            targets[prefix + "_mean"], targets[prefix + "_var"] = population_targets(values, ids)
            targets[prefix + "_sigma"] = norm.sigma(v, s).mean(axis=0)
            rows[prefix + "_raw"], rows[prefix + "_spatial"] = v, s
            rows[label + "_normalizer"] = norm
    sht = FullSkySHT(nside, nthreads=1)
    windows, centers = lp_windows(config.lmax)
    lo, hi = float(np.min(physical)), float(np.max(physical))
    support = np.array([lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)])
    pdf, mf = CICPDF(*support), Minkowski(sht)
    feats = {"power": [], "pdf": [], "mf": []}
    gcl = []
    for i in range(nref):
        x = np.asarray(physical[i], dtype=np.float64)
        feats["power"].append(windows @ sht.power(sht.analysis(x)))
        feats["pdf"].append(pdf.forward(x)[0])
        feats["mf"].append(mf.forward(x)[0])
        gcl.append(sht.power(sht.analysis(np.asarray(gauss[i], dtype=np.float64))))
    for name, arr in feats.items():
        targets[name + "_mean"], targets[name + "_var"] = population_targets(np.stack(arr), ids)
    targets["pdf_support"] = support
    targets["power_centers"] = centers
    targets["initialization_cl"] = np.stack(gcl)[ids].mean(axis=0)
    targets["initialization_reference_mean"] = np.asarray(np.mean(gauss[int(ids[0])], dtype=np.float64))
    # Members: Gaussian maps near the references (a non-trivial point of the loss).
    g = np.asarray(gauss[:members], dtype=np.float64) * 0.9 + 0.2 * rng.normal(size=(members, config.npix))
    with NativeEngine(nside, config.scales, batch=members, backend="cpu", block_pixels=256) as engine:
        obj = EnsembleObjective(config, engine, targets, lut, rows["gaussian_normalizer"],
                                rows["physical_normalizer"], scientific=False)
        loss, grad = obj(g)
        last = obj.last
        kappa = np.stack([lut.inverse(g[b])[0] for b in range(members)])
        gsc_raw, _ = engine.forward(g, generation=99, member=members, objective="dump")
        psc_raw, _ = engine.forward(kappa, generation=99, member=members, objective="dump")
    arrays = {"references": physical.astype(np.float64), "bootstrap_ids": ids.astype(np.int64),
              "probabilities": lut.probabilities, "physical_knots": lut.physical_knots,
              "gaussian_knots": lut.gaussian_knots, "gaussian_references": gauss.astype(np.float64),
              "windows": windows, "members": g, "kappa": kappa, "gsc_raw": gsc_raw, "psc_raw": psc_raw,
              "gradient": grad, **targets, **rows_out(rows),
              **{f"stat_{k}": v for k, v in last["statistics"].items()}}
    save("objective_n16", arrays, {"nside": nside, "scales": config.scales, "references": nref,
                                   "members": members, "block_pixels": 256, "ps_variance_upper": 40.0,
                                   "loss": loss, "blocks": last["blocks"]})


def rows_out(rows):
    return {k: v for k, v in rows.items() if not k.endswith("_normalizer")}


if __name__ == "__main__":
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    sht_case()
    objective_case()
    (ROOT / "index.json").write_text(json.dumps(["sht_n16", "objective_n16"]) + "\n")
    print("fixtures written to", ROOT)
