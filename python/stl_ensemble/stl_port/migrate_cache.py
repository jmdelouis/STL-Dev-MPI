"""Explicit, non-destructive migration of the known v1 calibration to v2.

No FITS file is reread and no scattering/LUT/harmonic target is recomputed.
Compact saved TRAIN54 features are rechecked against target moments. All five
array files are copied byte-for-byte. The old manifest remains untouched.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .config import Config, BASELINE_REVISION, SOURCE_SHA256, canonical_hash, sha256_file
from .calibration import (BLOCK_DEFINITIONS, science_config, implementation_digest,
                          population_targets, load_calibration)
from .sc_normalization import SCNormalization
from .sc_reports import variance_sensitivity_issues, variance_floor_report, mean_sigma_report

SCRIPT_VERSION = "2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: migrate_cache.py {SCRIPT_VERSION}", flush=True)
ROOT = Path(__file__).resolve().parents[1]
ARRAY_FILES = ("targets.npz", "lut.npz", "training_features.npz",
               "training_physical.npy", "training_gaussian.npy")


def verify_v1_package(directory):
    """Compare against bundled authoritative v1 hashes, not an edited manifest."""
    root = Path(directory).resolve()
    expected = json.loads((ROOT / "provenance/v1_PACKAGE_SHA256.json").read_text())
    bad = []
    for name, digest in expected.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha256_file(p) != digest:
            bad.append(name)
    if bad:
        raise ValueError("v1 package differs from the approved baseline: " + ", ".join(bad))
    return {"files_verified": len(expected), "path": str(root)}


def _close(label, actual, expected, reports):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Saved calibration consistency failed: {label} shape/finite values")
    # There is NO absolute floor here: even ~1e-28 variances are compared relatively.
    if not np.allclose(a, b, rtol=2e-12, atol=0):
        delta = np.max(np.abs(a-b), initial=0.)
        raise ValueError(f"Saved calibration consistency failed: {label}; max absolute error {delta}")
    reports[label] = {"passed": True, "rtol": 2e-12, "atol": 0.,
                      "max_absolute_difference": float(np.max(np.abs(a-b), initial=0.))}


def check_saved_features(root, meta, targets):
    """Verify stored moments and unchanged native normalization, not new targets."""
    plan, ids = meta["plan"], np.asarray(meta["bootstrap_ids"], dtype=int)
    checks = {}
    with np.load(root/"training_features.npz", allow_pickle=False) as features:
        for label, prefix in (("gaussian", "gsc"), ("physical", "psc")):
            norm = SCNormalization(plan, meta["normalizers"][label])
            if norm.spec.get("feature_rescaling") != "none":
                raise ValueError("Unexpected v1 feature rescaling; this migration does not convert coordinates")
            raw, spatial = features[prefix+"_raw"], features[prefix+"_spatial_variance"]
            if raw.shape != (54, len(plan)) or spatial.shape != raw.shape:
                raise ValueError(f"{prefix}: expected TRAIN54 raw/spatial [54,Ncoeff]")
            _close(prefix+"/native_normalization", norm.forward(raw), features[prefix], checks)
            _close(prefix+"/spatial_sigma", norm.sigma(raw, spatial).mean(0), targets[prefix+"_sigma"], checks)
        for prefix in ("gsc", "psc", "power", "pdf", "mf"):
            f = features[prefix]
            if f.shape[0] != 54:
                raise ValueError(f"{prefix}: stored population must contain exactly 54 maps")
            mean, var = population_targets(f, ids)
            _close(prefix+"/bootstrap_mean", mean, targets[prefix+"_mean"], checks)
            _close(prefix+"/population_variance", var, targets[prefix+"_var"], checks)
        _close("initialization/cl", features["gaussian_cl"][ids].mean(0), targets["initialization_cl"], checks)
    nside = int(meta["config"]["nside"])
    for name in ("training_physical.npy", "training_gaussian.npy"):
        data = np.load(root/name, mmap_mode="r", allow_pickle=False)
        if data.shape != (54, 12*nside**2) or data.dtype != np.float32:
            raise ValueError(f"{name}: expected source float32 TRAIN54 map bank")
        if name == "training_gaussian.npy":
            _close("initialization/reference_mean", np.mean(data[int(ids[0])], dtype=np.float64),
                   targets["initialization_reference_mean"], checks)
        del data
    return checks


def migrate_v1(config, source, destination, *, native_library, v1_package):
    """Create a NEW v2 cache; refuses unknown sources, existing output, or bad targets."""
    config.validate()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Migration output must be separate from the v1 cache")
    if destination.exists():
        raise FileExistsError(f"Migration destination already exists: {destination}")
    source_check = verify_v1_package(v1_package)
    known = json.loads((ROOT/"provenance/v1_compatibility.json").read_text())
    old_path = source/"manifest.json"
    old_hash = sha256_file(old_path)
    old = json.loads(old_path.read_text())
    old_identity = {k:v for k,v in old.items() if k not in ("cache_identity", "variance_floor_report")}
    if canonical_hash(old_identity) != old.get("cache_identity"):
        raise ValueError("v1 manifest integrity check failed")
    if (old.get("library"), old.get("native_revision"), old.get("source_sha256")) != (
            "STL-Dev-MPI", BASELINE_REVISION, SOURCE_SHA256):
        raise ValueError("Unexpected v1 library/source revision")
    if old.get("implementation_sha256") != known["implementation_sha256"] or old.get("losses") != known["losses"]:
        raise ValueError("v1 implementation/loss contract is not the reviewed baseline")
    expected_v1_config = science_config(config)
    expected_v1_config.pop("sc_variance_contract")
    expected_v1_config["script_version"] = "2026-10-28-v1"
    if old["config"] != expected_v1_config:
        raise ValueError("Scientific configuration mismatch: preserve resolution, bootstrap seed, plan and PS cutoff; "
                         "set V41_BOOTSTRAP_TARGET_SEED to the v1 manifest value")
    ids = np.random.default_rng(config.bootstrap_seed).choice(np.arange(54), size=10, replace=True)
    if old.get("bootstrap_ids") != ids.tolist():
        raise ValueError("v1 bootstrap IDs differ from the configured seed")
    expected_names = [f"allskymap_nres12r{i:03d}.zs1.mag.nside2048.fits" for i in range(54)]
    if [s.get("id") for s in old.get("sources", [])] != list(range(54)) or [s.get("name") for s in old["sources"]] != expected_names:
        raise ValueError("v1 cache does not describe exactly TRAIN54 r000..r053")
    lib_hash = sha256_file(native_library)
    if old.get("native_library_sha256") != lib_hash:
        raise ValueError("Native bridge differs from the binary used for v1 calibration; reuse that exact binary")
    for filename, key in (("targets.npz", "target_sha256"), ("lut.npz", "lut_sha256")):
        if sha256_file(source/filename) != old[key]:
            raise ValueError(f"v1 cached bytes no longer match its manifest: {filename}")
    with np.load(source/"targets.npz", allow_pickle=False) as data:
        targets = {k:data[k] for k in data.files}
    if not np.array_equal(targets["bootstrap_ids"], ids):
        raise ValueError("targets.npz bootstrap IDs differ from manifest")
    issues = variance_sensitivity_issues(targets, old["plan"])
    if issues:
        raise ValueError("Cannot migrate: relative loss needs positive finite selected variances.\n"+"\n".join(issues))
    checks = check_saved_features(source, old, targets)
    hashes = {name:sha256_file(source/name) for name in ARRAY_FILES}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name+".migrating-", dir=destination.parent))
    try:
        for filename in ARRAY_FILES:
            shutil.copy2(source/filename, staging/filename)
            if sha256_file(staging/filename) != hashes[filename]:
                raise ValueError(f"Copy verification failed: {filename}")
        # Retain original metadata as evidence, not as the active v2 contract.
        shutil.copy2(old_path, staging/"manifest.v1.json")
        if sha256_file(old_path) != old_hash or sha256_file(staging/"manifest.v1.json") != old_hash:
            raise RuntimeError("v1 manifest changed during migration")
        meta = copy.deepcopy(old)
        meta.pop("cache_identity", None)
        meta.update(cache_schema="stl-ensemble-calibration-v2", config=science_config(config),
                    losses=copy.deepcopy(BLOCK_DEFINITIONS), implementation_sha256=implementation_digest(),
                    scientific_variance_guard={"passed":True, "issues":[]},
                    variance_floor_report=variance_floor_report(targets, old["plan"]),
                    mean_sigma_report=mean_sigma_report(targets, old["plan"]),
                    retained_array_sha256=hashes,
                    migration={"version":SCRIPT_VERSION, "from_cache_identity":old["cache_identity"],
                               "from_manifest_sha256":old_hash, "from_cache_path":str(source),
                               "original_source_check":source_check,
                               "from_variance_guard":old.get("scientific_variance_guard"),
                               "compact_feature_consistency_checks":checks,
                               "native_library_unchanged":True, "arrays_byte_identical":True,
                               "source_arrays_without_v1_hashes":"Current bytes copied and hashed; compact features rechecked against authenticated target hashes; no retrospective claim of preexisting map-bank hashes"})
        meta["cache_identity"] = canonical_hash(meta)
        (staging/"manifest.json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
        load_calibration(config, old["plan"], staging, native_library_sha256=lib_hash,
                         require_variance_sensitivity=True)
        if destination.exists():
            raise FileExistsError("Migration destination appeared while copying")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[MIGRATE] {destination}", flush=True)
    print(f"[MIGRATE] New v2 identity: {meta['cache_identity']}", flush=True)
    print("[MIGRATE] All five array files byte-identical; old cache unchanged; mean weights unchanged.", flush=True)
    return meta
