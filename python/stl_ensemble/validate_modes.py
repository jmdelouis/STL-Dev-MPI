#!/usr/bin/env python3
"""Numerical execution equivalence on real native STL; synthetic data only.

This does not calibrate scientific targets or certify weak-lensing synthesis.
All modes consume the same ten maps, 54-map synthetic target population,
normalizers, coefficient plan, precision, and native L-BFGS options.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import time
import numpy as np
print("SCRIPT VERSION: validate_modes.py 2026-10-29-v2-variance-ratio",flush=True)
from stl_port.config import Config
from stl_port.driver import make_engine,json_ready
from stl_port.calibration import population_targets,variance_floor_report
from stl_port.ensemble import EnsembleObjective
from stl_port.lut import PooledLUT
from stl_port.pdf import CICPDF
from stl_port.harmonics import FullSkySHT,lp_windows
from stl_port.minkowski import Minkowski
from stl_port.native import native_lbfgs


def numerical_fixture(config, physical_scale=1.0):
    from stl_port.sc_normalization import SCNormalization
    rng=np.random.default_rng(28102026)
    z=rng.normal(size=(54,config.npix))
    # Smooth asymmetric marginal, no astrophysical interpretation.
    if not np.isfinite(physical_scale) or physical_scale <= 0:
        raise ValueError("physical_scale must be finite and positive")
    physical=(physical_scale*(z+.06*(z*z-1))).astype(np.float32)
    lut=PooledLUT.from_training(physical)
    gauss=lut.gaussianize(physical)
    bootstrap=np.random.default_rng(config.bootstrap_seed).choice(np.arange(54),10,replace=True)
    t={}; norms={}
    with make_engine(dataclasses.replace(config,mode="stream-replay")) as engine:
        plan=engine.plan
        for label,maps,prefix in (("gaussian",gauss,"gsc"),("physical",physical,"psc")):
            rows=[];spatial=[]
            for b in range(54):
                v,m=engine.forward(np.asarray(maps[b:b+1],dtype=np.float64),generation=0,member=b,objective="fixture-"+label,moments=True)
                rows.append(v[0]);spatial.append(m[0])
            rows,spatial=np.stack(rows),np.stack(spatial)
            norm=SCNormalization.fit(plan,rows,spatial,nside=config.nside)
            norms[label]=norm
            t[prefix+"_mean"],t[prefix+"_var"]=population_targets(norm.forward(rows),bootstrap)
            t[prefix+"_sigma"]=norm.sigma(rows,spatial).mean(axis=0)
    sht=FullSkySHT(config.nside,nthreads=config.threads)
    win,centers=lp_windows(config.lmax)
    pdf=CICPDF.from_training(physical)
    mf=Minkowski(sht)
    powers=[];pdfs=[];mfs=[]
    for x in physical:
        powers.append(win@sht.power(sht.analysis(x)))
        pdfs.append(pdf.forward(x)[0])
        feature,state=mf.forward(x)
        mfs.append(feature)
        del state
    for name,arr in (("power",powers),("pdf",pdfs),("mf",mfs)):
        t[name+"_mean"],t[name+"_var"]=population_targets(np.stack(arr),bootstrap)
    t["pdf_support"]=np.asarray([pdf.x_min,pdf.x_max])
    # Same independent numerical input in all execution modes.
    x=rng.normal(size=(10,config.npix))*.95
    return x,t,lut,norms,plan


def discrepancy(reference,actual,rtol=5e-9,atol=5e-11):
    a=np.asarray(reference);b=np.asarray(actual)
    if a.shape!=b.shape: return {"passed":False,"reference_shape":a.shape,"actual_shape":b.shape}
    difference=np.abs(a-b)
    absmax=float(difference.max(initial=0))
    scale=float(np.abs(a).max(initial=0))
    return {"passed":bool(np.all(difference<=atol+rtol*np.abs(a))),"max_absolute":absmax,
            "max_relative_to_reference_max":absmax/max(scale,1e-300),
            "l2_relative":float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-300)),
            "rtol":rtol,"atol":atol}


def amplitude_aware_discrepancy(reference, actual):
    """Per-feature checks cannot hide tiny physical coefficients under a fixed atol."""
    a=np.asarray(reference); b=np.asarray(actual)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed":False,"error":"shape/nonfinite mismatch"}
    scale=np.max(np.abs(a),axis=0,keepdims=True) if a.ndim>1 else np.max(np.abs(a),initial=0.)
    nonzero=scale>0
    aa=np.divide(a,scale,out=np.zeros_like(a),where=nonzero)
    bb=np.divide(b,scale,out=np.zeros_like(b),where=nonzero)
    result=discrepancy(aa,bb)
    zeros=np.broadcast_to(~nonzero,a.shape)
    result["zero_reference_entries_exact"]=bool(np.all(b[zeros]==0))
    result["passed"] &= result["zero_reference_entries_exact"]
    result["normalization"]="per-feature max over reference ensemble; exact-zero columns checked separately"
    return result


def compare_records(reference,record):
    result={"combined_loss":discrepancy(reference["loss"],record["loss"])}
    for name in reference["blocks"]:
        result[f"loss/{name}"]=discrepancy(reference["blocks"][name],record["blocks"][name])
        result[f"statistics/{name}"]=discrepancy(reference["statistics"][name],record["statistics"][name])
        result[f"cotangents/{name}"]=discrepancy(reference["cotangents"][name],record["cotangents"][name])
        result[f"statistics_scaled/{name}"]=amplitude_aware_discrepancy(reference["statistics"][name],record["statistics"][name])
        result[f"cotangents_scaled/{name}"]=amplitude_aware_discrepancy(reference["cotangents"][name],record["cotangents"][name])
    for b in range(10): result[f"gradient/member{b:02d}"]=discrepancy(reference["gradient"][b],record["gradient"][b])
    result["joint_optimizer_step"]=discrepancy(reference["updated_maps"],record["updated_maps"],rtol=1e-8,atol=1e-10)
    result["accepted_optimizer_loss"]=discrepancy(reference["accepted_loss"],record["accepted_loss"],rtol=1e-8,atol=1e-10)
    result["optimizer_loss_history"]=discrepancy(reference["optimizer_report"]["loss_history"],record["optimizer_report"]["loss_history"],rtol=1e-8,atol=1e-10)
    result["optimizer_evaluations"]=discrepancy(reference["optimizer_report"]["evaluations"],record["optimizer_report"]["evaluations"],rtol=0,atol=0)
    return result


def rejection_check(action):
    try: action()
    except (ValueError,RuntimeError) as exc: return {"passed":True,"error":str(exc)}
    return {"passed":False,"error":"invalid state was accepted"}


def validate_ownership_and_native_vjp(config,x):
    report={}
    cfg=dataclasses.replace(config,mode="stream-replay")
    with make_engine(cfg) as engine:
        values,_=engine.forward(x[0:1],generation=11,member=0,objective="gaussian")
        snapshot=engine.checkpoint()
        report["checkpoint_wrong_member"]=rejection_check(lambda:engine.restore(snapshot,generation=11,member=1,objective="gaussian"))
        report["checkpoint_stale_generation"]=rejection_check(lambda:engine.restore(snapshot,generation=12,member=0,objective="gaussian"))
        report["checkpoint_wrong_objective"]=rejection_check(lambda:engine.restore(snapshot,generation=11,member=0,objective="physical"))
        engine.forward(x[1:2],generation=11,member=1,objective="gaussian")
        report["vjp_wrong_member"]=rejection_check(lambda:engine.vjp(np.ones_like(values),generation=11,member=0,objective="gaussian"))
        engine.restore(snapshot,generation=11,member=0,objective="gaussian")
        snapshot.close()
        rng=np.random.default_rng(8)
        cot=rng.normal(size=values.shape)+1j*rng.normal(size=values.shape)
        gradient=engine.vjp(cot,generation=11,member=0,objective="gaussian")
        direction=rng.normal(size=(1,config.npix));direction/=np.linalg.norm(direction)
        eps=2e-6
        vp,_=engine.forward(x[:1]+eps*direction,generation=20,member=0,objective="gaussian")
        vm,_=engine.forward(x[:1]-eps*direction,generation=21,member=0,objective="gaussian")
        numerical=float(np.vdot(cot,(vp-vm)/(2*eps)).real)
        analytic=float(np.sum(gradient*direction))
        report["native_coefficient_vjp"]=discrepancy(numerical,analytic,rtol=2e-5,atol=1e-7)
    sht=FullSkySHT(config.nside,nthreads=config.threads)
    state=sht.cache_analysis(x[0],generation=1,member=0,objective="power")
    ones=np.ones(config.lmax+1)
    report["alm_wrong_member"]=rejection_check(lambda:sht.cached_power_vjp(state,ones,generation=1,member=1,objective="power"))
    report["alm_stale_generation"]=rejection_check(lambda:sht.cached_power_vjp(state,ones,generation=2,member=0,objective="power"))
    report["alm_wrong_objective"]=rejection_check(lambda:sht.cached_power_vjp(state,ones,generation=1,member=0,objective="mf"))
    return report


def run_validation(config,output,physical_scale=1.0):
    config.validate(scientific=False)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    x,t,lut,norms,plan=numerical_fixture(config,physical_scale=physical_scale)
    np.save(out/"input_gaussian.npy",x)
    np.savez(out/"synthetic_targets.npz",**t)
    records={};times={}
    options={"max_iter":1,"max_eval":25,"history_size":10,"lr":1.0,"tolerance_grad":1e-12,"tolerance_change":1e-15}
    for mode in ("resident","stream-replay","stream-checkpoint"):
        cfg=dataclasses.replace(config,mode=mode)
        with make_engine(cfg) as engine:
            obj=EnsembleObjective(cfg,engine,t,lut,norms["gaussian"],norms["physical"],scientific=False)
            start=time.perf_counter()
            value,gradient=obj(x)
            record={"loss":value,"gradient":gradient,"blocks":obj.last["blocks"],
                    "statistics":obj.last["statistics"],"cotangents":obj.last["cotangents"]}
            times[mode]={"evaluation_s":time.perf_counter()-start,"phases":obj.last["timing"]}
            updated,optimization=native_lbfgs(obj,x.copy(),options)
            record["updated_maps"]=updated
            accepted_loss,_=obj(updated)
            displacement=float(np.max(np.abs(updated-x)))
            record["accepted_loss"]=float(accepted_loss)
            record["optimizer_step_validation"]={"max_absolute_displacement":displacement,"initial_loss":float(value),"accepted_loss":float(accepted_loss),"nonzero_step":displacement>0.,"loss_decreased":float(accepted_loss)<float(value)-1e-12*max(1.,abs(float(value)))}
            record["optimizer_report"]=optimization
            records[mode]=record
            np.savez(out/f"{mode}.npz",loss=value,accepted_loss=accepted_loss,gradient=gradient,updated_maps=updated,
                     **{"stat_"+k:v for k,v in record["statistics"].items()},
                     **{"cot_"+k:v for k,v in record["cotangents"].items()})
            print(f"[VALIDATE] {mode}: initial loss={value:.12g}; accepted loss={accepted_loss:.12g}; max step={displacement:.8g}; gradient norm={np.linalg.norm(gradient):.8g}",flush=True)
    comparisons={mode:compare_records(records["resident"],records[mode]) for mode in ("stream-replay","stream-checkpoint")}
    guards=validate_ownership_and_native_vjp(config,x)
    passed=all(v["passed"] for group in comparisons.values() for v in group.values()) and all(v["passed"] for v in guards.values()) and all(v["optimizer_step_validation"]["nonzero_step"] and v["optimizer_step_validation"]["loss_decreased"] for v in records.values())
    report={"version":"2026-10-29-v2-variance-ratio","purpose":"NUMERICAL ONLY: synthetic 54-map calibration, not TRAIN54/scientific validation",
            "passed":passed,"physical_fixture_scale":physical_scale,"config":config.manifest(),"plan":plan,"comparisons":comparisons,"state_and_vjp_checks":guards,
            "variance_floor_report":variance_floor_report(t,plan),"timings":times,"total_seconds":time.perf_counter()-started,
            "optimizer_settings":options,"optimizer_step_validation":{k:v["optimizer_step_validation"] for k,v in records.items()},"optimizer_reports":{k:v["optimizer_report"] for k,v in records.items()},
            "normalization_specs":{k:v.spec for k,v in norms.items()},
            "tolerance_rationale":"float64 execution-mode differences use 5e-9 relative/5e-11 absolute; one-step map test 1e-8/1e-10. Native directional finite difference separately uses truncation/roundoff allowance. MF detached-floor derivatives are tested separately with fixed floors, never by recomputed-floor total finite differences."}
    (out/"validation_report.json").write_text(json.dumps(json_ready(report),indent=2,allow_nan=False)+"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=("cpu","gpu"),default="cpu")
    p.add_argument("--nside",type=int,default=8)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--output",required=True)
    p.add_argument("--physical-scale",type=float,default=1.0)
    p.add_argument("--block-pixels",type=int,default=256)
    args=p.parse_args(argv)
    if args.nside not in (4,8,16):p.error("Numerical comparison deliberately bounded to nside4/8/16")
    config=Config(nside=args.nside,backend=args.backend,threads=args.threads,ps_variance_upper=min(383,3*args.nside-1),cache_bytes=256*1024**2,block_pixels=args.block_pixels)
    report=run_validation(config,args.output,physical_scale=args.physical_scale)
    print(f"[VALIDATE] {'PASS' if report['passed'] else 'FAIL'}: {Path(args.output)/'validation_report.json'}",flush=True)
    return 0 if report["passed"] else 1

if __name__=="__main__":raise SystemExit(main())
