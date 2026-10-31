"""Command-line workflow for the source-preserving native STL ensemble port."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
from pathlib import Path
import time
import numpy as np
from .config import Config, SCRIPT_VERSION, sha256_file

print(f"SCRIPT VERSION: driver.py {SCRIPT_VERSION}")

def json_ready(obj):
    if isinstance(obj,dict): return {str(k):json_ready(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [json_ready(v) for v in obj]
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,Path): return str(obj)
    return obj

def write_json(path,obj):
    Path(path).write_text(json.dumps(json_ready(obj),indent=2,allow_nan=False)+"\n")

def make_engine(config):
    from .native import NativeEngine
    return NativeEngine(config.nside,config.scales,batch=config.active_batch,backend=config.backend,
                        kernel=config.kernel,orientations=config.orientations,
                        block_pixels=config.block_pixels,cache_bytes=config.cache_bytes,threads=config.threads)

def make_objective(config,engine,cache):
    from .calibration import load_calibration
    from .ensemble import EnsembleObjective
    targets,lut,norms,manifest=load_calibration(config,engine.plan,cache,native_library_sha256=engine.library_sha256,require_variance_sensitivity=True)
    objective=EnsembleObjective(config,engine,targets,lut,norms["gaussian"],norms["physical"])
    return objective,manifest

def new_output(path):
    p=Path(path)
    if p.exists() and any(p.iterdir()): raise FileExistsError(f"Output directory must be empty: {p}")
    p.mkdir(parents=True,exist_ok=True)
    return p

def summary(info):
    return {k:v for k,v in info.items() if k not in ("statistics","cotangents")}

def save_trial_details(out,obj,gradient):
    np.save(out/"gradient.npy",gradient)
    np.savez(out/"statistics.npz",**obj.last["statistics"])
    np.savez(out/"member_cotangents.npz",**obj.last["cotangents"])
    write_json(out/"evaluation.json",summary(obj.last))

def main(argv=None):
    parser=argparse.ArgumentParser(description="Five-block B=10 native-resolution STL synthesis; joint native L-BFGS only")
    parser.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs/native128.json"))
    parser.add_argument("--mode",choices=("stream-replay","stream-checkpoint","resident"))
    parser.add_argument("--backend",choices=("cpu","gpu"))
    parser.add_argument("--threads",type=int)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("calibrate",help="Rebuild all native targets and the pooled TRAIN54 LUT")
    p.add_argument("--reference-dir",required=True); p.add_argument("--cache",required=True)
    p=sub.add_parser("run",help="Joint L-BFGS pilot; iteration limit must be explicit")
    p.add_argument("--cache",required=True);p.add_argument("--output",required=True);p.add_argument("--iterations",type=int,required=True)
    p.add_argument("--max-evaluations",type=int);p.add_argument("--fits",action="store_true",help="Also write each physical member to NESTED FITS")
    p=sub.add_parser("evaluate",help="One complete five-block loss plus gradient")
    p.add_argument("--cache",required=True);p.add_argument("--output",required=True);p.add_argument("--maps",help="Current Gaussian ensemble .npy; omit for source GRF initialization")
    p=sub.add_parser("benchmark",help="Bounded complete evaluations, without optimizer updates")
    p.add_argument("--cache",required=True);p.add_argument("--output",required=True);p.add_argument("--evaluations",type=int,default=2)
    args=parser.parse_args(argv)
    config=Config.read(args.config)
    changes={k:getattr(args,k) for k in ("mode","backend","threads") if getattr(args,k) is not None}
    config=dataclasses.replace(config,**changes)
    config.validate()
    print(json.dumps(config.manifest(),indent=2),flush=True)
    from .calibration import build_calibration,draw_initial_ensemble
    if args.command=="calibrate":
        # Targets are always read one map at a time; the resident execution reference
        # later consumes these same bytes without rebuilding a differently sized target engine.
        calibration_config=dataclasses.replace(config,mode="stream-replay")
        with make_engine(calibration_config) as engine:
            _,_,_,manifest=build_calibration(config,engine,args.reference_dir,args.cache)
            print(f"[CALIBRATE] cache identity {manifest['cache_identity']}",flush=True)
            guard=manifest["scientific_variance_guard"]
            print(f"[CALIBRATE] scientific variance guard: {guard}",flush=True)
            if not guard["passed"]: print("[CALIBRATE] All targets and reports are saved. Scientific run is blocked until the recorded nonpositive/nonfinite SC variance targets is resolved.",flush=True)
        return 0
    if args.command=="run" and args.iterations<1: parser.error("--iterations must be positive")
    if args.command=="benchmark" and not 1<=args.evaluations<=10: parser.error("Bounded benchmark allows 1 through 10 complete evaluations")
    out=new_output(args.output)
    setup_start=time.perf_counter()
    with make_engine(config) as engine:
        native_setup_s=time.perf_counter()-setup_start
        objective,cache_manifest=make_objective(config,engine,args.cache)
        total_setup_s=time.perf_counter()-setup_start
        run_manifest={"config":config.manifest(),"target_cache_identity":cache_manifest["cache_identity"],
                      "target_bootstrap_ids":cache_manifest["bootstrap_ids"],"normalizers":cache_manifest["normalizers"],
                      "loss_contract":cache_manifest["losses"],"native_engine":engine.info(),
                      "setup_seconds":{"native_engine":native_setup_s,"total_engine_cache_and_other_branches":total_setup_s},
                      "command":args.command,"scientific_validation":"not established by optimization/evaluation", "pid":os.getpid()}
        write_json(out/"run_manifest.json",run_manifest)
        if args.command=="evaluate" and args.maps:
            maps=np.asarray(np.load(args.maps,allow_pickle=False),dtype=np.float64)
            if maps.shape!=(config.members,config.npix): raise ValueError("Expected Gaussian NEST ensemble [10,12*nside^2]")
        else: maps=draw_initial_ensemble(config,objective.targets,objective.sht)
        np.save(out/"initial_gaussian.npy",maps)
        log=open(out/"evaluations.jsonl","w",buffering=1)
        evaluation_seconds=[]
        def report(info):
            evaluation_seconds.append(info["elapsed_s"])
            line=summary(info)
            log.write(json.dumps(json_ready(line),allow_nan=False)+"\n")
            print(f"[EVAL {info['generation']}] loss={info['loss']:.10g} blocks={info['blocks']} seconds={info['elapsed_s']:.3f}",flush=True)
        objective.evaluation_callback=report
        try:
            if args.command=="run":
                from .native import native_lbfgs
                options={"lr":config.learning_rate,"max_iter":args.iterations,"history_size":config.history_size,
                         "tolerance_grad":config.tolerance_grad,"tolerance_change":config.tolerance_change}
                if args.max_evaluations is not None: options["max_eval"]=args.max_evaluations
                run_manifest["optimizer_settings"]=options
                write_json(out/"run_manifest.json",run_manifest)
                optimizer_started=time.perf_counter()
                maps,optimizer_info=native_lbfgs(objective,maps,options)
                optimizer_info["total_wall_s"]=time.perf_counter()-optimizer_started
                optimizer_info["objective_wall_s"]=sum(evaluation_seconds)
                optimizer_info["optimizer_vector_ffi_and_logging_remainder_s"]=max(0.,optimizer_info["total_wall_s"]-optimizer_info["objective_wall_s"])
                write_json(out/"optimizer_report.json",optimizer_info)
                # Evaluate the returned iterate, because last Wolfe trial may differ.
                _,gradient=objective(maps)
            elif args.command=="evaluate":
                _,gradient=objective(maps)
            else:
                durations=[]
                for _ in range(args.evaluations):
                    start=time.perf_counter();_,gradient=objective(maps);durations.append(time.perf_counter()-start)
                write_json(out/"benchmark.json",{"setup_seconds":run_manifest["setup_seconds"],"complete_evaluation_seconds":durations,"first":"cold evaluation after engine setup","remaining":"warm evaluations","engine":engine.info(),"last_phases":objective.last["timing"]})
            save_trial_details(out,objective,gradient)
            np.save(out/"gaussian.npy",maps)
            phys=np.lib.format.open_memmap(out/"physical.npy",mode="w+",dtype=np.float64,shape=maps.shape)
            for b in range(config.members):
                phys[b]=objective.lut.inverse(maps[b],remove_monopole=True)[0]
                if getattr(args,"fits",False):
                    import healpy as hp
                    hp.write_map(str(out/f"physical_{b:02d}_nside{config.nside}_NEST.fits"),phys[b],nest=True,dtype=np.float64,overwrite=False)
            phys.flush()
            write_json(out/"native_final_report.json",engine.info())
        finally: log.close()
    print(f"[DONE] {out.resolve()}",flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
