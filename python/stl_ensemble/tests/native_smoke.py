#!/usr/bin/env python3
"""Real-native bridge execution checks; requires built libstl_bridge."""
print("SCRIPT VERSION: native_smoke 2026-10-28-v1", flush=True)
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeEngine, native_lbfgs


def run(backend="cpu", nside=2, scales=2):
    rng = np.random.default_rng(83257)
    npix = 12*nside*nside
    x = rng.normal(size=(10, npix))
    kwargs = dict(nside=nside, j=scales, backend=backend, block_pixels=256)
    results = {"backend": backend, "nside": nside, "scales": scales, "members": 10, "kernel": 5, "orientations": 4}
    with NativeEngine(batch=10, **kwargs) as resident, NativeEngine(batch=1, **kwargs) as stream:
        ref, refvar = resident.forward(x, generation=1, moments=True)
        cot = rng.normal(size=ref.shape) + 1j * rng.normal(size=ref.shape)
        gradref = resident.vjp(cot, generation=1)
        got, gotvar, snapshots = [], [], []
        for b in range(10):
            a, v = stream.forward(x[b], generation=1, member=b, moments=True)
            got.append(a[0]); gotvar.append(v[0]); snapshots.append(stream.checkpoint())
        replay, checkpoint = [], []
        for b in range(10):
            stream.forward(x[b], generation=1, member=b)
            replay.append(stream.vjp(cot[b], generation=1, member=b)[0])
            stream.restore(snapshots[b], generation=1, member=b)
            checkpoint.append(stream.vjp(cot[b], generation=1, member=b)[0])
        for name,a,b in [("statistics",got,ref),("spatial_variance",gotvar,refvar),("replay_gradient",replay,gradref),("checkpoint_gradient",checkpoint,gradref)]:
            a,b = np.asarray(a),np.asarray(b)
            err=float(np.max(np.abs(a-b)))
            rel=err / max(float(np.max(np.abs(b))),1e-30)
            results[name] = dict(max_abs=err,max_rel=rel)
            np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)
        rejected=[]
        for key in [dict(generation=2,member=0,objective="gaussian"),dict(generation=1,member=1,objective="gaussian"),dict(generation=1,member=0,objective="physical")]:
            try: stream.restore(snapshots[0],**key)
            except (ValueError,RuntimeError): rejected.append(key)
            else: raise AssertionError("invalid checkpoint was accepted")
        results["rejected_checkpoint_keys"] = rejected
        try: resident.restore(snapshots[0], generation=1, member=0)
        except (ValueError,RuntimeError): results["wrong_owner_rejected"] = True
        else: raise AssertionError("wrong-owner checkpoint accepted")
        direction = rng.normal(size=npix); direction /= np.linalg.norm(direction)
        stream.restore(snapshots[0],generation=1,member=0)
        analytic=float(np.dot(stream.vjp(cot[0],generation=1,member=0)[0],direction))
        eps=1e-5
        plus=stream.forward(x[0]+eps*direction,generation=2,member=0)[0][0]
        minus=stream.forward(x[0]-eps*direction,generation=3,member=0)[0][0]
        fd=float(np.vdot(cot[0],plus-minus).real/(2*eps))
        results["native_vjp"] = dict(finite_difference=fd,analytic=analytic,abs_error=abs(fd-analytic))
        np.testing.assert_allclose(fd,analytic,rtol=3e-6,atol=3e-7)
        try: stream.vjp(cot[0],generation=2,member=0)
        except RuntimeError: results["stale_vjp_rejected"] = True
        else: raise AssertionError("stale VJP accepted")
        for s in snapshots: s.close()
    initial = rng.normal(size=20)
    optimum = rng.normal(size=20)
    def quadratic(p): return float(0.5*np.sum((p-optimum)**2)), p-optimum
    solved, info = native_lbfgs(quadratic,initial.copy(),options={"max_iter":10})
    np.testing.assert_allclose(solved,optimum,rtol=1e-9,atol=1e-9)
    results["native_lbfgs_quadratic"] = dict(error=float(np.max(abs(solved-optimum))),evaluations=info["evaluations"])
    # Check a genuine joint native-scattering first step in all execution modes.
    with NativeEngine(batch=10, **kwargs) as resident, NativeEngine(batch=1, **kwargs) as stream:
        target=resident.forward(x,generation=0)[0].mean(axis=0)*0.9
        generations={"resident":0,"replay":0,"checkpoint":0}
        def evaluate(p,mode):
            generations[mode]+=1; gen=generations[mode]
            if mode=="resident":
                vals=resident.forward(p,generation=gen)[0]
                diff=vals.mean(axis=0)-target
                cot=np.broadcast_to(2*diff/(10*len(target)),vals.shape).copy()
                grad=resident.vjp(cot,generation=gen)
            else:
                vals=[]; snaps=[]
                for b in range(10):
                    vals.append(stream.forward(p[b],generation=gen,member=b)[0][0])
                    if mode=="checkpoint": snaps.append(stream.checkpoint())
                vals=np.array(vals);diff=vals.mean(axis=0)-target
                one=2*diff/(10*len(target));grad=[]
                for b in range(10):
                    if mode=="checkpoint": stream.restore(snaps[b],generation=gen,member=b)
                    else: stream.forward(p[b],generation=gen,member=b)
                    grad.append(stream.vjp(one,generation=gen,member=b)[0])
                grad=np.array(grad)
                for s in snaps:s.close()
            return float(np.mean(abs(diff)**2)),grad
        updated={}
        for mode in generations:
            initial_loss=evaluate(x,mode)[0]
            updated[mode],info=native_lbfgs(lambda p:evaluate(p,mode),x.copy(),options={"max_iter":1,"max_eval":10})
            displacement=float(np.max(abs(updated[mode]-x)))
            final_loss=evaluate(updated[mode],mode)[0]
            assert displacement>0, "joint optimizer step was vacuous"
            assert final_loss<initial_loss, "accepted step did not decrease scalar"
            results[f"step_progress_{mode}"]=dict(max_map_change=displacement,initial_loss=initial_loss,final_loss=final_loss)
        for mode in ("replay","checkpoint"):
            err=float(np.max(abs(updated[mode]-updated["resident"])))
            results[f"one_step_{mode}"] = err
            np.testing.assert_allclose(updated[mode],updated["resident"],rtol=1e-9,atol=1e-10)
    return results


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",choices=["cpu","gpu"],default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--nside",type=int,default=2)
    parser.add_argument("--scales",type=int,default=2)
    args=parser.parse_args()
    result=run(args.backend,args.nside,args.scales)
    result["status"]="passed"
    rendered=json.dumps(result,indent=2)
    print(rendered)
    if args.output: Path(args.output).write_text(rendered+"\n")
