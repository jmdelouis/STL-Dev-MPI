#!/usr/bin/env python3
"""Variance-only native map VJP checks in Gaussian and tiny physical units.

This is deliberately isolated from all mean, power, PDF and MF losses so another
term cannot conceal a broken SC variance cotangent. No mock native engine.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from stl_port.config import Config
from stl_port.driver import make_engine
from stl_port.sc_normalization import SCNormalization
from stl_port.losses import moments, moments_vjp
from stl_port.sc_variance import variance_ratio_loss, CONTRACT
from stl_port.sc_reports import selected_mask

print("SCRIPT VERSION: validate_sc_ratio_native.py 2026-10-29-v2-variance-ratio",flush=True)


def run(backend, threads):
    cfg=Config(nside=8,backend=backend,threads=threads,block_pixels=256,
               ps_variance_upper=23,cache_bytes=256*1024**2)
    rng=np.random.default_rng(292026)
    cases={}
    for physical, scale in ((False,1.),(True,1e-4)):
        x=scale*rng.normal(size=(10,cfg.npix))
        with make_engine(cfg) as engine:
            gen=0
            def forward(y, need_moments=False):
                nonlocal gen
                gen+=1;rows=[];spatial=[]
                for b in range(10):
                    v,sv=engine.forward(y[b:b+1],generation=gen,member=b,objective='variance-only',moments=need_moments)
                    rows.append(v[0])
                    if need_moments:spatial.append(sv[0])
                return np.stack(rows),np.stack(spatial) if need_moments else None
            raw,sv=forward(x,True)
            norm=SCNormalization.fit(engine.plan,raw,sv,nside=cfg.nside)
            values=norm.forward(raw);mu,v=moments(values)
            mask=selected_mask(engine.plan,physical)
            # All selected targets must be positive: zero entries are never skipped.
            reference=v[mask]*np.resize(np.array([.1,1.,10.]),int(mask.sum()))
            value,dv_selected=variance_ratio_loss(v[mask],reference)
            dv=np.zeros_like(v);dv[mask]=dv_selected
            q=norm.vjp(raw,moments_vjp(values,np.zeros_like(mu),dv))
            gradient=np.empty_like(x)
            for b in range(10):
                gen+=1
                engine.forward(x[b:b+1],generation=gen,member=b,objective='variance-only')
                gradient[b]=engine.vjp(q[b:b+1],generation=gen,member=b,objective='variance-only')[0]
            direction=rng.normal(size=x.shape)
            direction*=scale/np.linalg.norm(direction)
            analytic=float(np.sum(gradient*direction))
            def value_only(y):
                rr,_=forward(y)
                _,vv=moments(norm.forward(rr))
                return variance_ratio_loss(vv[mask],reference)[0]
            checks=[]
            for h in (1e-4,1e-5,1e-6):
                fd=(value_only(x+h*direction)-value_only(x-h*direction))/(2*h)
                rel=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12)
                checks.append({'dimensionless_step':h,'finite_difference':float(fd),
                               'analytic':analytic,'relative_error':float(rel)})
            passed=bool(np.isfinite(gradient).all() and np.linalg.norm(gradient)>0 and
                        sum(c['relative_error']<3e-5 for c in checks)>=2)
            cases['physical' if physical else 'gaussian']={
                'passed':passed,'input_scale':scale,'selected_coefficients':int(mask.sum()),
                'target_variance_min':float(reference.min()),'target_variance_max':float(reference.max()),
                'loss':value,'gradient_norm':float(np.linalg.norm(gradient)),
                'directional_checks':checks,'native_engine':engine.info()}
            print(json.dumps(cases['physical' if physical else 'gaussian'],indent=2),flush=True)
    return {'version':'2026-10-29-v2-variance-ratio','contract':CONTRACT,
            'purpose':'NUMERICAL synthetic variance-only native VJP, not TRAIN54 scientific validation',
            'backend':backend,'passed':all(c['passed'] for c in cases.values()),'cases':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('Output exists; preserve existing evidence')
    report=run(a.backend,a.threads)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
