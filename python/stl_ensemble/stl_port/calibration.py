"""TRAIN54 calibration with one fixed ordered bootstrap and source preprocessing."""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path
import numpy as np
from .config import Config, canonical_hash, sha256_file, BASELINE_REVISION, SOURCE_SHA256
from .lut import PooledLUT
from .pdf import CICPDF
from .sc_variance import CONTRACT as SC_VARIANCE_CONTRACT
from .sc_reports import variance_sensitivity_issues, variance_floor_report, mean_sigma_report

SCRIPT_VERSION="2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: calibration.py {SCRIPT_VERSION}")

BLOCK_DEFINITIONS={
    "version":SCRIPT_VERSION,"average_divisor":5,"members":10,"ddof":1,
    "sc_population_variance":dict(SC_VARIANCE_CONTRACT),"sc_variance_weight":1,
    "physical_sc_mean":"sum of four family means; S1,S2,S3,S4",
    "gaussian_sc_mean":"coefficient-count mean of available native counterparts",
    "power":{"bands":16,"window":"cosine2 ell+0.5; no row normalization","mean_weight":10,"variance_weight":2.5,"var_epsilon":"0.01*max(trainvar,1e-30)"},
    "pdf":{"bins":100,"support_margin":0.02,"mean_weight":1,"variance_weight":1,"sigma_floor_rel":0.05,"variance_floor_rel":1e-4},
    "mf":{"thresholds":65,"limits":[-4,4],"epsilon":0.125,"mean_weight":1,"variance_weight":10,"gradient_floor":"stop_gradient(1e-4*mean(q)+1e-12)","angular_beam":None},
    "lut":{"knots":8192,"coordinates":"Gaussian quantiles","tails":"linear extrapolation","physical_monopole":"per synthesized member; differentiated","reference_recenter":False},
}

def science_config(config):
    data=config.manifest()
    # Execution modes must consume exactly the same target bytes.
    for k in ("mode","backend","threads","block_pixels","cache_bytes","active_batch","initialization_seed","history_size","learning_rate","tolerance_grad","tolerance_change"):
        data.pop(k,None)
    return data

def implementation_digest():
    root=Path(__file__).parent
    return canonical_hash({p.name:sha256_file(p) for p in sorted(root.glob("*.py"))})

def reference_paths(directory):
    return [Path(directory)/f"allskymap_nres12r{i:03d}.zs1.mag.nside2048.fits" for i in range(54)]

def prepare_maps(config, files, out_path):
    """Source-exact n128 path or direct native2048; references are not recentered."""
    import healpy as hp
    out=np.lib.format.open_memmap(out_path,mode="w+",dtype=np.float32,shape=(54,config.npix))
    for i,p in enumerate(files):
        print(f"[CALIBRATE] read TRAIN54 r{i:03d} {p}",flush=True)
        raw=hp.read_map(str(p),field=0,nest=False,dtype=np.float64)
        if hp.get_nside(raw)!=2048: raise ValueError(f"TRAIN54 map is not native2048: {p}")
        if config.nside==128:
            at512=hp.reorder(hp.ud_grade(raw,512),r2n=True).astype(np.float32)
            out[i]=at512.reshape(config.npix,16).mean(axis=1,dtype=np.float32)
        elif config.nside==2048:
            out[i]=hp.reorder(raw,r2n=True).astype(np.float32)
        else: raise ValueError("Production calibration only supports nside128 and native2048")
        if not np.all(np.isfinite(out[i])): raise ValueError(f"Nonfinite calibration map {p}")
    out.flush()
    return out

def population_targets(values, bootstrap_ids):
    values=np.asarray(values)
    mean=values[np.asarray(bootstrap_ids)].mean(axis=0)
    centered=values-values.mean(axis=0)
    variance=(centered.real**2+centered.imag**2).sum(axis=0)/(len(values)-1) if np.iscomplexobj(values) else np.var(values,axis=0,ddof=1)
    return mean,variance

def build_calibration(config, engine, reference_dir, output_dir):
    """Build immutable native target cache; never accepts a FOSCAT cache as input."""
    from .sc_normalization import SCNormalization
    from .harmonics import FullSkySHT, lp_windows
    from .minkowski import Minkowski
    config.validate()
    if engine.batch!=1: raise ValueError("Calibration engine must be constructed for one active member")
    files=reference_paths(reference_dir)
    missing=[str(p) for p in files if not p.is_file()]
    if missing: raise FileNotFoundError("Missing TRAIN54 inputs:\n"+"\n".join(missing))
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Calibration destination must be empty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    ids=np.random.default_rng(config.bootstrap_seed).choice(np.arange(54),size=10,replace=True)
    sources=[{"id":i,"name":p.name,"sha256":sha256_file(p),"bytes":p.stat().st_size} for i,p in enumerate(files)]
    base={"library":"STL-Dev-MPI","native_revision":BASELINE_REVISION,"source_sha256":SOURCE_SHA256,
          "config":science_config(config),"plan":engine.plan,"preprocessing":"2048 RING -> ud_grade512 -> NEST float32 -> exact blockmean128" if config.nside==128 else "native2048 RING -> NEST float32",
          "sources":sources,"bootstrap_ids":ids.tolist(),"losses":BLOCK_DEFINITIONS,
          "native_library_sha256":engine.library_sha256,
          "implementation_sha256":implementation_digest()}
    physical=prepare_maps(config,files,out/"training_physical.npy")
    lut=PooledLUT.from_training(physical,work_path=out/"lut_quantile_work.dat",nbins=8192)
    np.savez(out/"lut.npz",probabilities=lut.probabilities,physical_knots=lut.physical_knots)
    gauss=np.lib.format.open_memmap(out/"training_gaussian.npy",mode="w+",dtype=np.float32,shape=physical.shape)
    for i in range(54): gauss[i]=lut.gaussianize(physical[i])
    gauss.flush()
    # The physical references deliberately retain the source preprocessing mean.
    raw_stats={}; spatial={}
    for label,maps in (("gaussian",gauss),("physical",physical)):
        v=[]; s=[]
        for i in range(54):
            print(f"[CALIBRATE] native {label} r{i:03d}",flush=True)
            values,moments=engine.forward(np.asarray(maps[i:i+1],dtype=np.float64),generation=1,member=i,objective="calibration-"+label,moments=True)
            v.append(values[0]); s.append(moments[0])
        raw_stats[label]=np.stack(v); spatial[label]=np.stack(s)
    # Calibrated native normalizers must remain distinct from across-map variance.
    normalizers={}; targets={}; features={}
    for label,prefix in (("gaussian","gsc"),("physical","psc")):
        norm=SCNormalization.fit(engine.plan,raw_stats[label],spatial[label],nside=config.nside)
        normalizers[label]=norm.spec
        values=norm.forward(raw_stats[label])
        sigma=norm.sigma(raw_stats[label],spatial[label]).mean(axis=0)
        targets[prefix+"_mean"],targets[prefix+"_var"]=population_targets(values,ids)
        targets[prefix+"_sigma"]=sigma
        features[prefix]=values
        features[prefix+"_raw"]=raw_stats[label]
        features[prefix+"_spatial_variance"]=spatial[label]
    sht=FullSkySHT(config.nside,lmax=config.lmax,nthreads=config.threads)
    windows,centers=lp_windows(config.lmax)
    lo=float(np.min(physical)); hi=float(np.max(physical)); span=hi-lo
    support=np.array([lo-0.02*span,hi+0.02*span])
    pdf=CICPDF(*support)
    mf=Minkowski(sht)
    power=[]; hist=[]; mink=[]; gaussian_cl=[]
    for i in range(54):
        print(f"[CALIBRATE] power/PDF/MF r{i:03d}",flush=True)
        x=np.asarray(physical[i],dtype=np.float64)
        power.append(windows @ sht.power(sht.analysis(x)))
        hist.append(pdf.forward(x)[0])
        mink.append(mf.forward(x)[0])
        # Source _large_avg_mean_cl uses each uncentered Gaussian reference.
        # The GRF initializer subsequently sets C_0=0; do not recenter here.
        g=np.asarray(gauss[i],dtype=np.float64)
        gaussian_cl.append(sht.power(sht.analysis(g)))
    for name,arr in (("power",power),("pdf",hist),("mf",mink)):
        values=np.stack(arr)
        targets[name+"_mean"],targets[name+"_var"]=population_targets(values,ids)
        features[name]=values
    targets["pdf_support"]=support
    targets["power_centers"]=centers
    targets["initialization_cl"]=np.stack(gaussian_cl)[ids].mean(axis=0)
    targets["initialization_reference_mean"]=np.asarray(np.mean(gauss[int(ids[0])],dtype=np.float64))
    targets["bootstrap_ids"]=ids
    np.savez(out/"targets.npz",**targets)
    np.savez(out/"training_features.npz",**features,gaussian_cl=np.stack(gaussian_cl))
    base["normalizers"]=normalizers
    base["lut_sha256"]=sha256_file(out/"lut.npz")
    base["target_sha256"]=sha256_file(out/"targets.npz")
    base["scientific_variance_guard"]={"passed":not variance_sensitivity_issues(targets,engine.plan),"issues":variance_sensitivity_issues(targets,engine.plan)}
    base["variance_floor_report"]=variance_floor_report(targets,engine.plan)
    base["mean_sigma_report"]=mean_sigma_report(targets,engine.plan)
    base["cache_schema"]="stl-ensemble-calibration-v2"
    base["cache_identity"]=canonical_hash(base)
    (out/"manifest.json").write_text(json.dumps(base,indent=2,allow_nan=False)+"\n")
    return load_calibration(config,engine.plan,out,native_library_sha256=engine.library_sha256,require_variance_sensitivity=False)

def load_calibration(config,plan,directory,*,strict_code=True,native_library_sha256=None,require_variance_sensitivity=False):
    from .sc_normalization import SCNormalization
    root=Path(directory)
    meta=json.loads((root/"manifest.json").read_text())
    if meta.get("cache_schema") != "stl-ensemble-calibration-v2":
        raise ValueError("v2 requires explicit v1 cache migration; do not edit the old manifest")
    identity_payload={k:v for k,v in meta.items() if k != "cache_identity"}
    if canonical_hash(identity_payload)!=meta["cache_identity"]: raise ValueError("Calibration manifest identity check failed")
    if meta.get("losses") != BLOCK_DEFINITIONS:
        raise ValueError("Cache loss contract differs from this v2 implementation")
    if meta["library"]!="STL-Dev-MPI" or meta["native_revision"]!=BASELINE_REVISION: raise ValueError("Incompatible native target cache")
    if meta["config"]!=science_config(config) or meta["plan"]!=plan: raise ValueError("Cache scientific configuration/coefficient plan mismatch; rebuild targets")
    if native_library_sha256 is not None and meta.get("native_library_sha256")!=native_library_sha256: raise ValueError("Native binary changed; rebuild or explicitly revalidate target compatibility")
    if strict_code and meta["implementation_sha256"]!=implementation_digest(): raise ValueError("Implementation identity changed; use the explicit versioned migration/revalidation workflow")
    for file,key in (("targets.npz","target_sha256"),("lut.npz","lut_sha256")):
        if sha256_file(root/file)!=meta[key]: raise ValueError(f"Cache bytes changed: {file}")
    with np.load(root/"targets.npz",allow_pickle=False) as f: targets={k:f[k] for k in f.files}
    issues=variance_sensitivity_issues(targets,plan)
    guard={"passed":not issues,"issues":issues}
    if guard != meta.get("scientific_variance_guard"):
        raise ValueError("Recorded scientific guard disagrees with the measured targets")
    if require_variance_sensitivity and issues:
        raise ValueError("SC ratio loss requires every selected target variance to be finite and positive.\n"+"\n".join(issues))
    with np.load(root/"lut.npz",allow_pickle=False) as f: lut=PooledLUT(f["probabilities"],f["physical_knots"])
    norms={label:SCNormalization(plan,spec) for label,spec in meta["normalizers"].items()}
    return targets,lut,norms,meta

def draw_initial_ensemble(config,targets,sht=None):
    """Source RNG ordering and genuine target-C_l GRFs, without shell rescaling."""
    import healpy as hp
    from .harmonics import FullSkySHT
    sht=sht or FullSkySHT(config.nside,lmax=config.lmax,nthreads=config.threads)
    cl=np.maximum(np.nan_to_num(np.asarray(targets["initialization_cl"],dtype=np.float64),nan=0,posinf=0,neginf=0),0)
    cl=cl.copy(); cl[0]=0.0
    if cl.shape!=(config.lmax+1,): raise ValueError("Initial target C_l length mismatch")
    rng=np.random.default_rng(config.initialization_seed)
    maps=np.empty((config.members,config.npix),dtype=np.float64)
    for b in range(config.members):
        alm=np.zeros(hp.Alm.getsize(config.lmax),dtype=np.complex128)
        for m in range(config.lmax+1):
            ell=np.arange(m,config.lmax+1)
            idx=hp.Alm.getidx(config.lmax,ell,m)
            if m==0: alm[idx]=rng.standard_normal(len(ell))*np.sqrt(cl[ell])
            else: alm[idx]=np.sqrt(0.5*cl[ell])*(rng.standard_normal(len(ell))+1j*rng.standard_normal(len(ell)))
        # Preserve source addition of the first bootstrap Gaussian reference mean.
        # Native float64 is the chosen optimizer precision; source stored float32.
        maps[b]=sht.synthesis(alm)+float(targets["initialization_reference_mean"])
    return maps
