#!/usr/bin/env python3
"""Scientific comparison of native/FOSCAT ensembles using one calibrated plan.

This workflow reads maps and computes diagnostics; it does not optimize or
correct them. Every physical map is evaluated with the same native STL plan,
LUT, PDF support, harmonic convention and MF definitions. FOSCAT-produced maps
are compared as maps, never by treating FOSCAT coefficient caches as native SC.
"""
from __future__ import annotations

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: diagnose.py {SCRIPT_VERSION}", flush=True)

import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np

from stl_port.config import Config, sha256_file
from stl_port.calibration import load_calibration
from stl_port.driver import make_engine, write_json, new_output
from stl_port.harmonics import FullSkySHT, lp_windows
from stl_port.minkowski import Minkowski
from stl_port.pdf import CICPDF


def _input_members(paths, nside, ordering):
    """Yield maps unchanged except an explicitly requested ordering permutation."""
    import healpy as hp
    npix = 12*nside*nside
    for name in paths:
        path = Path(name)
        if path.suffix.lower() == ".npy":
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.ndim == 1:
                data = data[None, :]
            if data.ndim != 2 or data.shape[1] != npix:
                raise ValueError(f"Expected physical map(s) at nside{nside}: {path}, shape={data.shape}")
            for row in data:
                x = np.asarray(row, dtype=np.float64)
                yield hp.reorder(x, r2n=True) if ordering == "RING" else x
        elif path.suffix.lower() in ("fits", ".fits", ".fit", ".gz"):
            # FITS ORDERING is interpreted by healpy; output is always NESTED.
            x = hp.read_map(path, nest=True, dtype=np.float64)
            if x.shape != (npix,):
                raise ValueError(f"FITS is not native diagnostic resolution: {path}")
            yield x
        else:
            raise ValueError(f"Use physical .npy ensembles or HEALPix FITS maps: {path}")


def _heldout(directory, nside):
    """Exactly r054..r107, with the same preprocessing as calibration."""
    import healpy as hp
    files = [Path(directory)/f"allskymap_nres12r{i:03d}.zs1.mag.nside2048.fits" for i in range(54,108)]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing held-out inputs:\n"+"\n".join(missing))
    for p in files:
        raw = hp.read_map(p, nest=False, dtype=np.float64)
        if hp.get_nside(raw) != 2048:
            raise ValueError(f"Expected native2048 held-out source: {p}")
        if nside == 128:
            fine = hp.reorder(hp.ud_grade(raw, 512), r2n=True).astype(np.float32)
            yield fine.reshape(12*nside*nside, 16).mean(axis=1, dtype=np.float32).astype(np.float64)
        elif nside == 2048:
            yield hp.reorder(raw, r2n=True).astype(np.float32).astype(np.float64)
        else:
            raise ValueError("Held-out scientific comparison requires nside128 or native2048")


def moments(x):
    x = np.asarray(x)
    mean = x.mean(axis=0)
    if len(x) < 2:
        variance = np.full(mean.shape, np.nan)
    else:
        delta = x-mean
        variance = np.sum(np.abs(delta)**2, axis=0)/(len(x)-1)
    return mean, variance


def evaluate_members(members, *, engine, norms, lut, sht, pdf, mf, windows,
                     label, gaussian_members=None, generation=1):
    result = {key: [] for key in ("cl", "power", "pdf", "mf", "psc", "gsc", "gaussian_cl")}
    outside = []
    first_maps = []
    for b, x in enumerate(members):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"Nonfinite input {label} member {b}")
        print(f"[DIAGNOSE] {label} member {b}", flush=True)
        # Diagnostic reference arrays retain their physical monopole convention.
        # Supplied synthesized arrays already contain the synthesis mean removal.
        x = np.asarray(x, dtype=np.float64)
        raw, _ = engine.forward(x[None, :], generation=generation, member=b,
                                objective="diagnostic-physical-"+label)
        result["psc"].append(norms["physical"].forward(raw)[0])
        if gaussian_members is None:
            g = lut.gaussianize(x)
        else:
            try:
                g = np.asarray(next(gaussian_members), dtype=np.float64)
            except StopIteration as exc:
                raise ValueError("Fewer Gaussian maps than physical maps") from exc
        raw, _ = engine.forward(g[None, :], generation=generation, member=b,
                                objective="diagnostic-gaussian-"+label)
        result["gsc"].append(norms["gaussian"].forward(raw)[0])
        result["gaussian_cl"].append(sht.power(sht.analysis(g)))
        cl = sht.power(sht.analysis(x))
        result["cl"].append(cl)
        result["power"].append(windows@cl)
        hist, pdfstate = pdf.forward(x)
        result["pdf"].append(hist)
        outside.append(pdfstate.outside_fraction)
        curves, state = mf.forward(x)
        result["mf"].append(curves)
        del state
        if b < 10:
            # Keep only plotting samples, with a bounded native2048 preview.
            # Full-resolution diagnostics above always use every input pixel.
            if sht.nside <= 128:
                first_maps.append(x.copy())
            else:
                import healpy as hp
                first_maps.append(hp.ud_grade(x, 128, order_in="NESTED", order_out="NESTED"))
    if not result["cl"]:
        raise ValueError(f"No maps supplied for {label}")
    if gaussian_members is not None:
        try:
            next(gaussian_members)
        except StopIteration:
            pass
        else:
            raise ValueError("More Gaussian maps than physical maps")
    return {k: np.stack(v) for k,v in result.items()}, outside, first_maps


def metric_report(features, targets):
    out = {}
    for key in ("gsc", "psc", "power", "pdf", "mf"):
        mean, var = moments(features[key])
        tm, tv = targets[key+"_mean"], targets[key+"_var"]
        valid = np.isfinite(tv) & (tv > 0)
        ratio = var[valid]/tv[valid]
        finite = np.isfinite(ratio)
        out[key] = {
            "member_count": int(len(features[key])),
            "mean_relative_l2": float(np.linalg.norm(mean-tm)/max(np.linalg.norm(tm), np.finfo(float).tiny)),
            "target_positive_variance_count": int(np.count_nonzero(valid)),
            "median_variance_ratio_positive_targets": float(np.median(ratio[finite])) if finite.any() else None,
            "note": "Descriptive summaries; no covariance-aware chi-square or significance claim. Tiny positive targets are included here; inspect saved per-feature arrays and floor report.",
        }
    return out


def plot_diagnostics(out, datasets, targets, plan, centers, pdf_centers, nu, galleries, preview_nside):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import healpy as hp
    labels = list(datasets)
    colors = {name: plt.get_cmap("tab10")(i) for i,name in enumerate(labels)}
    stats = {label: {key: moments(arr) for key,arr in data.items()} for label,data in datasets.items()}

    def curves(ax, key, x, variance=False, component=None, target=True):
        for label in labels:
            y = stats[label][key][int(variance)]
            if component is not None:
                y = y[component]
            ax.plot(x, y, label=label, color=colors[label], linewidth=1)
        if target and key+"_mean" in targets:
            y = targets[key+("_var" if variance else "_mean")]
            if component is not None:
                y = y[component]
            ax.plot(x, y, "k--", linewidth=1.4, label="TRAIN54 variance" if variance else "fixed bootstrap mean target")
        ax.grid(alpha=.2)

    fig, axes = plt.subplots(2, 2, figsize=(12,7))
    ell = np.arange(next(iter(datasets.values()))["cl"].shape[-1])
    for row in range(2):
        curves(axes[row,0], "cl", ell, variance=bool(row))
        curves(axes[row,1], "power", centers, variance=bool(row))
        for ax in axes[row]:
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(left=2)
            ax.set_ylabel("population variance (ddof=1)" if row else "ensemble mean")
            ax.set_xlabel(r"$\ell$")
    axes[0,0].set_title("Physical full-sky Cℓ; no pixel-window correction")
    axes[0,1].set_title("Physical LP weighted sums (16 bands)")
    axes[0,1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/"physical_power.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9,7), sharex=True)
    for row in range(2):
        curves(axes[row], "pdf", pdf_centers, variance=bool(row))
        axes[row].set_yscale("log")
        axes[row].set_ylabel("density variance" if row else "CIC density mean")
    axes[0].legend(fontsize=7); axes[1].set_xlabel("physical κ")
    fig.tight_layout(); fig.savefig(out/"physical_pdf.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12,10), sharex=True)
    for component in range(3):
        for col in range(2):
            curves(axes[component,col], "mf", nu, variance=bool(col), component=component)
            axes[component,col].set_ylabel(f"V{component} "+("variance" if col else "mean"))
            if col:
                axes[component,col].set_yscale("log")
    axes[0,0].legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel("per-member normalized threshold ν")
    fig.tight_layout(); fig.savefig(out/"physical_minkowski.png", dpi=150); plt.close(fig)

    families = list(dict.fromkeys(p["family"] for p in plan))
    for key in ("gsc", "psc"):
        chosen = families if key == "gsc" else [f for f in families if f in ("S1","S2","S3","S4")]
        fig, axes = plt.subplots(len(chosen), 3, figsize=(15,3*len(chosen)), squeeze=False)
        for row,family in enumerate(chosen):
            indices = np.array([i for i,p in enumerate(plan) if p["family"] == family])
            # Display thinning only, never reduction of the saved feature vectors.
            stride = max(1, int(np.ceil(len(indices)/3000)))
            shown = indices[::stride]
            for label in labels:
                mean, var = stats[label][key]
                axes[row,0].plot(shown, mean[shown].real, color=colors[label], label=label, linewidth=.7)
                axes[row,1].plot(shown, mean[shown].imag, color=colors[label], linewidth=.7)
                axes[row,2].plot(shown, var[shown], color=colors[label], linewidth=.7)
            axes[row,0].plot(shown, targets[key+"_mean"][shown].real, "k--", linewidth=.8, label="bootstrap target")
            axes[row,1].plot(shown, targets[key+"_mean"][shown].imag, "k--", linewidth=.8)
            axes[row,2].plot(shown, targets[key+"_var"][shown], "k--", linewidth=.8)
            axes[row,2].set_yscale("log")
            for ax,colname in zip(axes[row], ("Re(mean)","Im(mean)","Var(Re)+Var(Im)")):
                ax.set_title(f"{family}: {colname}"); ax.set_xlabel("native plan index"); ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(("Gaussian" if key == "gsc" else "Physical")+" native SC, same calibrated plan for every map dataset")
        fig.tight_layout(); fig.savefig(out/(key+"_means_variances.png"), dpi=150); plt.close(fig)

    fig,axes = plt.subplots(2,1,figsize=(9,7),sharex=True)
    for row in range(2):
        curves(axes[row], "gaussian_cl", ell, variance=bool(row))
        axes[row].set_xscale("log"); axes[row].set_yscale("log"); axes[row].set_xlim(left=2)
        axes[row].set_ylabel("variance" if row else "mean")
    axes[0].legend(fontsize=7); axes[0].set_title("Secondary Gaussian spectrum diagnostics")
    axes[-1].set_xlabel(r"$\ell$")
    fig.tight_layout();fig.savefig(out/"gaussian_power.png",dpi=150);plt.close(fig)

    if galleries:
        ref = galleries.get("native", next(iter(galleries.values())))[0]
        low,high = np.quantile(ref,[.01,.99])
        for label,maps in galleries.items():
            count=len(maps); columns=min(5,count); rows=(count+columns-1)//columns
            fig=plt.figure(figsize=(4*columns,2.5*rows))
            for i,x in enumerate(maps):
                hp.mollview(x,nest=True,fig=fig.number,sub=(rows,columns,i+1),title=f"{label} {i}",
                            min=low,max=high,cbar=(i==0),notext=True)
            fig.suptitle(f"Common color limits; NEST maps; preview nside={preview_nside} (diagnostic display only)")
            fig.savefig(out/("maps_"+label+".png"),dpi=120,bbox_inches="tight");plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent/"configs/native128.json"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--maps", required=True, nargs="+", help="Physical native NEST ensemble .npy or FITS maps")
    parser.add_argument("--gaussian-maps", nargs="+", help="Optional actual Gaussian optimization variables; otherwise re-Gaussianize physical maps")
    parser.add_argument("--map-ordering", choices=("NEST","RING"), default="NEST", help="Ordering of .npy inputs (FITS uses its header)")
    parser.add_argument("--foscat-maps", nargs="+", help="Optional validated physical FOSCAT map ensemble at the same nside")
    parser.add_argument("--foscat-ordering", choices=("NEST","RING"), help="Required with --foscat-maps for explicit .npy convention")
    parser.add_argument("--heldout-dir", help="Optional raw r054..r107 FITS directory; never used for calibration")
    parser.add_argument("--backend", choices=("cpu","gpu"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", required=True)
    args=parser.parse_args(argv)
    if args.foscat_maps and args.foscat_ordering is None:
        parser.error("Supply --foscat-ordering NEST or RING; do not guess the reference .npy ordering")
    config=Config.read(args.config)
    changes={"mode":"stream-replay"}
    if args.backend is not None: changes["backend"]=args.backend
    if args.threads is not None: changes["threads"]=args.threads
    config=dataclasses.replace(config, **changes)
    config.validate()
    out=new_output(args.output)
    cache=Path(args.cache)
    with make_engine(config) as engine:
        targets,lut,norms,meta=load_calibration(config,engine.plan,cache,
            native_library_sha256=engine.library_sha256)
        sht=FullSkySHT(config.nside, nthreads=config.threads)
        windows,centers=lp_windows(config.lmax)
        pdf=CICPDF(*targets["pdf_support"])
        mf=Minkowski(sht)
        with np.load(cache/"training_features.npz",allow_pickle=False) as f:
            training={key:f[key].copy() for key in ("gsc","psc","power","pdf","mf","gaussian_cl")}
        training_maps=np.load(cache/"training_physical.npy",mmap_mode="r",allow_pickle=False)
        training["cl"]=np.stack([sht.power(sht.analysis(np.asarray(x,dtype=float))) for x in training_maps])
        # Auxiliary full-C_ell diagnostic targets, without adding a loss block.
        ids=np.asarray(meta["bootstrap_ids"])
        for name in ("cl","gaussian_cl"):
            targets[name+"_mean"]=training[name][ids].mean(axis=0)
            targets[name+"_var"]=moments(training[name])[1]
        datasets={"TRAIN54":training}
        galleries={};outside={}
        common=dict(engine=engine,norms=norms,lut=lut,sht=sht,pdf=pdf,mf=mf,windows=windows)
        gaussian=None if args.gaussian_maps is None else iter(_input_members(args.gaussian_maps,config.nside,args.map_ordering))
        datasets["native"],outside["native"],galleries["native"]=evaluate_members(
            _input_members(args.maps,config.nside,args.map_ordering),label="native",gaussian_members=gaussian,generation=1,**common)
        if len(datasets["native"]["cl"]) != 10:
            raise ValueError("The synthesized statistical ensemble must contain exactly ten members")
        if args.foscat_maps:
            datasets["FOSCAT_maps"],outside["FOSCAT_maps"],galleries["FOSCAT_maps"]=evaluate_members(
                _input_members(args.foscat_maps,config.nside,args.foscat_ordering),label="FOSCAT_maps",generation=2,**common)
        if args.heldout_dir:
            datasets["heldout54"],outside["heldout54"],galleries["heldout54"]=evaluate_members(
                _heldout(args.heldout_dir,config.nside),label="heldout54",generation=3,**common)
        arrays={}
        for label,data in datasets.items():
            for key,values in data.items():
                arrays[label+"__"+key]=values
                mean,variance=moments(values)
                arrays[label+"__"+key+"__mean"]=mean
                arrays[label+"__"+key+"__variance"]=variance
        arrays.update({"target__"+k:v for k,v in targets.items()})
        np.savez(out/"diagnostic_statistics.npz",**arrays)
        report={"script_version":SCRIPT_VERSION,"target_cache_identity":meta["cache_identity"],
                "config":config.manifest(),"native_plan":engine.plan,"variance_ddof":1,
                "bootstrap_ids":meta["bootstrap_ids"],"heldout_ids":list(range(54,108)) if args.heldout_dir else [],
                "metrics":{label:metric_report(data,targets) for label,data in datasets.items()},
                "pdf_outside_support":outside,"sc_variance_floor_report":meta["variance_floor_report"],
                "gaussian_diagnostics":{
                    "native":"actual supplied optimization variables" if args.gaussian_maps else "physical maps Gaussianized using pooled TRAIN54 LUT",
                    "FOSCAT_maps":"physical maps Gaussianized using pooled TRAIN54 LUT",
                    "TRAIN54":"cached pooled-LUT Gaussian calibration maps"},
                "comparison_limits":["Native STL 5x5/four orientations uses the optimization grid directly.",
                    "Validated original FOSCAT uses 7x7/eight orientations and internal finest-grid oversampling.",
                    "All SC plotted here is recomputed with the same native STL plan; no FOSCAT/native coefficient equality is assumed.",
                    "A single B=10 ensemble is a small variance sample; displayed differences alone do not establish scientific success.",
                    "Native2048 map galleries are downgraded only for display; every statistic uses the native-resolution input."],
                "input_files":[{"path":str(Path(p).resolve()),"sha256":sha256_file(p)} for p in args.maps],
                "scientific_success":"requires inspection of means, variances and map structure; not inferred automatically"}
        write_json(out/"diagnostic_report.json",report)
        plot_diagnostics(out,datasets,targets,engine.plan,centers,pdf.centers,mf.nu,galleries,min(128,config.nside))
    print(f"[DONE] diagnostic files in {out.resolve()}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
