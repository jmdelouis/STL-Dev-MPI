"""Ensemble synthesis with the legacy (full-sky, autograd) path of STL-Dev.

The reference for docs/bench_n128.md: the objective of `stl-synth --weights
none --ensemble B`,

    loss = sum_i | (1/B) sum_m S_i(x_m) - S_i(target) |^2,

with the ScatCov of `ST_Operator.apply` (no pyramid, no explicit adjoint)
differentiated by autograd, and `torch.optim.LBFGS` (strong Wolfe, the
defaults of `benchmark_synthesis.py --mode legacy`) on the B maps.

Modes:

* ``batch``      - the B maps go through one `apply` as a batch [B, 1, N];
                   one autograd graph of the whole ensemble (minimum time).
* ``sequential`` - one member at a time (minimum memory): a forward pass of
                   every member without graph gives the mean statistics,
                   then every member is evaluated again with its graph and
                   back-propagated with the cotangent 2 (mean - target) / B.
                   Same loss and gradient, one map's graph at a time, one
                   extra forward per member.

    python legacy_synth.py --stl-dev DIR --target T.npy --init 'init_m{m}.npy' \
        --ensemble 4 --L 4 --niter 100 --mode batch --device cuda --out run.json

The initial maps are read in standardised units (`stl-synth --init-units
standardized --init-noise 0` reads the same files).
"""

import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stl-dev", required=True, help="directory holding STL_main (and healpix_analyse)")
    p.add_argument("--target", required=True)
    p.add_argument("--init", required=True, help="per-member path, {m} = three-digit member index")
    p.add_argument("--ensemble", type=int, default=4)
    p.add_argument("--J", type=int, default=None, help="default log2(nside) - 1")
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--niter", type=int, default=100)
    p.add_argument("--history-size", type=int, default=10)
    p.add_argument("--mode", choices=("batch", "sequential"), default="batch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True, help="JSON report")
    p.add_argument(
        "--geometry-cache", default=None,
        help="HEALPIXCONV_CACHE of healpix-analyse; default: a fresh temporary directory, removed at the end",
    )
    args = p.parse_args()

    # healpix-analyse saves the convolution geometry on its first build and
    # reloads it in later processes with float32 weights (statistics then
    # differ by ~1e-9 relative); a fresh directory builds it exactly.
    scratch = None
    if args.geometry_cache is None:
        scratch = tempfile.mkdtemp(prefix="hpxconv_", dir=os.path.dirname(os.path.abspath(args.out)))
    os.environ["HEALPIXCONV_CACHE"] = args.geometry_cache or scratch
    sys.path.insert(0, args.stl_dev)
    import numpy as np
    import torch
    from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch

    torch.set_num_threads(args.threads)
    started_setup = time.perf_counter()
    raw = np.load(args.target).astype(np.float64).reshape(-1)
    npix = raw.size
    nside = int(round((npix / 12) ** 0.5))
    if 12 * nside**2 != npix:
        raise SystemExit(f"{args.target}: not a full-sky map")
    # Same standardisation as stl-synth (population moments).
    mean, std = float(raw.mean()), float(raw.std())
    target_map = torch.as_tensor((raw - mean) / std)
    B = args.ensemble
    inits = [np.load(args.init.replace("{m}", f"{m:03d}")).astype(np.float64).reshape(-1) for m in range(B)]
    J = args.J if args.J is not None else max(1, nside.bit_length() - 2)

    # The data class takes its device from the array.
    target = STL_Healpix_Kernel_Torch(target_map.to(args.device))
    device = target.device
    op = target.get_ST_op(
        J=J,
        L=args.L,
        norm="vanilla",
        compute_PS=False,
        has_fewer_convolutions=False,
        wavelet_op_kwargs=dict(kernel_size=args.kernel_size, ellipsoid="sphere"),
    )
    plan = op.build_plan()

    def values(x):
        """Plan coefficients [b, n_coef] of the maps x [b, 1, N] (legacy apply)."""
        stats = op.apply(target.new_like(x))
        return torch.stack([getattr(stats, c.family)[c.legacy_index] for c in plan], -1)

    with torch.no_grad():
        t_values = values(target_map.to(device).reshape(1, 1, npix))[0]
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    sync()
    setup_seconds = time.perf_counter() - started_setup
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    u = torch.as_tensor(np.stack(inits)).reshape(B, 1, npix).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [u],
        lr=1.0,
        max_iter=args.niter,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )
    history, stamps = [], []
    started = time.perf_counter()

    def closure():
        optimizer.zero_grad()
        if args.mode == "batch":
            residual = values(u).mean(0) - t_values
            loss = residual.abs().square().sum()
            loss.backward()
        else:
            with torch.no_grad():
                residual = sum(values(u[m : m + 1])[0] for m in range(B)) / B - t_values
                loss = residual.abs().square().sum()
            cot = (2.0 / B) * residual.conj()
            grads = []
            for m in range(B):
                um = u.detach()[m : m + 1].clone().requires_grad_(True)
                (g,) = torch.autograd.grad((cot * values(um)[0]).real.sum(), um)
                grads.append(g)
            u.grad = torch.cat(grads, 0)
        sync()
        history.append(float(loss.detach()))
        stamps.append(time.perf_counter() - started)
        print(f"[legacy-{args.mode}] evaluation {len(history):4d}  loss = {history[-1]:.15e}  t = {stamps[-1]:8.2f} s", flush=True)
        return loss

    optimizer.step(closure)
    sync()
    seconds = time.perf_counter() - started
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(os.path.splitext(args.out)[0] + "_maps_standardized.npy", u.detach().cpu().numpy().reshape(B, npix))
    report = dict(
        code="STL-Dev legacy (ST_Operator.apply + autograd)",
        mode=args.mode,
        nside=nside,
        J=J,
        L=args.L,
        ensemble=B,
        coefficients=len(plan),
        niter=args.niter,
        evaluations=len(history),
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if cuda else None,
        torch=torch.__version__,
        threads=args.threads,
        target=dict(file=args.target, mean=mean, std=std),
        init=args.init,
        setup_seconds=round(setup_seconds, 2),
        seconds=round(seconds, 2),
        seconds_per_evaluation=round(seconds / max(1, len(history)), 4),
        loss_history=history,
        evaluation_seconds=stamps,
        cuda_max_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        cuda_max_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("loss_history", "evaluation_seconds")}))


if __name__ == "__main__":
    main()
