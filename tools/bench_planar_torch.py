"""Seconds per evaluation (statistics + gradient) and peak memory of
STL-Dev's planar synthesis loss (`Synthesis.synthesize_from_maps`, kernel
data class, pbc=True, L = 4, power spectrum, mean field), for comparison
with `stl-synth2d --mode stldev --bench`.

    PYTHONPATH=<STL-Dev> python tools/bench_planar_torch.py --target T.npy \
        --ntarget 10 --nbatch 10 --evals 5 --out bench.json

The L-BFGS driver is replaced by K timed closures (forward, loss,
backward) on the model `optimize_from_maps` builds; peak memory is the
process's maximum resident set (CPU) or torch's maximum allocated device
memory (CUDA).  One configuration per process (the RSS peak is monotone).
"""

import argparse
import json
import resource
import time

import numpy as np
import torch

from STL_main import Synthesis
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--ntarget", type=int, default=10)
    p.add_argument("--nbatch", type=int, default=10)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--no-ps", action="store_true", help="ScatCov only (compute_PS=False)")
    p.add_argument("--out")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    # STL-Dev picks its device at import (CUDA when visible): hide the GPUs
    # with CUDA_VISIBLE_DEVICES="" for a CPU measurement.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = np.load(a.target)[: a.ntarget].astype(np.float64)
    data = STL_2D_Kernel_Torch(target[:, None], pbc=True)
    times, losses = [], []

    def timed(model, loss_fn, lr, max_iter, history_size, verbose, print_iter):
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(a.evals):
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            if model.u.grad is not None:
                model.u.grad = None
            loss = loss_fn(model())
            loss.backward()
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            losses.append(loss.item())
        return model.u.detach()

    Synthesis.optimize_lbfgs = timed
    t0 = time.perf_counter()
    Synthesis.synthesize_from_maps(data, nbatch=a.nbatch, pbc_running=True, max_iter=1, verbose=False, compute_PS=not a.no_ps)
    total = time.perf_counter() - t0
    report = dict(
        target=a.target,
        shape=list(target.shape[1:]),
        ntarget=a.ntarget,
        nbatch=a.nbatch,
        power_spectrum=not a.no_ps,
        device=device,
        threads=torch.get_num_threads(),
        torch=torch.__version__,
        seconds=times,
        loss=losses[-1],
        total_seconds=total,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    if device == "cuda":
        report["peak_device_bytes"] = torch.cuda.max_memory_allocated()
        report["gpu"] = torch.cuda.get_device_name()
    print(json.dumps({k: report[k] for k in ("shape", "nbatch", "device", "seconds", "peak_rss_bytes")} | ({"peak_device_bytes": report["peak_device_bytes"]} if "peak_device_bytes" in report else {})))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
