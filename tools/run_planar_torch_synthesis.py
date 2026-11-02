"""STL-Dev's planar synthesis (`Synthesis.synthesize_from_maps`, kernel data
class, pbc=True, defaults: L = 4, power spectrum, S3/S4 weights 3.5 and
3.5^2, L-BFGS 100 iterations, seed 26), with its initial maps and loss
history saved so that `stl-synth2d --mode stldev --init` can replay it.

    PYTHONPATH=<STL-Dev> python tools/run_planar_torch_synthesis.py \
        --target Turb_6.npy --ntarget 10 --nbatch 10 --niter 100 --out torch256

Writes PREFIX_init.npy (standardised units, [B, H, W]), PREFIX_maps.npy
([B, H, W]) and PREFIX.json (loss_history, seconds, device).
"""

import argparse
import json
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
    p.add_argument("--niter", type=int, default=100)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    target = np.load(a.target)[: a.ntarget].astype(np.float64)
    data = STL_2D_Kernel_Torch(target[:, None], pbc=True)
    rec = {}
    original = Synthesis.optimize_lbfgs

    def recording(model, loss_fn, lr, max_iter, history_size, verbose, print_iter):
        rec["u0"] = model.u.detach().cpu().numpy()[:, 0].copy()
        history = []

        def fn(o):
            v = loss_fn(o)
            history.append(v.item())
            return v

        t = time.perf_counter()
        u = original(model, fn, lr, max_iter, history_size, False, print_iter)
        rec["seconds"] = time.perf_counter() - t
        rec["history"] = history
        return u

    Synthesis.optimize_lbfgs = recording
    u = Synthesis.synthesize_from_maps(data, nbatch=a.nbatch, pbc_running=True, max_iter=a.niter, verbose=False)
    np.save(a.out + "_init.npy", rec["u0"])
    np.save(a.out + "_maps.npy", u.detach().cpu().numpy())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = dict(device=device, threads=torch.get_num_threads(), seconds=rec["seconds"], evaluations=len(rec["history"]), loss_history=rec["history"])
    with open(a.out + ".json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(dict(device=device, seconds=rec["seconds"], evaluations=len(rec["history"]), loss_first=rec["history"][0], loss_last=rec["history"][-1])))


if __name__ == "__main__":
    main()
