"""Memorisation and calibration test of the per-level models.

    python diffusion/memorisation.py TARGET.npy MODEL_DIR LEVEL [LEVEL ...] [--samples 4]

For every LEVEL, draws S detail fields conditioned on the target's own
parent map `x_{LEVEL/2}` and compares, separately on the training faces and
on the held-out faces:

  r_truth  = corr(generated details, target details)
  r_pair   = corr(generated details, another generated draw)
  var      = variance of the generated details / target's

The target's details are one draw of the true conditional law, so a
calibrated model that has not memorised gives r_truth ~ r_pair (both equal
the fraction of variance of the conditional mean) and var ~ 1, on training
and held-out faces alike.  r_truth well above r_pair on the training faces
is memorisation.
"""

import argparse
import json

import numpy as np
import torch

import healpix_faces as hf
from sample import load_level, sample_details


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("models")
    p.add_argument("levels", type=int, nargs="+")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--churn", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    pyr = {hf.nside_of(m.size): m for m in hf.pyramid(np.load(a.target).astype(np.float64), 2)}
    report = {}
    for n in a.levels:
        model, ck = load_level(a.models, n, dev)
        parent, h_true = hf.haar(pyr[n])
        face = np.arange(parent.size) // (parent.size // 12)
        groups = {"train": np.isin(face, ck["train_faces"]), "held_out": ~np.isin(face, ck["train_faces"])}
        draws = [sample_details(model, ck, parent, a.steps, gen, a.churn) for _ in range(a.samples)]
        rep = {}
        for g, sel in groups.items():
            if not sel.any():
                continue
            t = h_true[:, sel].ravel()
            r_truth = [corr(d[:, sel].ravel(), t) for d in draws]
            r_pair = [corr(draws[i][:, sel].ravel(), draws[j][:, sel].ravel())
                      for i in range(len(draws)) for j in range(i + 1, len(draws))]
            var = [float(np.mean(d[:, sel] ** 2) / np.mean(h_true[:, sel] ** 2)) for d in draws]
            rep[g] = dict(r_truth=float(np.mean(r_truth)), r_pair=float(np.mean(r_pair)),
                          var=float(np.mean(var)), var_std=float(np.std(var)))
            print(f"level {n} {g:8s}: r_truth {rep[g]['r_truth']:.3f}  r_pair {rep[g]['r_pair']:.3f}  "
                  f"var {rep[g]['var']:.3f} +- {rep[g]['var_std']:.3f}", flush=True)
        report[n] = rep
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
