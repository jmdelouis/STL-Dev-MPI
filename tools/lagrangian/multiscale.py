"""Coarse-to-fine synthesis with detail noise conditioned on the coarse scale
(docs/lagrangian_synthesis.md, step 3; the user's multi-scale Lagrangian idea).

    python multiscale.py --bin stl-synth2d --target Turb_6.npy --out DIR \
        --levels 32,64,128,256 --niter 300,200,200,400 --warm0 100 \
        --detail conditional --opt "--lagrangian both --anchor 50"

Level 0 (coarsest) is an ordinary synthesis (pixel L-BFGS, optionally with
--warm0 iterations on the stldev loss).  From level l-1 to level l, the
synthesis is upsampled (Fourier zero padding) and a detail field is added:

- white: Gaussian, with the rms of the targets' detail at that level (the
  usual coarse-to-fine recipe);
- conditional: for every pixel, a value drawn from the targets' detail
  pixels that share its bin of the joint (coarse value, coarse gradient
  modulus) quantiles, so that the new small scales fall on the structures
  already in place (heteroscedastic, histogram-exact per bin).

Then stl-synth2d optimises level l from that start with the options --opt
(pixel L-BFGS when empty).  Targets per level: Fourier truncation of the
Nyquist-filtered targets.  Writes DIR/level_<N>* (stl-synth2d outputs) and
DIR/multiscale.json (per-level evaluations, seconds, loss histories, and the
cost in level-0-of-the-finest equivalent evaluations, sum evals (N / N_max)^2).
"""
import argparse
import json
import os
import shlex
import subprocess
import time

import numpy as np


def nyquist(x):
    n = x.shape[-1]
    k = np.fft.fftfreq(n) * n
    m = (k[:, None] ** 2 + k[None, :] ** 2) <= (n / 2) ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(x) * m))


def resample(x, m):
    """Periodic Fourier resampling of [..., n, n] to [..., m, m] (pixel means kept)."""
    n = x.shape[-1]
    X = np.fft.fftshift(np.fft.fft2(x), axes=(-2, -1))
    if m < n:
        a = (n - m) // 2
        Y = X[..., a:a + m, a:a + m].copy()
        Y[..., 0, :] = 0
        Y[..., :, 0] = 0
    else:
        a = (m - n) // 2
        Y = np.zeros(x.shape[:-2] + (m, m), complex)
        Y[..., a:a + n, a:a + n] = X
        Y[..., a, :] = 0
        Y[..., :, a] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(Y, axes=(-2, -1)))) * (m / n) ** 2


def grad_mod(x):
    return np.hypot(np.roll(x, -1, -2) - np.roll(x, 1, -2), np.roll(x, -1, -1) - np.roll(x, 1, -1))


def standardize(x, ref):
    """stl-synth2d's standardisation of its targets `ref` [B, C, H, W], applied to `x`."""
    mean = ref.mean(axis=(0, 2, 3))
    std = np.sqrt(((ref - mean[None, :, None, None]) ** 2).mean(axis=(0, 2, 3)))
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


def detail_noise(s_up, t_fine, t_coarse_up, kind, rng, q=8):
    d = t_fine - t_coarse_up
    if kind == "white":
        return rng.normal(0.0, d.std(), s_up.shape)
    feats_t = [t_coarse_up, grad_mod(t_coarse_up)]
    feats_s = [s_up, grad_mod(s_up)]
    bins_t = np.zeros(d.shape, int)
    bins_s = np.zeros(s_up.shape, int)
    for ft, fs in zip(feats_t, feats_s):
        edges = np.quantile(ft, np.linspace(0, 1, q + 1)[1:-1])
        bins_t = bins_t * q + np.searchsorted(edges, ft)
        bins_s = bins_s * q + np.searchsorted(edges, fs)
    out = np.empty(s_up.shape)
    dflat, bt = d.ravel(), bins_t.ravel()
    order = np.argsort(bt, kind="stable")
    bt_sorted = bt[order]
    for b in np.unique(bins_s):
        lo, hi = np.searchsorted(bt_sorted, b), np.searchsorted(bt_sorted, b, side="right")
        pool = dflat[order[lo:hi]] if hi > lo else dflat
        sel = bins_s == b
        out[sel] = rng.choice(pool, size=int(sel.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ntarget", type=int, default=10)
    ap.add_argument("--nbatch", type=int, default=10)
    ap.add_argument("--levels", default="32,64,128,256")
    ap.add_argument("--niter", default="300,200,200,400")
    ap.add_argument("--warm0", type=int, default=100)
    ap.add_argument("--detail", choices=["white", "conditional"], default="conditional")
    ap.add_argument("--opt", default="", help="extra stl-synth2d options at levels > 0")
    ap.add_argument("--trace", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    levels = [int(v) for v in a.levels.split(",")]
    niter = [int(v) for v in a.niter.split(",")]
    rng = np.random.default_rng(a.seed)
    t = np.load(a.target)[: a.ntarget].astype(float)
    if t.ndim == 3:
        t = t[:, None]
    t = nyquist(t)
    tl = {n: (t if n == t.shape[-1] else resample(t, n)) for n in levels}
    report = {"levels": levels, "niter": niter, "warm0": a.warm0, "detail": a.detail, "opt": a.opt, "per_level": []}
    s = None
    for i, n in enumerate(levels):
        tp = os.path.join(a.out, f"target_{n}.npy")
        np.save(tp, tl[n])
        prefix = os.path.join(a.out, f"level_{n}")
        cmd = [a.bin, "--target", tp, "--ntarget", str(a.ntarget), "--nbatch", str(a.nbatch), "--mode", "ensemble",
               "--niter", str(niter[i]), "--trace", str(a.trace), "--out", prefix]
        if a.gpu:
            cmd.append("--gpu")
        if i == 0:
            if a.warm0 > 0:
                cmd += ["--warm", str(a.warm0)]
        else:
            s_up = resample(s, n)
            c_up = resample(tl[levels[i - 1]], n)
            init = s_up + detail_noise(s_up, tl[n], c_up, a.detail, rng)
            ip = prefix + "_init.npy"
            np.save(ip, standardize(init, nyquist(tl[n])))
            cmd += ["--init", ip] + shlex.split(a.opt)
        t0 = time.time()
        with open(prefix + ".log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        rep = json.load(open(prefix + ".json"))
        s = np.load(prefix + "_maps.npy")
        ev = rep["evaluations"]
        report["per_level"].append({"n": n, "evaluations": ev, "wall_seconds": time.time() - t0, "loss_first": rep["loss_first"],
                                    "loss_last": rep["loss_last"], "equivalent_evaluations": ev * (n / levels[-1]) ** 2})
        print(json.dumps(report["per_level"][-1]), flush=True)
    report["equivalent_evaluations"] = sum(p["equivalent_evaluations"] for p in report["per_level"])
    report["wall_seconds"] = sum(p["wall_seconds"] for p in report["per_level"])
    json.dump(report, open(os.path.join(a.out, "multiscale.json"), "w"), indent=1)
    print(json.dumps({k: report[k] for k in ["equivalent_evaluations", "wall_seconds"]}))


if __name__ == "__main__":
    main()
