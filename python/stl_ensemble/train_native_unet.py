#!/usr/bin/env python3
"""Train the Healpix-STL-IA U-Net with the native trainer (NativeUNet) on
the PR3 maps, frozen noise biases (the --real --bias frozen run of stlia
train_denoise.py), with Adam (cosine schedule) or L-BFGS by rounds.

    python train_native_unet.py --data DIR --out RUN --nside 256 --optimizer adam --steps 3000
    python train_native_unet.py --data DIR --out RUN --nside 64 --optimizer lbfgs --rounds 20 --lbfgs-iter 20

Data (as train_denoise.py): DIR/planck353_{Q,U}_n{nside}.npy, the half
missions planck353hm{1,2}_{f}_n{nside}.npy, planck857_I_n{nside}.npy, the
FFP10 noise planck353_noise{,hm1,hm2}_{f}_n{nside}_mc{k}.npy (--sims: the
simulations present for all three are used).

Adam: field f = fields[step % 2], biases refreshed every --bias-every steps
(per field, at the current output), zero on the coefficients without noise
bias unless --bias-all (train_denoise.py >= a2141c1); the first
--bias-warmup steps use live biases (--realizations fresh draws per step,
gradient through them), the frozen estimates start at step warmup + 1.
L-BFGS: round r trains field fields[r % 2] (or both with --lbfgs-fields
joint) for --lbfgs-iter iterations after a refresh of the biases (no cl, no
warmup). --seconds stops the training at that wall time.

Options with the names and meaning of train_denoise.py: --loss terms data,
l1, l2, l3 and cl (the binned half-mission cross spectrum C_b(m d1, m d2)
against C_b(m u, m u) under the apodised galactic masks --cl-masks, without
noise bias; stl_port.denoise over SpectrumEngine); --stat-masks (the
scattering statistics averaged under the galactic cuts |b| > b0, each
coefficient once per mask with its ergodic weight and its own frozen bias,
zero on the unbiased coefficients); --cond none, log857, sigma (the log of
the local noise rms over the training simulations, Q and U averaged) or
log857+sigma. --real is implied (the PR3 maps are always the data).

Evaluations (not counted in the training time): every --eval-every steps
or rounds, and every --eval-seconds of training if given, the loss terms of
both fields with the current biases, and with --eval-draws N the common
objective of the comparisons: both fields with biases from one fixed set
of N draws.

Writes, as train_denoise.py: RUN/train.json (args, j, coefficients,
parameters, noise_rms, loss_weights, seconds, loss_history, evals),
RUN/eval_{Q,U}.npz (sky = d, d, u) and RUN/model.pt (when torch is
importable; stlia.unet.load reads it), plus RUN/params.npy and
RUN/history.json (the native records, with the training times).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stl_port.native import NativeUNet  # noqa: E402
from stl_port.denoise import BinnedSpectra, SpectrumLoss, conditioning, stat_masks  # noqa: E402

TERMS = ("data", "res", "cross", "l1", "l2", "l3", "cl")
p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--nside", type=int, default=64)
p.add_argument("--j", type=int, default=None)
p.add_argument("--levels", type=int, default=None, help="default: down to nside 8")
p.add_argument("--channels", default=None)
p.add_argument("--gauge", default="phi", choices=["phi", "cosmo"])
p.add_argument("--ksz", type=int, default=3)
p.add_argument("--cond", default="none", choices=["none", "log857", "sigma", "log857+sigma"])
p.add_argument("--loss", default="data=1,l1=1,l2=1,l3=1")
p.add_argument("--real", action="store_true", help="accepted for train_denoise.py compatibility (always the PR3 maps)")
p.add_argument("--bias", default="frozen", choices=["frozen"])
p.add_argument("--bias-warmup", type=int, default=0, help="first Adam steps with live biases")
p.add_argument("--realizations", type=int, default=4, help="live biases: noise draws per step (B)")
p.add_argument("--stat-masks", default=None, help="galactic cuts b0 (deg), comma-separated: statistics under |b| > b0")
p.add_argument("--cl-masks", default="0,30,45", help="cl: galactic cuts b0 (deg) of the apodised masks")
p.add_argument("--cl-lmin", type=int, default=10)
p.add_argument("--cl-lmax", type=int, default=None, help="default 2 nside - 1")
p.add_argument("--cl-width", type=int, default=30)
p.add_argument("--cl-apod", type=float, default=5.0, help="FWHM (deg) of the mask apodisation")
p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
p.add_argument("--steps", type=int, default=3000, help="Adam steps (cosine schedule length)")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bias-every", type=int, default=50)
p.add_argument("--bias-realizations", type=int, default=40)
p.add_argument("--bias-all", action="store_true", help="estimate the bias of every coefficient (train_denoise.py --bias-all)")
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--lbfgs-iter", type=int, default=20)
p.add_argument("--lbfgs-history", type=int, default=10)
p.add_argument("--lbfgs-max-eval", type=int, default=None)
p.add_argument("--lbfgs-fields", choices=["alternate", "joint"], default="alternate")
p.add_argument("--sims", default="10-49")
p.add_argument("--seconds", type=float, default=None, help="wall-time budget of the training")
p.add_argument("--log-every", type=int, default=10)
p.add_argument("--eval-every", type=int, default=250, help="steps (Adam) or rounds (L-BFGS) between evaluations")
p.add_argument("--eval-seconds", type=float, default=None, help="also evaluate every this much training time")
p.add_argument("--eval-draws", type=int, default=0, help="draws of the fixed-bias common objective (0: not computed)")
p.add_argument("--engine-cache-bytes", type=int, default=None, help="GPU geometry cache (0: from the free memory)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", type=int, default=0)
p.add_argument("--backend", default="gpu")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
n = a.nside
load = lambda name: np.load(os.path.join(a.data, f"{name}_n{n}.npy")).astype(np.float64)
lo, _, hi = a.sims.partition("-")
sims = [k for k in range(int(lo), int(hi or lo) + 1)
        if all(os.path.exists(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy"))
               for kd in ("noise", "noisehm1", "noisehm2") for f in ("Q", "U"))]
assert sims, "no simulation with full- and half-mission noise"
levels = a.levels or int(math.log2(n // 8)) + 1
channels = [int(c) for c in a.channels.split(",")] if a.channels else [32, 64, 96, 128, 128, 128, 128][:levels]
j = a.j or int(math.log2(n)) - 1
weights = {}
for item in a.loss.split(","):
    k, _, v = item.partition("=")
    assert k in ("data", "l1", "l2", "l3", "cl"), f"loss term {k}: data, l1, l2, l3 or cl in the native trainer"
    weights[k] = float(v or 1)
w_cl = weights.get("cl", 0.0)
assert a.optimizer == "adam" or (not w_cl and not a.bias_warmup), "L-BFGS: no cl term, no bias warmup"
engine = {} if a.engine_cache_bytes is None else {"cache_bytes": a.engine_cache_bytes}
kinds = [] if a.cond == "none" else a.cond.split("+")
net = NativeUNet(n, channels, ksz=a.ksz, gauge=a.gauge, cond=len(kinds), j=j, loss=weights, backend=a.backend,
                 device=a.device, seed=a.seed, engine=engine, bias_all=a.bias_all)
if a.stat_masks:
    net.set_masks(stat_masks(n, [float(b) for b in a.stat_masks.split(",")]))
i857 = load("planck857_I")
net.set_common(i857)
fields = ("Q", "U")
obs, rms, noise_var, cl_loss = {}, {}, [], {}
spectra = BinnedSpectra(n, masks=[float(b) for b in a.cl_masks.split(",")], lmin=a.cl_lmin, lmax=a.cl_lmax,
                        width=a.cl_width, apod=a.cl_apod, backend=a.backend, device=a.device) if w_cl else None
for f in fields:
    pools = [np.stack([np.load(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy")) for k in sims])
             for kd in ("noise", "noisehm1", "noisehm2")]
    obs[f] = load(f"planck353_{f}")
    rms[f] = float(np.sqrt(np.mean(pools[0] ** 2)))
    noise_var.append(np.mean(np.square(pools[0].astype(np.float64)), 0))
    d1, d2 = load(f"planck353hm1_{f}"), load(f"planck353hm2_{f}")
    net.set_field(f, obs[f], d1, d2, *pools)
    if w_cl:
        cl_loss[f] = SpectrumLoss(spectra, d1, d2)
    del pools
if kinds:
    net.set_cond(conditioning(kinds, i857, noise_var))
rng = np.random.default_rng(a.seed + 777)
fixed_draws = (np.arange(a.eval_draws) % len(sims), np.where(np.arange(a.eval_draws) % 2, -1.0, 1.0))
print(f"nside {n} J {j} levels {levels} channels {channels} params {net.size} sims {len(sims)} loss {weights} "
      f"optimizer {a.optimizer} bias {'all' if a.bias_all else 'biased coefficients only'} warmup {a.bias_warmup} "
      f"cond {a.cond} stat masks {a.stat_masks or 'none'}"
      + (f" cl masks {a.cl_masks}, {len(spectra.edges) - 1} bins" if w_cl else ""), flush=True)


def draws(k=None):
    """(sims, signs): k draws with replacement (bias estimates), or, as the
    live draws of train_denoise.py, k distinct simulations."""
    if k is None:
        return rng.integers(0, len(sims), a.bias_realizations), rng.choice([-1.0, 1.0], a.bias_realizations)
    return rng.permutation(len(sims))[:k], rng.choice([-1.0, 1.0], k)


def add_cl(f, terms, backward):
    """Adds the cl term of field f to `terms`; with backward, its parameter cotangents."""
    if not w_cl:
        return terms
    u = net.apply(obs[f])
    if backward:
        v, g = cl_loss[f](u, grad=True)
        net.apply(obs[f], cot=w_cl * g)
    else:
        v = cl_loss[f](u)
    terms["cl"] = v
    terms["total"] += w_cl * v
    return terms


hist = {"step": [], "loss": [], **{t: [] for t in TERMS}}
native = {"eval": [], "steps": []}
evals = {}
clock = {"train": 0.0, "next_eval": a.eval_seconds if a.eval_seconds else float("inf")}
started = time.time()


def evaluate(step, save=False):
    """Terms of both fields with their current biases; the fixed-draw objective with --eval-draws."""
    out = {}
    for f in fields:
        u = net.apply(obs[f])
        out[f] = {"pixel_rms_error_over_noise": float(np.sqrt(np.mean((u - obs[f]) ** 2))) / rms[f],
                  "loss_terms": {k: v for k, v in net.loss(f).items() if k != "total"}}
        if w_cl:
            out[f]["loss_terms"]["cl"] = cl_loss[f](u)
        out[f]["data_loss"] = out[f]["loss_terms"].get("data", float("nan"))
        if save:
            np.savez(os.path.join(a.out, f"eval_{f}.npz"), sky=obs[f], d=obs[f], u=u)
    rec = {"step": step, "train_seconds": clock["train"],
           **{f: sum(out[f]["loss_terms"][k] * weights.get(k, 0.0) for k in out[f]["loss_terms"]) for f in fields}}
    if a.eval_draws:
        saved = {}
        for f in fields:
            net.estimate_bias(f, *fixed_draws)
            saved[f] = net.loss(f)["total"] + (w_cl * cl_loss[f](net.apply(obs[f])) if w_cl else 0.0)
        rec["fixed_draws"] = {**saved, "total": sum(saved.values())}
        for f in fields:  # back to training biases at the next refresh
            net.estimate_bias(f, *draws())
    native["eval"].append(rec)
    evals[step] = out
    print(f"eval {step} t={clock['train']:.1f}s " + " ".join(
        f"{f} err/noise {out[f]['pixel_rms_error_over_noise']:.3f} loss {rec[f]:.5g}" for f in fields)
        + (f" | fixed draws {rec['fixed_draws']['total']:.5g}" if a.eval_draws else ""), flush=True)
    np.save(os.path.join(a.out, "params.npy"), net.get_params())


def log(step, terms):
    hist["step"].append(step)
    hist["loss"].append(terms["total"])
    for t in TERMS:
        hist[t].append(terms.get(t) if t in weights else None)


def out_of_time():
    return a.seconds is not None and clock["train"] >= a.seconds


def timed_eval(step):
    if clock["train"] >= clock["next_eval"]:
        evaluate(step)
        clock["next_eval"] = clock["train"] + a.eval_seconds


for f in fields:  # biases of the untrained network, for the first evaluation
    net.estimate_bias(f, *draws())
evaluate(0)
last = 0
if a.optimizer == "adam":
    for step in range(1, a.steps + 1):
        f = fields[step % 2]
        t0 = time.perf_counter()
        live = step <= a.bias_warmup
        if not live and (step - 1 - a.bias_warmup) % a.bias_every < len(fields):
            net.estimate_bias(f, *draws())
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps))
        if live or w_cl:
            net.zero_grad()
            if live:
                terms = net.live(f, draws(a.realizations), draws(a.realizations), backward=True)
            else:
                terms = net.loss(f, backward=True)
            add_cl(f, terms, True)
            net.adam_step(lr)
        else:
            terms = net.step(f, lr)
        clock["train"] += time.perf_counter() - t0
        native["steps"].append({"step": step, "field": f, "seconds": clock["train"], "live": live, **terms})
        last = step
        if step % a.log_every == 0 or step == 1:
            log(step, terms)
            print(f"step {step} {f} loss {terms['total']:.6g} " + " ".join(f"{k} {terms[k]:.4g}" for k in weights)
                  + f" {clock['train'] / step:.3f} s/step", flush=True)
        if step % a.eval_every == 0 and step < a.steps:
            evaluate(step)
        timed_eval(step)
        if out_of_time():
            break
else:
    opts = {"max_iter": a.lbfgs_iter, "history_size": a.lbfgs_history}
    if a.lbfgs_max_eval:
        opts["max_eval"] = a.lbfgs_max_eval
    for r in range(1, a.rounds + 1):
        names = list(fields) if a.lbfgs_fields == "joint" else [fields[r % 2]]
        t0 = time.perf_counter()
        for f in names:
            net.estimate_bias(f, *draws())
        rep = net.lbfgs(names, opts)
        clock["train"] += time.perf_counter() - t0
        h = rep["loss_history"]
        native["steps"].append({"round": r, "fields": names, "seconds": clock["train"], "evaluations": rep["evaluations"],
                                "loss_start": h[0], "loss_end": min(h)})
        last = r
        log(r, dict(zip(rep["terms"], rep["terms_history"][int(np.argmin(h))])))
        print(f"round {r} {names} {h[0]:.6g} -> {min(h):.6g} in {rep['evaluations']} evaluations, "
              f"t={clock['train']:.1f}s", flush=True)
        if r % a.eval_every == 0 and r < a.rounds:
            evaluate(r)
        timed_eval(r)
        if out_of_time():
            break
evaluate(last, save=True)
cfg = net.info()["config"]
with open(os.path.join(a.out, "train.json"), "w") as fh:
    json.dump({"args": vars(a), "j": j, "coefficients": net.info()["coefficients"], "parameters": net.size,
               "noise_rms": rms, "loss_weights": weights, "seconds": time.time() - started,
               "train_seconds": clock["train"], "loss_history": hist, "evals": evals,
               "implementation": "STL-Dev-MPI native U-Net (NativeUNet)", "library": net.info()["version"]}, fh, indent=1)
with open(os.path.join(a.out, "history.json"), "w") as fh:
    json.dump({"args": vars(a), "params": net.size, "sims": sims, **native}, fh, indent=1)
try:
    import torch
    state = {k: torch.as_tensor(v) for k, v in net.state_dict().items()}
    torch.save({"state": state, "nside": n, "levels": levels, "channels": channels, "conv": "sphere", "gauge": a.gauge,
                "ksz": a.ksz, "norm": "group", "scale": None, "cond": len(kinds), "j": j, "noise_rms": rms},
               os.path.join(a.out, "model.pt"))
except ImportError:
    print("torch not importable: model.pt not written (params.npy and state names in train.json)", flush=True)
