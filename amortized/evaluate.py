"""Statistics of a population of maps against the target, as in the
`stl-synth` report, plus diversity checks.

    python evaluate.py --target T.npy --nside 64 --out DIR --model RUN/model.pt [--count 1000]
    python evaluate.py --target T.npy --nside 64 --out DIR --maps 'RUN/ens_m*_synthesis_nested.npy'

With `--model`, draws `--count` maps from the generator (timed) and writes
the first `--save` of them as DIR/synthesis_mMMM_nested.npy (target units,
the names `tools/cv10.py` reads).  With `--maps`, evaluates existing maps
(e.g. an L-BFGS ensemble).  Writes DIR/eval.json: per family the residual
of the population mean, the per-map residual and scatter in ergodic sigma,
the scatter over the ergodic variance; the correlation between maps and
with the target, on the map and on its nside-8 means.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python", "stl_ensemble"))
from generator import Generator  # noqa: E402
from stl_torch import EnsembleLoss  # noqa: E402
from stl_port.native import NativeEngine  # noqa: E402
from train import load_target, target_statistics  # noqa: E402


def degrade(m, nside):
    while m.shape[-1] > 12 * nside * nside:
        m = m.reshape(m.shape[:-1] + (-1, 4)).mean(-1)
    return m


def correlations(maps, target, nside_low=8):
    out = {}
    for name, f in (("map", lambda m: m), (f"nside{nside_low}", lambda m: degrade(m, nside_low))):
        a = f(maps)
        a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
        t = f(target[None])[0]
        t = (t - t.mean()) / t.std()
        c = a @ a.T / a.shape[1]
        off = c[~np.eye(len(a), dtype=bool)]
        ct = a @ t / a.shape[1]
        out[name] = {"pairwise_abs_mean": float(np.abs(off).mean()), "pairwise_abs_max": float(np.abs(off).max()),
                     "with_target_abs_mean": float(np.abs(ct).mean()), "with_target_abs_max": float(np.abs(ct).max())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--nside", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--maps", default=None)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--save", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(f"cuda:{a.device}")
    raw = load_target(a.target, a.nside)
    mu, sd = raw.mean(), raw.std()
    target = (raw - mu) / sd
    ck = torch.load(a.model, map_location=dev) if a.model else None
    j = ck["j"] if ck else int(math.log2(a.nside)) - 1
    tstats, tvar, plan = target_statistics(target, a.nside, j, a.device)
    report_fn = EnsembleLoss(tstats, tvar, plan, a.nside, 1, 0.0)
    engine = NativeEngine(a.nside, j, batch=a.batch, backend="gpu", device=a.device)

    gen_seconds = None
    if ck:
        gen = Generator(a.nside, ck["nside0"], ck["channels"]).to(dev).eval()
        gen.load_state_dict(ck["state"])
        g = torch.Generator(device=dev).manual_seed(a.seed)
        maps = []
        torch.cuda.synchronize(dev)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(math.ceil(a.count / a.batch)):
                maps.append(gen(gen.noise(a.batch, generator=g, device=dev)).double().cpu().numpy())
        torch.cuda.synchronize(dev)
        gen_seconds = time.time() - t0
        maps = np.concatenate(maps)[: a.count]
    else:
        files = sorted(glob.glob(a.maps))
        maps = np.stack([(degrade(np.load(f).astype(np.float64), a.nside) - mu) / sd for f in files])
    m = len(maps)

    stats = []
    t0 = time.time()
    for k in range(0, m, a.batch):
        chunk = maps[k:k + a.batch]
        if len(chunk) < a.batch:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], a.batch - len(chunk), 0)])
        v, _ = engine.forward(chunk)
        stats.append(v.view(np.float64).reshape(v.shape + (2,)))
    stats_seconds = time.time() - t0
    stats = np.concatenate(stats)[:m]
    np.save(os.path.join(a.out, "stats.npy"), stats)
    for k in range(min(a.save, m)):
        np.save(os.path.join(a.out, f"synthesis_m{k:03d}_nested.npy"), maps[k] * sd + mu)
    sub = maps[: min(m, 200)]
    report = {"maps": m, "nside": a.nside, "j": j, "source": a.model or a.maps,
              "generation_seconds_per_map": gen_seconds / m if gen_seconds else None,
              "statistics_seconds_per_map": stats_seconds / m,
              "families": report_fn.report(stats),
              "map_moments": {"std_mean": float(maps.std(1).mean()), "std_scatter": float(maps.std(1).std()),
                              "skew_mean": float((((maps - maps.mean(1, keepdims=True)) / maps.std(1, keepdims=True)) ** 3).mean()),
                              "target_skew": float((target ** 3).mean())},
              "correlations": correlations(sub, target)}
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
