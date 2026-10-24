"""Sample full-sky NESTED maps from coarse to fine with the per-level models.

    python diffusion/sample.py TARGET.npy MODEL_DIR NSIDE OUT.npy [options]

The coarsest map (nside `--coarse-nside`) is a Gaussian random field with
the angular power of the target (`--coarse grf`: healpy synfast at 8 times
that nside, then averaged down; monopole set to the target's mean), the
target's own coarse map (`--coarse target`: conditional super-resolution),
or any NESTED map averaged down (`--coarse FILE`, e.g. an STL synthesis:
STL for the large scales, diffusion for the finest levels; a stack
`[S, npix]` gives one sample per map, e.g. the members of an STL ensemble).  Every finer
level `n` draws its Haar details with the model
`MODEL_DIR/level{n}.pt` on the 12 face images with HEALPix padding (EDM
Heun sampler), and rebuilds `x_n` from `x_{n/2}`.  OUT.npy is `[S, npix]`.
"""

import argparse
import os
import time

import healpy as hp
import numpy as np
import torch

import healpix_faces as hf
from model import Denoiser, amp_dtype, heun_sample


def coarse_grf(target, nside, rng, oversample=8):
    """GRF with the target's power at nside `oversample * nside`, averaged
    down to `nside`: the NESTED means then carry the pixel window and the
    power above `3 nside` that a GRF drawn at `nside` would miss (13 % of the
    pixel variance at nside 8)."""
    x = target
    hi = min(nside * oversample, hf.nside_of(target.size))
    while hf.nside_of(x.size) > hi:
        x = hf.down(x)
    cl = hp.anafast(hp.reorder(x, n2r=True), lmax=3 * hi - 1)
    cl[0] = 0
    np.random.seed(int(rng.integers(2**31)))
    m = hp.reorder(hp.synfast(cl, hi, lmax=3 * hi - 1, new=True), r2n=True) + x.mean()
    while hf.nside_of(m.size) > nside:
        m = hf.down(m)
    return m


def load_level(models, n, dev):
    ck = torch.load(os.path.join(models, f"level{n}.pt"), map_location=dev, weights_only=False)
    model = Denoiser(**ck["unet"]).to(dev).eval()
    model.load_state_dict(ck["state"])
    model.net.set_padding("healpix")
    return model, ck


def sample_details(model, ck, x, steps, gen, churn=0.0):
    """Haar details `[3, npix(x)]` of the level above the NESTED map `x`, unwhitened."""
    dev = next(model.parameters()).device
    cond = torch.tensor(hf.to_faces(x) / ck["cond_std"], dtype=torch.float32, device=dev)[:, None]
    with torch.autocast(dev.type, dtype=amp_dtype(dev)):
        h = heun_sample(model, cond, steps=steps, generator=gen, s_churn=churn)
    h = h.float().cpu().numpy().transpose(1, 0, 2, 3)  # [3, 12, n, n]
    return hf.from_faces(h) * np.array(ck["det_std"])[:, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("models")
    p.add_argument("nside", type=int)
    p.add_argument("out")
    p.add_argument("--coarse-nside", type=int, default=8)
    p.add_argument("--coarse", default="grf", help="grf, target, or a NESTED map file (e.g. an STL synthesis)")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--churn", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-levels", action="store_true", help="also save every intermediate level")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    target = np.load(a.target).astype(np.float64)
    if a.coarse == "grf":
        xs = np.array([coarse_grf(target, a.coarse_nside, rng) for _ in range(a.samples)])
    else:
        # One map (repeated --samples times) or a stack [S, npix] (one sample per map).
        x0 = target if a.coarse == "target" else np.load(a.coarse).astype(np.float64)
        while hf.nside_of(x0.shape[-1]) > a.coarse_nside:
            x0 = hf.down(x0)
        xs = x0 if x0.ndim == 2 else np.repeat(x0[None], a.samples, axis=0)
    levels = {a.coarse_nside: xs}
    n = a.coarse_nside
    while n < a.nside:
        n *= 2
        model, ck = load_level(a.models, n, dev)
        t0 = time.time()
        xs = np.array([hf.unhaar(x, sample_details(model, ck, x, a.steps, gen, a.churn)) for x in xs])
        levels[n] = xs
        print(f"level {n}: {time.time() - t0:.1f} s for {len(xs)} samples, std {xs.std():.4f}", flush=True)
    np.save(a.out, xs)
    if a.keep_levels:
        np.savez(a.out.rsplit(".", 1)[0] + "_levels.npz", **{f"nside{k}": v for k, v in levels.items()})


if __name__ == "__main__":
    main()
