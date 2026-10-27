"""Training and sampling of the sphere-native denoiser of one pyramid level
(`sphere_conv.SphereDenoiser`).

    python diffusion/sphere_diffusion.py TARGET.npy LEVEL OUT.pt [options]

The level's variable is the scalar detail field `d = x_n - up(x_{n/2})` at
the fine nside `n` (zero mean over every group of 4 siblings), whitened by
its standard deviation; the condition is the coarse map seen at the fine
level, whitened by its own.  Training batches are regions (a central cell
of `cell x cell` fine pixels and `rings` rings of `sub x sub` sub-cells
around it); the loss is on the central cells.  Augmentation: the 4 exact
symmetries of the HEALPix grid that a north-oriented stencil does not
already average over (north-south mirror, east-west mirror).  Cells of
faces 5 and 10 are held out for validation unless `--all-faces`.
Sampling (`sample_level`) covers the sky with the regions of all the cells
and runs the EDM Heun sampler on the full-sky detail field.
"""

import argparse
import copy
import json
import math
import time

import healpy as hp
import numpy as np
import torch

import healpix_faces as hf
import sphere_conv as sc
from model import amp_dtype

VAL_FACES = (5, 10)


def parity_perms(nside):
    """NESTED permutations of the north-south and east-west mirrors (exact on the grid)."""
    npix = 12 * nside * nside
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    ns = hp.ang2pix(nside, np.pi - theta, phi, nest=True)
    ew = hp.ang2pix(nside, theta, (-phi) % (2 * np.pi), nest=True)
    assert np.array_equal(np.sort(ns), np.arange(npix)) and np.array_equal(np.sort(ew), np.arange(npix))
    return [np.arange(npix), ns, ew, ew[ns]]


def level_fields(target, n):
    x = np.load(target).astype(np.float64)
    while hf.nside_of(x.size) > n:
        x = hf.down(x)
    parent = hf.down(x)
    d = x - hf.up(parent)
    return d, hf.up(parent), float(d.std()), float(parent.std())


def cell_regions(cells, cell_nside, n, sub, rings, depth, ksz, device):
    return [sc.RegionGeometry(ids, central, n, depth, ksz, device)
            for ids, central in sc.region_ids(cells, cell_nside, n, sub=sub, rings=rings)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("level", type=int)
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--regions", type=int, default=8, help="regions per batch")
    p.add_argument("--cell", type=int, default=64, help="central cell side, fine pixels")
    p.add_argument("--sub", type=int, default=8, help="sub-cells per cell side (halo granularity)")
    p.add_argument("--rings", type=int, default=2, help="rings of sub-cells in the halo")
    p.add_argument("--channels", default="64,128,128")
    p.add_argument("--nres", type=int, default=2)
    p.add_argument("--ksz", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--all-faces", action="store_true")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = a.level
    d, cond, det_std, cond_std = level_fields(a.target, n)
    perms = [torch.as_tensor(q, device=dev) for q in parity_perms(n)]
    d_t = torch.tensor(d / det_std, dtype=torch.float32, device=dev)
    c_t = torch.tensor(cond / cond_std, dtype=torch.float32, device=dev)
    unet = dict(channels=tuple(int(c) for c in a.channels.split(",")), nres=a.nres, ksz=a.ksz, dropout=a.dropout)
    depth = len(unet["channels"]) - 1
    cell_nside = n // a.cell
    ncell = 12 * cell_nside**2
    face = np.arange(ncell) // cell_nside**2
    val = np.isin(face, VAL_FACES) & (not a.all_faces)
    t0 = time.time()
    geo = cell_regions(np.arange(ncell), cell_nside, n, a.sub, a.rings, depth, a.ksz, dev)
    print(f"level {n}: {ncell} cells of {a.cell}^2 fine pixels, regions of {len(geo[0].ids[0])} pixels, "
          f"geometry {time.time() - t0:.1f} s; detail std {det_std:.4f}, cond std {cond_std:.4f}", flush=True)
    train_cells, val_cells = np.flatnonzero(~val), np.flatnonzero(val)

    model = sc.SphereDenoiser(**unet).to(dev)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    nparam = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    amp = amp_dtype(dev)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp == torch.float16)

    def batch(cells, perm):
        pyr = sc.Pyramid([geo[c] for c in cells])
        ids = pyr.levels[0].ids
        return pyr, d_t[perm][ids][:, None], c_t[perm][ids][:, None]

    g = torch.Generator(device=dev).manual_seed(1234)
    fixed = {}
    for name, cells in (("train", train_cells), ("val", val_cells)):
        if len(cells) == 0:
            continue
        pick = np.random.default_rng(7).choice(cells, size=min(2 * a.regions, len(cells)), replace=False)
        pyr, dd, cc = batch(pick, perms[0])
        sigma = torch.exp(-1.2 + 1.2 * torch.randn(len(pick), device=dev, generator=g))
        noise = sc.project(torch.randn(dd.shape, device=dev, generator=g)) * math.sqrt(4 / 3)
        fixed[name] = (pyr, dd, cc, sigma, noise)

    @torch.no_grad()
    def evaluate(net):
        out = {}
        for name, (pyr, dd, cc, sigma, noise) in fixed.items():
            s = sigma[pyr.levels[0].region][:, None]
            with torch.autocast(dev.type, dtype=amp):
                den = net(dd + s * noise, sigma, cc, pyr)
            err = (s**2 + 1) / s**2 * (den - dd) ** 2
            out[name] = float(err[pyr.central].mean())
        return out

    print(f"{nparam / 1e6:.2f} M parameters, {len(train_cells)} training cells", flush=True)
    history, best, run, t0 = [], (math.inf, 0, None), 0.0, time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for grp in opt.param_groups:
            grp["lr"] = lr
        cells = rng.choice(train_cells, size=a.regions, replace=False)
        pyr, dd, cc = batch(cells, perms[rng.integers(4)])
        with torch.autocast(dev.type, dtype=amp):
            loss = model.loss(dd, cc, pyr, pyr.central)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        decay = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for qe, q in zip(ema.parameters(), model.parameters()):
                qe.lerp_(q, 1 - decay)
        run += float(loss.detach())
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(ema)
            k = a.eval_every if step % a.eval_every == 0 else step % a.eval_every
            rec = dict(step=step, loss=run / k, seconds=time.time() - t0, **{f"ema_{n_}": v for n_, v in ev.items()})
            history.append(rec)
            run = 0.0
            print(" ".join(f"{k_} {v:.4g}" if isinstance(v, float) else f"{k_} {v}" for k_, v in rec.items()), flush=True)
            if "val" in ev and ev["val"] < best[0]:
                best = (ev["val"], step, copy.deepcopy(ema.state_dict()))
    state, best_step = (best[2], best[1]) if best[2] is not None else (ema.state_dict(), a.steps)
    print(f"saved the EMA weights of step {best_step}", flush=True)
    meta = dict(kind="sphere", level=n, unet=unet, cell=a.cell, sub=a.sub, rings=a.rings, det_std=det_std,
                cond_std=cond_std, best_step=best_step, history=history, args=vars(a))
    torch.save(dict(state=state, **meta), a.out)
    with open(a.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(dict(parameters=nparam, **meta), f, indent=1)


def load_sphere(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = sc.SphereDenoiser(**ck["unet"]).to(dev).eval()
    model.load_state_dict(ck["state"])
    return model, ck


@torch.no_grad()
def sample_level(model, ck, parent, steps=32, gen=None, regions=8, sigma_max=80.0, sigma_min=0.002, rho=7.0):
    """The map at nside `2 nside(parent)` drawn given the NESTED map `parent`
    (EDM Heun sampler on the full-sky detail field, region by region)."""
    dev = next(model.parameters()).device
    n = 2 * hf.nside_of(parent.size)
    depth = len(ck["unet"]["channels"]) - 1
    cell_nside = n // ck["cell"]
    ncell = 12 * cell_nside**2
    geo = cell_regions(np.arange(ncell), cell_nside, n, ck["sub"], ck["rings"], depth, ck["unet"]["ksz"], dev)
    batches = [sc.Pyramid([geo[c] for c in range(i, min(i + regions, ncell))]) for i in range(0, ncell, regions)]
    cond = torch.tensor(hf.up(parent) / ck["cond_std"], dtype=torch.float32, device=dev)
    amp = amp_dtype(dev)

    def denoise(x, s):
        out = torch.empty_like(x)
        for pyr in batches:
            ids = pyr.levels[0].ids
            sig = torch.full((len(pyr.regions),), s, device=dev)
            with torch.autocast(dev.type, dtype=amp):
                den = model(x[ids][:, None], sig, cond[ids][:, None], pyr)[:, 0].float()
            out[ids[pyr.central]] = den[pyr.central]
        return out

    i = torch.arange(steps, dtype=torch.float64)
    t = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros(1, dtype=torch.float64)]).tolist()
    x = sc.project(torch.randn(cond.shape, device=dev, generator=gen)) * math.sqrt(4 / 3) * t[0]
    for k in range(steps):
        tc, tn = t[k], t[k + 1]
        dx = (x - denoise(x, tc)) / tc
        xn = x + (tn - tc) * dx
        if tn > 0:
            dn = (xn - denoise(xn, tn)) / tn
            xn = x + (tn - tc) * 0.5 * (dx + dn)
        x = xn
    return hf.up(parent) + x.cpu().numpy().astype(np.float64) * ck["det_std"]


if __name__ == "__main__":
    main()
