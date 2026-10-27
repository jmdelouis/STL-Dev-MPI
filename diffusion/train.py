"""Train the conditional denoiser of one pyramid level on a single map.

    python diffusion/train.py TARGET.npy LEVEL OUT.pt [options]

LEVEL is the fine nside of the level: the model generates the Haar details
of `x_LEVEL` on the parent grid (nside LEVEL / 2) given `x_{LEVEL/2}`.
TARGET is a NESTED map at nside >= LEVEL; it is averaged down to LEVEL.

Training pairs are random crops of the face images of `x_LEVEL`, padded by
the neighbouring faces so that crops straddle face edges, with the 8
symmetries of the square (transposition = mirror of the sky about the face
meridian; flips): `crop` parent pixels (`2 crop` fine pixels), the coarse
map and the details are the 2 x 2 means and Haar details of the crop.
Faces 5 (equatorial) and 10 (south) are held out for validation unless
`--all-faces`.  Whitening: the coarse map by its standard deviation, the
details by the RMS of every Haar channel, over the whole level.
"""

import argparse
import copy
import json
import math
import time

import numpy as np
import torch

import healpix_faces as hf
from model import Denoiser, amp_dtype

VAL_FACES = (5, 10)


def haar_images(img):
    """Fine images `[B, 2K, 2K]` -> (means `[B, 1, K, K]`, details `[B, 3, K, K]`)."""
    c0, c1 = img[:, 0::2, 0::2], img[:, 1::2, 0::2]
    c2, c3 = img[:, 0::2, 1::2], img[:, 1::2, 1::2]
    mean = (c0 + c1 + c2 + c3) / 4
    h = torch.stack([(c1 + c3 - c0 - c2) / 2, (c2 + c3 - c0 - c1) / 2, (c0 + c3 - c1 - c2) / 2], dim=1)
    return mean[:, None], h


class Crops:
    def __init__(self, fine, crop, faces, cond_std, det_std, device):
        n = hf.nside_of(fine.size)
        self.k = min(crop, n // 2)
        self.pad = self.k
        padded = fine[hf.pad_nest(n, self.pad)]
        self.img = torch.tensor(padded[list(faces)], dtype=torch.float32, device=device)
        self.span = (n + 2 * self.pad - 2 * self.k) // 2
        self.cond_std = cond_std
        self.det_std = torch.tensor(det_std, dtype=torch.float32, device=device)[None, :, None, None]
        self.device = device

    def sample(self, b, gen):
        dev = self.device
        f = torch.randint(self.img.shape[0], (b,), device=dev, generator=gen)
        ox = 2 * torch.randint(self.span + 1, (b,), device=dev, generator=gen)
        oy = 2 * torch.randint(self.span + 1, (b,), device=dev, generator=gen)
        r = torch.arange(2 * self.k, device=dev)
        crop = self.img[f[:, None, None], (ox[:, None] + r)[:, :, None], (oy[:, None] + r)[:, None, :]]
        sym = torch.rand(3, b, 1, 1, device=dev, generator=gen) < 0.5
        crop = torch.where(sym[0], crop.transpose(1, 2), crop)
        crop = torch.where(sym[1], crop.flip(1), crop)
        crop = torch.where(sym[2], crop.flip(2), crop)
        mean, h = haar_images(crop)
        return mean / self.cond_std, h / self.det_std


def interior_mask(k, margin, device):
    m = torch.zeros(1, 1, k, k, device=device)
    m[..., margin:k - margin, margin:k - margin] = 1
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("level", type=int)
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--crop", type=int, default=64, help="crop size in parent pixels")
    p.add_argument("--channels", default="64,128,128")
    p.add_argument("--nres", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--all-faces", action="store_true")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.load(a.target).astype(np.float64)
    while hf.nside_of(x.size) > a.level:
        x = hf.down(x)
    assert hf.nside_of(x.size) == a.level, f"target nside below {a.level}"
    mean, h = hf.haar(x)
    cond_std = float(mean.std())
    det_std = np.sqrt(np.mean(h**2, axis=-1)).tolist()
    train_faces = range(12) if a.all_faces else [f for f in range(12) if f not in VAL_FACES]
    train = Crops(x, a.crop, train_faces, cond_std, det_std, dev)
    k = train.k
    unet = dict(channels=tuple(int(c) for c in a.channels.split(",")), nres=a.nres, dropout=a.dropout)
    assert k % 2 ** (len(unet["channels"]) - 1) == 0, f"crop {k} not divisible by the U-Net depth"
    model = Denoiser(**unet).to(dev)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    nparam = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
    amp = amp_dtype(dev)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp == torch.float16)
    mask = interior_mask(k, max(1, k // 8), dev)

    # Fixed evaluation batches: same crops, noise levels and noise at every evaluation.
    g = torch.Generator(device=dev).manual_seed(1234)
    evals = {"train": Crops(x, a.crop, train_faces, cond_std, det_std, dev)}
    if not a.all_faces:
        evals["val"] = Crops(x, a.crop, VAL_FACES, cond_std, det_std, dev)
    fixed = {}
    for name, c in evals.items():
        cond, det = c.sample(256, g)
        sigma = torch.exp(-1.2 + 1.2 * torch.randn(256, device=dev, generator=g))
        noise = torch.randn(det.shape, device=dev, generator=g)
        fixed[name] = (cond, det, sigma, noise)

    @torch.no_grad()
    def evaluate(net):
        out = {}
        for name, (cond, det, sigma, noise) in fixed.items():
            s = sigma.reshape(-1, 1, 1, 1)
            with torch.autocast(dev.type, dtype=amp):
                d = net(det + s * noise, sigma, cond)
            err = ((s**2 + 1) / s**2) * (d - det) ** 2
            out[name] = float((err * mask).sum() / (mask.sum() * det.shape[0] * det.shape[1]))
        return out

    print(f"level {a.level}: parent nside {a.level // 2}, crop {k}, {len(list(train_faces))} faces, "
          f"cond std {cond_std:.4f}, detail rms {np.round(det_std, 4).tolist()}, {nparam / 1e6:.2f} M parameters",
          flush=True)
    history = []
    best = (math.inf, 0, None)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    t0 = time.time()
    run = 0.0
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for grp in opt.param_groups:
            grp["lr"] = lr
        cond, det = train.sample(a.batch, gen)
        with torch.autocast(dev.type, dtype=amp):
            loss = model.loss(det, cond, mask)
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
            rec = dict(step=step, loss=run / (a.eval_every if step % a.eval_every == 0 else step % a.eval_every),
                       seconds=time.time() - t0, **{f"ema_{k_}": v for k_, v in ev.items()})
            history.append(rec)
            run = 0.0
            print(" ".join(f"{k_} {v:.4g}" if isinstance(v, float) else f"{k_} {v}" for k_, v in rec.items()),
                  flush=True)
            if "val" in ev and ev["val"] < best[0]:
                best = (ev["val"], step, copy.deepcopy(ema.state_dict()))
    # With held-out faces, keep the EMA weights of the lowest validation loss
    # (early stopping: the coarse levels over-fit their few thousand pixels).
    state, best_step = (best[2], best[1]) if best[2] is not None else (ema.state_dict(), a.steps)
    print(f"saved the EMA weights of step {best_step}", flush=True)
    torch.save(dict(level=a.level, unet=unet, state=state, best_step=best_step, cond_std=cond_std, det_std=det_std,
                    crop=k, train_faces=list(train_faces), history=history, args=vars(a)), a.out)
    with open(a.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(dict(level=a.level, unet=unet, parameters=nparam, best_step=best_step, cond_std=cond_std,
                       det_std=det_std, history=history, args=vars(a)), f, indent=1)


if __name__ == "__main__":
    main()
