"""Amortised multi-scale Lagrangian generator with local attention
(docs/lagrangian_synthesis.md, step 4), on STL-Dev's 2D loss (torch autograd).

    PYTHONPATH=<STL-Dev> python attention_generator.py --target Turb_6.npy --out DIR

Pipeline (levels 32 -> 64 -> 128 -> 256):
- level 32: pixel L-BFGS syntheses ("bases"), 3 batches for training, 1 held out;
- each 2x step: the coarse synthesis is upsampled (Fourier zero padding),
  a detail field is drawn conditionally on the bin of (coarse value, coarse
  gradient modulus), as in step 3, and the generator moves it:
  fine pixels are queries, the 5 x 5 coarse tokens around their parent are
  keys, the displacement is sum_k softmax(q.k / sqrt(d) + b_k) (pos_k - x)
  (a learnt, non-linear Zel'dovich step), plus an amplitude exp(a(x));
  x_fine = up(coarse) + exp(a) * warp(detail, displacement), bicubic warp.
- one generator shared by the three steps, trained with Adam on
  sum_levels log loss_level (STL-Dev's synthesize_from_maps loss at each level).
Evaluation on the held-out base: the untrained pipeline (zero displacement,
unit amplitude: step 3's conditional draw) and the trained one, each followed
by pixel L-BFGS at 256 (torch.optim.LBFGS, strong Wolfe, history 50).
Writes DIR/attention.json.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from STL_main import Synthesis
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch

DT = torch.float64


def nyquist_np(x):
    n = x.shape[-1]
    k = np.fft.fftfreq(n) * n
    m = (k[:, None] ** 2 + k[None, :] ** 2) <= (n / 2) ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(x) * m))


def resample(x, m):
    """Periodic Fourier resampling of a torch [..., n, n] to [..., m, m] (pixel means kept)."""
    n = x.shape[-1]
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    if m < n:
        a = (n - m) // 2
        Y = X[..., a:a + m, a:a + m].clone()
        Y[..., 0, :] = 0
        Y[..., :, 0] = 0
    else:
        a = (m - n) // 2
        Y = torch.zeros(x.shape[:-2] + (m, m), dtype=X.dtype, device=x.device)
        Y[..., a:a + n, a:a + n] = X
        Y[..., a, :] = 0
        Y[..., :, a] = 0
    return torch.real(torch.fft.ifft2(torch.fft.ifftshift(Y, dim=(-2, -1)))) * (m / n) ** 2


def grads(x):
    gr = (torch.roll(x, -1, -2) - torch.roll(x, 1, -2)) / 2
    gc = (torch.roll(x, -1, -1) - torch.roll(x, 1, -1)) / 2
    return gr, gc, torch.sqrt(gr * gr + gc * gc)


def capture_loss(target, nbatch):
    """STL-Dev's synthesis loss for the targets `target` [Bt, H, W] as a function of u [B, 1, H, W]."""
    rec = {}
    orig = Synthesis.optimize_lbfgs

    def hook(model, loss_fn, *a, **k):
        rec["m"], rec["f"] = model, loss_fn
        return model.u.detach()

    Synthesis.optimize_lbfgs = hook
    try:
        Synthesis.synthesize_from_maps(STL_2D_Kernel_Torch(target[:, None], pbc=True), nbatch=nbatch, pbc_running=True, max_iter=1, verbose=False)
    finally:
        Synthesis.optimize_lbfgs = orig
    m, f = rec["m"], rec["f"]

    def loss(u):
        m.u = u
        return f(m())

    return loss


def lbfgs(loss, u0, max_eval):
    u = u0.detach().clone().requires_grad_()
    opt = torch.optim.LBFGS([u], lr=1, max_iter=max_eval, max_eval=max_eval, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)
    hist = []

    def closure():
        opt.zero_grad()
        v = loss(u)
        v.backward()
        hist.append(v.item())
        return v

    opt.step(closure)
    return u.detach(), hist


class Detail:
    """Conditional draw of the detail of one level (step 3), in that level's standardised units."""

    def __init__(self, cu_t, d_t, q=8):
        self.q = q
        _, _, g = grads(cu_t)
        p = torch.linspace(0, 1, q + 1, dtype=DT, device=cu_t.device)[1:-1]
        self.e1, self.e2 = torch.quantile(cu_t.flatten(), p), torch.quantile(g.flatten(), p)
        b = self.bins(cu_t, g).flatten()
        order = torch.argsort(b, stable=True)
        self.pool = d_t.flatten()[order]
        counts = torch.bincount(b, minlength=q * q)
        self.start = torch.cumsum(counts, 0) - counts
        self.count = counts

    def bins(self, c, g):
        return torch.bucketize(c.contiguous(), self.e1) * self.q + torch.bucketize(g.contiguous(), self.e2)

    def draw(self, c, gen):
        _, _, g = grads(c)
        b = self.bins(c, g)
        cnt = self.count[b]
        r = torch.rand(c.shape, generator=gen, dtype=DT, device=c.device)
        idx = torch.where(cnt > 0, self.start[b] + (r * cnt).long(), (r * self.pool.numel()).long())
        return self.pool[idx.clamp(max=self.pool.numel() - 1)]


class AttentionDisplacement(nn.Module):
    def __init__(self, d=16, r=2, hidden=32):
        super().__init__()
        self.d, self.r, self.M = d, r, (2 * r + 1) ** 2
        conv = lambda i, o, k: nn.Conv2d(i, o, k, padding=k // 2, padding_mode="circular")
        self.q = nn.Sequential(conv(5, hidden, 3), nn.GELU(), conv(hidden, d, 1))
        self.k = nn.Sequential(conv(4, hidden, 3), nn.GELU(), conv(hidden, d, 1))
        self.a = nn.Sequential(conv(5, hidden, 3), nn.GELU(), conv(hidden, 1, 1))
        nn.init.zeros_(self.a[-1].weight)
        nn.init.zeros_(self.a[-1].bias)
        self.bias = nn.Parameter(torch.zeros(self.M))
        di = torch.arange(-r, r + 1, dtype=torch.float32)
        # Offsets pos_k - x in fine pixels, by parity a of the fine index: 2 di + 0.5 - a.
        off = torch.stack([2 * di + 0.5 - a for a in (0, 1)])  # [2, 2r+1]
        self.register_buffer("off", off)

    def forward(self, coarse, c_up, delta):
        B, _, n, _ = coarse.shape
        N = 2 * n
        gr, gc, gm = grads(c_up)
        q = self.q(torch.cat([c_up, gr, gc, gm, delta], 1))  # [B, d, N, N]
        cgr, cgc, cgm = grads(coarse)
        k = self.k(torch.cat([coarse, cgr, cgc, cgm], 1))  # [B, d, n, n]
        r = self.r
        kp = F.pad(k, (r, r, r, r), mode="circular")
        kn = F.unfold(kp, 2 * r + 1).view(B, self.d, self.M, n, n)
        kn = kn.repeat_interleave(2, -2).repeat_interleave(2, -1)  # [B, d, M, N, N]
        logits = (q.unsqueeze(2) * kn).sum(1) / math.sqrt(self.d) + self.bias[None, :, None, None]
        w = torch.softmax(logits, 1).view(B, 2 * r + 1, 2 * r + 1, N, N)
        par = torch.arange(N, device=coarse.device) % 2
        o = self.off.to(coarse.dtype)[par]  # [N, 2r+1]: offset by fine index parity and token index
        ur = torch.einsum("bijxy,xi->bxy", w, o)
        uc = torch.einsum("bijxy,yj->bxy", w, o)
        P = 4 * r + 4
        dp = F.pad(delta, (P, P, P, P), mode="circular")
        S = N + 2 * P
        rows = torch.arange(N, device=coarse.device, dtype=coarse.dtype)
        gy = (rows[None, :, None] + ur + P) * 2 / (S - 1) - 1
        gx = (rows[None, None, :] + uc + P) * 2 / (S - 1) - 1
        moved = F.grid_sample(dp, torch.stack([gx, gy], -1), mode="bicubic", align_corners=True)
        amp = torch.exp(self.a(torch.cat([c_up, gr, gc, gm, delta], 1)))
        return c_up + amp * moved, (ur, uc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--ntarget", type=int, default=10)
    ap.add_argument("--nbatch", type=int, default=10)
    ap.add_argument("--base-iter", type=int, default=300)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--refine", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = nyquist_np(np.load(a.target)[: a.ntarget].astype(np.float64))
    tt = torch.tensor(t, dtype=DT, device=dev)
    levels = [32, 64, 128, 256]
    tl = {n: (tt if n == 256 else resample(tt, n)) for n in levels}
    stats = {n: (tl[n].mean().item(), torch.sqrt(((tl[n] - tl[n].mean()) ** 2).mean()).item()) for n in levels}
    z = lambda x, n: (x - stats[n][0]) / stats[n][1]
    phys = lambda u, n: u * stats[n][1] + stats[n][0]
    losses = {n: capture_loss(tl[n].cpu().numpy(), a.nbatch) for n in levels}
    details = {}
    for i, n in enumerate(levels[1:]):
        cu = z(resample(tl[levels[i]], n), n)
        details[n] = Detail(cu, z(tl[n], n) - cu)
    rep = {"levels": levels, "steps": a.steps}
    # Bases at 32^2.
    gen = torch.Generator(device=dev).manual_seed(0)
    bases, base_evals = [], []
    for s in range(4):
        u0 = torch.randn((a.nbatch, 1, 32, 32), generator=gen, dtype=DT, device=dev)
        u, h = lbfgs(losses[32], u0, a.base_iter)
        bases.append(u)
        base_evals.append(len(h))
    rep["base_evals"] = base_evals
    model = AttentionDisplacement().to(dev, DT)

    def generate(base, g, train=True):
        x, out = base, {}
        for i, n in enumerate(levels[1:]):
            m = levels[i]
            cu = z(resample(phys(x, m), n), n)
            delta = details[n].draw(cu, g)
            x, _ = model(x, cu, delta) if train is not None else (cu + delta, None)
            out[n] = x
        return out

    def total(out):
        return sum(torch.log(losses[n](out[n])) for n in out), {n: losses[n](out[n]).item() for n in out}

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist, t0 = [], time.time()
    for step in range(a.steps):
        out = generate(bases[step % 3], gen)
        L, parts = total(out)
        opt.zero_grad()
        L.backward()
        opt.step()
        hist.append(parts)
        if step % 20 == 0:
            print(step, {k: round(v, 4) for k, v in parts.items()}, flush=True)
    rep["train_seconds"], rep["train_history"] = time.time() - t0, hist
    # Held-out base: untrained pipeline (step 3 draw) and trained generator, then pixel L-BFGS at 256.
    g_eval = torch.Generator(device=dev).manual_seed(123)
    with torch.no_grad():
        plain = generate(bases[3], g_eval, train=None)[256]
        g_eval = torch.Generator(device=dev).manual_seed(123)
        t1 = time.time()
        learnt = generate(bases[3], g_eval)[256]
        rep["forward_seconds"] = time.time() - t1
    for name, x in [("conditional_draw", plain), ("attention", learnt)]:
        t1 = time.time()
        _, h = lbfgs(losses[256], x, a.refine)
        rep[name] = {"forward_loss": h[0], "refine_history": h, "refine_seconds": time.time() - t1}
        print(name, "forward", h[0], "after", len(h), "evals", min(h), flush=True)
    rep["equivalent_base_evals"] = base_evals[3] * (32 / 256) ** 2
    json.dump(rep, open(f"{a.out}/attention.json", "w"))
    torch.save(model.state_dict(), f"{a.out}/attention.pt")


if __name__ == "__main__":
    main()
