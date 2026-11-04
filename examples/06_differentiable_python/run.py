#!/usr/bin/env python3
"""Losses made of differentiable STL statistics and pseudo-spectra, optimised
with torch.optim.LBFGS on a map at nside 32 (the Takahashi et al. 2017
weak-lensing target of the other examples, degraded from nside 64).

    python run.py OUT TARGET_NSIDE64.npy FIGURE.png

0. gradient check: torch's gradient of each wrapper against a central
   finite difference along a random direction;
1. masked synthesis: white noise -> scattering statistics averaged over
   |b| > 20 deg (NativeEngine with a mask) and the pseudo-C_l on the same
   mask (SpectrumEngine);
2. patch synthesis: the statistics of 6 discs (R 30 deg, core 20 deg) of the
   target imposed on the same discs (PatchEngine);
3. denoising with cross statistics (Delouis et al. 2022): d = s + n, the map
   u such that <S(u + n_b, u + n_b)> = S(d, d) and <S(u + n_b, u)> = S(d, u)
   over noise draws n_b (cross statistics; PatchEngine with full-sky discs,
   which spreads the batch over CPU threads).
"""
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python" / "stl_ensemble"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from stl_port.native import NativeEngine, PatchEngine, SpectrumEngine
from stl_torch import discs, patches, scattering, spectrum

out, target_path, figure = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
NSIDE, J = 32, 4
NPIX = 12 * NSIDE**2
rng = np.random.default_rng(0)
torch.manual_seed(0)

t = np.load(target_path).astype(np.float64).reshape(-1, 4).mean(1)     # nside 64 -> 32 (NESTED children)
t = (t - t.mean()) / t.std()
target = torch.from_numpy(t)
report = {"nside": NSIDE, "J": J}


def lbfgs(x, loss_fn, max_eval, label):
    """torch LBFGS with strong Wolfe line search; returns the loss history."""
    opt = torch.optim.LBFGS([x], max_iter=max_eval, max_eval=max_eval, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=0, tolerance_change=0)
    history = []

    def closure():
        opt.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        history.append(loss.item())
        return loss

    t0 = time.perf_counter()
    opt.step(closure)
    dt = time.perf_counter() - t0
    print(f"{label}: {len(history)} evaluations in {dt:.1f} s ({dt / len(history):.2f} s each), "
          f"loss {history[0]:.4g} -> {history[-1]:.4g}", flush=True)
    report[label] = {"loss_history": history, "seconds": dt}
    return history


def chi2(s, ref, sigma2):
    """Mean of |s - ref|^2 / sigma^2 over the coefficients (real and imaginary parts)."""
    return ((s - ref).abs() ** 2 / sigma2).mean()


# Galactic-like mask |b| > 20 deg about the z axis.
with PatchEngine(NSIDE, 1) as probe:
    z = probe.centre_vectors(np.arange(NPIX))[:, 2]
mask = (np.abs(z) > np.sin(np.radians(20.0))).astype(np.float64)

# ---------------------------------------------------------------- 0. gradient check
def check(name, f, x0):
    x = x0.clone().requires_grad_(True)
    w = torch.from_numpy(rng.normal(size=tuple(f(x0).shape)))
    loss = lambda y: (w * (y.real if y.is_complex() else y)).sum() + (w * (y.imag if y.is_complex() else 0 * y)).sum()
    loss(f(x)).backward()
    h = torch.from_numpy(rng.normal(size=tuple(x0.shape)))
    eps = 1e-5
    fd = (loss(f(x0 + eps * h)) - loss(f(x0 - eps * h))).item() / (2 * eps)
    ad = (x.grad * h).sum().item()
    print(f"gradient check {name}: autograd {ad:.10e}, finite difference {fd:.10e}, relative {abs(ad - fd) / abs(fd):.1e}")
    report.setdefault("gradient_check", {})[name] = abs(ad - fd) / abs(fd)


x0 = torch.from_numpy(1.0 + rng.normal(size=NPIX))
with NativeEngine(NSIDE, J, masks=[np.flatnonzero(mask)]) as e:
    check("NativeEngine, mask", lambda x: scattering(e, x[None]), x0)
with NativeEngine(NSIDE, J, cross=True) as e:
    check("NativeEngine, cross", lambda x: scattering(e, torch.stack([x, x.flip(0)])[None]), x0)
with PatchEngine(NSIDE, J) as e:
    e.select(np.array([5, 7000]), 30.0, 20.0)
    check("PatchEngine", lambda x: patches(e, x), x0)
with SpectrumEngine(NSIDE, masks=mask[None]) as e:
    check("SpectrumEngine, mask", lambda x: spectrum(e, x[None]), x0)

# ---------------------------------------------------------------- 1. masked synthesis
stats = NativeEngine(NSIDE, J, masks=mask[None])
cls = SpectrumEngine(NSIDE, masks=mask[None])
s_t, var_t = stats.forward(t, moments=True)
sigma2_t = torch.from_numpy(var_t / stats.native_pixels())                 # ergodic sigma^2 per coefficient
s_t, c_t = torch.from_numpy(s_t), torch.from_numpy(cls.forward(t))
ell = torch.arange(2, c_t.shape[-1], dtype=torch.float64)
w_l = (2 * ell + 1) / 2                                                     # cosmic-variance weights, l >= 2


def masked_loss(x):
    s = scattering(stats, x[None])
    c = spectrum(cls, x[None])
    return chi2(s, s_t, sigma2_t) + (w_l * (c[..., 2:] / c_t[..., 2:] - 1) ** 2).sum() / w_l.sum()


x = torch.from_numpy(rng.normal(size=NPIX)).requires_grad_(True)
lbfgs(x, masked_loss, 30, "masked synthesis")
masked = x.detach().numpy()
c_masked = cls.forward(masked).reshape(-1)

# ---------------------------------------------------------------- 2. patch synthesis
pe = PatchEngine(NSIDE, J)
centres = rng.choice(12 * 2**2, 6, replace=False) * (NSIDE // 2) ** 2       # nside-2 pixels, as NESTED ids at nside 32
pe.select(centres, 30.0, 20.0)
p_t, pvar_t = pe.forward(pe.gather(t))
p_t, psigma2 = torch.from_numpy(p_t), torch.from_numpy(pvar_t / pe.native_pixels)
x = torch.from_numpy(rng.normal(size=NPIX)).requires_grad_(True)
lbfgs(x, lambda x: chi2(patches(pe, x), p_t, psigma2), 200, "patch synthesis")
patched = x.detach().numpy()
in_discs = np.zeros(NPIX, bool)
in_discs[np.concatenate(pe.ids)] = True

# ---------------------------------------------------------------- 3. denoising with cross statistics
# The 2B + 1 pairs of the loss go into one batch. On the CPU a disc of 180 deg in
# PatchEngine is the full-sky NativeEngine (same values to 1e-14) and runs the
# patches of a batch on parallel threads: 0.7 s per evaluation here, against
# 5.3 s for NativeEngine(cross=True, batch=7), which runs on one thread.
B = 3
d = t + rng.normal(size=NPIX)                                               # noise of the pixel std of s
n_b = torch.from_numpy(rng.normal(size=(B, NPIX)))
dt_ = torch.from_numpy(d)
cross = PatchEngine(NSIDE, J, cross=True)
cross.select(np.zeros(2 * B + 1, dtype=np.int64), 180.0)                  # 2B + 1 full-sky "discs"
with PatchEngine(NSIDE, J, cross=True) as e:
    e.select(np.zeros(1, dtype=np.int64), 180.0)
    s_dd, var_dd = e.forward(np.stack([d, d])[None])
    sigma2_dd = torch.from_numpy(var_dd[0] / e.native_pixels[0])
s_dd = torch.from_numpy(s_dd[0])


def denoise_loss(u):
    pairs = [torch.stack([u + n, u + n]) for n in n_b] + [torch.stack([u + n, u]) for n in n_b] + [torch.stack([dt_, u])]
    s = discs(cross, torch.stack(pairs))                                    # [2B + 1, ncoeff]
    return chi2(s[:B].mean(0), s_dd, sigma2_dd) + chi2(s[B:2 * B].mean(0), s[2 * B], sigma2_dd)


u = dt_.clone().requires_grad_(True)
lbfgs(u, denoise_loss, 100, "denoising")
denoised = u.detach().numpy()

full = SpectrumEngine(NSIDE)
spectra = {k: full.forward(m).reshape(-1) for k, m in (("target", t), ("noisy d", d), ("denoised u", denoised))}
# A statistical denoiser restores the statistics of s, not its pixels: compare spectra.
cl_err = {k: float(np.mean(np.abs(np.log(c[2:] / spectra["target"][2:])))) for k, c in spectra.items() if k != "target"}
rms = lambda a: float(np.sqrt(np.mean(a**2)))
report["denoising"].update(cl_log_error_d=cl_err["noisy d"], cl_log_error_u=cl_err["denoised u"],
                           rms_d_minus_s=rms(d - t), rms_u_minus_s=rms(denoised - t))
print(f"denoising: mean |log C_l / C_l(s)| for l >= 2: d {cl_err['noisy d']:.3f}, u {cl_err['denoised u']:.3f}; "
      f"pixel rms of d - s {rms(d - t):.3f}, of u - s {rms(denoised - t):.3f}")
np.save(out / "maps.npy", np.stack([t, masked, patched, d, denoised]))
json.dump(report, open(out / "report.json", "w"), indent=1)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotting import hist_norm, mollweide_index

idx = mollweide_index(NSIDE, 400)
f = hist_norm(t)
maps = [("target (nside 32)", t, None), ("1. masked synthesis, |b| > 20°", masked, mask > 0),
        ("2. patch synthesis, 6 discs", patched, in_discs), ("3. noisy d = s + n", d, None),
        ("3. denoised u", denoised, None)]
fig = plt.figure(figsize=(16, 6.2))
grid = fig.add_gridspec(2, 15, height_ratios=[1, 1.15], left=0.04, right=0.99, top=0.95, bottom=0.09, wspace=1.2, hspace=0.15)
for k, (title, m, keep) in enumerate(maps):
    ax = fig.add_subplot(grid[0, 3 * k:3 * k + 3])
    img = np.where(idx >= 0, f(m)[np.maximum(idx, 0)], np.nan)
    if keep is not None:                                                    # outside the constrained region: grey
        img = np.where((idx >= 0) & ~keep[np.maximum(idx, 0)], -1.0, img)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_under("0.85")
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
ax = fig.add_subplot(grid[1, 0:5])
for label in ("masked synthesis", "patch synthesis", "denoising"):
    h = report[label]["loss_history"]
    ax.semilogy(np.arange(1, len(h) + 1), h, label=label)
ax.set_xlabel("evaluation")
ax.set_ylabel("loss")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = fig.add_subplot(grid[1, 5:10])
l = np.arange(c_t.shape[-1])
ax.loglog(l[1:], c_t.numpy().reshape(-1)[1:], "k", label="target, |b| > 20°")
ax.loglog(l[1:], c_masked[1:], "C0--", label="masked synthesis")
ax.set_xlabel("$\\ell$")
ax.set_ylabel("pseudo-$C_\\ell$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = fig.add_subplot(grid[1, 10:15])
for (label, c), style in zip(spectra.items(), ("k", "C3:", "C2--")):
    ax.loglog(l[1:], c[1:], style, label=label)
ax.set_xlabel("$\\ell$")
ax.set_ylabel("$C_\\ell$ (full sky)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.savefig(figure, dpi=80)
print("figure", figure)
