"""Writes the tutorial notebooks (without outputs) from the cells below.

    python docs/tutorials/make_notebooks.py           # writes docs/tutorials/*.ipynb
    jupyter nbconvert --to notebook --execute --inplace docs/tutorials/NAME.ipynb

The committed notebooks hold the outputs of a run on a machine with a GPU
(garoupe: Tesla T4); the documentation build never executes them.  Each
notebook needs the binaries (`STL_BIN`, default target/release) and, for the
Python engines, `STL_NATIVE_LIBRARY` (libstl_bridge.so).
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = r'''import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
while not (ROOT / "Cargo.toml").exists():          # the repository root, from docs/tutorials
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "python" / "stl_ensemble"))
BIN = Path(os.environ.get("STL_BIN", ROOT / "target" / "release"))
WORK = Path(os.environ.get("STL_WORK", ROOT / "docs" / "tutorials" / "out")); WORK.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 80, "image.cmap": "plasma"})

def run(*args):
    """Runs a command-line tool, prints its last lines and its wall-clock time."""
    t0 = time.time()
    p = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    print(p.stdout.strip()[-600:], f"\n[{time.time() - t0:.1f} s]")'''

MAPS = r'''from plotting import mollweide_index, face_image, hist_norm
get_ipython().run_line_magic("matplotlib", "inline")    # plotting.py selects Agg for scripts

def show_maps(maps, titles, face=4, width=420):
    """Mollweide views and one base face of NESTED maps, on the histogram-equalised colour
    scale of the first map (healpy's norm='hist'), shared by all panels."""
    f = hist_norm(maps[0])
    nside = int(round((maps[0].size / 12) ** 0.5))
    idx = mollweide_index(nside, width)
    fig, ax = plt.subplots(2, len(maps), figsize=(4.0 * len(maps), 5.2), squeeze=False,
                           gridspec_kw={"height_ratios": [1, 1.5]})
    for k, (m, t) in enumerate(zip(maps, titles)):
        v = np.where(np.isnan(m), np.nan, f(m))                    # NaN: not shown (grey)
        ax[0, k].imshow(np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan), vmin=0, vmax=1, interpolation="nearest")
        ax[0, k].set_title(t)
        ax[1, k].imshow(face_image(v, face), vmin=0, vmax=1, interpolation="nearest")
        ax[1, k].set_title(f"base face {face}", fontsize=9)
    for a in ax.flat:
        a.axis("off")
        a.set_facecolor("0.85")
    plt.tight_layout(); plt.show()'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


TAKAHASHI = [
    md(r"""
# Synthesis of a weak-lensing map (Takahashi κ)

This tutorial synthesises a full-sky weak-lensing convergence map whose scattering-covariance
statistics match those of a simulated map, with `stl-synth` on a GPU, and checks the result.

The target is the convergence κ of the full-sky ray-tracing simulations of Takahashi et al. (2017,
ApJ 850, 24), realisation r000, source redshift `zs1`, block-averaged to nside 64 and shipped with
the repository (`fixtures/takahashi/`, the target of the command-line examples). κ is strongly
non-Gaussian (positive tail of haloes, voids), so we follow the recommended recipe for skewed
fields:

1. Gaussianise the target by its rank (Gaussian anamorphosis);
2. interpolate it to `2 nside` and synthesise it there with `stl-synth` (the finest level of a
   synthesis is the least well constrained), then cut back to `nside`;
3. give the synthesis the target's histogram back by rank matching.

[Example 01](healpix_synthesis) runs the same recipe as a script; here we look at the result.
Run time: about a minute and a half on one Tesla T4.
"""),
    code(SETUP + "\n\n" + MAPS),
    code(r'''
kappa = np.load(ROOT / "fixtures/takahashi/takahashi_r000_zs1_nside64_nested.npy").astype(np.float64)
nside = int(round((kappa.size / 12) ** 0.5))
z = (kappa - kappa.mean()) / kappa.std()
print(f"nside {nside}, {kappa.size} pixels, mean {kappa.mean():.3e}, std {kappa.std():.3e}, "
      f"skewness {np.mean(z**3):.2f}, excess kurtosis {np.mean(z**4) - 3:.1f}")
show_maps([kappa], ["Takahashi κ, r000, zs1"])
'''),
    md(r"""
## 1. Gaussian anamorphosis

The pixel of rank `r` among `N` becomes `Φ⁻¹((r + 1/2) / N)`; the sorted values of κ are kept to
go back (this is `tools/gaussianize.py forward`).
"""),
    code(r'''
from scipy.stats import norm
order = np.argsort(kappa, kind="stable")
g = np.empty_like(kappa)
g[order] = norm.ppf((np.arange(kappa.size) + 0.5) / kappa.size)
kappa_sorted = kappa[order]
np.save(WORK / "target_g.npy", g)
print(f"g: mean {g.mean():.1e}, std {g.std():.4f}")
# to 2 nside, each group of 4 children keeping its parent's value as mean (tools/resample_nested.py)
run(sys.executable, ROOT / "tools/resample_nested.py", "up", WORK / "target_g.npy", WORK / "target_g_up.npy", "--keep-mean")
'''),
    md(r"""
## 2. Synthesis with `stl-synth`

`--weights ergodic` divides every squared residual by the spatial variance of that coefficient
on the target (a reduced χ²), so that every family is fitted; `L = 4` orientations and
`J = log2(128) − 1 = 6` scales are the defaults. One T4, from white noise.
"""),
    code(r'''
run(BIN / "stl-synth", "--target", WORK / "target_g_up.npy", "--weights", "ergodic", "--niter", 400,
    "--seed", 1, "--gpu", "--quiet", "--out", WORK / "synth_g")
rep = json.load(open(WORK / "synth_g.json"))
print(f"{rep['evaluations']} evaluations, {rep['seconds_per_evaluation']} s each")
print("final residual per family, in ergodic sigma:", rep["final_residual_in_ergodic_sigma"])
plt.figure(figsize=(5, 3)); plt.semilogy(rep["loss_history"]); plt.xlabel("evaluation"); plt.ylabel("loss")
plt.grid(alpha=0.3); plt.show()
'''),
    md(r"""
## 3. Back to κ

The synthesis is in the target's units (here those of `g`), at nside 128: the mean of each group
of 4 children brings it back to nside 64. Then its pixel of rank `r` receives the `r`-th smallest
value of κ (`tools/gaussianize.py match`): the histogram is exactly the target's, and the spatial
structure is the synthesis'.
"""),
    code(r'''
s_g = np.load(WORK / "synth_g_synthesis_nested.npy").reshape(-1, 4).mean(1)
np.save(WORK / "synth_g64.npy", s_g)
s_kappa = np.empty_like(s_g)
s_kappa[np.argsort(s_g, kind="stable")] = kappa_sorted
np.save(WORK / "synth_kappa.npy", s_kappa)
show_maps([kappa, s_kappa], ["target κ", "synthesis κ"])
'''),
    md(r"""
## 4. Checks

Angular power spectra with the differentiable anafast of the package (`SpectrumEngine`, healpy's
conventions), and the scattering statistics of both maps with `stl-stats`, in the units of the
target (`--norm`).
"""),
    code(r'''
from stl_port.native import SpectrumEngine
spec = SpectrumEngine(nside, batch=2)
cl = spec.forward(np.stack([kappa, s_kappa]))
ell = np.arange(cl.shape[1])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].loglog(ell[2:], (ell * (ell + 1) * cl[0])[2:], "k", label="target")
ax[0].loglog(ell[2:], (ell * (ell + 1) * cl[1])[2:], "C3", lw=1, label="synthesis")
ax[0].set_xlabel("ℓ"); ax[0].set_ylabel("ℓ(ℓ+1) C_ℓ"); ax[0].legend()
bins = np.linspace(kappa.min(), np.percentile(kappa, 99.9), 80)
ax[1].hist(kappa, bins, histtype="step", color="k", label="target")
ax[1].hist(s_kappa, bins, histtype="step", color="C3", label="synthesis")
ax[1].set_yscale("log"); ax[1].set_xlabel("κ"); ax[1].legend()
plt.tight_layout(); plt.show()
print("C_l ratio synthesis / target, mean over l = 2..3 nside-1:", float(np.mean(cl[1, 2:] / cl[0, 2:])))
'''),
    code(r'''
np.save(WORK / "white.npy", np.random.default_rng(0).standard_normal(kappa.size))
run(BIN / "stl-stats", "--out", WORK / "stats.json", "--norm", WORK / "target_g.npy", "--gpu", "--quiet", "--",
    WORK / "target_g.npy", WORK / "synth_g64.npy", WORK / "white.npy")
r = json.load(open(WORK / "stats.json"))
fam = np.array([c["family"] for c in r["plan"]]); j1 = np.array([c["j1"] for c in r["plan"]])
vals = [np.array(m["re"]) + 1j * np.array(m["im"]) for m in r["maps"]]
names = ["target g", "synthesis g", "white noise"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for k, f in enumerate(["S1", "S2"]):
    for n, v in zip(names, vals):
        js = np.unique(j1[fam == f])
        ax[k].semilogy(js, [v[(fam == f) & (j1 == j)].real.mean() for j in js], "o-", ms=3, label=n)
    ax[k].set_title(f"{f}, mean over orientations"); ax[k].set_xlabel("scale j")
ax[0].legend()
sel = (fam == "S3") | (fam == "S4")
ax[2].plot(vals[0][sel].real, vals[1][sel].real, ".", ms=2, label="synthesis")
ax[2].plot(vals[0][sel].real, vals[2][sel].real, ".", ms=2, label="white noise")
lim = np.abs(vals[0][sel].real).max(); ax[2].plot([-lim, lim], [-lim, lim], "k", lw=0.5)
ax[2].set_title(f"S3 and S4 (Re), {sel.sum()} coefficients"); ax[2].set_xlabel("target"); ax[2].legend(markerscale=4)
ax[2].ticklabel_format(style="sci", scilimits=(0, 0))
plt.tight_layout(); plt.show()
'''),
    md(r"""
## Going further

- **More resolution**: with a target at a higher nside, start the finer synthesis from this one
  (`--init synth_g64.npy --init-noise conditional`): the details drawn conditionally on the
  coarse map start the run close to the solution ({doc}`../user-guide/stl-synth`).
- **Ensembles** of maps whose mean statistics match this target (`--ensemble 10
  --scatter-weight 1`), or whose mean and variance match ten realisations (`stl-ensemble`,
  {doc}`../user-guide/stl-ensemble`).
- **Hold the physical map** directly: keep `g` as the variable and add the statistics of `κ = f(g)`
  with `--kappa-table`, the level power and the histogram tails (`--marginal-tail-weight`).
"""),
]

STATISTICS = [
    md(r"""
# Statistics of a map, with masks, cross statistics and gradients

This tutorial computes the scattering-covariance statistics of the Takahashi κ map with the Python
engine `NativeEngine`: the plan, the statistics per family, their spatial variances (ergodic error
bars), averages over masks, cross statistics of two maps, and the exact gradient.

`NativeEngine` loads `libstl_bridge.so` (`cargo build --release -p stl-bridge --features gpu`,
`STL_NATIVE_LIBRARY`). The map is the Takahashi κ map at nside 64 of `fixtures/takahashi/`.
Everything here runs in about a minute on a CPU.
"""),
    code(SETUP),
    code(r'''
from stl_port.native import NativeEngine
kappa = np.load(ROOT / "fixtures/takahashi/takahashi_r000_zs1_nside64_nested.npy").astype(np.float64)
x = (kappa - kappa.mean()) / kappa.std()
nside, j = 64, 5
eng = NativeEngine(nside, j, batch=1, backend="cpu", threads=8)
plan = eng.plan
fam = np.array([c["family"] for c in plan])
print(len(plan), "coefficients:", {str(f): int((fam == f).sum()) for f in dict.fromkeys(fam)})
print("one coefficient of each family:", [plan[int(np.flatnonzero(fam == f)[0])] for f in ("S1", "S3", "S4")])
'''),
    md(r"""
## 1. Statistics and ergodic error bars

`forward(x, moments=True)` returns the values and the spatial variance of each coefficient's
terms; divided by the number of pixels each coefficient averages (`native_pixels()`), it estimates
the variance of the coefficient between realisations (under ergodicity).
"""),
    code(r'''
t0 = time.time(); s, var = eng.forward(x, moments=True); print(f"forward: {time.time() - t0:.2f} s")
s, var = s[0], var[0]
sigma = np.sqrt(var / eng.native_pixels())
j1 = np.array([c["j1"] for c in plan]); l1 = np.array([c["l1"] for c in plan])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for k, f in enumerate(["S1", "S2"]):
    for l in range(4):
        sel = (fam == f) & (l1 == l)
        ax[k].errorbar(j1[sel] + 0.05 * l, s[sel].real, 3 * sigma[sel], fmt="o-", ms=3, label=f"l = {l}")
    ax[k].set_yscale("log"); ax[k].set_xlabel("scale j"); ax[k].set_title(f"{f} (±3σ ergodic)")
ax[0].legend(fontsize=8)
sel = fam == "S4"
ax[2].hist(np.abs(s[sel]) / sigma[sel], bins=np.logspace(-2, 3, 60)); ax[2].set_xscale("log")
ax[2].set_title("S4: |value| / σ"); plt.tight_layout(); plt.show()
'''),
    md(r"""
## 2. Masks

With `K` masks, every coefficient is averaged over each mask at its native level (the wavelets and
moduli stay those of the full sky). Here: the whole sky, and the pixels of latitude
`|lat| > 20°` and `|lat| > 45°` of the HEALPix grid (galactic cuts of a real sky would be the same). The values become `[batch, K, ncoeff]`.
"""),
    code(r'''
from healpix_geo.nested import healpix_to_lonlat
lon, lat = healpix_to_lonlat(np.arange(12 * nside**2), int(np.log2(nside)), ellipsoid="sphere")
masks = [np.arange(lat.size), np.flatnonzero(np.abs(lat) > 20), np.flatnonzero(np.abs(lat) > 45)]
eng.set_masks(masks)
t0 = time.time(); sm, varm = eng.forward(x, moments=True); print(f"forward with 3 masks: {time.time() - t0:.2f} s")
sm, sigm = sm[0], np.sqrt(varm[0] / eng.native_pixels())
print("full sky equals mask 0:", np.allclose(sm[0], s, rtol=1e-12, atol=1e-14))
sel = fam == "S2"
plt.figure(figsize=(6, 3.4))
for k, name in enumerate(["whole sky", "|lat| > 20°", "|lat| > 45°"]):
    plt.errorbar(np.arange(sel.sum()) + 0.2 * k, sm[k, sel].real, sigm[k, sel], fmt=".", label=name)
plt.yscale("log"); plt.xlabel("S2 coefficient (j, l)"); plt.legend(); plt.show()
eng.set_masks(None)
'''),
    md(r"""
## 3. Cross statistics

`cross=True` gives the statistics `S(a, b)` of two maps (the terms with `c1 ≠ c2`; S3 and S4 in
both orders). Here κ against its Gaussianised version `g`, which has the same phases.
"""),
    code(r'''
from scipy.stats import norm
g = np.empty_like(x); g[np.argsort(x)] = norm.ppf((np.arange(x.size) + 0.5) / x.size)
ce = NativeEngine(nside, j, batch=1, cross=True, threads=8)
cfam = np.array([c["family"] for c in ce.plan])
sc, _ = ce.forward_cross(x[None], g[None])
print(len(ce.plan), "cross coefficients:", {str(f): int((cfam == f).sum()) for f in dict.fromkeys(cfam)})
sel = cfam == "S2"
print("cross S2 <z_kappa conj z_g> per scale (orientation 0):", np.round(sc[0, sel].real[::4], 4))
'''),
    md(r"""
## 4. Exact gradient

`vjp` returns the gradient of `Re Σ conj(c_i) S_i` with respect to the map. For the loss
`Σ_i w_i |S_i(y) − S_i(x)|²` the cotangent is `2 w (S(y) − S(x))`. A directional finite difference
confirms it.
"""),
    code(r'''
w = 1 / sigma**2
y = np.random.default_rng(1).standard_normal(x.size)
def loss(m):
    v, _ = eng.forward(m)
    return float(np.sum(w * np.abs(v[0] - s) ** 2)), v[0]
L0, v0 = loss(y)
t0 = time.time(); grad = eng.vjp(2 * w * (v0 - s))[0]; print(f"vjp: {time.time() - t0:.2f} s")
d = np.random.default_rng(2).standard_normal(x.size); eps = 1e-4
fd = (loss(y + eps * d)[0] - loss(y - eps * d)[0]) / (2 * eps)
print(f"loss {L0:.6e}; <grad, d> = {grad @ d:.8e}, finite difference {fd:.8e}, relative difference {abs(grad @ d / fd - 1):.1e}")
'''),
]

TURBULENCE = [
    md(r"""
# 2D synthesis: turbulence images

This tutorial synthesises a batch of 10 periodic images of 256 × 256 whose statistics have the
mean and the variance of 10 images of a 2D turbulence simulation, STL-Dev's 2D example
(`data/test/Turb_6.npy` of the [STL-Dev](https://github.com/STL-Proj/STL-Dev) repository; set
`TURB` to its path). It uses `stl-synth2d` with the ensemble loss (scattering covariance, power
spectrum and Minkowski functionals, batch mean and variance) and a coarse-to-fine start:
synthesis at 64², then 128², each level started from the previous one with details drawn
conditionally on the coarse image.

Run time: about 6 minutes on one Tesla T4.
"""),
    code(SETUP),
    code(r'''
get_ipython().run_line_magic("config", "InlineBackend.figure_formats = ['jpeg']")   # images: JPEG keeps the notebook small
TURB = Path(os.environ.get("TURB", ROOT.parent / "STL-Dev" / "data" / "test" / "Turb_6.npy"))
target = np.load(TURB)
print(TURB.name, target.shape, target.dtype, "- the first 10 images are the targets")
'''),
    md(r"""
## 1. Synthesis

`--c2f 64,128`: pixel L-BFGS at 64² (from white noise, with a warm start on STL-Dev's loss), then
at 128², then at 256² for `--niter` iterations; the details added at each step are drawn from the
targets' details in bins of the coarse value and gradient.
"""),
    code(r'''
run(BIN / "stl-synth2d", "--target", TURB, "--ntarget", 10, "--nbatch", 10, "--mode", "ensemble",
    "--c2f", "64,128", "--warm", 100, "--niter", 300, "--gpu", "--out", WORK / "turb")
rep = json.load(open(WORK / "turb.json"))
print(f"{rep['evaluations']} evaluations at 256², {rep['seconds_per_evaluation']:.3f} s each; total {rep['seconds']:.0f} s")
'''),
    code(r'''
maps = np.load(WORK / "turb_maps.npy")[:, 0]; tgt = np.load(WORK / "turb_target.npy")[:, 0]
lo, hi = np.percentile(tgt, [0.5, 99.5])
fig, ax = plt.subplots(2, 5, figsize=(13, 5.6))
for i in range(5):
    ax[0, i].imshow(tgt[i], vmin=lo, vmax=hi); ax[0, i].set_title(f"target {i}")
    ax[1, i].imshow(maps[i], vmin=lo, vmax=hi); ax[1, i].set_title(f"synthesis {i}")
for a in ax.flat: a.axis("off")
plt.tight_layout(); plt.show()
'''),
    md(r"""
## 2. Checks

The statistics of the batch (`turb_features_*.npy`, in the plan's order, with the family of each
column in the report) against the targets' mean and scatter, the pixel distribution and the
isotropic power spectrum.
"""),
    code(r'''
fam = np.array(rep["families"])
feat = {k: np.load(WORK / f"turb_features_{k}.npy") for k in ("target", "init", "final")}
feat = {k: v[..., 0] + 1j * v[..., 1] for k, v in feat.items()}
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
for f in ("S1", "S2", "S3", "S4", "PS"):
    sel = fam == f
    t = feat["target"][:, sel]; tm, ts = t.mean(0), np.abs(t.std(0, ddof=1))
    pull = np.abs(feat["final"][:, sel].mean(0) - tm) / np.where(ts > 0, ts, np.inf)
    ax[0].hist(np.clip(pull, 1e-3, 1e2), bins=np.logspace(-3, 2, 40), histtype="step", label=f"{f} ({sel.sum()})")
ax[0].set_xscale("log"); ax[0].axvline(1, color="k", lw=0.5); ax[0].legend(fontsize=8)
ax[0].set_title("|mean − target mean| / target σ")
bins = np.linspace(min(tgt.min(), maps.min()), max(tgt.max(), maps.max()), 80)
ax[1].hist(tgt.ravel(), bins, density=True, histtype="step", color="k", label="targets")
ax[1].hist(maps.ravel(), bins, density=True, histtype="step", color="C3", label="syntheses")
ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("pixel PDF")
def iso_ps(a):
    p = np.abs(np.fft.fft2(a - a.mean(axis=(-2, -1), keepdims=True))) ** 2
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(a.shape[-1]), np.fft.fftfreq(a.shape[-2]))) * a.shape[-1]
    kb = np.arange(1, a.shape[-1] // 2)
    return kb, np.array([p[..., (k >= b - 0.5) & (k < b + 0.5)].mean() for b in kb])
for a, c, n in ((tgt, "k", "targets"), (maps, "C3", "syntheses")):
    kb, ps = iso_ps(a); ax[2].loglog(kb, ps, c, label=n)
ax[2].set_xlabel("k (cycles per image)"); ax[2].legend(); ax[2].set_title("power spectrum")
plt.tight_layout(); plt.show()
'''),
    md(r"""
The batch mean of nearly every coefficient lies within one standard deviation of the targets'
mean, and the pixel distribution and the power spectrum follow the targets'. The dip of the
spectrum around `k = 32` is the Nyquist frequency of the 64² level: the Fourier upsampling of the
coarse-to-fine start drops the Nyquist modes of the smaller grid, and the 256² iterations have
not fully restored them.

More: the options of `stl-synth2d` ({doc}`../user-guide/stl-synth2d`), the benchmarks against
STL-Dev ({doc}`../planar_2d`) and the coarse-to-fine study ({doc}`../lagrangian_synthesis`,
section 6.4).
"""),
]

TORCH = [
    md(r"""
# A PyTorch loss on the GPU with `PatchEngine` and `SpectrumEngine`

The Python engines give the statistics **and their exact vector–Jacobian product**, so they plug
into PyTorch as `torch.autograd.Function`s. [Example 06](differentiable) wraps the three engines
for host tensors (`examples/06_differentiable_python/stl_torch.py`) and runs masked synthesis,
patch synthesis and denoising on a laptop CPU. This notebook shows the **GPU path**, the one for
training a network on a GPU:

- `PatchEngine(backend="gpu")` reads the patches from a CUDA tensor and returns their gradient as a
  CUDA tensor (`forward_device`, `vjp_device`): the maps never leave the device;
- `SpectrumEngine(backend="gpu")` runs its transforms on the device (through host arrays, with
  the `spectrum` wrapper of example 06).

The loss holds the scattering statistics of 32 disc patches and the full-sky power spectrum of the
Takahashi κ map at nside 64; it is minimised over the pixels of a map with Adam, as the output of
a network would be. Run time: about two minutes on one Tesla T4.
"""),
    code(SETUP),
    code(r"""
import torch
sys.path.insert(0, str(ROOT / "examples" / "06_differentiable_python"))
from stl_torch import spectrum                       # host wrapper of SpectrumEngine (example 06)
from stl_port.native import PatchEngine, SpectrumEngine
dev = torch.device("cuda")

class DevicePatches(torch.autograd.Function):
    # Statistics of the selected discs of a GPU PatchEngine, from the patches concatenated in one
    # CUDA tensor (patch i: u[ids[i]]); complex [B, ncoeff] on the device.
    @staticmethod
    def forward(ctx, flat, engine):
        values, _ = engine.forward_device(flat.contiguous(), moments=False)
        ctx.engine = engine
        return torch.from_numpy(values).to(flat.device)

    @staticmethod
    def backward(ctx, g):
        return ctx.engine.vjp_device(g.detach().cpu().numpy()), None

def patch_stats(engine, u, index):
    return DevicePatches.apply(u[index], engine)       # u[index]: autograd scatters the gradient back
"""),
    md(r"""
## 1. Engines and targets

32 discs of radius 15° (statistics averaged over a 10° core), centred on pixels of the nside-8
grid, and the spectrum up to `ℓ = 3 nside − 1`. The targets are those of the κ map, standardised;
the weights of the statistics are ergodic (`1 / σ²`).
"""),
    code(r"""
kappa = np.load(ROOT / "fixtures/takahashi/takahashi_r000_zs1_nside64_nested.npy").astype(np.float64)
t = torch.from_numpy((kappa - kappa.mean()) / kappa.std()).to(dev)
nside = 64
t0 = time.time()
pe = PatchEngine(nside, j=4, backend="gpu")
centres = np.arange(0, 768, 24) * (nside // 8) ** 2     # 32 nside-8 pixels, as nside-64 ids
ids = pe.select(centres, radius=15.0, core=10.0)
index = torch.from_numpy(np.concatenate(ids)).to(dev)
print(f"{len(ids)} discs of {min(map(len, ids))} to {max(map(len, ids))} pixels, {pe.ncoeff} coefficients each; "
      f"setup {time.time() - t0:.1f} s")
se = SpectrumEngine(nside, backend="gpu")
with torch.no_grad():
    s_t = patch_stats(pe, t, index)
    _, var_t = pe.forward_device(t[index].contiguous(), moments=True)
    cl_t = spectrum(se, t[None])[0]
w = torch.from_numpy(pe.native_pixels / np.maximum(var_t, 1e-30)).to(dev)   # ergodic weights [B, ncoeff]
ell = slice(2, None)

def loss_fn(u):
    stats, cl = patch_stats(pe, u, index), spectrum(se, u[None])[0]
    l_sc = (w * (stats - s_t).abs() ** 2).mean()
    l_ps = ((torch.log(cl[ell]) - torch.log(cl_t[ell])) ** 2).mean()
    return l_sc, l_ps
"""),
    md(r"""
## 2. Optimisation through autograd

`loss.backward()` calls `vjp_device` for the patches (gradient on the device) and the spectrum's
VJP; Adam updates the map on the device.
"""),
    code(r"""
u = torch.randn(12 * nside**2, dtype=torch.float64, device=dev, generator=torch.Generator(dev).manual_seed(0)).requires_grad_()
opt = torch.optim.Adam([u], lr=0.03)
history = []
t0 = time.time()
for step in range(600):
    opt.zero_grad()
    l_sc, l_ps = loss_fn(u)
    (l_sc + l_ps).backward()
    opt.step()
    history.append((l_sc.item(), l_ps.item()))
h = np.array(history)
print(f"600 steps in {time.time() - t0:.0f} s; statistics {h[0, 0]:.3g} -> {h[-1, 0]:.3g}, spectrum {h[0, 1]:.3g} -> {h[-1, 1]:.3g}")
plt.figure(figsize=(5, 3)); plt.semilogy(h[:, 0], label="patch statistics (reduced χ²)"); plt.semilogy(h[:, 1], label="log C_ℓ")
plt.xlabel("Adam step"); plt.legend(); plt.grid(alpha=0.3); plt.show()
"""),
    md(r"""
Inside the discs the statistics approach the target's (the reduced χ² keeps falling: a network
or a longer run would continue); outside them only the power spectrum constrains the map, which
stays noise-like there.
"""),
    code(MAPS + r"""

inside = np.full(t.numel(), np.nan); inside[np.concatenate(ids)] = 1.0
show_maps([t.cpu().numpy(), u.detach().cpu().numpy(), u.detach().cpu().numpy() * inside],
          ["target κ (standardised)", "optimised map", "inside the discs"])
"""),
    md(r"""
## 3. Gradient check

The gradient of the patch term computed on the GPU (through `vjp_device` and autograd's scatter
of `u[index]`) against the same gradient from the CPU `PatchEngine` on the same discs (its `vjp`,
summed back onto the sky by hand). A finite difference would not do: the GPU evaluations agree
with each other to about 1e-12, which a step small enough to avoid the kinks of `|z|` amplifies
to about 1e-4 of the directional derivative.
"""),
    code(r"""
v = u.detach().clone().requires_grad_(True)
l_sc, _ = loss_fn(v)
l_sc.backward()
g_gpu = v.grad.cpu().numpy()

cpu = PatchEngine(nside, j=4, threads=8)
cpu.select(centres, radius=15.0, core=10.0)
values, _ = cpu.forward(cpu.gather(v.detach().cpu().numpy()), moments=False)
wn, st = w.cpu().numpy(), s_t.cpu().numpy()
cot = 2 * wn * (values - st) / wn.size                       # d l_sc / d Re + i d l_sc / d Im
g_cpu = np.zeros(v.numel())
for i, gi in zip(cpu.ids, cpu.vjp(cot)):
    np.add.at(g_cpu, i, gi)
print(f"statistics: GPU vs CPU {np.abs(s_t.cpu().numpy() - cpu.forward(cpu.gather(t.cpu().numpy()), moments=False)[0]).max():.1e}")
print(f"gradient: max |GPU - CPU| / max |CPU| = {np.abs(g_gpu - g_cpu).max() / np.abs(g_cpu).max():.1e}")
"""),
]


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / f"{name}.ipynb")
    print("wrote", HERE / f"{name}.ipynb")


NOTEBOOKS = {"takahashi_synthesis": TAKAHASHI, "map_statistics": STATISTICS,
             "turbulence_2d": TURBULENCE, "torch_loss": TORCH}

if __name__ == "__main__":
    for name in sys.argv[1:] or NOTEBOOKS:
        write(name, NOTEBOOKS[name])
