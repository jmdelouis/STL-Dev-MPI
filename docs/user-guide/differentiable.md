# Differentiable statistics in Python

The engines of STL-Dev-MPI are also available from Python as **differentiable functions** of the
maps. They return the statistics and, for any cotangent, their exact vector–Jacobian product
(VJP). This lets them serve as loss terms for a neural network, a denoiser, a component separation
or a custom synthesis. There is no autograd graph: the gradient comes from the same block-wise
reverse sweep as the command-line tools. The memory is that of the engine, and the results are
those of the validated Rust code.

Three engines in `stl_port.native` (`python/stl_ensemble/stl_port/native.py`):

| Engine | Statistics | Domain | Pages with the definitions and the measurements |
|---|---|---|---|
| `NativeEngine` | scattering covariance (`mean`, `var`, S1–S4), auto or cross `S(a, b)` | full sky, or averaged over `K` masks | [masked statistics](../masked_statistics.md), [cross statistics](../disc_cross_statistics.md) |
| `PatchEngine` | the same plan, auto or cross | disc patches (support radius `R`, averaging core `r`) | [disc patches](../disc_cross_statistics.md) |
| `SpectrumEngine` | pseudo-`C_ℓ` (anafast, `iter=0`): spin 0 (`TT`), spin `s` (`EE`, `BB`, `EB`), auto or cross | full sky, or `K` masks (weights) | [pseudo-spectra](../spectrum.md) |

A worked example with all three, wrapped for torch, is
[the tutorial on differentiable losses](../tutorials/differentiable.md).

## Setup

```bash
cargo build --release -p stl-bridge                  # CPU; add --features gpu for backend="gpu"
export STL_NATIVE_LIBRARY=$PWD/target/release/libstl_bridge.so
export PYTHONPATH=$PWD/python/stl_ensemble           # for `import stl_port`
```

Conventions shared by the three engines:

- **Maps.** Full-sky maps are NESTED, `float64`, `npix = 12 nside²`, batched as `[B, npix]`
  (`[B, 2, npix]` for a pair `(a, b)` with `cross=True`, or for `(Q, U)` with `spin > 0`).
- **Statistics.** Scattering statistics are `complex128`, in the order of `engine.plan`, a list
  of dictionaries with the family, scales, orientations and native level. Spectra are `float64`
  `[.., lmax + 1]`. With `K` masks, an axis `K` follows the batch axis.
- **Cotangents.** A cotangent has the shape of the output. For complex statistics it is
  `dL/dRe + i dL/dIm`, which is also what torch passes to the backward of a complex tensor. `vjp`
  returns `dL/dmaps`, shaped like the maps.
- **Forward state.** An engine keeps the state of its last forward only. Call `vjp` after the
  `forward` with the same `generation` (`member` and `objective` are optional parts of the same
  key). A `vjp` with another key raises an error instead of returning a wrong gradient.
  Changing the masks or the selected discs drops the state.
- **Backends.** `backend="cpu"` or `"gpu"` (`device=k`), with the same results to 1e-12 or better.
  The CPU `NativeEngine` runs on one thread; `PatchEngine` and `SpectrumEngine` use
  `STL_THREADS` threads (default: every CPU).
- **Full sky on CPU threads.** A disc of 180° in `PatchEngine` has the full-sky statistics of
  `NativeEngine` (to 1e-14), and the discs of a batch run on parallel threads. For a batch of
  full-sky maps on the CPU it is the faster engine: 7 cross pairs at nside 32 take 0.7 s per
  forward and VJP on a laptop, against 5.3 s for `NativeEngine(cross=True, batch=7)`. Masks remain
  a `NativeEngine` feature.

## `NativeEngine`: full sky, masks, cross

```python
from stl_port.native import NativeEngine

eng = NativeEngine(nside, j, batch=B, backend="gpu")          # J scales, L = 4, kernel 5
values, var = eng.forward(x, generation=g, moments=True)      # x [B, npix] -> complex [B, ncoeff], float [B, ncoeff]
grad = eng.vjp(cot, generation=g)                             # cot complex [B, ncoeff] -> [B, npix]
eng.plan, eng.ncoeff, eng.native_pixels()                     # coefficient list; pixels each coefficient averages
```

- **Masks.** `NativeEngine(..., masks=m)` or `eng.set_masks(m)` takes `K` masks, given as a list
  of NESTED id arrays at `nside` or as `[K, npix]` indicators (nonzero means in the mask). Every
  coefficient is then averaged over each mask at its native level: level `n + 1` holds the
  parents of the mask at level `n`. The fields (wavelets, moduli, pyramid) remain those of the
  full sky, so the terms near the mask edge read pixels outside it. Values, variances and
  cotangents become `[B, K, ncoeff]`, and `native_pixels()` returns `[K, ncoeff]`. One sweep
  serves all the masks: five galactic cuts cost +16 % over the full sky on a T4.
  `set_masks(None)` returns to the whole sky.
- **Cross.** `cross=True` holds the cross terms of two maps, `S(a, b)`: S1 to S4 with
  `c1 ≠ c2`, with S3 and S4 in both orders (there is no cross `mean` or `var`).
  `forward_cross(a, b)` and `vjp_cross(cot) -> (ga, gb)` take the two maps separately. Drop
  `gb` to keep `b` constant. Masks combine with `cross`.
- **Ergodic weights.** `var` holds the spatial variances of the terms over the averaged pixels.
  `var / native_pixels()` estimates the variance of each coefficient, so a reduced χ² is
  `mean(|S − T|² / (var / N))`, as `stl-synth --weights ergodic` does.

## `PatchEngine`: disc patches

```python
from stl_port.native import PatchEngine

pe = PatchEngine(nside, j, cross=False, backend="cpu")    # backend="gpu", device=0 for the device engine
ids = pe.select(centres, radius=10.0, core=7.0)           # degrees; centres: NESTED ids [B] or vectors [B, 3]
values, var = pe.forward(pe.gather(x))                    # patches x[ids[i]] -> complex [B, ncoeff]
grads = pe.vjp(cot)                                       # one gradient per patch, laid out as ids[i]
pe.native_pixels                                          # [B, ncoeff]
```

- **Disc and core.** The *support* holds every pixel whose centre lies within `radius` of the
  centre. The wavelet is bound to it with renormalised weights, and the decimation is restricted
  to it, so nothing outside the disc is read. The coefficients average over the *core*: the
  pixels within `core` and their parents. The ring between core and support absorbs the edge
  effects of the convolutions. Without `core` the averages cover the whole support. A disc of
  180° gives the full-sky statistics.
- **Batches.** The discs of a batch may differ in size, so patches and gradients are lists. A
  dense `[B, n]` array works when all discs have `n` pixels. The plan is that of
  `NativeEngine(nside, j, cross=...)`.
- **CPU cache.** `cache` (256) disc geometries are kept, keyed by `(centre, R, r)`. Sixteen new discs
  at nside 256 and `R = 10°` take 0.6 s on 20 threads; a cached disc takes less than 1 ms. Drawing
  centres from a finite set (for example the pixels of a coarse nside) makes the geometry free
  after the first epoch.
- **GPU cache and `reserve_bytes`.** With `backend="gpu"`, the discs of a batch are grouped by
  centre and evaluated in one launch per operator. Their geometries are cached on the device,
  least recently used first out: before a new disc is built, cached discs are evicted until
  `reserve_bytes` (2 GiB) stay free on the device. The free memory is read at that time, so a
  network training on the same GPU is accounted for. `cache_bytes` optionally caps the cache.
  `forward_device(x)` and `vjp_device(cot)` take and return CUDA tensors (the concatenated
  patches) without copying the maps to the host.

## `SpectrumEngine`: pseudo-`C_ℓ`, spin 0 and spin `s`

```python
from stl_port.native import SpectrumEngine

se = SpectrumEngine(nside, lmax=None, batch=B, backend="gpu", masks=w)   # lmax: 3 nside - 1 by default
cl = se.forward(x, generation=g)          # x [B, npix] -> [B, K, lmax + 1] (no K axis without masks)
grad = se.vjp(cot, generation=g)          # cot like cl -> [B, npix]
se.fsky                                   # mean(w²) per mask

pol = SpectrumEngine(nside, spin=2, masks=w)
cl = pol.forward(qu)                      # qu [B, 2, npix] (Q, U) -> [B, K, 3, lmax + 1]: pol.components = EE, BB, EB
```

- **Definition.** The spectra are those of `healpy.anafast(x, lmax, iter=0)`, with no ring
  weights, on NESTED maps; spin `s` follows `healpy.map2alm_spin`. The tests compare with
  healpy to 1e-11.
- **Masks.** `[K, npix]` weights (binary or apodised) or NESTED id lists. Each map (`Q` and `U`
  alike) is multiplied by each weight before the analysis. The output is the pseudo-spectrum, not
  divided by `f_sky` (`se.fsky` provides it).
- **Cross.** `cross=True`: pairs `[B, 2, npix]` (`[B, 2, 2, npix]` for spin `s`) and the cross
  spectra only (`E1E2`, `B1B2`, `E1B2`, `B1E2` for spin `s`).
- **GPU.** Spin 0 runs its harmonic transforms on the device. Spin `s` uses the host transforms
  with either backend.

## What is differentiable

| Engine | Output | Gradient with respect to | Constant (no gradient) |
|---|---|---|---|
| `NativeEngine` | statistics `[B, (K,) ncoeff]` | the maps; both maps `a` and `b` with `cross=True` | masks, `var` (moments), the plan |
| `PatchEngine` | statistics `[B, ncoeff]` | the patch pixels (both maps with `cross=True`); to the full sky through `gather` | disc centres, radii, `var` |
| `SpectrumEngine` | `C_ℓ` `[B, (K,) (C,) lmax + 1]` | the maps (`Q`, `U` for spin `s`; both fields with `cross=True`) | mask weights, `fsky` |
| `delouis_transform` | `sign(Re)·sqrt|Re| + i·sign(Im)·sqrt|Im|` | its input (`delouis_transform_vjp`, unbounded at 0) | – |

## Backends and typical costs

| Engine | CPU | GPU | Typical cost of forward + VJP (measured) |
|---|---|---|---|
| `NativeEngine` | yes (one thread) | yes | nside 32, `J = 4`: 2.4 s on one laptop core; nside 64, `J = 5`: 7.5 s. T4, nside 256, `J = 4`: 1.78 s (auto), 2.06 s (cross); 5 masks +16 % (auto), +27 % (cross) |
| `PatchEngine` | yes (`threads` host threads) | yes (`forward_device` for CUDA tensors) | 20 laptop threads, nside 256, 16 discs of 10° (5973 pixels), cached geometry: 0.18 s. T4, 240 cross patches on 16 discs: 1.33 s per step, including geometry rebuilds |
| `SpectrumEngine` | yes (`STL_THREADS`) | spin 0 on the device, spin `s` on the host | nside 256, `lmax = 511`, one map: 0.064 s spin 0, 0.24 s spin 2 (20 threads); T4 with 5 masks: 0.19 s spin 0 |

The full measurements are in the reports linked at the top of this page. A VJP costs about the
same as a forward for the spectra and about 1.2 to 1.5 forwards for the statistics.

## A torch wrapper

`examples/06_differentiable_python/stl_torch.py` turns each engine into a torch operation with one
`torch.autograd.Function`. Its forward copies the input to numpy, calls the engine with a fresh
generation and returns a tensor on the input's device. Its backward calls the engine's `vjp` with
the same generation:

```{literalinclude} ../../examples/06_differentiable_python/stl_torch.py
:language: python
```

Because the forward state is kept per engine, every term of a loss that runs in the same
backward pass needs its own engine, or all the terms go into one batched call (the denoising
loss below does this). For training at scale, use `forward_device` / `vjp_device` of the GPU
`PatchEngine` to keep the maps on the device.

## Example losses

The three losses below run in [the tutorial](../tutorials/differentiable.md). `chi2(s, t, σ²)` is
`mean(|s − t|² / σ²)` over the coefficients.

**Patch loss.** Make the statistics of a map match those of a target on a set of discs (for
example random discs drawn at every training step of a network):

```python
pe = PatchEngine(nside, j)
pe.select(centres, 30.0, 20.0)                        # R = 30°, core 20°
t_values, t_var = pe.forward(pe.gather(target))
t_values, sigma2 = torch.from_numpy(t_values), torch.from_numpy(t_var / pe.native_pixels)
loss = chi2(patches(pe, x), t_values, sigma2)         # x: full-sky torch tensor [npix]
```

**Cross loss (Delouis et al. 2022, A&A 668, A122).** Denoise `d = s + n` with known noise
statistics. Find `u` such that adding noise realisations `n_b` to `u` reproduces the statistics
of the data, `<S(u + n_b, u + n_b)> = S(d, d)`, and its cross statistics with the data,
`<S(u + n_b, u)> = S(d, u)`. All the pairs go into one cross engine:

```python
cross = NativeEngine(nside, j, batch=2 * B + 1, cross=True)         # or backend="gpu"
def denoise_loss(u):
    pairs = ([torch.stack([u + n, u + n]) for n in noise]           # B pairs
             + [torch.stack([u + n, u]) for n in noise]             # B pairs
             + [torch.stack([d, u])])                               # 1 pair
    s = scattering(cross, torch.stack(pairs))                       # [2B + 1, ncoeff]
    return chi2(s[:B].mean(0), s_dd, sigma2) + chi2(s[B:2 * B].mean(0), s[2 * B], sigma2)
```

The tutorial runs the same loss on CPU threads with full-sky discs: `PatchEngine(nside, j,
cross=True)`, `select(np.zeros(2 * B + 1, dtype=np.int64), 180.0)`, and `discs(cross, pairs)`
instead of `scattering`. This estimator restores the statistics of the signal, not its pixel
values: judge it on statistics or spectra, not on the pixel error.

The third loss of the paper, `S(T, d)` against `<S(T, u + n)>` for a template `T`, uses the same
engine with `T` as the first map. S3 and S4 are not symmetric in `(a, b)`: keep the pairs in the
written order. `delouis_transform` gives the paper's `sqrt` compression of the coefficients,
which makes them comparable across scales.

**C_ℓ loss.** Match a target pseudo-spectrum on a mask, with cosmic-variance weights:

```python
se = SpectrumEngine(nside, masks=mask[None])            # mask [npix] weights
c_t = torch.from_numpy(se.forward(target))
l = torch.arange(2, c_t.shape[-1])
w = (2 * l + 1) / 2
loss = (w * (spectrum(se, x[None])[..., 2:] / c_t[..., 2:] - 1) ** 2).sum() / w.sum()
```

With `spin=2` the same loss applies to `EE`, `BB` and `EB` (`[.., 3, lmax + 1]`) of a `(Q, U)`
pair. With `cross=True` it applies to the cross spectrum of two maps, for example a map and a
template.
