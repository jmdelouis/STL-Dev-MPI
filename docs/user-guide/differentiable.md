# Differentiable statistics in Python

The engines of STL-Dev-MPI are also available from Python as **differentiable functions** of the
maps. They return the statistics and, for any cotangent, their exact vector–Jacobian product
(VJP). This lets them serve as loss terms for a neural network, a denoiser, a component separation
or a custom synthesis. There is no autograd graph: the gradient comes from the same block-wise
reverse sweep as the command-line tools. The memory is that of the engine, and the results are
those of the validated Rust code.

Three engines in `stl_port.native` (`python/stl_ensemble/stl_port/native.py`), and a network
trained with them:

| Engine | Statistics | Domain | Reports with the definitions and the measurements |
|---|---|---|---|
| `NativeEngine` | scattering covariance (`mean`, `var`, S1–S4), auto or cross `S(a, b)`; E/B statistics of `(Q, U)` with `spin=2` | full sky, or averaged over `K` masks | [masked statistics](../masked_statistics.md), [cross statistics](../disc_cross_statistics.md), [spin 2](../spin_scattering.md) |
| `PatchEngine` | the same plan, auto or cross | disc patches (support radius `R`, averaging core `r`) | [disc patches](../disc_cross_statistics.md) |
| `SpectrumEngine` | pseudo-`C_ℓ` (anafast, `iter=0`): spin 0 (`TT`), spin `s` (`EE`, `BB`, `EB`), auto or cross | full sky, or `K` masks (weights) | [pseudo-spectra](../spectrum.md) |
| `NativeUNet` | a spherical U-Net denoiser and its training by the losses of Delouis et al. (2022), on one device | full sky, or `K` masks | [native U-Net](../native_unet.md) |

Worked examples: the three engines wrapped for torch on a laptop CPU
({doc}`../tutorials/differentiable`, example 06), and the GPU path with CUDA tensors
({doc}`../tutorials/torch_loss`).

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

## The engines

| Engine | Page |
|---|---|
| `NativeEngine`: full sky, `K` masks, cross statistics | {doc}`native-engine` |
| `PatchEngine`: disc patches, CPU threads or GPU, CUDA tensors | {doc}`patch-engine` |
| `SpectrumEngine`: pseudo-`C_ℓ`, spin 0 and spin `s`, masks | {doc}`spectrum-engine` |
| `NativeEngine(spin=2)`: E/B statistics of polarisation `(Q, U)`, auto and cross | {doc}`spin-statistics` |
| `NativeUNet`: U-Net denoiser trained by scattering losses, Adam or L-BFGS | {doc}`native-unet` |

The signatures are in {doc}`../api/python`.

## What is differentiable

| Engine | Output | Gradient with respect to | Constant (no gradient) |
|---|---|---|---|
| `NativeEngine` | statistics `[B, (K,) ncoeff]` | the maps; both maps `a` and `b` with `cross=True` | masks, `var` (moments), the plan |
| `NativeEngine(spin=2)` | E/B statistics `[B, 2, (K,) ncoeff]` | `(Q, U)` (and `a`, `Q`/`U` or `T`, with `cross=True`) | masks (eroded), `var`, the plan |
| `NativeUNet` | `u = d − scale r(d / scale)`, loss terms | the network parameters (`apply(d, cot=...)`, `step`, `lbfgs`) | frozen biases, targets, conditioning maps |
| `PatchEngine` | statistics `[B, ncoeff]` | the patch pixels (both maps with `cross=True`); to the full sky through `gather` | disc centres, radii, `var` |
| `SpectrumEngine` | `C_ℓ` `[B, (K,) (C,) lmax + 1]` | the maps (`Q`, `U` for spin `s`; both fields with `cross=True`) | mask weights, `fsky` |
| `delouis_transform` | `sign(Re)·sqrt|Re| + i·sign(Im)·sqrt|Im|` | its input (`delouis_transform_vjp`, unbounded at 0) | – |

## Backends and typical costs

| Engine | CPU | GPU | Typical cost of forward + VJP (measured) |
|---|---|---|---|
| `NativeEngine` | yes (one thread) | yes | nside 32, `J = 4`: 2.4 s on one laptop core; nside 64, `J = 5`: 7.5 s. T4, nside 256, `J = 4`: 1.78 s (auto), 2.06 s (cross); 5 masks +16 % (auto), +27 % (cross) |
| `PatchEngine` | yes (`threads` host threads) | yes (`forward_device` for CUDA tensors) | 20 laptop threads, nside 256, 16 discs of 10° (5973 pixels), cached geometry: 0.18 s. T4, 240 cross patches on 16 discs: 1.33 s per step, including geometry rebuilds |
| `SpectrumEngine` | yes (`STL_THREADS`) | spin 0 on the device, spin `s` on the host | nside 256, `lmax = 511`, one map: 0.064 s spin 0, 0.24 s spin 2 (20 threads); T4 with 5 masks: 0.19 s spin 0 |
| `NativeEngine(spin=2)` | yes | yes (spin front end on the host) | T4, nside 256, one `(Q, U)` map: forward 0.87 s, VJP 1.24 s |
| `NativeUNet` | yes | yes | T4, one frozen-bias training step: 0.30 s at nside 64 (4 levels), 4.2–6.4 s at nside 256 (6 levels) |

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
