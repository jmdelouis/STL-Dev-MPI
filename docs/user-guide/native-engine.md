# `NativeEngine`: differentiable full-sky statistics in Python

`stl_port.native.NativeEngine` gives Python the forward and the exact vector–Jacobian product
(VJP) of the full-sky scattering-covariance statistics, through the C ABI of `crates/stl-bridge`
(`libstl_bridge.so`). Use it to build a loss of STL statistics in NumPy or PyTorch, for example
to train a network or to drive your own optimiser. The engine runs on the CPU or on one GPU; its
statistics are those of `stl-stats` and `stl-synth`.

It also computes:

- **masked statistics**: every coefficient averaged over each of `K` masks at its native level,
  with the wavelets and moduli of the full sky (galactic cuts, survey footprints);
- **cross statistics** `S(a, b)` of two maps (`cross=True`): the terms with `c1 ≠ c2`, S3 and S4
  in both orders.

```python
from stl_port.native import NativeEngine

eng = NativeEngine(nside=64, j=5, batch=1, backend="cpu")     # or backend="gpu"
values, var = eng.forward(x, moments=True)    # x [batch, npix] NESTED -> complex [batch, ncoeff]
grad = eng.vjp(cotangent)                     # cotangent = dL/dRe + i dL/dIm -> [batch, npix]
```

The gradient returned by `vjp` is that of `Re sum(conj(cotangent) * values)`: for the loss
`sum |S - S*|^2`, the cotangent is `2 (S - S*)`.

---

## Key parameters

| Argument | Default | Description |
|---|---|---|
| `nside`, `j` | required | resolution and number of scales of the plan |
| `batch` | 1 | maps per call (fixed for the engine's life) |
| `backend` | `"cpu"` | `"cpu"` or `"gpu"` (needs a library built with `--features gpu`) |
| `kernel`, `orientations` | 5, 4 | stencil and `L` |
| `block_pixels` | 4096 | CPU block size |
| `threads` | 1 | host threads of the CPU engine |
| `cache`, `cache_bytes` | `"transposition"`, 1 GiB | GPU geometry cache policy and budget |
| `device` | 0 | CUDA device |
| `cross` | `False` | cross statistics of map pairs `[batch, 2, npix]` |
| `masks` | `None` | `K` masks: NESTED id arrays or `[K, npix]` indicators; values become `[batch, K, ncoeff]` |
| `library` | `$STL_NATIVE_LIBRARY` | path of `libstl_bridge.so` |

| Method | Returns |
|---|---|
| `forward(maps, generation=0, member=0, objective=..., moments=False)` | `(values, variances)`: complex `[batch, (K,) ncoeff]`, and the spatial variances of the terms if `moments` |
| `vjp(cotangents, generation=0, ...)` | map gradient shaped like `maps` |
| `forward_cross(a, b)`, `vjp_cross(cot)` | cross statistics of `a`, `b`; gradients `(grad_a, grad_b)` |
| `set_masks(masks)`, `native_pixels()` | change the masks; pixels averaged by each coefficient `[K, ncoeff]` |
| `checkpoint()`, `restore(snapshot)` | save and restore the forward state (several objectives on one engine) |
| `plan`, `ncoeff`, `info()` | the plan (family, scales, orientations, native level) and the engine's description |

Full signatures: {doc}`../api/python`.

---

## Minimal example

```python
import numpy as np
from stl_port.native import NativeEngine

t = np.load("target_nested.npy"); t = (t - t.mean()) / t.std()
eng = NativeEngine(nside=64, j=5, batch=1, backend="cpu", threads=8)
s_target, var = eng.forward(t, moments=True)
w = eng.native_pixels() / var[0]                 # ergodic weights 1 / sigma^2

x = np.random.default_rng(0).standard_normal(t.size)
s, _ = eng.forward(x)
loss = np.sum(w * np.abs(s - s_target) ** 2)
grad = eng.vjp(2 * w * (s - s_target))[0]        # d loss / d x, [npix]

# galactic cuts: the same statistics averaged over |b| > 20 deg and |b| > 45 deg
# (b: the galactic latitude of every NESTED pixel, e.g. from healpix_geo)
eng.set_masks([np.flatnonzero(np.abs(b) > 20), np.flatnonzero(np.abs(b) > 45)])
```

`native_lbfgs(callback, x)` runs the same strong-Wolfe L-BFGS as `stl-synth` on a Python loss.
Walk-throughs: {doc}`../tutorials/map_statistics` and {doc}`../tutorials/torch_loss`.

---

## Pitfalls and costs

- **One forward state.** `vjp` must follow the `forward` of the same `(generation, member,
  objective)`; it refuses a stale state. To interleave several objectives on one engine, use
  `checkpoint` / `restore`. `set_masks` drops the forward state.
- **One caller thread per engine**; the batch size is fixed at construction.
- **Full sky only**: every pixel finite. Masks change the averages, not the fields: level
  `n + 1` of a mask holds the parents of level `n`, and the terms near the mask edge read pixels
  outside it. For statistics that only see a patch, use {doc}`patch-engine`.
- **Cross statistics** have no `mean` or `var` terms; S3 and S4 are not symmetric in `(a, b)` and
  come in both orders. Drop `grad_b` to keep `b` constant (a template, the data).
- **Batches of full-sky maps on the CPU**: the CPU `NativeEngine` runs on one thread; a
  `PatchEngine` with discs of 180° gives the same statistics (to 1e-14) with the maps on parallel
  threads (7 cross pairs at nside 32: 0.7 s against 5.3 s per forward and VJP on a laptop).
- **Weights**: under ergodicity, `var / native_pixels()` estimates the variance of each
  coefficient (for each mask).
- **Cost** (T4, nside 256, `J = 4`, `L = 4`): forward 0.79 s, VJP 0.99 s; five masks add 16 %,
  cross statistics with five masks 47 %. The GPU setup (CUDA context, NVRTC) takes about 15 s once
  per process.
- The module prints its version on import; the native library is checked (SHA-256) and loaded at
  the first engine.

Definitions, validation and timings: {doc}`../masked_statistics`, {doc}`../disc_cross_statistics`
(section 2.2).
