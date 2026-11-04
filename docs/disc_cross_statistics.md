# Disc patches and cross statistics in `stl-bridge`

Two additions to the Python bridge (`stl_port.native`, C ABI of `crates/stl-bridge`) for
training networks whose loss is made of STL statistics (Healpix-STL-IA): the statistics of a map
known only on a **disc patch** of the sphere, and the **cross statistics** `S(a, b)` of two maps,
both with exact gradients. Branch `claude/disc-cross-stats`, bridge version `2026-11-04-v3`.

## 1. Definitions

### 1.1 Disc patch

- **Support.** Level 0 holds every NESTED pixel of `nside` whose centre lies within the angular
  distance `R` of the disc centre (a true disc, not NESTED cells). Level `d + 1` holds the parents
  of level `d`: a parent is kept when at least one of its children is in the disc.
- **Pyramid.** A kept parent is the masked mean of the fine pixels present, with the weights of
  the engine's own decimation: the smooth decimation (Gaussian weights of the fine pixels within
  3 fine pixels of the parent centre) restricted to the support and l1-renormalised over the
  pixels present (`smooth_decimation` on the support). It is the full-sky decimation wherever the
  parent's whole cone is inside the disc. A plain mean of the four children would differ from the
  full-sky pyramid everywhere, and the disc of radius 180° would no longer give the full-sky
  statistics (section 3).
- **Wavelet.** The convolution of every level is bound to the support of that level with
  renormalised bilinear weights, as `bind_support` / `TorchConv` do for a partial grid; a
  stencil row whose reads are all inside the support is the full-sky row.
- **Averages.** Every coefficient is a pixel mean at its native level over the averaging pixels
  of that level: the whole support, or with a core radius `r < R` the support pixels whose centre
  lies within `r` (`core_0`) and their parents at the coarser levels (`core_{d+1}` =
  parents of `core_d`, never empty). The ring `R − r` then absorbs the edge effects of the
  convolutions. The spatial variances returned with `moments=True` are taken over the same
  pixels, and `native_pixels` gives their number for the ergodic weights `V / N`.
- **Formulas.** Same families, plan and term formulas as the full-sky engine (the term sums and
  their cotangents are shared code, `stl_core::engine::term_sums` / `operand_cotangents`).
  The whole patch is one domain: no blocks, no halos (nothing outside the disc is read).

### 1.2 Cross statistics

`S(a, b)` are the cross terms the engine already computes for two channels, `c1 = a`, `c2 = b`:

| Family | Cross term (pixel mean at the native level) |
|---|---|
| S1 | `< a_z / sqrt(|a_z| + 1e-8) · conj(b_z / sqrt(|b_z| + 1e-8)) >`, `a_z = psi * a` |
| S2 | `< a_z · conj(b_z) >` |
| S3 | `< [psi * D^(n-d1)|a_z|](l1, l3) · conj(b_z(l3)) >`, then the same with `a` and `b` swapped |
| S4 | `< [psi * m_a](l1, l3) · conj([psi * m_b](l2, l3)) >`, then swapped |

There are no cross `mean` and `var`. With `a = b` S2, S3 and S4 equal the auto statistics (to
rounding) and S1 equals the auto S1 to about `1e-8` absolute (the `1e-8` of the root
normalisation). The Delouis et al. (2022, A&A 668, A122, Sect. 2.2) form
`sign(Re) sqrt|Re| + i sign(Im) sqrt|Im|` is available as a post-transform
(`delouis_transform`, `delouis_transform_vjp`; its derivative is unbounded at 0).

The VJP returns the cotangents of both maps; to keep `b` gradient-free, drop its cotangent.

## 2. Python API (`stl_port.native`)

`STL_NATIVE_LIBRARY` points to `libstl_bridge.so` (`cargo build --release -p stl-bridge`,
`--features gpu` for the GPU full-sky engine). Arrays are `float64` / `complex128`; radii are in
**degrees**; a cotangent is `dL/dRe + i dL/dIm` (the gradient is that of
`Re sum(conj(cotangent) * values)`).

### 2.1 `PatchEngine`

```python
from stl_port.native import PatchEngine

pe = PatchEngine(nside, j, kernel=5, orientations=4, cross=False, threads=0, cache=256)
# threads=0: STL_THREADS, else every CPU.  cache: disc geometries kept (key: centre, R, r).

ids = pe.select(centres, radius, core=None)
#   centres: NESTED pixel indices [B] at nside (int array) or vectors [B, 3] (float array);
#   radius, core: degrees (core < radius, or None: average on the whole disc; radius >= 180: full sky).
#   ids: list of B sorted int64 arrays, the NESTED pixels of each disc (sizes differ by disc).
#   A repeated identical selection is free; a cached disc costs < 0.1 ms.

values, var = pe.forward(maps, centres=None, radius=None, core=None, moments=True, generation=0)
#   maps: list of B arrays [n_i] (cross=True: [2, n_i]), maps[i] = x[ids[i]] in that order;
#         or one ndarray [B, n] when all discs have n pixels.  centres/radius/core: select() first.
#   values: complex128 [B, ncoeff]; var: float64 [B, ncoeff] spatial variances of the terms
#   over the averaging pixels (empty [B, 0] if moments=False).
pe.native_pixels      # float64 [B, ncoeff]: pixels each coefficient averages (ergodic sigma^2 = var / N)
pe.plan, pe.ncoeff    # the plan: identical to NativeEngine(nside, j[, cross]).plan

grads = pe.vjp(cotangents, generation=0)
#   cotangents: complex128 [B, ncoeff]; grads: same structure as maps (list of [n_i] or ndarray).

pe.gather(x)          # list of patches x[ids[i]] from full-sky maps x [npix] or [B, npix] ([B, 2, npix])

# cross=True
values, var = pe.forward_cross(a, b, centres, radius, core)   # a, b: lists of [n_i] (or [B, n])
ga, gb = pe.vjp_cross(cotangents)
```

`forward` keeps one forward state (the last call); `vjp` must follow with the same
`generation`. The patches of a batch run in parallel on host threads.

### 2.2 Full-sky cross statistics

```python
from stl_port.native import NativeEngine

ce = NativeEngine(nside, j, batch=B, backend="gpu", cross=True)     # or backend="cpu"
values, var = ce.forward_cross(a, b, generation=g, moments=True)    # a, b: [B, npix]; values [B, ncoeff]
ga, gb = ce.vjp_cross(cot, generation=g)                            # each [B, npix]
# equivalently ce.forward(np.stack([a, b], 1)) and ce.vjp(cot) -> [B, 2, npix]
```

### 2.3 Torch use (sketch for `stlia/stl_torch.py`)

```python
class PatchStatsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, engine, generation):          # x: [B, n] patches (all discs the same size) or use lists
        values, _ = engine.forward(x.detach().cpu().numpy(), generation=generation, moments=False)
        ctx.engine, ctx.generation, ctx.device = engine, generation, x.device
        return torch.from_numpy(values.view(np.float64).reshape(values.shape + (2,)).copy()).to(x.device)

    @staticmethod
    def backward(ctx, g):
        g = g.detach().cpu().numpy()
        grad = ctx.engine.vjp(g[..., 0] + 1j * g[..., 1], generation=ctx.generation)
        return torch.as_tensor(np.asarray(grad), device=ctx.device), None, None
```

Disc sizes vary with the centre (pixel centres within `R`), so patches of a batch are lists in
general; gather them from the network output with `u[ids_i]` (a torch index), concatenate for
the call, and split the gradient back. For the three losses of Delouis et al. (2022):
`S(d1, d2)` vs `<S(u + n1, u + n2)>`, `S(d, u)` vs `<S(u + n, u)>` and `S(T, d)` vs
`<S(T, u + n)>`, use a cross engine with the pair in the order written (S3/S4 are not symmetric
in `a`, `b`; both orders are in the plan).

## 3. Validation

| Test | Where | Result |
|---|---|---|
| Disc pixels = brute-force centres within `R` | `stl-core/tests/disc.rs` | equal |
| Disc of radius 180° = `LocalEngine` (values, variances, gradient), C = 1 and 2, both S3/S4 definitions | `stl-core/tests/disc.rs` | 1e-11 abs / 1e-10 rel |
| Disc VJP vs central finite differences (R = 50°, r = 35°, nside 16, J 3, L 4), auto and cross, both definitions | `stl-core/tests/disc.rs` | 1e-6 rel |
| Cross of `(a, a)` = auto (disc) | `stl-core/tests/disc.rs` | S2–S4 1e-12, S1 1e-7 |
| C ABI: patch handle at 180° = full-sky handle (auto and cross), geometry cache, stale state refused | `stl-bridge/tests/bridge.rs` | 1e-10 |
| Cross handle on the GPU vs CPU (both convolution forms and field stores) | `stl-bridge/tests/bridge.rs` (`--features gpu`) | see section 4 |
| Python: 180° = full sky, FD of disc / cross (both maps, full sky and patch), `S(a, a)` = auto, Delouis transform VJP | `python/stl_ensemble/tests/test_patch_cross.py` | pass |

The existing fixture tests (`stl-core`, `stl-bridge`) pass unchanged after the term formulas
were factored out of `engine.rs`.

## 4. Speed

Laptop (WSL, 20 threads), `R = 10°`, `r = 7°`, `J = log2(nside) − 1`, `L = 4`, random centres
(`python/stl_ensemble/tests/bench_patch.py`):

| nside | pixels / disc | B | new geometry | cached geometry | forward | VJP |
|---|---|---|---|---|---|---|
| 64 | 374 | 16 | 0.058 s | 0.2 ms | 0.011 s | 0.008 s |
| 64 | 373 | 64 | 0.18 s | 0.4 ms | 0.032 s | 0.024 s |
| 128 | 1493 | 16 | 0.15 s | 0.2 ms | 0.026 s | 0.028 s |
| 256 | 5973 | 16 | 0.59 s | 0.3 ms | 0.086 s | 0.096 s |
| 256 | 5974 | 64 | 2.1 s | 1 ms | 0.27 s | 0.31 s |

The geometry of a new disc is dominated by the binding of the convolution (21 µs per row and
thread: 4 gauges × 25 taps of the library-exact bilinear interpolation); the support and the
decimations are 10 % (`cargo run --release -p stl-core --example disc_profile NSIDE R`). Draw centres from a finite set (e.g. the NESTED pixels of a coarse nside)
to reuse cached geometries (a disc of 10° costs about 2.4 MB at nside 64, 38 MB at nside 256).
Everything runs on the CPU; the device geometry of `stl-gpu` is the route if patches at nside
≥ 512 become necessary.
