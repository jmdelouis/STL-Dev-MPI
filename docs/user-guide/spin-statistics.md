# Spin-2 statistics: polarisation `(Q, U)` with `NativeEngine(spin=2)`

`NativeEngine(nside, j, spin=2)` computes scattering-covariance statistics of polarisation maps
`(Q, U)` and their exact VJP, on the CPU or on one GPU. The maps are first turned into two
frame-independent scalar fields, the local E and B fields

    chi = chi_E + i chi_B = -(d_x - i d_y)^2 (Q + iU)        (eth-bar squared, lengths in pixels)

and the engine then returns the statistics of `chi_E` and `chi_B`, each with the plan of one
scalar channel. Every step is local: no spherical harmonic transform, so masks and holed data
are supported. In the flat limit `chi = (k h)^2 (E + iB)`: the E and B modes weighted by
`(ℓ / ℓ_j)^2`, of order one in the band of the wavelet of level `j`.

It also computes:

- **masked statistics** (`masks=`, indicators of the valid data), with the masks eroded so that
  the averages only take pixels whose field reads valid data;
- **cross statistics** (`cross=True`) of pairs `(a, b)`, `b` a `(Q, U)` field and `a` either
  another `(Q, U)` field (`spin_a=2`, QU × QU) or a scalar map `T` (`spin_a=0`, T × QU, for
  example the 857 GHz intensity against the 353 GHz polarisation).

```python
from stl_port.native import NativeEngine

eng = NativeEngine(nside=64, j=5, batch=1, spin=2, backend="cpu")    # or backend="gpu"
values, var = eng.forward(qu, moments=True)    # qu [batch, 2, npix] NESTED -> [batch, 2 (E, B), ncoeff]
grad = eng.vjp(cotangent)                      # [batch, 2, ncoeff] -> [batch, 2 (Q, U), npix]
```

`values[:, 0]` holds the statistics of `chi_E`, `values[:, 1]` those of `chi_B`; `eng.plan` is the
one-channel plan, the same for both.

---

## Key parameters

The arguments of {doc}`native-engine` apply; the spin ones are:

| Argument | Default | Description |
|---|---|---|
| `spin` | 0 | 2 for `(Q, U)` maps `[batch, 2, npix]` (HEALPix/COSMO convention) |
| `spin_kernel` | 5 | stencil of the derivative: the pixels within `spin_kernel // 2` rings (3, 5, 7 or 9; 5 is 25 pixels) |
| `spin_pyramid` | `"scalar"` | `"scalar"`: `chi` at level 0 only, the coarser levels its decimations by the engine's scalar pyramid; `"transported"`: `(Q, U)` averaged level by level with parallel transport, the stencil applied at every level |
| `spin_a` | 2 | with `cross=True`: 2 if `a` is a `(Q, U)` field, 0 if it is a scalar map |
| `masks` | `None` | `K` masks of the valid data; eroded by the stencil (and, for the scalar pyramid, through the decimation cones) |

| Method | Returns |
|---|---|
| `forward(qu, moments=False, ...)` | `(values, variances)`: complex `[batch, 2 (E, B), (K,) ncoeff]` |
| `vjp(cotangents, ...)` | gradient `[batch, 2 (Q, U), npix]` |
| `forward_cross(a, b, moments=False)` | `cross=True`: the cross plan of `(chi_E^a, chi_E^b)` and `(chi_B^a, chi_B^b)` (or `(T, chi_E^b)`, `(T, chi_B^b)`), `[batch, 2, (K,) ncoeff]` |
| `vjp_cross(cotangents)` | `(grad_a, grad_b)`, in the shapes of `a` and `b` |
| `spin_fields(qu, level=0)` | `((Q, U), (chi_E, chi_B))` of a level, each `[batch, 2, npix_level]` (scalar pyramid: level 0 only) |
| `native_pixels()` | the eroded mask sizes `[K, ncoeff]`, for the ergodic weights |
| `forward_levels(levels)` | statistics of fields given at every level, without the spin front end (validation) |

Full signatures: {doc}`../api/python`.

---

## Minimal example

E and B statistics of a polarisation map on a galactic cut, a loss and its gradient:

```python
import numpy as np
from stl_port.native import NativeEngine

qu = np.load("planck353_QU_n64_nested.npy")          # [2, npix]
qu = qu / qu.std()                                    # b: galactic latitude of every NESTED pixel
eng = NativeEngine(nside=64, j=5, spin=2, masks=[np.flatnonzero(np.abs(b) > 20)])
s_t, var = eng.forward(qu[None], moments=True)       # [1, 2, 1, ncoeff]
w = eng.native_pixels() / var[0]                     # ergodic weights per E/B, mask, coefficient

x = np.random.default_rng(0).standard_normal(qu.shape)
s, _ = eng.forward(x[None])
loss = np.sum(w * np.abs(s - s_t) ** 2)
grad = eng.vjp(2 * w * (s - s_t))[0]                 # d loss / d (Q, U), [2, npix]

qu0, chi0 = eng.spin_fields(qu[None])                # the local E and B fields of level 0
```

Cross statistics of an intensity template with the polarisation:

```python
x = NativeEngine(nside=64, j=5, spin=2, cross=True, spin_a=0)
values, var = x.forward_cross(t857[None], qu[None], moments=True)   # (T, chi_E), (T, chi_B)
ga, gb = x.vjp_cross(cotangents)                                     # [1, npix], [1, 2, npix]
```

---

## Pitfalls and costs

- **Level 0 sits at the pixel scale.** The stencil is accurate for `k h < 1` (transfer 0.99) and
  degrades towards the pixel scale (0.72 at `k h ≈ 1.5–2`), so S1 and S2 of `j = 0` and the S3/S4
  terms built on the level-0 modulus differ from the exact (SHT) route by 30–75 %. With the default
  scalar pyramid, the terms with `j ≥ 1` agree with the exact route to 0.4–5.5 % (S1, S2) and
  5–14 % (S3, S4 with `j1 > 0`). Weight the level-0 terms accordingly, or leave them out.
- **`"scalar"` is the default** for that reason: every wavelet of level `j` reads `chi` at
  `k h_0 ≈ 1.57 / 2^j`, where the stencil is accurate. `"transported"` applies the stencil at the
  pixel scale of every level (30–70 % errors at every `j`); cross statistics need `"scalar"`.
- **E/B leakage**: a pure-E input gives `S2(B) / S2(E)` of 0.15–3 % at `j = 1..4` (scalar
  pyramid), 1–2 % (transported).
- **Cross statistics** compute E × E and B × B pairs only, not E × B. With `spin_a=0`, `T` is used
  as it is and its two rows (against `chi_E` and `chi_B`) add up in `grad_a`.
- **Masks are eroded**: a coarse level that the erosion empties (a small mask at a coarse level)
  falls back to the parents of the level below. Check `native_pixels()` before weighting.
- **Cost** (T4, nside 256, one `(Q, U)` map = two engine rows): forward 0.87 s, VJP 1.24 s,
  1.8 GB on the device, against 0.81 s and 1.01 s for one scalar map. The spin front end runs on
  the host and stores about 1 kB per pixel of level 0 (0.8 GB at nside 256, 13–18 GB at nside
  1024); building it adds a few seconds to the engine setup.

Definition, stencil, validation against the SHT on nside 64 and Planck 353 GHz, and timings:
{doc}`../spin_scattering`.
