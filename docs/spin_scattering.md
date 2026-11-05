# Spin-2 scattering statistics: local E / B by eth-bar squared

`NativeEngine(nside, j, spin=2)` takes polarisation maps `(Q, U)`
`[batch, 2, 12 nside^2]` (NESTED, HEALPix/COSMO convention) and returns the
scattering-covariance statistics of two frame-independent scalar fields,
`chi_E` and `chi_B`, with the plan of one channel. Everything is local:
no spherical harmonic transform at any step, masks and holed data
supported, CPU and GPU.

## Definition

With `P = Q + iU`, the field of level `j` is

    chi_j = chi_E + i chi_B = -(d_x - i d_y)^2 P        (lengths in pixels h_j = sqrt(4 pi / N_j))

the spin-lowering operator applied twice (eth-bar squared) in the tangent
plane of each pixel, axes `(e_theta, e_phi)`. In the flat limit
`chi = (k h_j)^2 (E + iB)`: the E and B modes weighted by `(l / l_j)^2`,
of order one in the band of the wavelet of level `j`. The exact spherical
counterpart, used for validation, is

    chi_lm = h_j^2 sqrt((l + 2)! / (l - 2)!) (E_lm, B_lm).

**Stencil** (`stl_core::spin::eth_bar2`): for every pixel, the pixels
within `rings = spin_kernel // 2` rings (2 by default, 25 pixels), each
rotated into the centre's frame by parallel transport along the geodesic
(`P -> P e^{-2 i psi}`), with complex weights exact on the polynomials of
degree `2 rings` of the gnomonic coordinates of the actual pixel positions,
of least weighted norm `sum |w|^2 exp(r^2 / rings^2)`. A regular 5x5 grid
read by bilinear interpolation was tried first: interpolation makes an
O(1) relative error on second derivatives (8 % E -> B leakage on smooth
fields, against 0.5 % for the fitted weights). Larger supports (3, 4 rings)
are not better: the error comes from the pixel scale, where neither a local
stencil nor the SHT reference is accurate.

**Pyramid** (`spin_pyramid`):

- `"transported"`: `P_j` is the mean of the four NESTED
  children of `P_{j-1}`, each rotated into the parent's frame by parallel
  transport, and `chi_j` is the stencil applied to `P_j` at level `j`; the
  engine reads the `chi_j` of every level as they are
  (`LocalEngine::reload_levels`, `GpuEngine::load_levels`).
- `"scalar"` (default): `chi_0` only; the coarser levels are its decimations by the
  engine's own scalar pyramid, as for a scalar map. The wavelet of level
  `j` then sees `chi` at `k h_0 ~ 1.57 / 2^j`, where the stencil is
  accurate, instead of `k h_j ~ 1.57` at every level (see the
  validation; chosen as the default).

**Gradient** (pyramidal): the engine's reverse sweep gives the cotangents of
the maps of every level; `transported`: each goes through the transposed
stencil, then down the transposed transported means to level 0;
`scalar`: the engine chains them to level 0 (`map_cotangent`) and the
transposed stencil of level 0 follows.

**Masks** (`masks=`, indicators of the valid data): eroded at every level,
so that the averages only take pixels whose field reads valid data:
`transported`: a parent is valid when its four children are, and a stencil
row when every pixel it reads is; `scalar`: level 0 eroded by the stencil,
then a coarse pixel is dropped when the decimation cone of its children
meets a dropped pixel. The wavelets still read the zeroed data within their
halo near an edge (as for scalar maps). A coarse level that the erosion empties (a small
mask, e.g. |b| > 45 deg at nside 2-4) falls back to the parents of the
level below, the rule of the scalar masks (a parent kept when one of its
children is).

## API (Python, `stl_port.native`)

```python
eng = NativeEngine(nside, j, batch=B, backend="gpu", spin=2,
                   spin_kernel=5, spin_pyramid="scalar", masks=None)
values, var = eng.forward(qu, moments=True)   # qu [B, 2, npix] -> [B, 2 (E, B), (K,) ncoeff]
grad = eng.vjp(cotangents)                     # [B, 2, (K,) ncoeff] -> [B, 2 (Q, U), npix]
qu_j, chi_j = eng.spin_fields(qu, level)       # each [B, 2, npix_level] (transported: every level; scalar: level 0)
eng.forward_levels([...])                      # statistics of maps given per level (validation)
```

`eng.plan` is the one-channel plan (`family`, `c1 = c2 = 0`, scales,
orientations), the same for E and B: index `[:, 0]` is E, `[:, 1]` is B.
`native_pixels()` gives the eroded mask sizes. C ABI: `stlp_new` with
`"spin": 2`, `"spin_kernel"`, `"spin_pyramid"` (`"scalar"` by default),
`"spin_a"`; `stlp_spin_fields`, `stlp_forward_levels`; ABI version
`2026-11-08-v11`.

**Cross statistics** (scalar pyramid only), pairs `(a, b)` with `b` a
`(Q, U)` field and `a` another one (`spin_a=2`) or a scalar map `T`
(`spin_a=0`, e.g. the 857 GHz intensity, used as it is):

```python
x = NativeEngine(nside, j, batch=B, spin=2, cross=True, spin_a=0, masks=...)
values, var = x.forward_cross(a, b, moments=True)   # a [B, npix] (spin_a=0) or [B, 2, npix]; b [B, 2, npix]
ga, gb = x.vjp_cross(cotangents)                    # shapes of a and b
```

`values[:, 0]` is the cross plan of `(chi_E^a, chi_E^b)` (or `(T,
chi_E^b)`), `values[:, 1]` that of `(chi_B^a, chi_B^b)` (or `(T,
chi_B^b)`), `[B, 2, (K,) ncoeff]`, with the plan of the scalar cross
engine (`c1 != c2`; `x.plan == NativeEngine(..., cross=True).plan`). E x B
terms are not computed. The engine runs the scalar cross plan on the pairs
`(chi_k^a, chi_k^b)`, `k = E, B`, as two batch rows per pair; the transposed
stencil maps the cotangents back to `(Q, U)` (and the two rows of `T` add
up).

## Validation (nside 64, J = 5, L = 4; `python/stl_ensemble/validate_spin.py`)

Inputs: two Gaussian realisations with `C_l^E = C_l^B ~ l^-2.4`, one pure-E
realisation, and the Planck 353 GHz `(Q, U)` at nside 64. Reference: the
exact operator by SHT (`healpy.map2alm(pol=True, iter=3)`) on the same
`P_j` (transported) or on `P_0` (scalar), then the same engine on these
exact fields. Logs and JSON: `docs/runs/spin_n64/`.

**Fields, against the exact operator, by multipole** (level 0, dust):
transfer `<chi, ref> / <ref, ref>` 0.99 for `k h < 1`, 0.89 at
`1-1.5`, 0.72 at `1.5-2`, 0.38 at `2-3`; pure-E leakage
`|chi_B| / |chi_E|` 3-5 % per band for `k h < 2` (the SHT reference
itself, `iter = 5`, is 12 % from the truth over the whole band). Smooth
pure-E field (`l < 24` at nside 32): `chi_E` within 1.3 %, leakage 0.5 %.
Over the whole band the field error is ~60 %, all at the pixel scale.

**Statistics, relative error against the exact route** (E; B alike;
dust, Planck 353 in brackets):

| | transported | scalar |
|---|---|---|
| S1, j = 0 | 0.32 (0.32) | 0.32 (0.32) |
| S2, j = 0 | 0.54 (0.53) | 0.54 (0.53) |
| S1, j = 1 / 2 / 3 / 4 | 0.31 / 0.31 / 0.33 / 0.44 | 0.027 / 0.004 / 0.010 / 0.027 (0.029 / 0.005 / 0.005 / 0.013) |
| S2, j = 1 / 2 / 3 / 4 | 0.52 / 0.53 / 0.55 / 0.68 | 0.054 / 0.009 / 0.024 / 0.055 (0.059 / 0.004 / 0.003 / 0.021) |
| S3, j1 = 0 / j1 > 0 | 0.75 / 0.58 | 0.65 / 0.14 (0.53 / 0.07) |
| S4, j1 = 0 / j1 > 0 | 0.54 / 0.59 | 0.54 / 0.055 (0.52 / 0.06) |

Level 0 sits at the pixel scale in both modes (the wavelet peaks at
`k h ~ 1.57`), and the S3/S4 terms built on the level-0 modulus inherit
it. Orientations behave alike (S1, scalar, dust, j = 1: 0.026-0.029 over
the four orientations). Pure-E input, `S2(B) / S2(E)`: scalar 0.15 / 0.32 /
1.2 / 3.1 % at j = 1..4 (0.96 % at j = 0); transported 1-2 %.

**Locality**: a 15-degree hole (zeroed data). `chi` changes only outside the
eroded masks (1113 of 49152 pixels at level 0; the eroded mask keeps
48039). Masked statistics, holed against full data, same eroded masks
(the remaining change is the wavelet halo at the edge): transported S1
0.01-0.2 %, S2 0.02-0.2 %, S3 2-4 %, S4 0.2-3 %; scalar (masks eroded
through the decimation cones: 48039 / 11821 / 2833 / 611 / 86 pixels kept
at levels 0-4): S1 and S2 below 0.023 % (0.007 % for j >= 1), S3 0.02-3.7 %,
S4 0.007-0.17 %.

**Gradient**: finite differences of `Re sum conj(c) S` along a random
direction, nside 16, relative error 2.3e-6 (transported) and 1.2e-6
(scalar), CPU and GPU. GPU against CPU: statistics equal to the printed 14
digits; `tests/test_spin.py` checks 1e-12.

**Cross statistics** (scalar pyramid, against the scalar cross engine on
the exact `chi_0`, E; `docs/runs/spin_n64/val_n64_scalar_cross_*.json`):
Planck 857 GHz `T` x Planck 353 `(Q, U)`: S1 j = 0..4 0.021 / 0.005 /
0.006 / 0.017 / 0.064, S2 0.26 / 0.021 / 0.004 / 0.017 / 0.079, S3 / S4
with j1 > 0 0.038 / 0.030; two correlated dust `(Q, U)` fields: S1 j = 1..4
0.027 / 0.005 / 0.009 / 0.022, S2 0.054 / 0.008 / 0.020 / 0.041. (A
synthetic `T` weakly correlated with `chi_E` gives larger relative errors
at j = 3, 4, 20-30 %, on cross coefficients close to zero.) GPU equals CPU
to the printed 14 digits; `tests/test_spin.py` checks 1e-12, the
finite-difference vjp of both kinds of pairs, and the identity with the
scalar cross engine on the same `chi` (1e-12). The galactic cuts |b| > 0,
20, 30, 45 deg at nside 64 (J = 5) keep 49152 / 30464 / 22528 / 12320
pixels at level 0 and 192 / 24 / 8 / 24 at level 4 (the last two after the
fallback).

## Timings (T4, garoupe; one `(Q, U)` map = two engine rows)

| nside | engine | forward | vjp | device peak | host operators |
|---|---|---|---|---|---|
| 64 | scalar (one map) | 0.037 s | 0.036 s | 0.48 GB | |
| 64 | spin 2, transported | 0.027 s | 0.055 s | 0.49 GB | 70 MB |
| 64 | spin 2, scalar | 0.023 s | 0.052 s | 0.48 GB | 50 MB |
| 256 | scalar (one map) | 0.81 s | 1.01 s | 1.63 GB | |
| 256 | spin 2, transported | 0.86 s | 1.25 s | 1.79 GB | 1.12 GB |
| 256 | spin 2, scalar | 0.87 s | 1.24 s | 1.83 GB | 0.80 GB |
| 256 | spin 2 cross, 2 pairs (QU x QU), forward with moments | 1.64 s | 2.93 s | 2.63 GB | 0.80 GB |
| 256 | spin 2 cross, 2 pairs (T x QU), forward with moments | 1.53 s | 2.92 s | 2.67 GB | 0.80 GB |

Building the engine and the operators takes 26 s (scalar pyramid) to 28 s
(transported) at nside 256, against 23 s for the scalar engine alone.
The front end runs on the host (complex sparse rows, both directions
stored, ~1 kB per pixel of level 0): 13-18 GB at nside 1024. Moving it to
the device, with the weights recomputed per block, is the next step for
large nside.
