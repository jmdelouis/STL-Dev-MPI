# Terminology

The vocabulary is that of STL-Dev, so that both code bases read the same.

---

## HEALPix, NESTED, level

HEALPix divides the sphere into `12 nside²` equal-area pixels; `nside` is a power of 2. In the
**NESTED** ordering, the 4 children of a pixel at `2 nside` have consecutive indices
`4p .. 4p + 3`, so a contiguous range of indices is a compact patch of the sky. Every map read or
written by the tools is a full-sky NESTED map (`.npy`, 1-D, or a HEALPix FITS binary table).

A **level** `d` of the pyramid is the map at `nside / 2^d`; level 0 is the input resolution.

## Pyramid

`maps[d] = D^d x`: the input map `x` and its anti-aliased decimations, one per level. `D` averages
the fine pixels within 3 fine-pixel sizes of each coarse centre with Gaussian weights. On 2D grids
it is the raised-cosine kernel of STL-Dev with stride 2.

## Wavelet, orientation, scale

`psi`: one complex directional wavelet with `L` orientations (`L = 4` by default), a
`kernel_size × kernel_size` stencil (5) in the tangent plane, the **same kernel at every level**.
The scale `j` is the level at which it is applied: `z_j = psi * maps[dg(j)]`, `dg(j) = j` by default.
`J` is the number of scales (default `log2(nside) − 1`).

## Coefficient families

| Family | Definition (one channel) |
|---|---|
| `mean`, `var` | `<x>`, `<x²>` of the standardised map |
| S1 | `<|z_j|>` per scale and orientation |
| S2 | `<|z_j|²>` |
| S3 | `<(psi * m[d1][n])_{l1,l3} conj(z_n,l3)>`, `j1 ≤ j3` |
| S4 | `<(psi * m[d1][n])_{l1,l3} conj((psi * m[d2][n])_{l2,l3})>`, `j1 ≤ j2 ≤ j3` |

Every coefficient is a pixel mean at its **native level** `n = dg(j3)`. `--fewer`
(`has_fewer_convolutions`) uses the cheaper STL-Dev definitions of S3 and S4.

## Plan

The ordered list of coefficients (family, scales, orientations, native level), as
`PyramidScattering.build_plan` builds it. `stl-stats` writes it next to the values.

## Stored moduli

`m[d][n] = D^(n−d) |z_d|`: the modulus of the wavelet field of level `d`, decimated to level
`n > d`. They are the operands of S3 and S4 at coarser levels and are kept for the whole pyramid
(about `4N/9 × L` values); their cotangents `gm[d][n]` are accumulated in the reverse sweep.

## Block, halo, domain

A **block** is a contiguous range of NESTED cells of one level (`--block-pixels`, default 16384),
the unit of work. Its **halo** is the ring of cells its computation reads: the stencil reach for
the convolution (`(kernel_size − 1) / 2 + 2` pixels), the Gaussian cone for the decimation. The
**domain** is block plus halo. Results are exact on the block; nothing outside the domain is
touched.

## Piece

The contribution of one fine block to a stored modulus,
`D^(n−d)(|z_d| · 1_block)`, pushed through the decimation chain restricted to its support.
The stored moduli are the sums of the pieces; the gradient pulls back through the transposed pieces.

## Sweep

One pass over the levels. The **upward** (forward) sweep goes from level 0 to the coarsest:
wavelet fields, partial sums of the statistics, pieces of the stored moduli. The **downward**
(reverse) sweep goes back from the coarsest level, applies the cotangents of the coefficients and
accumulates the cotangent of every level's map, which are finally chained to level 0 through the
transposed decimations.

## Explicit VJP

The gradient is not obtained by automatic differentiation: every operation has a hand-written
vector–Jacobian product (convolution adjoint, decimation transpose, modulus derivative), applied
block by block in the reverse sweep. Nothing of the forward pass needs to be kept except the
pyramid and the stored moduli.

## Rank, owned cells, ghosts

Under MPI, rank `r` owns a contiguous NESTED range of level 0 and the parents of its cells at
every level, so its whole pyramid is aligned. **Held** cells are the owned cells plus the halo
received from the neighbours before each sweep. Pieces and transposed pieces that fall on cells
owned by a neighbour are accumulated in local **ghost** buffers and summed by the owner
(additive ghost-cell pattern).

## Ergodic weights

`--weights ergodic` divides each squared residual by `σ_i²`, the spatial variance of the
coefficient's terms on the target divided by the number of pixels of its level: the loss becomes
a reduced χ², and a residual of 1 is the scatter expected between two realisations.

## Ensemble, member, group

`--ensemble B` synthesises `B` maps (members) whose **mean** statistics match the target;
`--scatter-weight` also holds their scatter at the ergodic variance. `--groups G` splits the
ranks into `G` groups that each hold `B / G` members. `stl-ensemble` implements P. Campeti's
method, where both the mean and the variance over the members match those of a set of reference
maps.
