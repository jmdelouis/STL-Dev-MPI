# Concepts: the block-wise pyramidal gradient

This page explains how STL-Dev-MPI computes the statistics and their gradient with a memory bounded
by the share of the sky of each process. The exact formulas are in the repository's `CLAUDE.md`
(section 3), the vocabulary in [Terminology](../terminology.md).

## The objective

For a map `x` at level 0, the statistics `S(x)` are pixel means of products of wavelet fields
computed on the pyramid `maps[d] = D^d x`. A synthesis minimises

$$
\mathcal{L}(x) = \sum_i w_i \, |S_i(x) - S_i(\text{target})|^2
$$

with L-BFGS (strong Wolfe line search, transcribed line by line from `torch.optim.LBFGS`), the
map itself being the variable. Extra terms can be added (ensemble scatter, level power, histogram,
Minkowski functionals, statistics of a transformed map; see {doc}`stl-synth` and {doc}`../api/cli/stl-synth`).

## The pyramid

```
level 0   x              nside N      z_0 = psi * x
level 1   D x            N/2          z_1 = psi * D x        m[0][1] = D |z_0|
level 2   D² x           N/4          z_2 = psi * D² x       m[0][2] = D² |z_0|,  m[1][2] = D |z_1|
...
```

The wavelet kernel is the same at every level, so the scale `j` is just the level where it is
applied. S1 and S2 at scale `j` only need `z_j`. S3 and S4 at scale `j3` need the second wavelet
of the moduli of the finer levels, `psi * m[d][n]` with `n = dg(j3)`. These **stored moduli**
`m[d][n]` are the only cross-level state: about `4N/9 × L` real values for the full pyramid,
against `4N/3` for the maps.

## Blocks and halos

Every operation is local on the sphere: the convolution reads a 5 × 5 stencil, the decimation a
Gaussian cone of three fine pixels. So a level is processed by **blocks** of constant size
(`--block-pixels`, default 16384 cells), each with a **halo** wide enough for the operations it
chains:

| Operation | Halo (pixels of the level) |
|---|---|
| convolution | `(kernel_size − 1) / 2 + 2` |
| decimation | `ceil(radius / pixel) + 1`, doubled for pieces and transposes |

The geometry (pixel centres, stencil rotation, bilinear interpolation weights, decimation
weights) is **computed on the fly for each block and never stored for the full sky**: the Python
HEALPix library STL-Dev used needed 3.2 kB per pixel, which is impossible at nside 2048. Inside
a block the results are exact to the rounding: the fixture tests compare with STL-Dev's full-sky
operators to 1e-12.

The stored moduli are built from **pieces**: the modulus of one fine block, decimated level by
level on its support, then summed at the coarse level,

$$
m[d][n] = \sum_k D^{n-d}\left(|z_d| \cdot 1_{F_k}\right).
$$

## Sweeps and the explicit VJP

**Upward sweep** (levels 0 to D). For every block of level `n`: `z_n` on the block and its halo;
the second wavelets of `|z_n|` and of the stored `m[d][n]`; the partial sums of every coefficient
whose native level is `n`; then the piece of `|z_n|` pushed to the coarser levels.

**Downward sweep** (levels D to 0). For every block: recompute the block's fields, apply the
cotangents `2 w_i (S_i − S_i^*)` of its coefficients, accumulate the cotangents `gm[d][n]` of the
stored moduli (complete once the coarser levels are done), pull back the block's own chain
through the transposed pieces, then the modulus derivative and the wavelet adjoint, which give
the cotangent of the map of level `n`.

**Chain to level 0.** The map cotangents of all levels are brought back to level 0 through the
transposed decimations.

Every step has a hand-written vector–Jacobian product, so no autograd graph is kept: the memory
of a process is the pyramid, the stored moduli, their cotangents, the L-BFGS history (all
proportional to its share of the sky) and the transients of one block.

## Distributed memory

```
level 0 :  |── rank 0 ──|── rank 1 ──|── rank 2 ──|── rank 3 ──|     contiguous NESTED ranges
level 1 :  |─ r0 ─|─ r1 ─|─ r2 ─|─ r3 ─|                             parents of the owned cells
...        halos received before each sweep; pieces on neighbours' cells summed by their owner
```

Each rank owns a contiguous range of NESTED cells at level 0 (a compact patch) and the parents of
those cells at every coarser level, so `maps[d]`, `m[d][n]` and `gm[d][n]` are sliced the same
way. Before each sweep, ranks exchange halos point to point; pieces and transposed pieces that
fall on a neighbour's cells go to local ghost buffers, summed by the owner after the level
(owner first, then ranks in order, so the result does not depend on timing). The coefficient
partial sums are reduced in rank order: **1, 2, 4, ... ranks give the same statistics to
1e-15**, and the loss values of a synthesis agree to 16 digits across rank and node counts.

On GPUs, each rank drives one device with the same decomposition; the blocks are sized from the
free device memory, and the fields can be kept between the sweeps or recomputed per block
([GPU guide](gpu.md)).

## Planar images

The 2D engine (`stl_core::planar`, `stl-synth2d`) is the same algorithm on a periodic grid:
square tiles instead of NESTED blocks, the 5 × 5 kernels and the stride-2 raised-cosine
decimation of STL-Dev's `STL_2D_Kernel_Torch`, and the same statistics to 1e-10. See
[the planar report](../planar_2d.md).

## Memory budget

At nside 2048 (50 M pixels), `L = 4`, history 10, on `P` ranks: maps and their cotangents
~0.5 GB / P, stored moduli and their cotangents ~1.4 GB / P, L-BFGS history ~10 GB / P, block
transients ~0.1 GB per rank. On a GPU, the level fields cost 188 bytes per pixel and member
with `--gpu-fields keep` and 60 with `recompute` ([GPU report](../gpu.md), section 2.7).
