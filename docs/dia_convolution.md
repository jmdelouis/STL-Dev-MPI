# DIA (diagonal) form of the wavelet convolution

Can the GPU wavelet convolution (`crates/stl-gpu`, a generic sparse gather
with a sorted transposition for the adjoint) be written as a "diagonal"
operator, `y[i] = sum_d a_d[i] x[i + off_d]`, in face-raster coordinates,
and what does it save, with the operator unchanged (same results to
rounding)? The idea comes from a discussion of the user with NVIDIA
(earth2grid / DLWP-HEALPix order the pixels of a face as an image).

Code: `stl_core::dia` (frames, offsets, CPU reference), `stl_gpu::dia`
and the `dia_*` kernels of `kernels.cu`, `examples/dia_offsets.rs`
(section 2), tests `crates/stl-core/tests/dia.rs`,
`crates/stl-gpu/tests/dia.rs`, benchmark `bench_kernels::bench_dia`.

## 1. The operator today

For output row `k` (a pixel), gauge `g` and output channel `o` (re, im):

    y[g, o, k] = sum_p ker[o, p] sum_n w[g, n, p, k] x[pos[g, n, p, k]]

over the `P = 25` taps of the rotated 5 x 5 stencil and the 4 cells of
the ring bilinear interpolation of each tap: `4 G P` entries per row (400
at `L = 4`), each a position (4 bytes) and a weight (8 bytes). The adjoint
is a deterministic gather from a transposition sorted by grid cell (4 + 8
more bytes per entry): 9608 bytes per row at `L = 4`, or 1608 when only
the transposition is kept and the weights are recomputed at every sweep
(`--gpu-cache transposition`, the production setting at nside >= 1024).

## 2. Feasibility: the cells of a row as offsets in face coordinates

A face is an `nside x nside` image in `(x, y)` (NESTED bits
de-interleaved). The frame of a face is extended across its 4 edges and 4
corners with the neighbour tables of the HEALPix C++ library
(`nb_facearray`, `nb_swaparray`: across an edge the frames differ by a
rotation of a multiple of 90 degrees): `ext_pix(face, x + dx, y + dy)` is
the pixel at an offset, `rel_offset(row, cell)` the offset of a cell seen
from a row. At the 8 vertices where only 3 faces meet, the quadrant
between the two other faces is empty (`ext_pix` returns nothing), but
every pair of faces there shares an edge, so every cell still has an
offset. `ext_pix` reproduces the 8 cdshealpix neighbours of every pixel at
nside 2 to 16, and `ext_pix(rel_offset)` is the identity on `[-6, 6]^2`
(unit tests).

`examples/dia_offsets.rs` takes every row of the full-sky geometry
(`ConvGeometry::rows`, kernel 5, `L = 4`, gauge cosmo) and every entry
with a nonzero weight, and expresses its cell as `(dx, dy)` from the row
(raw outputs in `benchmarks/dia/`):

| nside | rows | offsets used (all gauges) | per gauge | cells per (row, gauge) | cells per row (4 gauges) | max-norm |
|---|---|---|---|---|---|---|
| 8 | 768 | 81 | 73, 71, 73, 71 | 38.4 (29..44) | 43.6 (37..48) | 5 |
| 16 | 3 072 | 81 | 77, 67, 77, 67 | 38.8 (29..43) | 44.0 (37..47) | 5 |
| 64 | 49 152 | 81 | 77, 67, 77, 67 | 39.0 (29..45) | 44.1 (37..49) | 5 |
| 256 | 786 432 | 81 | 77, 67, 77, 67 | 39.0 (29..45) | 44.1 (37..49) | 5 |
| 1024 | 12 582 912 | 81 | 77, 67, 77, 67 | 39.0 (29..45) | 44.1 (37..49) | 5 |

- **Every** read has an offset, all within `max(|dx|, |dy|) <= 5`: the
  window is the 11 x 11 square. The set of offsets used is the same 81 at
  every nside from 16 to 1024 (nside 8 reads 4 others: 85 together; `L =
  8` reads 93). Seen from the cell, the row is at an offset of the same
  max-norm (0 exceptions in 5e9 reads at nside 1024), which the adjoint
  needs.
- Merging the entries that read the same cell leaves 39 cells per (row,
  gauge) instead of 100 entries (2.6 x fewer), 44 per row over the 4
  gauges.
- The offsets used, per mille of the rows (nside 1024; `x + y` is the
  ring index, the band is wider along the ring, `|dx - dy| <= 7`, than
  across, `|dx + dy| <= 6`):

```
        -5   -4   -3   -2   -1    0    1    2    3    4    5
    5    .    .    .   32    .    0    .    .    .    .    .
    4    .    .   14  148  361  258   34    0    .    .    .
    3    .   14  434  885  937  904  771  282    0    .    .
    2   32  148  885 1000 1000 1000 1000  872  282    0    .
    1    .  361  937 1000 1000 1000 1000 1000  771   34    .
    0    0  258  904 1000 1000 1000 1000 1000  904  258    0
   -1    .   34  771 1000 1000 1000 1000 1000  937  361    .
   -2    .    0  282  872 1000 1000 1000 1000  885  148   32
   -3    .    .    0  282  771  904  937  885  434   14    .
   -4    .    .    .    0   34  258  361  148   14    .    .
   -5    .    .    .    .    .    0    .   32    .    .    .
```

Rows that leave their face (the cells they read are on other faces):

| nside | interior | edge (one other face) | corner of 4 faces | corner of 3 faces |
|---|---|---|---|---|
| 64 | 80.2 % | 18.6 % | 272 rows | 352 rows |
| 256 | 94.8 % | 5.1 % | 360 rows | 352 rows |
| 1024 | 98.7 % | 1.3 % | 360 rows | 352 rows |

With the extended frames, none of them needs an index table: the edge
and corner rows read their cells through `ext_pix` like the others, and
no read falls outside the window. Index tables would be needed only with
a padded face-image layout whose corner quadrants at the 3-face vertices
have no pixel (712 rows at any nside), which the extended frames avoid.

**Conclusion: the operator is exactly a DIA operator on 81 (85 with nside
8) diagonals of the extended face frames, with no exception list.** The
diagonals are `Diagonals::observed(&[8, 16, 32], kernel, L, gauge)` (the
offsets read at those nsides); the device build counts the reads off the
diagonals (`misses`, 0 in every test and benchmark), so a configuration
outside this set would be detected, not silently wrong. DIA needs `nside
> 2 W = 10`; coarser levels keep the current geometry.

## 3. The DIA operator

Coefficients, one per (diagonal, row, gauge, channel):

    coef[d, k, g, o] = sum_p sum_n [cell(n, p) on diagonal d] ker[g, o, p] w[g, n, p, k]

over the bound weights of `geom_build` (same interpolation, binding and
renormalisation; incomplete rows included), weights equal to 0 skipped.
Forward and adjoint:

    y[g, o, k] = sum_d coef[d, k, g, o] x[cell(k, d)]
    xbar[q]    = sum_{v in [-5, 5]^2} sum_{g, o} coef[d(r, q), r, g, o] ybar[g, o, r],  r = ext_pix(q + v)

The cell of a row on a diagonal is computed (`ext_pix`, then the grid
position: direct in the rank's own range, a binary search for halo
cells), so no position is stored, and the adjoint needs neither a
transposition nor a sort: a cell finds its rows at the opposite offsets
(the same face: `d(r, q)` is the diagonal of `-v`; across an edge it is
computed from both pixels).

Storage. A plain DIA layout keeps all `D` diagonals of every row,
`coef[((d K + k) G + g) 2 + o]` (`2 G D` doubles per row: 5440 bytes at
`L = 4`); but a row uses only ~44 of the 85 diagonals over its 4 gauges
(37 to 49), and both the bytes and the per-diagonal work (cell lookup,
input loads) of the forward follow `D`. The kept layout is therefore a
**masked DIA**: a mask per row (`3 x u32`, the diagonals that receive a
term) and the coefficients of those diagonals only, packed
`coef[((s K + k) G + g) 2 + o]` for the `s`-th used diagonal of row `k`
(`S` slots, the most diagonals a row of the block uses: 46 to 49). The
forward walks a row's mask; the adjoint skips the diagonals a candidate
row does not use and finds a slot by popcount. The first build goes
through the dense layout (a transient), later rebuilds reuse the masks
and accumulate into the slots directly. Since a skipped diagonal has
zero coefficients for every gauge, the masked forward equals the plain
DIA one to the bit (up to the sign of zeros).

Summation orders (documented in `stl_gpu::dia`): coefficient = taps then
bilinear neighbours in order; forward = diagonals in increasing order;
adjoint = per row of the window, gauges and channels in order, then the
window rows by 4 lane-strided partial sums combined by a fixed shuffle
tree. `--fmad=false` as for every kernel. The results differ from
`conv_fwd` / `conv_adj` only by these orders (the kernel is folded into
the coefficients before the data are read).

Exactness (all to rounding, deterministic):

| Test | Result |
|---|---|
| CPU DIA vs the 66 STL-Dev `TorchConv` fixtures (nside 8 to 1024, kernels 3 / 5, `L` 2 / 4, gauges cosmo / phi, complete and incomplete rows) | forward <= 0.45 x the rounding-noise bound (the CPU convolution's own figure), <= 0.011 x from the CPU convolution; adjoint <= 1.9e-15 relative from `convolution_adjoint` |
| device DIA vs `conv_fwd` / `conv_adj` (9 blocks, nside 32 to 1024: the north pole, two 3-face vertices, the faces 2 / 3 junction, kernels 3 / 5, `L` 2 / 4; rows `a1` and the whole domain; the block's own rows only) | forward 0.000 x noise; adjoint <= 7e-16 relative |
| device DIA vs the CPU convolution and the CPU DIA | forward <= 0.008 x noise; adjoint as `conv_adj` vs the CPU (<= 4.8e-12 relative near the pole at nside 1024, where the noise bound is 3e-8) |
| coefficients stripped then recomputed (from the kept masks, into the slots) | identical to the bit |
| `cargo test --release -p stl-core -p stl-gpu` (fixtures, multi-rank, synthesis; the engine still uses the bound stencils) | all pass |
| `geom_build` after sharing its rotation and binding with `dia_build` | positions, weights and transposition identical to the bit (hashes, 4 blocks) |

## 4. Device kernels

| Kernel | Parallelism | Notes |
|---|---|---|
| `dia_reach` | thread per (row, diagonal) | flags the grid cells reached; compacted on the host into `dest` (kept: 4 bytes per cell) |
| `dia_build` | thread per (gauge, row) | `geom_build`'s rotation and tap binding (shared code), each weight folded into its diagonal: dense layout and row masks (first build), or into the slots of the kept masks (rebuild) |
| `dia_slots`, `dia_pack` | thread per row | slot count, then the used diagonals packed (first build only) |
| `dia_fwd` | thread per (row, 4 gauges, 4 input rows) | walks the row's mask; a diagonal's cell and inputs serve all gauges; coefficient loads coalesced |
| `dia_gin_t` | thread per element | cotangents copied row by row (transient of the cotangents' size) |
| `dia_adj` | 4 lanes per (cell, 4 input rows) | `STL_DIA_LANES` (1, 2, 4, 8, 32 tried: 2 and 4 best; 1 lane is 5 x slower at 16 input rows) |

Kept per row at `L = 4` (`L = 2`), bytes:

| | bound stencils + transposition | plain DIA | masked DIA |
|---|---|---|---|
| everything | 9608 (4808) | 5444 (2468) | 3152 (1456) |
| between sweeps (`strip`) | 1608 (808): the transposition | 4: the cells reached | 16: masks and cells reached |
| transient of a build | ~11 200 (5 600): entries, transposition, sort indices | the coefficients | + the dense coefficients, 5440 (2464), first build only |

## 5. Benchmark

`bench_kernels::bench_dia`: one block of 262 144 owned rows from pixel 0
(274 576 rows with the convolution halo, 287 296 grid cells; the rank's
own range is the block, so halo cells go through the binary search),
kernel 5, gauge cosmo, mean of 3 runs after a warm-up. "Old" is the
current geometry (`DeviceGeometry::build`, `refresh`, `conv_fwd`,
`conv_adj`); the input rows are the batch x channels (x `L` for the
second wavelets) of the engine. Every run checks old against DIA
(forward <= 2.5e-16, adjoint <= 4.1e-16 relative). Raw outputs in
`benchmarks/dia/`.

### 5.1 Tesla T4 (garoupe), plain DIA (all 85 diagonals stored)

ms, old / DIA (ratio > 1: DIA faster). nside 256, 1024 and 2048 give the
same times within 3 % (same block size); nside 1024 shown:

| kernel | `L = 4` old | `L = 4` DIA | ratio | `L = 2` old | `L = 2` DIA | ratio |
|---|---|---|---|---|---|---|
| build | 361.7 | 136.7 | 2.64 | 178.0 | 65.7 | 2.71 |
| weights / coefficients recomputed | 108.9 | 104.7 | 1.04 | 53.2 | 50.1 | 1.06 |
| forward, 1 input row | 12.3 | 14.3 | 0.86 | 6.0 | 12.3 | 0.49 |
| forward, 4 input rows | 12.7 | 15.0 | 0.85 | 6.3 | 12.8 | 0.49 |
| forward, 16 input rows | 54.1 | 56.7 | 0.95 | 27.0 | 50.0 | 0.54 |
| adjoint, 1 input row | 26.4 | 17.7 | 1.49 | 14.9 | 9.4 | 1.58 |
| adjoint, 4 input rows | 27.7 | 24.4 | 1.14 | 15.6 | 11.1 | 1.41 |
| adjoint, 16 input rows | 112.9 | 98.2 | 1.15 | 63.2 | 44.9 | 1.41 |

The plain DIA forward is slower: its cost follows the 85 (77) diagonals
of every row, half of them unused, whatever `L`, hence the masked layout.

### 5.2 Masked DIA

Tesla T4 (garoupe), same block as 5.1, ms; nside 256 and 2048 within 3 %
of nside 1024 (raw outputs `benchmarks/dia/t4_masked/`):

| kernel | `L = 4` old | `L = 4` masked DIA | ratio | `L = 2` old | `L = 2` masked DIA | ratio |
|---|---|---|---|---|---|---|
| build | 362.5 | 191.2 | 1.90 | 178.5 | 89.3 | 2.00 |
| weights / coefficients recomputed | 108.9 | 105.8 | 1.03 | 53.0 | 51.7 | 1.02 |
| forward, 1 input row | 12.1 | 7.5 | 1.60 | 6.0 | 6.8 | 0.89 |
| forward, 4 input rows | 12.8 | 8.2 | 1.57 | 6.3 | 7.2 | 0.88 |
| forward, 16 input rows | 54.5 | 31.2 | 1.75 | 27.1 | 27.7 | 0.98 |
| adjoint, 1 input row | 26.3 | 12.9 | 2.03 | 14.9 | 6.1 | 2.46 |
| adjoint, 4 input rows | 27.8 | 15.7 | 1.77 | 15.7 | 7.1 | 2.23 |
| adjoint, 16 input rows | 112.9 | 65.8 | 1.72 | 63.2 | 28.5 | 2.22 |
| kept, everything (MB) | 2638 | 866 | 3.05 | 1320 | 400 | 3.30 |
| kept between sweeps (MB) | 442 | 4.4 | 100 | 222 | 4.4 | 50 |

On the T4 the coefficient rebuild costs what the weight refresh costs
(both are the FP64 interpolation of `4 G P` taps per row, at 1/32 of the
FP32 rate); everything else is faster, and the first build no longer
sorts.

RTX A1000 laptop (WSL, sm_86, 1/64 FP64), nside 128, the whole sky as
one block (196 608 rows, no halo), ms:

| kernel | `L = 4` old | `L = 4` masked DIA | ratio | `L = 2` old | `L = 2` masked DIA | ratio |
|---|---|---|---|---|---|---|
| build | 307.3 | 224.7 | 1.37 | 152.6 | 103.7 | 1.47 |
| weights / coefficients recomputed | 154.8 | 126.8 | 1.22 | 73.4 | 63.4 | 1.16 |
| forward, 1 input row | 15.8 | 10.0 | 1.58 | 8.0 | 8.6 | 0.93 |
| forward, 4 input rows | 16.7 | 10.4 | 1.61 | 8.3 | 8.9 | 0.94 |
| forward, 16 input rows | 67.5 | 41.1 | 1.64 | 34.0 | 33.9 | 1.00 |
| adjoint, 1 input row | 34.3 | 16.3 | 2.11 | 19.6 | 7.8 | 2.51 |
| adjoint, 4 input rows | 35.9 | 19.8 | 1.81 | 20.4 | 9.1 | 2.23 |
| adjoint, 16 input rows | 143.4 | 81.7 | 1.76 | 81.3 | 38.5 | 2.11 |
| kept, everything (MB) | 1889 | 620 | 3.05 | 945 | 286 | 3.30 |
| kept between sweeps (MB) | 316 | 3.1 | 100 | 159 | 3.1 | 51 |

On the same laptop the plain DIA gave: build 1.91, refresh 1.17, forward
0.85-0.92, adjoint 0.99-1.09 (one warp per cell) at `L = 4`. Where the
masked first build loses against the plain one (225 against 162 ms) is
the dense transient and the packing; rebuilds from the kept masks do not
pay it.

### 5.3 Upper bound: a constant kernel on face images (not the operator)

`face_conv_f64` / `face_conv_f32`: the same 5 x 5 complex kernel at every
pixel of a face image, no rotation, no interpolation, zero outside the
face (orientations tied to the face axes: a different operator), same
262 144 output rows, T4, `L = 4`:

| input rows | current `conv_fwd` | masked DIA forward (exact operator, FP64) | constant kernel FP64 | constant kernel FP32 |
|---|---|---|---|---|
| 1 | 12.1 ms | 7.5 ms | 1.02 ms | 0.12 ms |
| 4 | 12.8 ms | 8.2 ms | 3.95 ms | 0.46 ms |
| 16 | 54.5 ms | 31.2 ms | 15.7 ms | 1.80 ms |

This is where "very large speedups" come from: the coefficients are the
same for every row (registers or shared memory) and FP32 runs 32 x faster
than FP64 on a T4 (a naive kernel; cuDNN would do better). The DIA layout
alone cannot give that to our operator, whose coefficients differ from
row to row (rotated stencil, gauge, ring interpolation): with 2 G
coefficients per used diagonal per row, the exact convolution reads ~44
x 2 G doubles per row whatever the format, and the masked DIA is close to
that minimum (it stores no positions and no transposition).

## 6. Recommendation

**Replace the bound stencils and their transposition by the masked DIA
geometry** for the levels with `nside >= 16` (the window needs `nside > 2
W = 10`; the coarser levels are tiny and keep the current code), first
behind an option of `stl-synth --gpu`, then by default once the engine
tests and one A100 measurement agree. It is the same operator to
rounding, and it dominates the current geometry on every axis measured:

- **Memory.** Everything kept: 3152 instead of 9608 bytes per row at `L =
  4` (3 x); between sweeps, 16 instead of 1608 bytes per row (100 x: the
  transposition cache disappears). The build transient is also smaller
  (the dense coefficients, 5440 bytes per row, instead of ~11 200 for the
  entries, transposition and sort indices), so device blocks can be
  larger.
- **Time.** No transposition and no sort: the first build is 1.9 x
  faster (T4; 1.4 x on the laptop); rebuilds from the kept masks cost
  what today's weight refresh costs (1.03 x on the T4, 1.2 x on the
  laptop); forward 1.6-1.75 x and adjoint 1.7-2.0 x faster at `L = 4`
  (T4, same on the laptop). At `L = 2` the forward is 0.9-1.0 x and the
  adjoint 2.0-2.5 x.
- **Determinism and exactness** are those of the current kernels (fixed
  summation orders, `--fmad=false`; differences from `conv_fwd` /
  `conv_adj` at 1e-16), validated against the STL-Dev fixtures through
  the CPU reference.

Measured in the engine (`stl-synth --gpu --gpu-conv dia`, one T4, `L =
4`, `gpu.md` §5.0): nside 256, 1.246 -> 0.505 s per evaluation and peak
9.8 -> 4.2 GB (the whole DIA geometry fits, 3.5 GB); nside 1024, 61.4 ->
22.9 s per evaluation (2.7 x) with the same cache budget, loss values
equal to 2e-15.

What it should change on the A100 (estimates from the per-row figures
above, to be measured):

- A100, nside 1024, 10 members (`--kappa-table`, 252 260-row blocks):
  today a 28.4 GB transposition cache (1608 bytes x 17.7 M rows) and the
  weights of every block recomputed at every sweep, peak 62.8 GB. With
  the masked DIA the masks and cells of all rows take 0.3 GB, and the ~40
  GB then free keep the whole coefficients of ~70 % of the rows (3152
  bytes per row), so ~70 % of the per-sweep stencil work disappears and
  the convolutions run ~1.7 x faster.
- A100, nside 2048 (the fields leave ~10 GB): today the transposition of
  ~70 M rows (112 GB) cannot be kept, so almost every block is rebuilt
  with its transposition at every sweep (362 ms per 262 144-row block on
  the T4, 3.3 x a weight refresh). With the masked DIA all masks and
  cells fit (1.1 GB) and every sweep only recomputes coefficients (106
  ms): about 3.4 x less geometry time per sweep, convolutions ~1.7 x
  faster, and no cache to size.

Integration (done, behind `--gpu-conv dia` / `STL_GPU_CONV=dia`; the
default stays `stencils`): `GpuEngine::take_geometry` / `put_geometry`
build a `DiaGeometry` on the levels with nside > 10 (`OwnRows` from the
topology's block), the four call sites (`z` forward on `a1`, second
wavelets on the block's rows, their adjoint through `dmap`, the `z`
adjoint into the map cotangent) go through `forward` / `adjoint`; the
cache policies are `full` = coefficients kept, `transposition` = masks
and cells kept (16 bytes per row) with whole geometries in the rest of
the budget, `none` = rebuilt; the per-row budget of `GpuEngine::new`
counts the DIA transients (dense coefficients at the first build, the
row-ordered cotangents of `dia_gin_t`). The cells reached are compacted
on the device (`flag_count`, `flag_fill`: only the chunk counts reach the
host). The engine's results differ from the bound stencils by summation
orders only (`stats::dia_matches_stencils`: statistics <= 5e-17,
gradients <= 5e-15 relative), so synthesis trajectories from white noise
may part after a few L-BFGS iterations, as with 3 GPUs (§5 of `gpu.md`).

## 7. Redundancy between rows, and index widths

The DIA removes the redundancy within a row (100 entries per gauge for
~39 distinct cells, positions shared by the gauges). Between rows,
`examples/dia_redundancy.rs` compares the coefficients of the rows of a
ring within a face (nside 64, `L = 4`):

| gauge | rows of an equatorial ring (>= 6 rings from the caps) | distinct rows (to 1e-12) |
|---|---|---|
| cosmo (production) | differ by up to 0.5 x the largest coefficient | 49 152 of 49 152 |
| phi | equal to 1e-13 on all 932 (ring, face) groups | 9 068 of 49 152 |

With the gauge cosmo the stencil makes the angle `-phi + g pi / L` with
the local meridian (regular at the poles), so it turns along a ring and
every row has its own coefficients: the masked DIA is then close to the
minimum the exact operator needs. With the gauge phi the angle is `g pi /
L`, the equatorial rings are invariant under the rotation by one pixel,
and their coefficients could be stored once per ring (2/3 of the sky,
then a constant kernel per ring: the regime of NVIDIA's speed-ups). That
is a different operator (other statistics), not pursued here.

Index widths: positions are `u32` on purpose (half the bytes of `u64`).
At nside 2048 on one rank the largest indices are the entries of the
bound stencils (`4 G P K` <= 8.4e8 for `K` <= 2^21 rows per block, the
reason for that cap), the decimation nonzeros of a level (~3.8e8) and the
grid positions (5.0e7), all below 2^32 = 4.3e9; the DIA indexes its
coefficients in 64 bits and has no entry indices. The decimation row
pointers would overflow at nside 8192 on one rank (~6e9 nonzeros):
`topology` now stops with a message there instead of wrapping.

Possible refinements: per-gauge masks (39 instead of 44 slots per row at
`L = 4`, ~10 % fewer bytes); 32-bit Morton arithmetic for the in-face
cell of a diagonal (the `L = 2` forward is bound by that integer work, not
by bytes); the `dest` compaction on the device instead of the host.
