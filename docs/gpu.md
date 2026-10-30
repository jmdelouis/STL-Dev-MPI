# GPU engine (`crates/stl-gpu`)

The statistics and gradients of `stl_core::engine::LocalEngine` on CUDA
devices, one device per MPI rank, with the same decomposition, halo and
ghost exchanges, distributed objective, L-BFGS and command line as the CPU
engine (`stl-synth --gpu`). Everything stays `f64` and is validated against
the STL-Dev fixtures with the CPU tolerances.

## 1. Hardware it was built and measured on

| Machine | Devices | FP64 | Driver / toolkit | MPI |
|---|---|---|---|---|
| WSL laptop (development) | 1 x RTX A1000 Laptop, 6 GB, sm_86 | 1/64 of FP32 | 553 (CUDA 12.4), NVRTC 12.8 | OpenMPI 4.1.6 (threads used) |
| garoupe (`ssh garoupe`, through datarmor) | 3 x Tesla T4, 15 GB, sm_75, PCIe gen3 x16, no NVLink | 1/32 of FP32, ~0.25 TFLOPS each | 470 (CUDA 11.4); toolkit 10.0 only, so NVRTC 11.8 from PyPI | OpenMPI 4.1.2, not CUDA-aware; no NCCL |

garoupe's host is 2 x Xeon Bronze 3204 (12 cores at 1.9 GHz, no
hyper-threading) with 62 GB of RAM: the host-side setup is slow there.
The T4 is a weak FP64 device (an A100 has ~40 x its FP64 throughput, an
H100 ~130 x): the engine is FP64-bound in the geometry kernel and
latency-bound in the gathers, so data-centre GPUs should do much better
than the timings below.

## 2. Design

### 2.1 One rank, one device

`stl_mpi::synthesis::Objective` takes an `Evaluator`
(`stl_core::engine::Evaluator`): the CPU sweeps by default,
`stl_gpu::GpuEvaluator` with `--gpu`. A rank picks its device from
`STL_GPU_DEVICE`, else its node-local rank (`OMPI_COMM_WORLD_LOCAL_RANK`,
`MPI_LOCALRANKID`, `SLURM_LOCALID`) modulo the visible devices; several
thread ranks may share one device (tests). The halo and ghost exchanges
are those of `stl_mpi::HaloExchange`, on fields copied to the host (MPI is
not CUDA-aware on garoupe); with one rank (`Exchange::trivial`) nothing
leaves the device.

The kernels (`src/kernels.cu`) are compiled at start-up by NVRTC to a cubin
for the device (`sm_XY`, loads on any driver of the major version, where
PTX would need a driver as recent as NVRTC) with `--fmad=false` and IEEE
division and square root: every expression is evaluated in the order of
its Rust counterpart, so the differences with the CPU engine come only from
the transcendental functions (CUDA libm, 1-2 ulp from glibc) and from the
summation orders listed in §2.6. cudarc loads `libcuda` and `libnvrtc` at
run time: the crate builds without a CUDA toolkit.

### 2.2 Fields and blocks

Every level's fields live on the device over the rank's layout (owned
cells + halo, the `Partition` of `stl-mpi`): maps, stored moduli
`m[d][n]`, and in the reverse sweep their cotangents `gm[d][n]` and the map
cotangents. The wavelet fields `z_n` and moduli `|z_n|` are either kept
over the layout from the forward sweep for the reverse one
(`--gpu-fields keep`), or recomputed by every block of the reverse sweep
from the level's map, as the CPU engine does (`--gpu-fields recompute`:
the forward sweep then holds `|z_n|` of the level it is on only, for its
decimations to the coarser levels); `auto`, the default, keeps them when
that takes at most half of the free device memory (§2.7). Both give the
same values to the bit. A level is processed by **device blocks**: contiguous
ranges of owned cells (up to 2^20, sized from the free memory), each with
its convolution rows `a1` = block + one convolution halo, as on the CPU but
hundreds of times larger. The layouts do not depend on the block size
(dilations distribute over unions), so the device blocks read the same
halo as the CPU blocks.

Per block, forward: `z` on `a1` (one convolution reading the level's map),
`|z|`, the second wavelets on the block for the levels its coefficients
need, the coefficient partial sums. Reverse: with `recompute`, `z` and
`|z|` on `a1` again (the second wavelet of `|z_n|` reads the block's own
copy through the map from grid cells to `a1` rows that its adjoint already
uses); the coefficient cotangents (accumulated per target in plan order),
the adjoints of the second wavelets into the modulus cotangents on `a1`,
the modulus VJP, the stored-moduli cotangents, the chain from the coarser
levels (the last transposed decimation applied to the block's own rows),
the wavelet adjoint into the map cotangent. A row of `z` or `|z|` is the
same whichever block computes it (its geometry and summation order do not
depend on the block), hence the identity of `keep` and `recompute`.

### 2.3 Convolution geometry on the device

`geom_build` computes, per output row and gauge, the rotated stencil, the
ring bilinear interpolation of its taps and the binding on the grid (port
of `ConvGeometry::rows`, same operation order). Entries are stored
`((g * 4 + n) * P + p) * K + k` (rows fastest: coalesced reads). The
transposition groups the entries by the grid cell they read (count, fill,
per-cell bitonic sort in the CPU scatter order: gauge, neighbour, row,
tap), each packed as `row << 11 | (g * 4 + n) << 6 | tap` with its
weight, so that the adjoint is a gather with a fixed order.

The geometry depends on the decomposition only. `--gpu-cache` keeps it
between sweeps and evaluations within a budget measured on the device:

| Policy | Kept per entry (`4 G P` entries per row) | Per sweep |
|---|---|---|
| `full` | 24 bytes (positions, weights, transposition) | nothing |
| `transposition` | 4 bytes | the weights (stencil kernel + gather) |
| `none` | 0 | stencils and transposition |

`auto` (default) takes `full` if everything fits, else `transposition`
for the blocks that fit. At `L = 2`, kernel 5, a row costs 4.8 kB in
`full` and 0.8 kB in `transposition`.

**DIA form** (the default; `--gpu-conv stencils` or
`STL_GPU_CONV=stencils` selects the bound stencils above;
`docs/dia_convolution.md`). The same operator written on the diagonals of
the extended face frames: every read of a row is an offset of max-norm
<= 5 in the frame of its face (81 offsets at nside >= 16, 85 with nside 8),
so no position is stored; the kernel and the bound bilinear weights are
folded into per-row coefficients, kept for the ~44 diagonals a row uses
(a mask per row and packed slots), and the adjoint gathers each cell from
the rows at the opposite offsets, without transposition or sort. It is
used on the levels with nside > 10 (the coarser ones keep the bound
stencils). The cache policies keep, per row at `L = 4`:

| Policy | Bound stencils | DIA |
|---|---|---|
| `full` | 9608 bytes | ~3150 bytes (coefficients, mask, cells reached) |
| `transposition` | 1608 bytes (transposition; weights recomputed per sweep) | 16 bytes (mask and cells reached; coefficients recomputed per sweep into their slots) |

with whole geometries in the rest of the budget in both cases. Results
differ from the bound stencils by summation orders only (<= 5.3e-15
relative on the fixtures, `stats::dia_matches_stencils`). It is the
default since its measurements on the T4 and the A100 (§5.0: 1.6-2.7 x
less time per evaluation); runs before it used the bound stencils, so
their trajectories from white noise may part from new ones after a few
L-BFGS iterations (summation orders, as with 3 GPUs in §5).

### 2.4 Kernels

| Kernel | Parallelism | Order of the sums |
|---|---|---|
| `conv_fwd` | thread per (row, gauge, 4 input rows) | CPU order (taps, then neighbours) |
| `conv_adj` | warp per (grid cell, 4 input rows) | lane-strided partial sums, fixed shuffle tree |
| `dia_fwd` (DIA) | thread per (row, 4 gauges, 4 input rows) | the row's diagonals in increasing order (coefficients: taps, then neighbours) |
| `dia_adj` (DIA) | 4 lanes per (grid cell, 4 input rows) | per window offset, gauges and channels in order; lane-strided offsets, fixed shuffle tree |
| `csr_apply` (decimation, transpose) | thread per (row, input row) | CPU order (columns increasing) |
| `coef_partial` | block per (coefficient, 2^16-pixel chunk) | fixed tree in a chunk, chunks then blocks then levels increasing |
| `coef_reverse` | thread per (target row, pixel) | plan order |

### 2.5 Decimations

The CPU engine pushes every block's modulus through decimation *pieces*
restricted to its support and sums them. On the device the decimations
use the full-sky rows (`stl_core::decimation::decimation_row`, the row
`smooth_decimation` builds on any grid holding its candidates, to the bit)
of the rank's owned coarse cells: `maps[d+1] = D maps[d]`, `m[d][d+1] = D
|z_d|`, `m[d][n] = D m[d][n-1]`, each after a halo exchange of the finer
field. The reverse chain of a level, `sum_{n2 > n} (D^T)^(n2-n) gm[n][n2]`,
is accumulated from the coarsest level by Horner's scheme through the
transposed rows of the owned fine cells (halo exchange between steps).
These are the values of the pieces (exact rows), summed in another order.

### 2.6 Determinism and summation orders

Every result is identical from run to run (no floating-point atomics). It
differs from the CPU engine in the last bits through: the transcendental
functions of the geometry; the decimation chain (whole rows instead of
pieces) and its Horner reverse; the convolution adjoint (lane-strided);
the coefficient partial sums (chunked tree); block boundaries.

### 2.6b The objective's terms on the device (`--gpu-terms device`)

With one rank per group of members (the case of every single-device
run), `stl-synth --gpu` evaluates the whole objective on the device
(`stl-cli/src/device_terms.rs`, kernels in `stl_gpu::terms`), not only
the statistics' sweeps:

- the synthesis sweep reads the members' maps from the device and leaves
  its gradient there (`GpuEvaluator::forward_device`, `gradient_device`,
  `take_map`);
- the level power and the kurtosis per level build the 4-children pyramid
  of the maps (or of kappa) on the device, its moments by chunks, and add
  their gradients there;
- the pixel marginal sorts every member on the device (a stable LSD radix
  sort on the `f64::total_cmp` order of the values, the pixel index as
  payload: the order of the host's sort by value then pixel) and adds
  `coef (x - t)` at the sorted positions;
- the kappa transform (PCHIP of the table) and its derivative are
  computed there, and the kappa sweep and kurtosis read kappa from the
  device.

The host keeps the losses on the statistics, the reductions of the
terms' sums over pixels (small vectors, with the communicators) and
L-BFGS. Every element-wise expression is the host's, in its order, so
the pyramid, the gradients of the level power and kurtosis, the marginal
and the kappa transform are identical to the bit; the sums over pixels
go by chunks of 65 536 cells reduced in a fixed tree and added in chunk
order (the loss and gradient agree with the host path to 2e-16 relative,
§3). During a sweep nothing but the engine's fields is on the device: the
synthesis gradient waits on the host during the kappa sweep, the maps go
up again for kappa's derivative afterwards, and the sort's buffers live
between the sweeps; the engine's memory budget, hence its cache, is the
host path's. Transfers go through a page-locked buffer (maps up twice,
gradient down twice and up once per evaluation with the kappa
statistics). `--gpu-terms host`, several ranks per group, `--minkowski-weight`
or `--freeze-nside` keep the host path.

L-BFGS (`stl_core::lbfgs`) runs its vector operations (dot products,
axpy, the line points, copies) on the host threads by chunks of 65 536
elements; a dot product adds its chunks' sums in chunk order, independent
of the number of threads (`STL_THREADS`, else the available CPUs), and a
slice of one chunk is summed sequentially as before.

### 2.7 Memory per rank

**One engine per rank.** A `GpuEngine` evaluates any plan and batch on
its decomposition, and `GpuEvaluator::share` makes evaluators of other
plans and batches on the same engine. `stl-synth` builds one engine per
rank, sized for the members of its group, and every objective evaluates
on it: the target statistics (batch 1), the synthesis, and with
`--kappa-table` the target statistics and the synthesis term of
`kappa = f(g)`. They share the decimation rows, the blocks' convolution
rows, the geometry cache and the level fields: an evaluation (forward
then gradient) of one objective ends before the next starts, so the
fields of one are free for the other (a change of batch reallocates
them). The kappa term therefore costs no device memory, and its sweeps
reuse the cached geometry.

Level fields (`B` members, `C` channels, `L` orientations, `H_n` held
cells of level `n`), doubles per held cell of level `n`:

| Field | Doubles | Plans | Held |
|---|---|---|---|
| maps and map cotangents | `2 B C` | all | maps always, cotangents in the reverse sweep |
| stored moduli `m[d][n]` and cotangents `gm[d][n]` | `2 n B C L` | with S3 or S4 | `m` always, `gm` in the reverse sweep |
| `z_n` (complex) | `2 B C L` | all | `keep` only |
| `\|z_n\|` | `B C L` | with S3 or S4 | `keep`; `recompute`: level `n` while the forward sweep is on it |
| chain of the reverse sweep | `B C L` on levels 1 and 2 | with S3 or S4 | transient |

A plan without S3 or S4 (the default kappa families `mean,var,S1,S2`)
reads neither `|z_n|` nor the stored moduli: its sweeps neither compute
them (no modulus decimation chain, no reverse chain) nor allocate them.
Over the pyramid (`H_n ~ 4^-n H_0`, `sum H_n = 4/3 H_0`, `sum n H_n =
4/9 H_0`) the full plan needs at its peak (the reverse sweep)

- `keep`: `8 B C H_0 (8/3 + L (4 + 8/9 + 5/16))` bytes, 188 bytes per
  pixel and member at `L = 4`: 2.4 GB per member at nside 1024, 9.5 GB at
  nside 2048 (one rank);
- `recompute`: `8 B C H_0 (8/3 + L (8/9 + 5/16))` bytes, 60 bytes per pixel
  and member at `L = 4` (the forward sweep peaks a little lower, `8 B C H_0
  (4/3 + 13 L / 9)`, with `|z_0|`): 0.75 GB per member at nside 1024, 3.0
  GB at nside 2048, a third of `keep`.

Until the reverse chain was applied per block it was held on all the owned
cells of a level, `B C L H_0` more doubles at level 0, which made `keep`
220 bytes per pixel and member. The decimation rows and block topology
add about 200 bytes per held cell (independent of `B`), the transients of
one block (its geometry while built, second wavelets and their
cotangents, with `recompute` its `z` and `|z|`) are sized by the block,
which is sized from the free memory. The report gives the field estimate
(`gpu.field_bytes_estimate`) and the decimation rows on the device
(`gpu.decimation_rows_bytes`).
The geometry cache then takes 4/5 of what is free after all that
(`STL_GPU_CACHE_MB` overrides it): the device peak in the report
(`gpu.memory_peak_bytes_per_rank`, whole device, CUDA context included)
is mostly the cache, which grows to what is left, and
`gpu.geometry_cache_used_bytes_per_rank` gives its final size. How many
members fit is set by the fields and the decimation rows.

Until c15e8ec `--kappa-table` built its own engines (one per objective
and batch), each sized from the device memory free when it was built,
before any sweep had allocated anything: the second engine duplicated
the fields, rows and topology, and the two caches together claimed 8/5
of the free memory. A 10-member ensemble at nside 1024 on one H200 ran
out of device memory before its first evaluation unless
`STL_GPU_CACHE_MB` capped both caches (§5.1).

## 3. Validation

All tests run on both devices above with identical results.

| Test (`cargo test --release -p stl-gpu`) | Result |
|---|---|
| `geometry`: device geometry vs `ConvGeometry` (nside 8 to 1024, kernels 3 and 5, `L` 2 and 4, complete and incomplete rows) | positions identical; weights <= 0.07 x the rounding-noise bound of the CPU tests; convolution forward <= 0.01 x that bound, adjoint <= 5e-13 relative; weights recomputed from the kept transposition identical to the bit |
| `stats`: the 8 STL-Dev fixtures (nside 16/32, `J` 3/4, `L` 2/4, kernels 3/5, 1-2 channels, batch 1-2, both S3/S4 definitions, shared resolutions), 1 to 333 device blocks per level | statistics <= 7e-6 x tolerance (1e-11 + 1e-10 relative); stored moduli <= 1e-12; map and band cotangents within tolerance (<= 1e-5 x) |
| `stats`: device vs CPU engine, same inputs | statistics <= 6e-16, spatial variances <= 3e-15, level-0 gradient <= 9e-13 relative to its maximum |
| `multirank`: 1, 2, 4 thread ranks sharing a device, small blocks, misaligned ranges (3, 5, 7 ranks), 52 ranks (some without coarse cells) | <= 2.2e-16 from one rank; vs CPU engine <= 4e-17 |
| `stats`: geometry cache policies (`full`, `transposition`, `none`, small budgets), two evaluations, both geometry forms | identical to the bit |
| `stats`: the fixtures and the CPU engine with the DIA geometry (`--gpu-conv dia`); DIA engine vs bound-stencil engine | same tolerances as above; statistics <= 5.3e-17, gradients <= 5.3e-15 relative from the bound stencils |
| `dia`: device DIA vs `conv_fwd` / `conv_adj` and the CPU (9 blocks, poles, 3-face vertices) | forward 0.000 x noise, adjoint <= 7e-16 relative (`docs/dia_convolution.md`) |
| the whole `stl-gpu` suite with the default (DIA) and with `STL_GPU_CONV=stencils` | all pass |
| `stats`: one engine shared by the full plan and a mean / var / S1 / S2 plan at batches 1 and 2 (target statistics, then alternated evaluations, as `stl-synth --kappa-table`) vs one engine each; the plan without S3 / S4 vs the full plan restricted to it (laptop) | identical to the bit (values, variances, gradients) |
| `stats`: wavelet fields recomputed per block vs kept (`--gpu-fields`), the 8 fixtures, both geometry forms, 1 to 333 device blocks per level, the full plan and a mean / var / S1 / S2 plan, three evaluations (laptop) | identical to the bit (values, variances, stored moduli, map cotangents of every level) |
| the whole `stl-gpu` suite with `STL_GPU_FIELDS=keep` and `=recompute` (laptop) | all pass |
| `terms`: the device radix sort vs a host sort by value then index (sizes 1 to 2^20 around the tile and scan boundaries, ties, signed zeros, extremes); pyramid, level-power and kurtosis gradients (with and without the kappa chain), marginal, `g += a b` vs the host expressions; staged transfers (laptop) | sort: the same permutation; gradients and marginal identical to the bit; moments and marginal loss <= 1e-13 relative |
| `stl-cli` `device_objective_matches_host_terms`: the device objective vs the host terms on the same sweeps, nside 64, 2 members, every term (ensemble with scatter, level power, marginal, kappa statistics, kurtosis of kappa), two maps (laptop) | kappa transform identical to the bit; loss identical at 16 digits, gradient <= 1.9e-16 of its maximum |
| `multirank`: synthetic nside 128, 1 / 2 / 3 ranks | statistics <= 1.6e-15, gradients <= 3e-16 |
| `synthesis`: objective (loss and gradient, three consecutive maps, state reused) vs CPU, 1 / 2 / 3 ranks, nside 64, 128, 256 | loss <= 1.1e-13, gradient <= 2.1e-12 relative |
| `synthesis`: WebSky kappa nside 64, 20 L-BFGS iterations (25 evaluations), `STL_RANKS` 1, 2, 3 | 25 / 25 loss values identical to STL-Dev at the printed precision, max relative difference 1.5e-12 to 1.8e-11 |
| `STL_CHECK_GPU=1 tools/mpi_check.sh 2 3` on garoupe (one T4 per rank, `mpirun`) | every fixture with 1, 2, 3 ranks: within tolerance, <= 6.7e-16 from one rank |

## 4. Building and running

```bash
# any machine with a CUDA driver; CUDARC_CUDA_VERSION = the driver's CUDA version
export CUDARC_CUDA_VERSION=12040
cargo test --release -p stl-gpu
cargo build --release -p stl-cli --features "gpu mpi"
mpirun -n 3 stl-synth --target map.npy --gpu --niter 50 --out run
```

On garoupe (`tools/garoupe/`): `setup.sh` installs rustup, NVRTC 11.8 and
libclang wheels under `$W` (default `$HOME/stl-gpu-work`); `env.sh` sets
`CUDARC_CUDA_VERSION=11040`, the library paths and a local cargo target
directory; `bench.sh` runs the benchmarks below.

Options: `--gpu-block-rows N` (owned cells per device block),
`--gpu-conv dia|stencils` (form of the convolution geometry, default
`STL_GPU_CONV` else `dia`), `--gpu-fields auto|keep|recompute` (wavelet
fields kept between the sweeps or recomputed per block, default
`STL_GPU_FIELDS` else `auto`, §2.7), `--gpu-terms device|host` (the
objective's terms on the device, the default, or on the host, §2.6b),
`--gpu-cache auto|full|transposition|none`, `STL_GPU_CACHE_MB` (cache
budget), `STL_GPU_THREADS` (host threads of the setup per rank),
`STL_GPU_DEVICE`, `STL_GPU_SYNC=1` (synchronise after every kernel, to
locate a failing one), `STL_GPU_PROFILE=1` (geometry phases).

## 5. Measurements

garoupe, `tools/garoupe/bench.sh`: `stl-synth --gpu`, lognormal target,
white-noise start, 3 L-BFGS iterations (4 evaluations, forward +
gradient), `J = log2(nside) - 1`, kernel 5, one T4 per MPI rank
(`mpirun --bind-to none`). Seconds per evaluation; setup = partition,
topology, target statistics (garoupe's host CPUs are slow); device memory
peak of the busiest rank (CUDA context included); geometry cache policy
chosen by `auto`; device blocks at level 0 per rank. Reports (JSON, with
the phase timers) in `benchmarks/garoupe/gpu_scaling_2026-10-20/`.

| nside | L | GPUs | coefficients | s / evaluation | setup s | cache | peak GB / GPU | blocks (level 0) |
|---|---|---|---|---|---|---|---|---|
| 256 | 2 | 1 | 814 | 0.37 | 11 | full | 5.4 | 2 |
| 256 | 2 | 2 | 814 | 0.23 | 7 | full | 2.8 | 1 |
| 256 | 2 | 3 | 814 | 0.19 | 6 | full | 2.0 | 1 |
| 256 | 4 | 1 | 5882 | 1.26 | 11 | transposition | 9.1 | 4 |
| 256 | 4 | 2 | 5882 | 0.42 | 7 | full | 5.5 | 2 |
| 256 | 4 | 3 | 5882 | 0.34 | 7 | full | 4.0 | 2 |
| 1024 | 2 | 1 | 1538 | 26.6 | 129 | transposition | 12.1 | 37 |
| 1024 | 2 | 2 | 1538 | 7.75 | 68 | transposition | 10.9 | 15 |
| 1024 | 2 | 3 | 1538 | 5.12 | 50 | transposition | 10.6 | 9 |
| 1024 | 4 | 1 | 11354 | 66.8 | 147 | transposition | 12.3 | 95 |
| 1024 | 4 | 2 | 11354 | 24.7 | 75 | transposition | 11.3 | 35 |
| 1024 | 4 | 3 | 11354 | 13.3 | 56 | transposition | 10.9 | 21 |
| 2048 | 2 | 1 | 2022 | out of memory | | | | |
| 2048 | 2 | 2 | 2022 | 107 | 291 | transposition | 13.6 | 171 |
| 2048 | 2 | 3 | 2022 | 45.6 | 203 | transposition | 12.8 | 62 |
| 2048 | 4 | 1 | 15042 | out of memory | | | | |
| 2048 | 4 | 2 | 15042 | 474 | 486 | transposition | 13.8 | 978 |
| 2048 | 4 | 3 | 15042 | 114 | 235 | transposition | 12.9 | 180 |

At nside 2048 the fields and decimation rows leave little room: blocks
are small (978 per rank at `L = 4` on 2 GPUs) and most of the time is the
geometry rebuilt at every sweep (2197 s of the 2 GPU, `L = 4` run, 469 s
of the 3 GPU one).

CPU reference (README, hpc cluster, 2 x AMD EPYC 9655 per node, same
`J`, `L = 2`, kernel 5): nside 1024, 10.2 s per evaluation on 192 ranks
(one node), 60.9 s on 24 ranks. Three T4s (5.1 s) are twice as fast as
the 192-core node; one T4 (26.6 s) is about 2.3 x faster than 24 cores.

Where the time goes (nside 1024, `L = 2`, 3 GPUs, totals over the 9
sweeps of a run): stencil weights recomputed for the blocks whose
geometry is not kept whole 11.0 s, convolutions and adjoints 6.1 s,
geometry built from scratch (first sweep, blocks beyond the budget)
4.1 s, exchanges and decimations 2.7 s, coefficients < 1 s. The speed-up
from 1 to 2 or 3 GPUs is superlinear while the cache covers more of each
rank's geometry; once it all fits (nside 256, `L = 2`) only the
convolutions remain, ~45 % parallel efficiency at 3 GPUs on that small
problem (exchanges through the host).

The kernels on one block (`tests/bench_kernels.rs`, nside 256, `L = 2`,
786 432 rows, one T4): convolution forward 17 ms, adjoint 43 ms, stencil
weights 150 ms (63 % of a rebuild without the transposition), full
geometry with its transposition 560 ms. During the development the
adjoint went from 143 ms (one thread per cell) to 43 ms (one warp per
cell), and the evaluation at nside 256 from 1.12 s to 0.37 s.

Synthesis trajectories: with 2 and 3 GPUs the 25 STL-Dev loss values of
the nside 64 fixture are reproduced (25 / 25 at the printed precision).
From a white-noise start, the first L-BFGS step is `1 / |g|_1`, so
`y = g_1 - g_0` loses ~9 digits and the second iteration amplifies
gradient differences of 1e-12 to the 1e-6 level: on garoupe the 3-GPU
runs at `L = 2` (nside 256 and 1024) leave the trajectory of the 1- and
2-GPU runs (and of the CPU) there, the others follow it to 1e-13. This is
not an error of the 3-rank path: `mpirun -n 3` and 3 thread ranks give
the same losses to the bit, and the objective's gradient with 3 ranks
differs from the CPU by no more than with one (nside 256, three maps:
loss <= 1e-13, gradient <= 2.1e-12 relative, `objective_gradient_rank_counts`).

### 5.0 DIA geometry (`--gpu-conv dia`)

garoupe, one T4 (`CUDA_VISIBLE_DEVICES=2`), `stl-synth --gpu`, the same
lognormal targets and 3 L-BFGS iterations (4 evaluations) as above, `L =
4`, cache `auto`, block size from the device memory; reports in
`benchmarks/dia/t4_synth/`:

| nside | geometry | s / evaluation | cache kept | peak GB | geometry built / recomputed (s, whole run) | blocks (level 0) |
|---|---|---|---|---|---|---|
| 256 | bound stencils | 1.246 | transposition, 7.0 GB | 9.8 | 1.5 / 1.7 | 4 |
| 256 | DIA | 0.505 | full, 3.5 GB | 4.2 | 0.9 / 0 | 3 |
| 1024 | bound stencils | 61.4 | transposition, 4.9 GB (part of the blocks) | 13.1 | 237.0 / 13.8 | 95 |
| 1024 | DIA | 22.9 | masks of all blocks + 4.9 GB of whole geometries | 12.0 | 13.0 / 64.2 | 68 |

The loss values agree to 2e-15 relative over the 4 evaluations. At
nside 1024 the bound stencils cannot keep even the transposition of every
block, so most blocks are rebuilt with their sort at every sweep; the DIA
keeps the masks of all blocks (16 bytes per row) and only recomputes
coefficients, and its convolutions are 1.6-2 x faster (`forward` 59.6
against 170.4 s, `reverse` 46.9 against 127.1 s over the run).

hpc, one A100-SXM4-80GB (job 2353875, `tools/hpc/synth_dia.slurm`), same
settings, one member; reports in `benchmarks/dia/a100_synth/`:

| nside | geometry | s / evaluation | cache kept | peak GB | geometry built / recomputed (s, whole run) | forward / reverse (s, whole run) | blocks (level 0) |
|---|---|---|---|---|---|---|---|
| 1024 | bound stencils | 4.873 | transposition, 45.3 GB | 62.0 | 4.4 / 9.3 | 12.0 / 7.3 | 12 |
| 1024 | DIA | 3.035 | masks of all blocks + whole geometries, 50.1 GB | 61.0 | 3.6 / 0.8 | 5.3 / 3.8 | 12 |
| 2048 | bound stencils | 46.25 | transposition, 33.0 GB (part of the blocks) | 67.4 | 159.9 / 8.2 | 113.1 / 77.4 | 58 |
| 2048 | DIA | 17.93 | masks of all blocks + whole geometries, 34.5 GB | 63.2 | 18.0 / 23.6 | 36.8 / 20.7 | 48 |

The DIA is 1.6 x faster at nside 1024 and 2.6 x at nside 2048; the loss
values agree to 1.8e-15 relative. The peak is set by the cache budget,
which fills the device in both forms: the DIA keeps more of the geometry
in it rather than lowering it. At nside 2048 the bound stencils rebuild
most blocks with their sort at every sweep (160 of the 185 s), the DIA
only recomputes coefficients (24 s). The CPU setup (topology, target
statistics: 77 s at nside 1024, 320 s at nside 2048) is the same in both
forms and not counted per evaluation.

### 5.1 Ensembles with the kappa term (`--kappa-table`)

One engine per rank for all the objectives (§2.7) against c15e8ec (one
engine per objective). Laptop (RTX A1000, 6 GB): nside 128 test map,
`L = 4`, `J = 5`, 2 members, `--weights ergodic --scatter-weight 1
--level-power-weight 10 --marginal-weight 100 --kappa-table` (41 knots,
`kappa = exp(0.9 g) - 1 + 0.05 g`), `--gpu-block-rows 30000`, 4 L-BFGS
iterations (5 evaluations). With two ranks on the one device both builds
cap the cache (each rank would size it from the same free memory); the
peak is then the whole device's.

| Build | Ranks | `STL_GPU_CACHE_MB` | s / evaluation | setup s | peak MB | cache |
|---|---|---|---|---|---|---|
| c15e8ec | 1 | unset | out of device memory | | | |
| c15e8ec | 1 | 500 (per engine) | 1.92 | 8.4 | 2810 | transposition |
| shared engine | 1 | unset | 0.73 | 3.7 | 4090 | full, 3.0 GB |
| c15e8ec | 2 | 500 | 1.80 | 7.8 | 2426 | transposition |
| shared engine | 2 | 500 | 1.63 | 3.5 | 1914 | transposition |
| c15e8ec | 2, `--groups 2` | 500 | 2.88 | 8.5 | 2682 | transposition |
| shared engine | 2, `--groups 2` | 500 | 2.72 | 4.4 | 2042 | transposition |

The loss histories, the kappa residuals of the report and the output maps
of the shared engine are those of c15e8ec to the bit in the three
configurations (fixed device blocks; with blocks sized from the device
memory the block boundaries, hence the last bits, depend on the memory
free at start-up in both builds).

hpc, Takahashi kappa ensemble (gaussianised target, `L = 4`, one rank,
`--weights ergodic --scatter-weight 1 --kappa-table knots.npy
--kappa-weight 0.1`, level-power and marginal weights 10 and 10000 at
nside 1024, 5 and 5000 at nside 2048, start from the syntheses of the
previous nside, 2 L-BFGS iterations = 3 evaluations), device blocks and
cache sized from the free memory. Peak: whole device, CUDA context
included (GB = 1e9 bytes); cache: policy chosen by `auto`, then its final
size (`geometry_cache_used_bytes_per_rank`) or, for c15e8ec, the budget of
each engine. Host threads of the setup: 16 on the H200, 6 on the A100
(`STL_GPU_THREADS`, the CPUs free on the A100 node). The H200 runs (26
October) started from earlier syntheses than the A100 ones, so their
losses differ.

| Device | nside x members | Build | `STL_GPU_CACHE_MB` | s / evaluation | setup s | peak GB | cache |
|---|---|---|---|---|---|---|---|
| H200 (141 GB) | 1024 x 10 | c15e8ec | unset | out of device memory | | | |
| H200 | 1024 x 10 | c15e8ec | 16000 | 44.5 | 108 | 98.0 | budget 16.8 GB per engine |
| H200 | 2048 x 5 | c15e8ec | 6000 | 156.7 | 521 | 139.2 | budget 6.3 GB per engine |
| A100 (80 GB) | 1024 x 10 | c15e8ec | unset | out of device memory | | | |
| A100 | 1024 x 10 | c15e8ec | 6000 | 86.6 | 156 | 70.2 | transposition, budget 6.3 GB per engine |
| A100 | 1024 x 10 | shared engine | unset | 64.4 | 86 | 62.8 | transposition, 28.4 GB |
| A100 | 2048 x 5 | shared engine | unset | 394.7 | 446 | 77.7 | transposition, 8.6 GB |

On the same A100 at nside 1024 the shared engine runs without a cap,
takes 26 % less time per evaluation and 45 % less setup (one topology
instead of one per engine), with a peak 7.4 GB lower although its one
cache (28.4 GB) is larger than the 6.3 GB budgets of all of c15e8ec's
engines together; its losses are c15e8ec's to 3e-16 (different block
boundaries). The device is busy 18 % of the time on average (`nvidia-smi`
every 20 s; 28 % for c15e8ec): its phase timers add up to 55 s of the
193 s of the three evaluations, the rest is host work with the 6 threads
of this node. Its 62.8 GB are the fields of the 10 members (27.7 GB,
§2.7), the decimation rows and topology (~3.4 GB), one block's transients
(252 260 rows, ~2.4 GB) and the cache, which takes the rest. At nside
2048 the 5 members fit on the A100 (c15e8ec needed 139 GB on the H200):
their fields (55.5 GB) and the decimation rows (~13 GB) leave 8.6 GB to
the cache, a small part of the geometry's transposition (~108 GB), so
the geometry is rebuilt for every block at every sweep (527 blocks at
level 0): 924 s of the device phase timers, against 73 s for the
convolutions, adjoints, decimations and coefficients together (the
device is busy 33 % of the time on average). Ten members at nside 2048
do not fit on one 80 GB device: their fields alone are 111 GB. `z_n` and
`|z_n|` with the reverse chain are 77 % of the fields (2 x 10.7 of the
27.6 doubles per pixel and member at `L = 4`), the maps 10 %, the stored
moduli 13 %. Recomputing `z_n` and `|z_n|` per block brings them within
the A100 (§5.2).

### 5.2 Kept or recomputed wavelet fields (`--gpu-fields`)

`stl-synth --gpu` with the DIA geometry, `--gpu-fields keep` against
`recompute` (§2.2, §2.7), one device, blocks and cache sized from the
device memory, commit 3177d9c. "fields" is the engine's estimate of the
level fields at their peak (`gpu.field_bytes_estimate`), "rows" the
decimation rows on the device, "cache" the geometry cache at the end;
the peak is the engine's (whole device, CUDA context included). Reports
in `benchmarks/fields/`.

garoupe, one T4 (15 GB), nside 1024, `L = 4`, lognormal target, 3
L-BFGS iterations (4 evaluations) from white noise, `--ensemble B`:

| B | fields | s / evaluation | peak GB | fields GB | rows GB | cache GB | block rows (blocks at level 0) |
|---|---|---|---|---|---|---|---|
| 1 | keep | 22.56 | 12.03 | 2.36 | 2.98 | 5.19 | 194 187 (65) |
| 1 | recompute | 23.14 | 11.35 | 0.75 | 2.98 | 5.95 | 227 598 (56) |
| 2 | keep | 30.56 | 12.70 | 4.73 | 2.98 | 3.90 | 104 388 (121) |
| 2 | recompute | 30.42 | 11.72 | 1.50 | 2.98 | 5.64 | 153 964 (82) |
| 4 | keep | 48.91 | 14.54 | 9.45 | 2.98 | 1.27 | 19 086 (660) |
| 4 | recompute | 46.18 | 11.93 | 3.01 | 2.98 | 4.77 | 84 451 (149) |

hpc, one A100-SXM4-80GB, the Takahashi kappa ensemble of §5.1 at nside
2048 (`--level-power-weight 5 --marginal-weight 5000 --kappa-table
--kappa-weight 0.1`, members started from the nside-1024 syntheses), 2
iterations (3 evaluations), 6 host threads (job 2404157):

| B | fields | s / evaluation | setup s | peak GB | fields GB | rows GB | cache GB | block rows (blocks at level 0) | device busy |
|---|---|---|---|---|---|---|---|---|---|
| 5 | keep | 163.2 | 324 | 75.1 | 47.3 | 11.9 | 13.1 | 173 901 (290) | 14 % |
| 5 | recompute | 162.6 | 322 | 63.8 | 15.0 | 11.9 | 30.9 | 431 433 (117) | 14 % |
| 10 | keep | does not fit (fields 94.5 GB) | | | | | | | |
| 10 | recompute | 332.8 | 341 | 67.5 | 30.1 | 11.9 | 22.3 | 175 644 (287) | 11 % |

The recomputation costs one wavelet convolution of the maps per block of
the reverse sweep: 0.3 to 0.6 s per evaluation on the T4 at nside 1024,
0.6 s on the A100 at nside 2048 (`gpu_reverse_conv` 11.9 -> 13.9 s over
the 2048 x 5 run). The memory it frees goes to larger blocks and a larger
geometry cache, which recompute fewer DIA coefficients: with one member
`recompute` is 2.6 % slower, with two members as fast, with four 5.6 %
faster (the kept fields leave 1.3 GB of cache and blocks of 19 086 rows
on the T4). At nside 2048 x 5 both take the same time (163 s per
evaluation, against 395 s with the bound stencils in §5.1); the fields
go from 47.3 to 15.0 GB, a third, and the peak from 75.1 to 63.8 GB. The
losses of the two modes agree to 2.7e-15 relative on both devices
(different block boundaries; with the same blocks they are identical,
§3), and at nside 2048 x 5 to 2.3e-15 with the bound-stencil run of
§5.1.

Ten members at nside 2048 now fit on one A100: 67.5 GB at the peak,
of which the level fields 30.1 GB (45 %), the DIA geometry cache 22.3 GB
(33 %: masks of every block and the coefficients of some), the
decimation rows 11.9 GB (18 %) and the rest (one block's transients,
topology, CUDA context) 3.2 GB. An evaluation takes 333 s, twice the 5
members', but the device is busy 11 % of the time: the engine's phases
(target statistics, synthesis and kappa sweeps) add up to 70 s per
evaluation, the rest is host work of the objective (ensemble, level-power,
marginal and kappa terms, L-BFGS) on 10 maps of 50 M pixels with 6
threads. §5.3 moves that work to the device.

### 5.3 The objective's terms on the device (`--gpu-terms`)

The terms of the production recipe at nside 1024 on garoupe, one T4:
target the nside-2048 gaussianised Takahashi map averaged over groups of
4 children, 10 members from the nside-256 ensemble (`--init-noise` auto),
`--L 4 --weights ergodic --scatter-weight 1 --level-power-weight 10
--marginal-weight 10000 --kappa-table knots1024.npy --kappa-weight 0.1
--kurtosis-weight 1.0 --kurtosis-nside 32-1024`, 3 iterations (4
evaluations), wavelet fields recomputed (`auto`), 12 host threads. The
timers (`timers_seconds`, per evaluation) are wall times of the
evaluation's parts: the objectives with their sweeps, the terms with
their gradient sums, L-BFGS the rest of the run. Reports in
`benchmarks/terms/t4/`.

| | host terms (f65ce8e) | device terms, room reserved (96c6bf1) | device terms (cceca63) |
|---|---|---|---|
| s / evaluation | 205.1 | 129.6 | 107.5 |
| device busy (whole run, setup included) | 42 % | 67 % | 65 % |
| synthesis objective (sweeps + copies) | 93.7 | 85.6 | 76.2 |
| kappa objective | 46.7 | 35.3 | 26.9 |
| marginal | 35.1 | 1.2 | 1.2 |
| kurtosis | 8.8 | 0.07 | 0.23 |
| kappa transform | 8.5 | 0.01 | 0.08 |
| level power | 7.2 | 0.02 | 0.02 |
| copies to and from the engine's layout, transfers | 19.0 | 1.8 | 1.7 |
| L-BFGS | 4.8 | 5.6 | 1.2 |

The losses of the three runs agree to 1e-15 relative at every evaluation.
The host work (terms, copies, L-BFGS) took 83 s of the 205 s, the
marginal's sorts 35 s of them; on the device the terms take 1.5 s.
Keeping room for the terms' buffers on the device (96c6bf1) took it from
the geometry cache and cost the sweeps about 20 s per evaluation (DIA
coefficients recomputed more often on the T4); with nothing but the
engine's fields during the sweeps (cceca63) they get the host path's
cache again. L-BFGS on 12 threads takes a quarter of its single-thread
time. What is left, 100 s of the 107 s, is the engine's sweeps.

## 6. Limitations and next steps

- **FP64 throughput.** On the T4 the stencil kernel (interpolation of
  `4 G P` taps per row: acos, atan2, ring search) is FP64-bound; it is
  recomputed at every sweep when the geometry does not fit whole. A
  data-centre GPU (A100, H100) removes most of it. The DIA geometry
  (`--gpu-conv dia`, §2.3, §5.0) merges the entries of a row that read
  the same cell and drops the transposition (3 x less memory whole, 100 x
  less between sweeps); measured 2.5-2.7 x faster per evaluation on the
  T4 and 1.6-2.6 x on the A100 (§5.0), it is the default.
- **Memory.** With `z_n` and `|z_n|` recomputed per block (§5.2) the
  level fields are 60 bytes per pixel and member at `L = 4`, and the
  decimation rows (~200 bytes per held cell, 11.9 GB for nside 2048 on
  one rank, independent of the members) are the next item: they could be
  computed on the device per sweep instead of stored. In the forward
  sweep `|z_0|` is still held over the level for its decimations (`B C L
  H_0` doubles); pushing each block's moduli down the decimation chain as
  the CPU pieces do would remove it, and freeing `m[d][n]` once the
  reverse sweep has passed level `n` would lower the reverse peak too.
- **Host work.** At nside 2048 the A100 was busy 11 to 14 % of the time
  with the objective's terms on the host (§5.2); they now run on the
  device (§5.3). L-BFGS stays on the host (its history, 2 x 10 x 4 GB at
  nside 2048 x 10, does not fit next to the fields): its two-loop
  recursion reads the history twice per iteration on the host threads.
  The engine's own host work (contribution lists of the coefficient
  cotangents per block, partial sums brought back per block) and its
  synchronisation at the end of every phase are the next.
- **Exchanges** copy whole level fields between device and host (MPI is
  not CUDA-aware on garoupe): packing only the halo cells on the device
  would cut the transfers by the halo fraction.
- **Setup** on the host: `build_partition` (CPU code of stl-mpi) and the
  decimation rows dominate on garoupe's 12 slow cores (1 to 5 minutes at
  nside 1024 to 2048); both could move to the device.
- Multi-node runs should work as is (host-staged exchanges) but were not
  tested; NCCL is not used.
