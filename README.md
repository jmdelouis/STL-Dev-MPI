# STL-Dev-MPI

Pyramidal scattering-covariance statistics and gradients on HEALPix maps,
in Rust with MPI, block-wise so that the memory of every rank is bounded by
its share of the sky. Port of the patch-local engine of
[STL-Dev](https://github.com/STL-Proj/STL-Dev) (branch `HealpixUpdate`);
see `CLAUDE.md` for the model, the architecture and the milestones.

## Layout

| Path | Content |
|---|---|
| `crates/stl-core` | HEALPix geometry (centres, `ring2nest`, bilinear interpolation, stencil rotation, binding), wavelet kernel, partial convolution and its adjoint, smooth decimation, block-wise `wave` / `decimate` / pieces and their transposes (`local.rs`); no MPI, `f64` |
| `crates/stl-mpi` | decomposition, halo exchange, ghost accumulation (`Comm`: threads or MPI), distributed objective (`synthesis.rs`), npy / HEALPix FITS I/O |
| `crates/stl-cli` | `stl-synth`: synthesis from a target map |
| `tools/dump_reference.py` | dumps STL-Dev outputs as fixtures |
| `tools/datarmor/` | environment and PBS script for Datarmor |
| `fixtures/<suite>/` | `.npy` arrays + `meta.json` per case, `index.json` per suite |
| `benchmarks/datarmor/` | reports of the Datarmor runs |

## Tests

From WSL (keep the build directory on WSL-native storage):

```bash
CARGO_TARGET_DIR=~/.cache/stl-dev-mpi-target cargo test --release
```

Under MPI (OpenMPI and `libclang-dev` installed):

```bash
tools/mpi_check.sh 2 4
```

Regenerate the fixtures (STL-Dev synced in `~/stl-work`, CPU venv in
`~/stl-deps/venv-cpu`; override with `STL_DEV` / `STL_PYTHON`):

```bash
tools/dump_reference.sh geometry block stats synthesis
```

The 20-iteration trajectory comparison (minutes):

```bash
STL_RANKS=4 cargo test --release -p stl-mpi --test synthesis -- --ignored --nocapture
```

## Synthesis

```bash
stl-synth --target kap_nside64_nested.npy --niter 20 --ranks 4 --out run
mpirun -n 28 stl-synth --target kap_nside64_nested.npy --niter 20 --out run   # --features stl-cli/mpi
```

writes `run.json` (fields of `benchmark_synthesis.py`: `loss_history`,
`memory_floor_bytes`, `timers_seconds`, plus per-rank peak memory),
`run_synthesis.fits` / `run_target.fits` (NESTED HEALPix binary tables) and
`run_synthesis_nested.npy`.

On Datarmor (PBS Pro, Intel MPI 2019; no Rust module, so a rustup toolchain
in the work directory; `datawork` was over quota, the test used `scratch`):

```bash
source $W/env.sh && cargo build --release -p stl-cli --features mpi
qsub -v W=$W $W/src/tools/datarmor/synth_nside64.pbs
```

On the hpc cluster (Slurm, RHEL 9, OpenMPI 5; no Rust or clang module:
rustup toolchain and the PyPI `libclang` wheel under `$W`, `$HOME/stl-mpi`
since `$WORK` was full; compilation on a compute node):

```bash
sbatch --export=ALL,W=$W $W/src/tools/hpc/build.slurm
sbatch --export=ALL,W=$W $W/src/tools/hpc/synth_nside64.slurm
```

## Status

Milestone 1 (geometry) against `STL_main/TorchGeometry.py`:

| Quantity | Agreement |
|---|---|
| `nest2xyf`, `ring2nest`, `kth_neighbourhood`, decimation sparsity, `complete` flags | exact |
| pixel centres, rotations, stencil grid | <= 1e-14 |
| ring interpolation, decimation weights | <= 1e-12 |
| NESTED interpolation weights | <= max(1e-12, 1e-14 nside) |
| convolution outputs on complete rows | <= max(1e-12, 128 eps nside / sin theta), observed 1.2e-12 (nside 32) to 1.7e-11 (nside 1024) |

The last two are bounded by the rounding of the reference path itself:
torch evaluates `acos` with SLEEF (1 ulp from glibc on ~10 % of inputs) and
the directions' colatitudes go through `acos(z)`, amplified by
`1 / sin(theta)` and by the pixel size in the bilinear weights.

Milestone 2 (one block) against `STL_main/LocalGeometry.py` and
`AdjointOps.py`, fixtures `fixtures/block` (nside 16, 32, 128; `L` 2 and 4;
kernels 3 and 5; all levels):

| Quantity | Agreement |
|---|---|
| wavelet kernel | <= 1e-15 |
| `wave` (wavelet on a block) and `wave_adjoint` | <= 8e-14 (0.02 of the noise bound above) |
| `decimate`, `decimate_adjoint`, `piece_forward`, `piece_adjoint`, supports and domains | exact ids, <= 4e-14 values |
| `modulus_vjp` | <= 1e-12 |
| `<W x, g> = <x, W^T g>` | 1e-12 relative |

Milestone 3 (single process) against `STL_main/LocalScattering.py`
(`LocalEngine.forward` / `reverse` as `apply_local` / `vjp_local` run them),
fixtures `fixtures/stats`: nside 16 and 32, `J` 3 and 4, `L` 2 and 4,
kernels 3 and 5, 1 and 2 channels, batch 1 and 2, both S3/S4 definitions,
blocks of 64 pixels to whole levels, shared resolutions `dg = (0, 1, 1, 2)`;
218 to 5860 coefficients:

| Quantity | Agreement |
|---|---|
| plan (`build_plan`) | identical |
| maps of every level, stored moduli `m[d][n]` | <= 1e-12 |
| statistics (O(1)) | <= 7e-16 (tolerance 1e-11 + 1e-10 relative) |
| map cotangents of every level, band gradients (O(1e-2)) | <= 9e-16 (tolerance 1e-10 + 1e-9 relative) |
| gradient vs central finite difference of the Rust statistics | 2.5e-9 relative |

`stl_core::engine::LocalEngine` keeps the STL-Dev memory model: per block
fields on `block + halo`, globally the maps, their cotangents and the
stored moduli with their cotangents.

Milestone 4 (`crates/stl-mpi`): contiguous NESTED ranges aligned to
`4^(levels-1)` (every rank owns the parents of its pixels), per-level
layouts of owned + halo cells computed from the rank's blocks, halo
exchange and additive ghost reduction (owner first, then ranks in order),
statistics summed in rank order on every rank. Backends: threads
(`cargo test`) and MPI (feature `mpi`, `tools/mpi_check.sh`).

| Check (8 stats fixtures, 1, 2, 4 ranks) | Result |
|---|---|
| 1 rank through the distributed path vs serial engine | bit-identical |
| 2 and 4 ranks vs 1 rank, threads and `mpirun` (OpenMPI 4.1.6) | <= 7e-16 |
| every rank count vs STL-Dev | <= 1.3e-5 of the tolerances |

Under `mpirun` on one 20-core WSL machine, the wall-clock time of one
evaluation (forward + gradient) drops by 1.8x at 2 ranks and 2.4x to 3.0x
at 4 ranks on the nside 16/32 fixtures, where the halo is still a large
fraction of each slice (at nside 32 and 4 ranks a rank holds 46 % to 65 %
of level 0).

Milestone 5: L-BFGS with strong Wolfe ported line by line from
`torch.optim.LBFGS` (torch 2.14), map variable, loss
`sum |S(x) - S_target|^2`, against STL-Dev's
`benchmark_synthesis.py --mode local --variables map` on the WebSky kappa
map at nside 64 (J 5, L 2, kernel 5, 362 coefficients, 20 iterations = 25
evaluations, same initial map). One Datarmor node (2 x Xeon E5-2680 v4,
28 cores), Intel MPI 2019, PBS job 7695947:

| Ranks | Blocks | Loss values identical to STL-Dev (`%.6e`) | Max relative difference | s / evaluation | Speed-up | Peak RSS per rank (MB) |
|---|---|---|---|---|---|---|
| 1 | 16384 | 25 / 25 | 2.3e-10 | 9.15 | 1.0 | 316 |
| 4 | 16384 | 25 / 25 | 9.7e-11 | 3.27 | 2.8 | 216 - 259 |
| 16 | 16384 | 25 / 25 | 3.1e-11 | 1.45 | 6.3 | 97 - 115 |
| 28 | 16384 | 25 / 25 | 2.0e-11 | 1.19 | 7.7 | 81 - 99 |
| 1 | 4096 | 25 / 25 | 9.4e-12 | 11.85 | 0.77 | 124 |
| 1 | 1024 | 25 / 25 | 1.3e-11 | 19.04 | 0.48 | 65 |

STL-Dev on the same case (WSL, 4 torch threads): 12.6 to 18 s per
evaluation, 1.90 GB peak RSS (1.55 GB above its baseline).

Memory, first version. The block-independent state (maps, stored moduli,
their cotangents, L-BFGS history) is 11 MB on one rank at nside 64 and
shrinks with the share of the sky (0.84 MB per rank at 28 ranks). The peak
was set by the transients of one block, about 18 kB per block pixel: the
convolution geometry of the whole block domain, with 8-byte positions and
the raw interpolation buffers kept until the binding was done.

Block transients reduced (commit "Bound the block transients"): binding
pixel by pixel, `u32` positions, geometry for the output rows only (`a1`),
second wavelets evaluated on the block only, optional geometry by
sub-blocks of rows (`--conv-chunk`), and a fixed glibc mmap threshold so
that freed block buffers are returned to the system. The geometry costs
`4 G P (4 + 8)` bytes per output row (2.4 kB at L = 2, kernel 5). Same
results to the bit in whole-block mode.

| Datarmor, 1 node (Intel MPI 2019), job 7696400 | s / evaluation | Peak RSS per rank | of which above the MPI/input baseline |
|---|---|---|---|
| 1 rank, blocks of 16384 | 8.06 (was 9.15) | 103 MB (was 316) | 85 MB |
| 1 rank, blocks of 4096 | 9.72 | 56 MB | 38 MB |
| 1 rank, blocks of 1024 | 14.69 | 41 MB | 23 MB |
| 1 rank, geometry by sub-blocks of 2048 rows | 15.43 | 50 MB | 32 MB |
| 28 ranks, blocks of 16384 | 0.81 (was 1.19) | 48 - 51 MB (was 81 - 99) | 21 - 24 MB |
| 28 ranks, sub-blocks of 2048 rows | 1.03 | 45 - 46 MB | 18 - 19 MB |

hpc cluster (Slurm, one node of 2 x AMD EPYC 9655, OpenMPI 5.0.11), job
1523174, same case: 25 / 25 loss values identical to STL-Dev at the
printed precision for every run (max relative difference 2.2e-10).

| Ranks | s / evaluation | Speed-up | Peak RSS per rank | of which above the MPI baseline |
|---|---|---|---|---|
| 1 | 3.26 | 1.0 | 278 MB | 87 MB |
| 4 | 1.05 | 3.1 | 399 - 407 MB | 57 - 62 MB |
| 16 | 0.49 | 6.7 | 385 - 391 MB | 38 - 42 MB |
| 64 | 0.29 | 11.3 | 290 - 300 MB | 72 - 81 MB |
| 128 | 0.28 | 11.5 | 291 - 355 MB | 59 - 122 MB |

OpenMPI 5 starts every rank at 190 to 350 MB resident (Intel MPI on
Datarmor: 18 to 29 MB); the code's own memory is the last column. At
nside 64 the work saturates beyond 64 ranks (a rank owns 384 pixels of
level 0 at 128 ranks) and the remaining per-rank memory is the halo of the
small levels and the all-to-all buffers. Every rank also holds the
full-sky target and initial maps in this version (0.8 MB at nside 64, but
800 MB at nside 2048): to be read by slices before milestone 6.

Milestone 6, first scaling runs at nside 1024 (WebSky kappa, J = 9, L = 2,
kernel 5, 1538 coefficients, blocks of 16384, white-noise start, 3 L-BFGS
iterations = 4 evaluations per run), hpc cluster (2 x AMD EPYC 9655 per
node, OpenMPI 5.0.11). No rank reads or writes a full-sky map: maps are
read and written by owned slices. The decomposition goes beyond the 192
cells of the coarsest level (ranks may own no coarse cell). The QOS of
the account limits a user to 192 CPUs, so the multi-node runs spread 192
ranks over 2 and 4 nodes.

| Nodes x ranks | s / evaluation | Speed-up vs 24 | Setup s | State per rank | Peak RSS per rank | of which above the MPI baseline | Level-0 cells held by rank 0 |
|---|---|---|---|---|---|---|---|
| 1 x 24 | 60.9 | 1.0 | 21.4 | 123 MB | 391 - 396 MB | 155 - 160 MB | 561 584 (owns 524 288) |
| 1 x 48 | 33.0 | 1.84 | 11.8 | 63 MB | 357 - 363 MB | 129 - 133 MB | 287 296 |
| 1 x 96 | 19.7 | 3.09 | 7.3 | 32 MB | 366 - 379 MB | 133 - 142 MB | 150 080 |
| 1 x 192 | 10.2 | 6.00 | 3.9 | 17 MB | 349 - 362 MB | 99 - 111 MB | 78 400 (owns 65 536) |
| 2 x 96 | 10.8 | 5.62 | 5.7 | 17 MB | 330 - 343 MB | 79 - 91 MB | 78 400 |
| 4 x 48 | 10.6 | 5.74 | 4.8 | 17 MB | 318 - 329 MB | 78 - 87 MB | 78 400 |

Every run gives the same loss values (identical to 16 digits across rank
and node counts). 75 % parallel efficiency from 24 to 192 ranks; spreading
the 192 ranks over 4 nodes costs 5 %. Per rank, the state (maps, stored
moduli, cotangents, L-BFGS history) shrinks as `1/P`; what remains is the
block transients (~80 to 110 MB at blocks of 16384) and the OpenMPI 5
runtime (230 to 255 MB).

Unsorted cell ids are rejected: with unsorted ids `TorchConv` (and
healpix-analyse's `HealPixConv`, which it reproduces) returns its rows in
neither the input nor the sorted order; STL-Dev always passes sorted ids.
