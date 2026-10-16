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
| `tools/dump_reference.py` | dumps STL-Dev outputs as fixtures |
| `fixtures/<suite>/` | `.npy` arrays + `meta.json` per case, `index.json` per suite |

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
tools/dump_reference.sh geometry block stats
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

Unsorted cell ids are rejected: with unsorted ids `TorchConv` (and
healpix-analyse's `HealPixConv`, which it reproduces) returns its rows in
neither the input nor the sorted order; STL-Dev always passes sorted ids.
