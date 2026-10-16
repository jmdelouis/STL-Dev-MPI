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

Regenerate the fixtures (STL-Dev synced in `~/stl-work`, CPU venv in
`~/stl-deps/venv-cpu`; override with `STL_DEV` / `STL_PYTHON`):

```bash
tools/dump_reference.sh geometry block
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

Unsorted cell ids are rejected: with unsorted ids `TorchConv` (and
healpix-analyse's `HealPixConv`, which it reproduces) returns its rows in
neither the input nor the sorted order; STL-Dev always passes sorted ids.
