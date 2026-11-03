# Reports and measurements

The design notes and measurement reports written while the package was built. Each one states
its setup (machine, commit, job) so that it can be reproduced.

## Validation

| Report | Content |
|---|---|
| [Validation and milestones](validation.md) | the fixture tests against STL-Dev and their results, milestone by milestone: geometry, one block, statistics and gradients, multi-rank, synthesis trajectories, scaling at nside 1024, GPU |

## Benchmarks

| Report | Content |
|---|---|
| [nside 128 × 4 on one A100](bench_n128.md) | STL-Dev legacy (autograd) against STL-Dev-MPI on the same synthesis: 11.6× faster, 21× less GPU memory; gradients agree to 2.7e-13 |
| [Planar 2D images](planar_2d.md) | the block-wise pyramid on periodic grids, CPU and GPU, against STL-Dev's 2D example: 11× faster on CPU, 3.5× on a T4, one 4096² image in 2.9 GB |
| [GPU engine](gpu.md) | design (one device per rank, fields, geometry, determinism, memory per rank), validation, T4 and A100 measurements, `--gpu-fields`, `--gpu-terms` |
| [DIA convolution](dia_convolution.md) | the wavelet convolution as masked diagonals in face-raster coordinates (`--gpu-conv dia`, the default) |
| [Geometry cache and preconditioners](geometry_cache_preconditioner.md) | `--geometry-cache host` and `auto`, per-scale L-BFGS preconditioners, projection to nside 2048 × 10 |
| [Harmonic terms: cost at nside 1024 and 2048](harmonic_speed.md) | profile of `stl-ensemble` and the options to reach nside 2048 on one A100 |

## Ensembles and productions

| Report | Content |
|---|---|
| [P. Campeti's ensemble method](campeti_merge.md) | the merge of his package: `stl-bridge`, `stl-ensemble`, validation against his Python, harmonic terms on the GPU |
| [Production at nside 1024](campeti_production_1024.md) | step by step: calibration, Slurm job on one A100, monitoring and restart, results against the ten Takahashi realisations |
| [Kappa power spectrum](kappa_power_spectrum.md) | the C_ℓ deficit of Gaussianised syntheses and its correction (`--kappa-table`, level power) |

## Research notes

| Report | Content |
|---|---|
| [Multi-scale conditional diffusion](diffusion_design.md) | a learned generative model on the HEALPix pyramid, complementary to the synthesis (`diffusion/`) |

```{toctree}
:hidden:
:maxdepth: 1

validation
bench_n128
planar_2d
gpu
dia_convolution
geometry_cache_preconditioner
harmonic_speed
campeti_merge
campeti_production_1024
kappa_power_spectrum
diffusion_design
```
