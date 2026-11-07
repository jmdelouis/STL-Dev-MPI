# User Guide

Everything you need to understand how STL-Dev-MPI works and how to use each of its tools: one page
per command-line tool and per Python engine, each with the same layout (what it does, key
parameters, a minimal example, pitfalls and costs).

## The workflow

```
1. prepare the target      tools/prepare_healpix_target.py, gaussianize.py          (NESTED, full sky)
2. look at its statistics  stl-stats --out stats.json -- target.npy
3. synthesise              stl-synth --target target.npy --weights ergodic --niter 200 --out run
   an ensemble             stl-synth --ensemble 10 ...   |   stl-ensemble calibrate / run
   2D images               stl-synth2d --target images.npy --out run2d
4. check                   stl-stats --norm target.npy -- target.npy run_synthesis_nested.npy
in Python                  NativeEngine / PatchEngine / SpectrumEngine: forward + exact VJP
                           NativeEngine(spin=2): E/B statistics of (Q, U)   |   NativeUNet: a denoiser trained by them
```

## Key parameters (all HEALPix tools)

| Parameter | Default | Description |
|---|---|---|
| `--L` / `orientations` | 4 | orientations of the wavelet (always 4: `L = 2` gives anisotropic syntheses) |
| `--J` / `j` | `log2(nside) − 1` | number of scales, one per level of the pyramid |
| `--kernel-size` / `kernel` | 5 | wavelet stencil in the tangent plane |
| `--block-pixels` | 16384 | cells per block: the memory knob of the CPU engine |
| `--weights` | `none` | `ergodic` makes the loss a reduced χ² that fits every family |
| `--ranks` / `mpirun -n` | 1 | thread ranks / MPI ranks |
| `--gpu` / `backend="gpu"` | off | one CUDA device per rank |

## Units and outputs

Maps are full-sky HEALPix maps in the **NESTED** ordering (`.npy`, or HEALPix FITS binary tables
for the command-line tools). A target is standardised (mean 0, std 1) before its statistics are
computed; a synthesis runs in those units and is written back in the target's units. Every run
writes a JSON report with the field names of STL-Dev's `benchmark_synthesis.py`
(`loss_history`, `timers_seconds`, `memory_peak_bytes_per_rank`, ...).

## What next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc
The model: pyramid, blocks and halos, stored moduli, sweeps, explicit VJP, distributed memory.
:::

:::{grid-item-card} stl-stats
:link: stl-stats
:link-type: doc
Statistics of HEALPix maps, with the plan, in one JSON file.
:::

:::{grid-item-card} stl-synth
:link: stl-synth
:link-type: doc
Synthesis of a HEALPix map, or of an ensemble matching one target.
:::

:::{grid-item-card} stl-ensemble
:link: stl-ensemble
:link-type: doc
Ensembles matching the mean and variance of a population (P. Campeti's method).
:::

:::{grid-item-card} stl-synth2d
:link: stl-synth2d
:link-type: doc
Synthesis of periodic 2D images, with coarse-to-fine starts.
:::

:::{grid-item-card} Differentiable statistics in Python
:link: differentiable
:link-type: doc
`NativeEngine`, `PatchEngine`, `SpectrumEngine`: statistics and pseudo-spectra with exact VJPs,
full sky, masks, discs and cross terms, as torch losses.
:::

:::{grid-item-card} NativeEngine
:link: native-engine
:link-type: doc
Full-sky statistics in Python with their exact VJP: masks, cross statistics.
:::

:::{grid-item-card} PatchEngine
:link: patch-engine
:link-type: doc
Statistics of disc patches with their VJP, batched, CPU or GPU.
:::

:::{grid-item-card} SpectrumEngine
:link: spectrum-engine
:link-type: doc
Differentiable pseudo-spectra (anafast), spin 0 and spin s, with masks.
:::

:::{grid-item-card} Spin-2 statistics
:link: spin-statistics
:link-type: doc
E and B scattering statistics of polarisation maps `(Q, U)`, local, with masks and cross terms.
:::

:::{grid-item-card} NativeUNet
:link: native-unet
:link-type: doc
A spherical U-Net denoiser trained by scattering losses on one device, Adam or L-BFGS.
:::

:::{grid-item-card} GPU
:link: gpu
:link-type: doc
One device per rank: fields, geometry cache, extra terms, measured speeds.
:::

:::{grid-item-card} MPI
:link: mpi
:link-type: doc
Decomposition, rank counts, groups for ensembles, batch systems.
:::


:::{grid-item-card} Computing centres
:link: ../USAGE
:link-type: doc
Installation without root, preparing targets, the recipe for non-Gaussian fields, batch scripts.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

concepts
stl-stats
stl-synth
stl-ensemble
stl-synth2d
differentiable
native-engine
patch-engine
spectrum-engine
spin-statistics
native-unet
gpu
mpi
../USAGE
```
