# User Guide

Everything you need to understand how STL-Dev-MPI works and how to run it.

## The workflow

```
1. prepare the target      tools/prepare_healpix_target.py, gaussianize.py     (NESTED, full sky)
2. look at its statistics  stl-stats --out stats.json -- target.npy
3. synthesise              stl-synth --target target.npy --weights ergodic --niter 200 --out run
   or an ensemble          stl-synth --ensemble 10 ...    |   stl-ensemble calibrate / run
   or 2D images            stl-synth2d --target images.npy --out run2d
4. check                   stl-stats --norm target.npy -- target.npy run_synthesis_nested.npy
```

## Key options (all HEALPix tools)

| Option | Default | Description |
|---|---|---|
| `--L` | 4 | orientations of the wavelet (always use 4: `L = 2` gives anisotropic syntheses) |
| `--J` | `log2(nside) − 1` | number of scales (levels of the pyramid) |
| `--kernel-size` | 5 | wavelet stencil in the tangent plane |
| `--block-pixels` | 16384 | cells per block: the memory knob of the CPU engine |
| `--weights` | `none` | `ergodic` makes the loss a reduced χ² that fits every family |
| `--ranks` / `mpirun -n` | 1 | thread ranks / MPI ranks |
| `--gpu` | off | one CUDA device per rank |

## Units and outputs

The target is standardised (mean 0, std 1) before its statistics are computed; the synthesis
runs in those units and is written back in the target's units (`PREFIX_synthesis_nested.npy`).
`stl-stats --norm target.npy` puts several maps in the same units. Every run writes a JSON report
with the fields of STL-Dev's `benchmark_synthesis.py` (`loss_history`, `timers_seconds`,
`memory_peak_bytes_per_rank`, ...).

## What next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc
Pyramid, blocks and halos, stored moduli, sweeps, explicit VJP, distributed memory.
:::

:::{grid-item-card} Command-line tools
:link: cli
:link-type: doc
`stl-synth`, `stl-stats`, `stl-ensemble`, `stl-synth2d`, `stl-minkowski`, `stl-mpi-check`: every
option.
:::

:::{grid-item-card} GPU guide
:link: gpu
:link-type: doc
One device per rank: fields, geometry cache, extra terms, measured speeds.
:::

:::{grid-item-card} MPI guide
:link: mpi
:link-type: doc
Decomposition, rank counts, groups for ensembles, batch systems.
:::

:::{grid-item-card} Differentiable statistics in Python
:link: differentiable
:link-type: doc
`NativeEngine`, `PatchEngine`, `SpectrumEngine`: statistics and pseudo-spectra with exact VJPs,
full sky, masks, discs and cross terms, as torch losses.
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
cli
gpu
mpi
differentiable
../USAGE
```
