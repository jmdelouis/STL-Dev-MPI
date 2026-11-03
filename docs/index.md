# STL-Dev-MPI

**Scattering-covariance statistics and map synthesis on HEALPix spheres and 2D grids, with a block-wise pyramidal gradient whose memory is bounded — in Rust, with MPI and CUDA.**

STL-Dev-MPI computes the scattering-covariance statistics (`mean`, `var`, S1, S2, S3, S4, with
`L = 4` orientations) of HEALPix maps and periodic images, and synthesises new maps whose
statistics match those of a target by L-BFGS on the map itself. It is the high-performance
port of the patch-local engine of [STL-Dev](https://github.com/STL-Proj/STL-Dev): the same
statistics to 1e-10, the same optimiser, but with the gradient computed block by block over the
multiresolution pyramid, so that the memory of a process is set by its share of the sky, not by the
full sky.

```
target map ──► statistics S(target)                       (stl-stats)
white noise x ──► S(x), ∂loss/∂x block by block ──► L-BFGS ──► synthesis   (stl-synth)
             loss = Σ_i w_i |S_i(x) − S_i(target)|²
```

The sky is split into contiguous NESTED ranges, one per MPI rank, and each rank works on blocks
of constant size with their halos: nside 2048 × 10 maps fit on one A100, nside 1024 runs on 192
CPU cores in 10 s per evaluation, and the same synthesis at nside 128 is 11.6× faster with 21×
less GPU memory than STL-Dev on the same A100.

## Where to go next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Build the binaries with cargo, with or without MPI and CUDA.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Five worked examples that run in minutes: HEALPix synthesis on CPU and GPU, ensembles,
statistics, 2D images, MPI.
:::

:::{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc
The model (pyramid, blocks and halos, stored moduli, explicit VJP), the command-line tools,
GPUs and MPI.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc
The four crates and their main types; the full rustdoc.
:::

:::{grid-item-card} Reports and measurements
:link: reports
:link-type: doc
Validation against STL-Dev, benchmarks, production runs at nside 1024 and 2048.
:::

:::{grid-item-card} Terminology
:link: terminology
:link-type: doc
Block, halo, piece, sweep, native level, stored moduli, plan, coefficient families.
:::

::::

```{toctree}
---
maxdepth: 2
caption: User guide
hidden: true
---
installation
user-guide/index
tutorials/index
```

```{toctree}
---
maxdepth: 2
caption: Reference
hidden: true
---
api
terminology
reports
```
