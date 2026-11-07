# STL-Dev-MPI

**Scattering-covariance statistics and synthesis of HEALPix maps and 2D images, with a block-wise gradient whose memory stays bounded: Rust, MPI, CUDA and Python.**

STL-Dev-MPI computes the scattering-covariance statistics (`mean`, `var`, S1, S2, S3, S4, with
`L = 4` orientations) of full-sky HEALPix maps, disc patches and periodic images, together with
their exact gradient. With them it synthesises new maps whose statistics match those of a target
map or of a population of maps. It is the high-performance port of the patch-local engine of
[STL-Dev](https://github.com/STL-Proj/STL-Dev): it gives the same statistics to 1e-10, but
computes the gradient block by block over the multiresolution pyramid. The memory of a process is
therefore set by its share of the sky, not by the full sky: nside 2048 maps on one GPU, nside 1024
on 192 CPU cores.

```
target map(s) ─► statistics S(target)                               stl-stats   · NativeEngine
white noise x ─► S(x) and ∂loss/∂x, block by block over the pyramid ─► L-BFGS ─► synthesis
                 loss = Σ_i w_i |S_i(x) − S_i(target)|²                stl-synth   · stl-ensemble · stl-synth2d
your network  ─► S(patches), C_ℓ and their exact VJP in PyTorch         PatchEngine · SpectrumEngine
(Q, U)        ─► E/B statistics, local, no SHT                          NativeEngine(spin=2)
noisy map d   ─► U-Net u = f(d) trained by scattering losses            NativeUNet
```

## Where to go next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Build the binaries and the Python bindings, with or without MPI and CUDA.
:::

:::{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc
One page per tool and engine: what it does, key parameters, a minimal example, pitfalls and
costs.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Notebooks on real data: lensing synthesis, statistics with masks, 2D turbulence, a PyTorch loss.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc
The Python engines, every command-line option, and the Rust crates.
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
api/index
terminology
```
