# Tutorials

Six worked examples, each a script in `examples/` that runs in 1 to 12 minutes on a
workstation with one GPU (the sixth on a laptop CPU). They were run on garoupe (12 Xeon cores, Tesla T4s, shared with other
jobs at the time), and the figures and timings below come from those runs.

```bash
cargo build --release -p stl-cli --features gpu       # examples 01 to 04 (drop gpu and set GPU=0 without a device)
bash examples/01_healpix_synthesis/run.sh
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} HEALPix synthesis
:link: healpix_synthesis
:link-type: doc
One map from the Takahashi weak-lensing map at nside 64, on CPU threads and on a GPU, with `stl-synth`.
:::

:::{grid-item-card} Ensemble synthesis
:link: healpix_ensemble
:link-type: doc
Ten members with P. Campeti's method (`stl-ensemble`), calibrated on ten reference maps.
:::

:::{grid-item-card} Statistics of maps
:link: statistics
:link-type: doc
`stl-stats` on a target, its synthesis and white noise; the plan and the values in JSON.
:::

:::{grid-item-card} 2D synthesis
:link: planar_synthesis
:link-type: doc
STL-Dev's 2D example, 10 images of 256², with `stl-synth2d` on CPU and GPU.
:::

:::{grid-item-card} Multi-rank runs
:link: mpi
:link-type: doc
The same synthesis on 1, 2 and 4 MPI ranks (and one GPU per rank); `stl-mpi-check`.
:::

:::{grid-item-card} Differentiable losses in Python
:link: differentiable
:link-type: doc
torch wrappers of `NativeEngine`, `PatchEngine` and `SpectrumEngine`: masked synthesis, patch
synthesis and denoising with cross statistics, on a laptop CPU.
:::

::::

Every script takes its paths from the environment: `STL_BIN` (the binaries, default
`target/release`), `PY` (a Python with numpy, matplotlib and healpix-geo, for the figures) and
`OUT` (outputs, default `examples/<name>/out`, not versioned); `GPU=0` skips the GPU runs.

```{toctree}
:hidden:
:maxdepth: 1

healpix_synthesis
healpix_ensemble
statistics
planar_synthesis
mpi
differentiable
```
