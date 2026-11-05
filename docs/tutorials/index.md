# Tutorials

Notebooks on real data, each run in a few minutes on one GPU (garoupe, Tesla T4) and committed
with their outputs: the figures and timings below come from those runs. They live in
`docs/tutorials/` (regenerate them with `make_notebooks.py`, then
`jupyter nbconvert --execute --inplace`). The command-line examples of `examples/` follow.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Weak-lensing synthesis
:link: takahashi_synthesis
:link-type: doc
Synthesise the Takahashi κ map at nside 128 with `stl-synth` on a GPU: Gaussian anamorphosis,
synthesis, histogram matching, spectra and statistics.
:::

:::{grid-item-card} Statistics of a map
:link: map_statistics
:link-type: doc
`NativeEngine` on the same map: the plan, ergodic error bars, latitude masks, cross statistics and
an exact gradient.
:::

:::{grid-item-card} 2D turbulence
:link: turbulence_2d
:link-type: doc
STL-Dev's 2D example with `stl-synth2d`: ensemble loss and coarse-to-fine start, 10 images of 256².
:::

:::{grid-item-card} A PyTorch loss
:link: torch_loss
:link-type: doc
`PatchEngine` and `SpectrumEngine` as `torch.autograd.Function`s: patch statistics plus the power
spectrum, optimised with Adam.
:::

::::

## Command-line examples

Scripts in `examples/`, with their READMEs, figures and timings:

| Example | Tool |
|---|---|
| {doc}`healpix_synthesis` | `stl-synth` on CPU and GPU, WebSky κ at nside 64 |
| {doc}`healpix_ensemble` | `stl-ensemble`, 10 members, P. Campeti's method |
| {doc}`statistics` | `stl-stats` on a target, its synthesis and white noise |
| {doc}`planar_synthesis` | `stl-synth2d`, STL-Dev's 2D example on CPU and GPU |
| {doc}`mpi` | `mpirun` on 1, 2, 4 ranks and 3 GPUs; `stl-mpi-check` |
| {doc}`differentiable` | the Python engines wrapped for torch: masked synthesis, patch synthesis, denoising with cross statistics (laptop CPU) |

```{toctree}
:hidden:
:maxdepth: 1
:caption: Notebooks

takahashi_synthesis
map_statistics
turbulence_2d
torch_loss
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Command-line examples

healpix_synthesis
healpix_ensemble
statistics
planar_synthesis
mpi
differentiable
```
