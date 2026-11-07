# API Reference

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python
:link: python
:link-type: doc
`stl_port.native`: `NativeEngine`, `PatchEngine`, `SpectrumEngine`, `NativeUNet`, `native_lbfgs`,
and the `stl_port.denoise` helpers, generated from the docstrings.
:::

:::{grid-item-card} Command line
:link: cli/index
:link-type: doc
Every option of `stl-synth`, `stl-stats`, `stl-ensemble`, `stl-synth2d`, `stl-minkowski`,
`stl-mpi-check`, generated from the usage strings in the code.
:::

:::{grid-item-card} Rust
:link: rust
:link-type: doc
The five crates, their main types, and the rustdoc.
:::

:::{grid-item-card} Reports and measurements
:link: ../reports
:link-type: doc
Validation against STL-Dev, benchmarks, design notes and production runs.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

python
cli/index
rust
../reports
```
