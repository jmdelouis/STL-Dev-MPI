# STL-Dev-MPI

[![Docs](https://img.shields.io/badge/docs-sphinx-blue.svg)](docs/index.md)
[![Rust](https://img.shields.io/badge/rust-stable%202021-orange.svg)](Cargo.toml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](python/stl_ensemble/stl_port/native.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](Cargo.toml)

**STL-Dev-MPI** computes scattering-covariance statistics of maps on the sphere
([HEALPix](https://healpix.sourceforge.io/)) and of periodic 2D images, together with their
exact gradient, and synthesises new maps whose statistics match those of a target map or of a
population of maps, at resolutions where automatic differentiation runs out of memory: nside 2048
on one GPU, nside 1024 on 192 CPU cores.

It is the Rust + MPI + CUDA port of the patch-local engine of
[STL-Dev](https://github.com/STL-Proj/STL-Dev). The statistics (`mean`, `var`, S1, S2, S3, S4 with
`L = 4` orientations) and their gradient are computed **block by block over the multiresolution
pyramid**, with hand-written vector–Jacobian products instead of an autograd graph, and the sky
is split between MPI ranks: the memory of a process is bounded by its share of the sky. The
results match STL-Dev to 1e-10, and 1, 2, 4, ... ranks give the same statistics to 1e-15. Python
bindings expose the same engines, with their VJPs, to NumPy and PyTorch.

## Tools at a glance

| Tool | What it does | Good for |
|---|---|---|
| `stl-stats` | statistics of HEALPix maps, with the plan, in JSON | looking at a target, checking a synthesis |
| `stl-synth` | synthesis of one HEALPix map, or of an ensemble matching one target; extra terms for level power, histogram tails, kurtosis, Minkowski functionals | maps from one target, at any nside, on CPU ranks or GPUs |
| `stl-ensemble` | ensembles whose mean and variance of statistics match a population (P. Campeti's method: ScatCov of `g` and `κ`, C_ℓ, PDF, Minkowski) | simulation-like ensembles of weak-lensing maps |
| `stl-synth2d` | synthesis of periodic 2D images, ensemble loss, coarse-to-fine starts | images, turbulence, textures |
| `NativeEngine` (Python) | full-sky statistics and their VJP; masks; cross statistics `S(a, b)` | losses in NumPy / PyTorch, masked skies |
| `NativeEngine(spin=2)` (Python) | E/B statistics of polarisation `(Q, U)` from local eth-bar² fields, no SHT; masks; QU × QU and T × QU cross statistics | polarisation losses, dust, masked skies |
| `PatchEngine` (Python) | statistics of disc patches and their VJP, batched, CPU or GPU | training networks on patches of the sky |
| `SpectrumEngine` (Python) | differentiable anafast: auto, cross, spin `s` (`EE`, `BB`, `EB`), masks | holding the power spectrum in a loss |
| `NativeUNet` (Python) | a spherical U-Net denoiser and its training by scattering losses (frozen or live noise biases), Adam or L-BFGS, on one device | denoising Planck-like maps at nside 256 on one GPU |

## Installation

### From source

```bash
git clone https://github.com/jmdelouis/STL-Dev-MPI.git      # private: ask jmdelouis for access
cd STL-Dev-MPI
cargo build --release -p stl-cli                            # CPU; ranks as threads (--ranks N)
cargo build --release -p stl-cli --features "gpu mpi"       # + one CUDA device per rank, mpirun
cargo build --release -p stl-bridge --features gpu          # Python bindings: libstl_bridge.so
export STL_NATIVE_LIBRARY=$PWD/target/release/libstl_bridge.so
export PYTHONPATH=$PWD/python/stl_ensemble:$PYTHONPATH      # import stl_port.native
```

### Requirements

- Rust stable (edition 2021) and a C compiler;
- feature `mpi`: an MPI library (OpenMPI 4/5, Intel MPI, MPICH) and `libclang` at build time;
- feature `gpu`: an NVIDIA driver and NVRTC at run time (no CUDA toolkit needed to build;
  `CUDARC_CUDA_VERSION=12040` sets the driver API version); cuFFT for the harmonic terms;
- Python bindings: Python 3.10+ and NumPy (PyTorch for the `*_device` methods);
- tools and figures: `scipy`, `matplotlib`, `healpix-geo`.

### Verifying the installation

```bash
cargo test --release                                     # fixture tests against STL-Dev
python -c "from stl_port.native import NativeEngine; print(NativeEngine(16, 3).ncoeff)"
```

Details, computing centres without root rights, batch scripts:
[installation](docs/installation.md) and [USAGE](docs/USAGE.md).

## Quickstart

```bash
# synthesis of a HEALPix map from a target (NESTED .npy or FITS) on a GPU
stl-synth --target map_nested.npy --weights ergodic --niter 200 --gpu --out run
# its statistics against the target's
stl-stats --out stats.json --norm map_nested.npy -- map_nested.npy run_synthesis_nested.npy
# 2D: STL-Dev's example, 10 images of 256 x 256
stl-synth2d --target Turb_6.npy --ntarget 10 --nbatch 10 --c2f 64,128 --niter 300 --gpu --out run2d
```

```python
from stl_port.native import NativeEngine
eng = NativeEngine(nside=128, j=6, backend="gpu")
values, var = eng.forward(x, moments=True)     # x [1, npix] NESTED -> complex [1, ncoeff]
grad = eng.vjp(2 * (values - target))          # d sum|S - target|^2 / dx
```

| Synthesis of a lensing ensemble at nside 64 | STL-Dev's 2D example |
|---|---|
| ![](examples/02_healpix_ensemble/maps.png) | ![](examples/04_planar_synthesis/ensemble_maps.png) |

Same synthesis as STL-Dev, nside 128 × 4 members, `L = 4`, one A100 ([report](docs/bench_n128.md)):

| Code | s / evaluation | GPU memory |
|---|---|---|
| STL-Dev (autograd) | 1.229 | 19.8 GB |
| STL-Dev-MPI, fastest | **0.106** | 2.2 GB |
| STL-Dev-MPI, least memory | 0.885 | **0.58 GB** |

## Documentation

The documentation site (Sphinx, PyData theme, MyST; `docs/build.sh`) starts at
[docs/index.md](docs/index.md):

- [Installation](docs/installation.md)
- [User Guide](docs/user-guide/index.md): one page per tool and engine
- [Tutorials](docs/tutorials/index.md): notebooks on real data and command-line examples
- [API Reference](docs/api/index.md): Python, every command-line option, Rust crates
- [Terminology](docs/terminology.md)
- [Reports and measurements](docs/reports.md), [validation against STL-Dev](docs/validation.md)

## Development

```bash
cargo test --release                                  # all fixture tests (threads backend)
tools/mpi_check.sh 2 4                                # under mpirun, 1, 2 and 4 ranks
CUDARC_CUDA_VERSION=12040 cargo test --release -p stl-gpu
pip install -r docs/requirements.txt && docs/build.sh # documentation site + rustdoc, warnings are errors
```

The model, architecture and validation rules are in [CLAUDE.md](CLAUDE.md); the vocabulary and
numerics follow STL-Dev (block, halo, piece, sweep, native level, stored moduli, plan).

## Target applications

- Weak-lensing convergence and other cosmological fields: emulation of simulations, ensembles,
  non-Gaussian statistics on the full sky and on masked skies
- Training neural networks with scattering-statistics losses (denoising, component separation)
  on the sphere, for intensity and polarisation
- Ocean and atmosphere fields on HEALPix grids, and 2D turbulence images
- Large syntheses on HPC clusters and GPUs (nside 1024 to 2048 and beyond)

## License

MIT (declared in `Cargo.toml`). If you use STL-Dev-MPI, please cite it with the metadata of
[CITATION.cff](CITATION.cff).
