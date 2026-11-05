# Installation

STL-Dev-MPI is a Cargo workspace. It builds with a recent stable Rust toolchain on Linux
(and in WSL); MPI and CUDA are optional features of the command-line tools.

## Requirements

| Need | For | Notes |
|---|---|---|
| Rust (stable, edition 2021) | everything | `rustup`; no root rights needed (`RUSTUP_HOME`, `CARGO_HOME` in a work directory) |
| C compiler (`gcc`) | linking | the system compiler |
| An MPI library with `mpicc` (OpenMPI 4/5, Intel MPI, MPICH) | feature `mpi` | usually a `module load` |
| `libclang` | feature `mpi`, build time only | bindgen generates the MPI bindings; a module, `apt install libclang-dev`, or the PyPI `libclang` wheel |
| NVIDIA driver (`libcuda.so`) and NVRTC (`libnvrtc.so`) | feature `gpu`, run time only | no CUDA toolkit is needed to build: the kernels are compiled by NVRTC when a run starts |
| cuFFT (`libcufft.so`) | `stl-ensemble --harmonic gpu` | or `STL_CUFFT=/path/to/libcufft.so` |
| Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `healpix-geo` | target preparation and figures (`tools/`, `examples/`) | optional |

The engine itself is pure Rust and `f64` throughout (no fast-math). Without MPI, every
tool still runs several ranks as threads of one process (`--ranks N`), which is enough up to
nside 256 on one node.

## Get the sources

```bash
git clone https://github.com/jmdelouis/STL-Dev-MPI.git
cd STL-Dev-MPI
```

The repository is private: ask `jmdelouis` for access. `main` is the reference branch; every
merge into it has passed the fixture tests.

## Build

::::{tab-set}

:::{tab-item} CPU (threads)

```bash
cargo build --release -p stl-cli
ls target/release/stl-synth target/release/stl-synth2d target/release/stl-stats target/release/stl-ensemble
```

:::

:::{tab-item} MPI

```bash
# libclang for bindgen, and the compiler's own headers
export LIBCLANG_PATH=/usr/lib/llvm-14/lib            # or the PyPI wheel: $W/libclang/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
cargo build --release -p stl-cli -p stl-mpi --features stl-cli/mpi,stl-mpi/mpi
mpirun -n 4 target/release/stl-synth --target map.npy --out run
```

:::

:::{tab-item} GPU

```bash
# the driver's CUDA version (nvidia-smi), e.g. 12040 for CUDA 12.4, 11040 for 11.4
export CUDARC_CUDA_VERSION=12040
cargo build --release -p stl-cli --features gpu
target/release/stl-synth --target map.npy --gpu --out run
```

:::

:::{tab-item} Python bindings

```bash
cargo build --release -p stl-bridge                  # add --features gpu for backend="gpu"
export STL_NATIVE_LIBRARY=$PWD/target/release/libstl_bridge.so
export PYTHONPATH=$PWD/python/stl_ensemble:$PYTHONPATH
python -c "from stl_port.native import NativeEngine; print(NativeEngine(16, 3).ncoeff)"
```

:::

:::{tab-item} GPU + MPI

```bash
export CUDARC_CUDA_VERSION=12040 LIBCLANG_PATH=... BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
cargo build --release -p stl-cli --features "gpu mpi"
mpirun -n 4 target/release/stl-synth --target map.npy --gpu --out run   # one device per rank
```

:::

::::

| Feature | Crate | Effect |
|---|---|---|
| `mpi` | `stl-cli`, `stl-mpi` | the tools run under `mpirun` (`MPI_COMM_WORLD`); `stl-mpi-check` is built |
| `gpu` | `stl-cli`, `stl-mpi` | `--gpu`: one CUDA device per rank (`crates/stl-gpu`) |
| `gpu` | `stl-bridge` | `backend="gpu"` in the Python engines |

Keep the build directory on local storage (`CARGO_TARGET_DIR`): network file systems and
`/mnt/c` in WSL make the build much slower. Compute nodes without internet access build with
`cargo fetch` on a login node first, then `cargo build --offline`.

## Verify

```bash
cargo test --release                              # fixture tests against STL-Dev (threads backend)
tools/mpi_check.sh 2 4                            # the same statistics under mpirun, 1, 2 and 4 ranks
CUDARC_CUDA_VERSION=12040 cargo test --release -p stl-gpu   # skipped without a device
```

Every test compares with fixtures dumped from STL-Dev (`fixtures/`, `tools/dump_reference.py`):
statistics to 1e-11 absolute and 1e-10 relative, gradients to 1e-10, and 1, 2, 4 ranks
identical to 1e-15.

## Python bindings

`python/stl_ensemble/stl_port/native.py` loads `libstl_bridge.so` with `ctypes`: it needs NumPy
only (PyTorch for `PatchEngine.forward_device` / `vjp_device`). The library is found through
`STL_NATIVE_LIBRARY`, or the `library=` argument of every engine; its SHA-256 is recorded in each
engine's `info()`. See {doc}`user-guide/native-engine`.

## Python tools

```bash
python3 -m venv venv
venv/bin/pip install numpy scipy matplotlib healpix-geo
```

They prepare targets (`tools/prepare_healpix_target.py`, `gaussianize.py`,
`resample_nested.py`) and draw the figures (`tools/plot_mollweide.py`, `plot_planar.py`,
`examples/plotting.py`).

## Computing centres

[Using STL-Dev-MPI on a computing centre](USAGE.md) covers installation without root rights,
libclang from PyPI, environment files, Slurm, PBS and TGCC batch scripts, and sizing. Ready-made
environments are in `tools/hpc/` (Slurm, OpenMPI 5, A100), `tools/datarmor/` (PBS, Intel MPI),
`tools/irene/` (TGCC) and `tools/garoupe/` (3 × Tesla T4, driver 470, NVRTC from PyPI).
