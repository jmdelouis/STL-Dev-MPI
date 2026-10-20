# Environment of the GPU build on garoupe (3 x Tesla T4, driver 470 / CUDA
# 11.4, OpenMPI 4.1 without CUDA support).  Source it from bash:
#   W=$HOME/stl-gpu-work; source $W/src/tools/garoupe/env.sh
: "${W:=$HOME/stl-gpu-work}"
export W
export RUSTUP_HOME=$W/rustup CARGO_HOME=$W/cargo
export PATH=$CARGO_HOME/bin:$PATH
# Build products on the local disk (the home is on NFS).
export CARGO_TARGET_DIR=/tmp/$USER-stl-gpu-target
# cudarc: driver API of CUDA 11.4.
export CUDARC_CUDA_VERSION=11040
# NVRTC 11.8 (PyPI wheel; the system one is 10.0) and libclang for the mpi crate.
NVRTC_DIR=$W/py/nvidia/cuda_nvrtc/lib
export LIBCLANG_PATH=$W/py/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
export LD_LIBRARY_PATH=$W/lib:$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
