# Environment of the GPU build and runs on the hpc cluster (partition gpu:
# 8 x A100 or 8 x H200 per node).  Reuses the toolchain, libclang and
# OpenMPI of env.sh (W, default $HOME/stl-mpi) and builds in G (default
# $HOME/stl-gpu: sources in $G/src, build in $G/target).
G=${G:-$HOME/stl-gpu}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env.sh"
export CARGO_TARGET_DIR=$G/target
# NVRTC, loaded at run time by stl-gpu (LD_LIBRARY_PATH from the module).
module load cuda/13.0.0
# cudarc: driver API version.
export CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION:-13000}
