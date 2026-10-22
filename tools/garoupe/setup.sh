#!/bin/bash
# One-time setup of the GPU build on garoupe, everything under $W:
# rustup toolchain, NVRTC 11.8 and libclang from PyPI wheels.
#   W=$HOME/stl-gpu-work bash $W/src/tools/garoupe/setup.sh
set -euo pipefail
: "${W:=$HOME/stl-gpu-work}"
source "$(dirname "$0")/env.sh"
mkdir -p "$W/lib" "$W/py"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
fi
python3 -m pip install --quiet --target "$W/py" --upgrade 'nvidia-cuda-nvrtc-cu11==11.8.*' 'libclang==18.*'
# cudarc looks for libnvrtc.so first: point it at the 11.8 library.
lib=$(ls "$W"/py/nvidia/cuda_nvrtc/lib/libnvrtc.so.* | head -1)
ln -sf "$lib" "$W/lib/libnvrtc.so"
cargo --version
ls -l "$W/lib"
