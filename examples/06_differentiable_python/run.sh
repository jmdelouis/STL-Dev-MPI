#!/bin/bash
# Differentiable STL statistics and pseudo-spectra from Python: torch
# wrappers of NativeEngine (masks, cross), PatchEngine and SpectrumEngine,
# three losses optimised with torch.optim.LBFGS at nside 32 (CPU, a few
# minutes on a laptop).  Needs libstl_bridge.so (cargo build --release -p
# stl-bridge) and a Python with numpy, torch, matplotlib and healpix-geo.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../common.sh"
: "${STL_NATIVE_LIBRARY:=$STL_BIN/libstl_bridge.so}"
export STL_NATIVE_LIBRARY

step "three differentiable losses (torch LBFGS)" "$PY" "$HERE/run.py" "$OUT" "$KAPPA64" "$HERE/losses.png"
