#!/bin/bash
# Synthesis of one HEALPix map from a target map (Takahashi weak-lensing kappa, nside 64),
# on the CPU (thread ranks) and on a GPU, then the figures.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../common.sh"
: "${NITER:=100}" "${RANKS:=8}" "${GPU:=1}"

# J defaults to log2(nside) - 1 = 5, L to 4, kernel 5; ergodic weights make
# the loss a reduced chi^2 that fits every family.
OPTS=(--target "$KAPPA64" --L 4 --weights ergodic --niter "$NITER" --seed 1 --quiet)

# Runs whose report exists are skipped (delete OUT to rerun them).
[ -f "$OUT/cpu.json" ] || step "CPU, $RANKS thread ranks" "$STL_BIN/stl-synth" "${OPTS[@]}" --ranks "$RANKS" --out "$OUT/cpu"
if [ "$GPU" = 1 ]; then
    [ -f "$OUT/gpu.json" ] || step "GPU" "$STL_BIN/stl-synth" "${OPTS[@]}" --gpu --out "$OUT/gpu"
fi

MAPS=(target="$KAPPA64" "synthesis (CPU)=$OUT/cpu_synthesis_nested.npy")
LOSS=("CPU, $RANKS thread ranks=$OUT/cpu.json")
if [ "$GPU" = 1 ]; then
    MAPS+=("synthesis (GPU)=$OUT/gpu_synthesis_nested.npy")
    LOSS+=("GPU=$OUT/gpu.json")
fi
"$PY" "$HERE/../plotting.py" summary "$OUT"/*.json
"$PY" "$HERE/../plotting.py" maps "$HERE/maps.png" "${MAPS[@]}"
"$PY" "$HERE/../plotting.py" loss "$HERE/loss.png" "${LOSS[@]}"
