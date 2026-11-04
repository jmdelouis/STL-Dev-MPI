#!/bin/bash
# Synthesis of one HEALPix map from a target map (Takahashi weak-lensing kappa, nside 64):
# 1. directly, from white noise, on the CPU (thread ranks) and on a GPU;
# 2. with the recipe for skewed fields (docs/USAGE.md, section 4): Gaussian
#    anamorphosis, synthesis at 2 nside, back to nside, histogram matching.
# Then the figures.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../common.sh"
: "${NITER:=100}" "${RANKS:=8}" "${GPU:=1}"
TOOLS=$ROOT/tools

# J defaults to log2(nside) - 1 (5 at nside 64, 6 at 128), L to 4, kernel 5;
# ergodic weights make the loss a reduced chi^2 that fits every family.
COMMON=(--L 4 --weights ergodic --niter "$NITER" --seed 1 --quiet)
OPTS=(--target "$KAPPA64" "${COMMON[@]}")
if [ "$GPU" = 1 ]; then DEVICE=(--gpu); else DEVICE=(--ranks "$RANKS"); fi

# Runs whose report exists are skipped (delete OUT to rerun them).
[ -f "$OUT/cpu.json" ] || step "direct, CPU, $RANKS thread ranks" "$STL_BIN/stl-synth" "${OPTS[@]}" --ranks "$RANKS" --out "$OUT/cpu"
if [ "$GPU" = 1 ]; then
    [ -f "$OUT/gpu.json" ] || step "direct, GPU" "$STL_BIN/stl-synth" "${OPTS[@]}" --gpu --out "$OUT/gpu"
fi

# The recipe: the Gaussianised target, interpolated to nside 128 so that its
# new finest level carries little power of its own (--keep-mean: every group
# of 4 children keeps its parent's value as mean, so cutting back gives the
# target exactly); the synthesis at 128 (GPU, or CPU thread ranks with GPU=0)
# is cut back to 64 and given the target's histogram.
if [ ! -f "$OUT/recipe.json" ]; then
    "$PY" "$TOOLS/gaussianize.py" forward "$KAPPA64" "$OUT/target_gauss.npy" "$OUT/table.npz"
    "$PY" "$TOOLS/resample_nested.py" up "$OUT/target_gauss.npy" "$OUT/target_gauss_up.npy" --keep-mean
    step "recipe, synthesis at nside 128" "$STL_BIN/stl-synth" --target "$OUT/target_gauss_up.npy" "${COMMON[@]}" "${DEVICE[@]}" --out "$OUT/recipe"
fi
"$PY" "$TOOLS/resample_nested.py" down "$OUT/recipe_synthesis_nested.npy" "$OUT/recipe_gauss.npy"
"$PY" "$TOOLS/gaussianize.py" match "$OUT/table.npz" "$OUT/recipe_gauss.npy" "$OUT/recipe_nested.npy"

MAPS=(target="$KAPPA64" "direct synthesis (CPU)=$OUT/cpu_synthesis_nested.npy")
LOSS=("direct, CPU, $RANKS thread ranks=$OUT/cpu.json")
if [ "$GPU" = 1 ]; then
    MAPS+=("direct synthesis (GPU)=$OUT/gpu_synthesis_nested.npy")
    LOSS+=("direct, GPU=$OUT/gpu.json")
fi
MAPS+=("recipe: Gaussian, 2 nside, histogram=$OUT/recipe_nested.npy")
LOSS+=("recipe, nside 128=$OUT/recipe.json")
"$PY" "$HERE/../plotting.py" summary "$OUT"/*.json
"$PY" "$HERE/../plotting.py" maps "$HERE/maps.png" "${MAPS[@]}"
"$PY" "$HERE/../plotting.py" loss "$HERE/loss.png" "${LOSS[@]}"
