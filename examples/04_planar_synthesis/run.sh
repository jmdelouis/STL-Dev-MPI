#!/bin/bash
# Planar (2D, periodic) synthesis: STL-Dev's example, 10 images of 256 x 256
# from the first 10 images of Turb_6.npy (STL-Dev repository, data/test/).
#   TURB=/path/to/STL-Dev/data/test/Turb_6.npy
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../common.sh"
: "${TURB:?set TURB to STL-Dev data/test/Turb_6.npy}"
: "${NITER:=100}" "${GPU:=1}"
COMMON=(--target "$TURB" --ntarget 10 --nbatch 10)

# 1. STL-Dev's own loss (synthesize_from_maps: ScatCov with L = 4, J = 6,
#    and the power spectrum), 100 L-BFGS iterations, CPU then GPU.
step "stldev loss, CPU" "$STL_BIN/stl-synth2d" "${COMMON[@]}" --mode stldev --niter "$NITER" --out "$OUT/stldev_cpu"
if [ "$GPU" = 1 ]; then
    step "stldev loss, GPU" "$STL_BIN/stl-synth2d" "${COMMON[@]}" --mode stldev --niter "$NITER" --gpu --out "$OUT/stldev_gpu"
fi
# 2. Ensemble loss (ScatCov + power spectrum + Minkowski functionals, batch
#    mean and variance against the targets'), after a warm start of 100
#    iterations on the stldev loss.
ENSDEV=()
[ "$GPU" = 1 ] && ENSDEV=(--gpu)
step "ensemble loss, warm start" "$STL_BIN/stl-synth2d" "${COMMON[@]}" --mode ensemble --warm 100 \
    --niter "${NITER_ENS:-300}" "${ENSDEV[@]}" --out "$OUT/ensemble"

"$PY" "$HERE/../plotting.py" summary "$OUT"/*.json
mkdir -p "$OUT/fig"
for r in stldev_cpu ensemble; do
    "$PY" "$ROOT/tools/plot_planar.py" --run "$OUT/$r" --fig "$OUT/fig" > "$OUT/fig/$r.summary.json"
done
cp "$OUT/fig/stldev_cpu_maps.png" "$HERE/stldev_maps.png"
cp "$OUT/fig/ensemble_maps.png" "$HERE/ensemble_maps.png"
cp "$OUT/fig/ensemble_stats.png" "$HERE/ensemble_stats.png"
