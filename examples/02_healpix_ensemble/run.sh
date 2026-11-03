#!/bin/bash
# Ensemble synthesis with P. Campeti's method (stl-ensemble): ten members
# whose mean and variance of statistics (ScatCov of g and of kappa, C_l,
# pixel PDF, Minkowski functionals) match those of ten reference maps.
#   REFS="r000.npy ... r009.npy"  NESTED reference maps (any nside >= NSIDE)
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../common.sh"
: "${REFS:?set REFS to the reference maps (NESTED .npy), e.g. the Takahashi r000-r009}"
: "${NSIDE:=64}" "${NITER:=300}" "${GPU:=1}"
GPUOPT=()
[ "$GPU" = 1 ] && GPUOPT=(--gpu)

# 1. Calibration: LUT g -> kappa pooled over the references, targets of the
#    five blocks, C_l of the initial Gaussian fields.
# shellcheck disable=SC2086
[ -f "$OUT/cal.json" ] || step "calibrate" "$STL_BIN/stl-ensemble" calibrate --references $REFS \
    --nside "$NSIDE" --out "$OUT/cal.json" "${GPUOPT[@]}" --quiet
# 2. Synthesis: ten members in one batch on one device.  The C_l variance
#    term covers the bands up to 3 nside - 1, the production rule.
step "run" "$STL_BIN/stl-ensemble" run --calibration "$OUT/cal.json" --niter "$NITER" \
    --ps-variance-upper $((3 * NSIDE - 1)) "${GPUOPT[@]}" --out "$OUT/ens" --quiet

"$PY" "$HERE/../plotting.py" summary "$OUT/ens.json"
# First reference block-averaged to NSIDE (NESTED: groups of 4^k children),
# mean removed as the members' kappa = f(g) - mean.
set -- $REFS
"$PY" -c "import numpy as np; r = np.load('$1').astype(float); r = r.reshape(12 * $NSIDE**2, -1).mean(1); np.save('$OUT/ref0.npy', r - r.mean())"
"$PY" "$HERE/../plotting.py" maps "$HERE/maps.png" --range 1,99.5 "reference r000=$OUT/ref0.npy"     "member 0=$OUT/ens_m000_physical_nested.npy" "member 1=$OUT/ens_m001_physical_nested.npy"
"$PY" "$HERE/../plotting.py" loss "$HERE/loss.png" "stl-ensemble, 10 members=$OUT/ens.json"
