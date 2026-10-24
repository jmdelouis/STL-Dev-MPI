#!/bin/bash
# Sample an ensemble from the per-level models and evaluate it against the
# target and the STL syntheses, at the target's nside and at nside 64.
#   diffusion/run_eval.sh TARGET.npy MODEL_DIR OUT_DIR COARSE_NSIDE SAMPLES LEVELS [LABEL=MAP.npy ...]
# LEVELS: comma-separated levels for the memorisation test (e.g. 32,64,128,256).
# PYTHON (default python), SAMPLE_OPTS (sampler options, e.g. --churn 40) and COARSE
# (coarsest map: grf, target or a NESTED map file; default grf) from the environment.
set -e
here=$(cd "$(dirname "$0")" && pwd)
target=$1 models=$2 out=$3 coarse=$4 samples=$5 levels=$6
shift 6
py=${PYTHON:-python}
mkdir -p "$out"
nside=$($py -c "import numpy as np; print(int(round((np.load('$target', mmap_mode='r').size / 12) ** 0.5)))")
$py -u "$here/sample.py" "$target" "$models" "$nside" "$out/samples.npy" --coarse-nside "$coarse" \
    --samples "$samples" --coarse "${COARSE:-grf}" ${SAMPLE_OPTS:-} 2>&1 | grep -v Warning | tee "$out/sample.log"
$py -u "$here/evaluate.py" "$out/eval$nside" "$target" "diffusion=$out/samples.npy" "$@" 2>&1 | grep -v Warning \
    | tee "$out/eval$nside.txt"
$py -u "$here/evaluate.py" --nside 64 "$out/eval64" "$target" "diffusion=$out/samples.npy" "$@" 2>&1 \
    | grep -v Warning | tee "$out/eval64.txt"
$py -u "$here/memorisation.py" "$target" "$models" ${levels//,/ } --samples 4 ${SAMPLE_OPTS:-} --json "$out/memorisation.json" 2>&1 \
    | grep -v Warning | tee "$out/memorisation.txt"
$py -c "import numpy as np; np.save('$out/sample0.npy', np.load('$out/samples.npy')[0])"
# Full sky, 20 x 20 deg zoom, histograms and small-scale anisotropy (STL tool).
$py "$here/../tools/compare_syntheses.py" "$out/compare.png" "$target" "diffusion=$out/sample0.npy" "$@" 2>&1 \
    | grep -v Warning | tee "$out/compare.txt"
