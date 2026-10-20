#!/usr/bin/env bash
# Fetch the target and synthesis maps of finished hpc jobs and compare them.
#   tools/hpc/fetch_compare.sh OUT_DIR TARGET_LABEL:JOB:RANKS LABEL:JOB:RANKS [...]
# e.g. tools/hpc/fetch_compare.sh runs/c2f_L2_L4 "L=2:1524067:192" "L=4:1526994:160"
# The target is taken from the first job; maps land in OUT_DIR (ignored by git).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=$1; shift
mkdir -p "$OUT"
PY=${STL_PYTHON:-$HOME/stl-deps/venv-cpu/bin/python}
specs=()
first=1
for item in "$@"; do
    IFS=: read -r label job ranks <<< "$item"
    tag=$(echo "$label" | tr -c 'A-Za-z0-9' '_')
    rsync -az "hpc:stl-mpi/runs/$job/p${ranks}_synthesis_nested.npy" "$OUT/${tag}_synthesis_nested.npy"
    rsync -az "hpc:stl-mpi/runs/$job/p${ranks}.json" "$OUT/${tag}_report.json"
    if [ $first = 1 ]; then
        rsync -az "hpc:stl-mpi/runs/$job/p${ranks}_target.fits" "$OUT/target.fits"
        first=0
    fi
    specs+=("$label=$OUT/${tag}_synthesis_nested.npy")
done
"$PY" -I tools/compare_syntheses.py "$OUT/compare.png" "$OUT/target.fits" "${specs[@]}"
