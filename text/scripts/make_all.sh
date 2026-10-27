#!/bin/bash
# Measure the realizations and the ensemble members at one nside (cached),
# then draw every data figure and table of the paper for that nside.
#
#   make_all.sh NSIDE "LABEL=RUN_DIR" ["LABEL=RUN_DIR" ...]
#
# Environment: PY (python with healpy, scipy, matplotlib), REAL (directory of
# rXXX_nside{N}_nested.npy), CACHE, OUT (figures), PROCS / THREADS (measure.py),
# MAPS_RUN (the label of the run shown in the map figures, default the last).
set -eu
N=$1; shift
S=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
: "${REAL:?directory of the realizations}" "${CACHE:?cache directory}" "${OUT:?figure directory}"
mkdir -p "$CACHE" "$OUT"
$PY "$S/measure.py" "$CACHE" "$N" "$REAL" "$@" --procs "${PROCS:-4}" --threads "${THREADS:-2}"
if [ $# -gt 0 ]; then
  for f in fig_cl fig_levels fig_peaks_pdf fig_minkowski table_zscores; do
    $PY "$S/$f.py" "$CACHE" "$N" "$OUT" "$@"
  done
  last=${MAPS_RUN:-${!#}}
  $PY "$S/fig_maps.py" "$N" "$OUT" "$REAL" "$last"
fi
echo "nside $N done $(date)"
