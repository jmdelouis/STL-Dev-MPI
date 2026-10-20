#!/bin/bash
# Coarse-to-fine production on Irene: one chain_up job per target nside,
# each started from the synthesis at 2N of the previous one (afterok).
#   bash tools/irene/submit_c2f.sh NAME "N:ranks:niter:seconds" [...]
# e.g. bash tools/irene/submit_c2f.sh takahashi_zs1 512:512:400:7200 1024:1024:150:7200 2048:2048:80:7200
# Targets in $S/prep/NAME_N, runs in $S/runs/NAME_c2f_N.
set -eu
W=${W:-/ccc/work/cont005/gen17889/delouisj/stl-mpi}
source "$W/src/tools/irene/env.sh"
NAME=$1; shift
dep=""; init=""
mkdir -p "$S/runs" && cd "$S/runs"
for step in "$@"; do
    IFS=: read -r N P NITER SECS <<< "$step"
    RUN=$S/runs/${NAME}_c2f_$N
    EXPORT="ALL,PREP=$S/prep/${NAME}_$N,RUN=$RUN,L=${L:-4},NITER=$NITER${init:+,INIT=$init}"
    E="--export=$EXPORT${dep:+ --dependency=afterok:$dep}"
    out=$(ccc_msub -r "stl-c2f-$N" -n "$P" -T "$SECS" -E "$E" "$W/src/tools/irene/chain_up.msub")
    id=$(echo "$out" | grep -oE '[0-9]+' | tail -n 1)
    echo "nside $N: $P ranks, $NITER iterations, ${SECS} s -> job $id ($out)"
    dep=$id; init=$RUN/up_synthesis_nested.npy
done
