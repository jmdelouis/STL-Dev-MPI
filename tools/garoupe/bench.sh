#!/bin/bash
# Seconds per evaluation (forward + gradient) and device memory of
# stl-synth --gpu on garoupe: 3 L-BFGS iterations (4 evaluations) from
# white noise, J = log2(nside) - 1, kernel 5, one T4 per MPI rank.
#   W=$HOME/stl-gpu-work bash $W/src/tools/garoupe/bench.sh "256 1024 2048" "2 4" "1 2 3"
# Runs in $BENCH (default /tmp/$USER-stl-gpu-bench: the home quota is
# small); the output maps are deleted, the JSON reports kept.  Targets:
# $BENCH/field_nside<N>_nested.npy (made here if missing: a lognormal
# field, its content does not change the cost).
set -u
: "${W:=$HOME/stl-gpu-work}"
source "$W/src/tools/garoupe/env.sh"
NSIDES=${1:-"256 1024 2048"}
LS=${2:-"2 4"}
RANKS=${3:-"1 2 3"}
BIN=$CARGO_TARGET_DIR/release/stl-synth
: "${BENCH:=/tmp/$USER-stl-gpu-bench}"
mkdir -p "$BENCH" && cd "$BENCH"
for N in $NSIDES; do
    f=field_nside${N}_nested.npy
    if [ ! -f "$f" ]; then
        PYTHONPATH=$W/py python3 - "$N" <<'EOF'
import sys
import numpy as np
nside = int(sys.argv[1])
n = 12 * nside * nside
rng = np.random.default_rng(5)
x = rng.standard_normal(n)
for _ in range(2):
    x = np.repeat(x.reshape(-1, 4).mean(axis=1), 4) + 0.5 * rng.standard_normal(n)
np.save(f"field_nside{nside}_nested.npy", np.exp(0.5 * x))
EOF
    fi
    for L in $LS; do
        for P in $RANKS; do
            out=bench_n${N}_L${L}_P${P}
            [ -f "$out.json" ] && continue
            echo "== nside $N L $L ranks $P $(date +%T)"
            STL_GPU_THREADS=$((12 / P)) mpirun --bind-to none -n "$P" "$BIN" --target "$f" --niter 3 --L "$L" --gpu --quiet --out "$out" > "$out.log" 2>&1
            tail -1 "$out.log"
            rm -f "$out"_*.fits "$out"_*.npy
        done
    done
done
