#!/bin/bash
# Multi-rank runs under MPI: the same synthesis on 1, 2 and 4 ranks gives
# the same losses (to the order of the sums); stl-mpi-check compares the
# distributed statistics and gradients with an STL-Dev fixture.
# Needs binaries built with --features mpi (and gpu for the GPU line):
#   cargo build --release -p stl-cli -p stl-mpi --features stl-cli/mpi,stl-cli/gpu,stl-mpi/mpi
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../common.sh"
: "${NITER:=10}" "${MPIRUN:=mpirun}" "${GPUS:=0}"
read -ra MPI <<< "$MPIRUN"          # e.g. MPIRUN="mpirun --bind-to none"

for n in 1 2 4; do
    step "stl-synth, $n MPI ranks" "${MPI[@]}" -n "$n" "$STL_BIN/stl-synth" --target "$KAPPA64" --L 4 \
        --weights ergodic --niter "$NITER" --seed 1 --quiet --out "$OUT/mpi$n"
done
if [ "$GPUS" -gt 0 ]; then
    step "stl-synth, $GPUS MPI ranks, one GPU each" "${MPI[@]}" -n "$GPUS" "$STL_BIN/stl-synth" --target "$KAPPA64" --L 4 \
        --weights ergodic --niter "$NITER" --seed 1 --quiet --gpu --out "$OUT/mpi_gpu$GPUS"
fi
"$PY" "$HERE/../plotting.py" summary "$OUT"/mpi*.json
"$PY" - "$OUT"/mpi*.json <<'PY'
import json, sys
import numpy as np
ref = np.array(json.load(open(sys.argv[1]))["loss_history"])
for p in sys.argv[2:]:
    h = np.array(json.load(open(p))["loss_history"])
    n = min(len(h), len(ref))
    print(f"{p.rsplit('/', 1)[-1]}: max relative difference of the loss history to 1 rank {np.max(np.abs(h[:n] / ref[:n] - 1)):.1e}")
PY

CASE=$ROOT/fixtures/stats/stats_n32_J4_L4_k5_C1_B1_full_b256
for n in 1 2 4; do
    step "stl-mpi-check, $n ranks" "${MPI[@]}" -n "$n" "$STL_BIN/stl-mpi-check" "$CASE"
done
