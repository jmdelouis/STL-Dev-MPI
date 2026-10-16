#!/usr/bin/env bash
# Multi-rank equality (CLAUDE.md validation 4) under mpirun: every stats
# fixture with 1 rank (saved as the reference), then with the other rank
# counts compared with the fixture and with the 1-rank run.
#
#   tools/mpi_check.sh [rank counts...]     (default: 2 4)
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/stl-dev-mpi-target}"
cargo build --release -p stl-mpi --features mpi --bin stl-mpi-check
BIN="$CARGO_TARGET_DIR/release/stl-mpi-check"
COUNTS=("${@:-2 4}")
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
status=0
for case in fixtures/stats/*/; do
    name="$(basename "$case")"
    mpirun -n 1 "$BIN" "$case" --save "$OUT/$name.bin" || status=1
    for n in ${COUNTS[@]}; do
        mpirun --oversubscribe -n "$n" "$BIN" "$case" --reference "$OUT/$name.bin" || status=1
    done
done
exit $status
