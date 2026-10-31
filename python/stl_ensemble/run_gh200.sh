#!/usr/bin/env bash
set -euo pipefail
echo 'SCRIPT VERSION: run_gh200.sh 2026-10-29-v2-variance-ratio'
port_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${STL_PYTHON:=${port_dir}/.venv/bin/python}"
: "${STL_THREADS:=${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="$STL_THREADS"
export MKL_NUM_THREADS="$STL_THREADS"
export OPENBLAS_NUM_THREADS="$STL_THREADS"
export NUMEXPR_NUM_THREADS="$STL_THREADS"
command -v numactl >/dev/null || { echo 'numactl is required to place optimizer history/checkpoints on CPU NUMA node 0.' >&2; exit 1; }
test -x "$STL_PYTHON" || { echo "Set STL_PYTHON to the isolated environment Python: $STL_PYTHON" >&2; exit 1; }
cd "$port_dir"
policy="${STL_NUMA_POLICY:-strict}"
case "$policy" in
 strict) args=(--cpunodebind=0 --membind=0);;
 cpu-only) args=(--cpunodebind=0);;
 *) echo 'STL_NUMA_POLICY must be strict or cpu-only' >&2; exit 2;;
esac
echo "NUMA policy: $policy (cpu-only does not enforce memory placement)"
exec numactl "${args[@]}" "$STL_PYTHON" synthesize.py --threads "$STL_THREADS" "$@"
