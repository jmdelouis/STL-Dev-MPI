#!/bin/bash
# stl-synth --gpu with the bound stencils and with the DIA geometry
# (--gpu-conv stencils | dia) on one GPU: seconds per evaluation, device
# memory, phase timers; L = 4, 3 L-BFGS iterations (4 evaluations) from
# white noise on a lognormal target, one member, nsides $NSIDES (default
# "1024 2048").  Sources in $G/src, build in $G/target (G default
# $SCRATCH/stl-dia; build with tools/hpc/build_gpu.slurm and G set), outputs
# in $G/runs/$SLURM_JOB_ID.  Submit (one A100): sbatch tools/hpc/synth_dia.slurm
#SBATCH --job-name=stl-dia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
set -euo pipefail
export G=${G:-$SCRATCH/stl-dia}
source "$G/src/tools/hpc/env_gpu.sh"
OUT=$G/runs/$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, GPU ${CUDA_VISIBLE_DEVICES:-?}, $(cat $G/src/COMMIT 2>/dev/null)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
export STL_GPU_THREADS=$SLURM_CPUS_PER_TASK
for N in ${NSIDES:-1024 2048}; do
    T=$G/field_nside${N}_nested.npy
    if [ ! -f "$T" ]; then
        python3 - "$N" "$T" <<'PY'
import sys
import numpy as np
nside, path = int(sys.argv[1]), sys.argv[2]
n = 12 * nside * nside
rng = np.random.default_rng(5)
x = rng.standard_normal(n)
for _ in range(2):
    x = np.repeat(x.reshape(-1, 4).mean(axis=1), 4) + 0.5 * rng.standard_normal(n)
np.save(path, np.exp(0.5 * x))
PY
    fi
    for C in stencils dia; do
        out=synth_n${N}_L4_$C
        echo "== nside $N $C $(date +%T)"
        "$CARGO_TARGET_DIR/release/stl-synth" --target "$T" --niter 3 --L 4 --gpu --gpu-conv "$C" --quiet --out "$out" > "$out.log" 2>&1 || echo "FAILED ($?)"
        tail -n 1 "$out.log"
        rm -f "${out}"_*.fits "${out}"_*.npy
        python3 - "$out.json" <<'PY' || true
import json, sys
r = json.load(open(sys.argv[1]))
g = r["gpu"]
print("  s/eval", r["seconds_per_evaluation"], "setup", r.get("setup_seconds"), "peak GB", [round(v / 1e9, 1) for v in g["memory_peak_bytes_per_rank"]],
      "cache", g["geometry_cache"], [round(v / 1e9, 1) for v in g["geometry_cache_used_bytes_per_rank"]], "blocks", g["blocks_per_level"][:2])
t = r["timers_seconds"]
print("  geometry", round(t.get("gpu_geometry", 0), 1), "weights", round(t.get("gpu_geometry_weights", 0), 1),
      "forward", round(t["forward"], 1), "reverse", round(t["reverse"], 1))
PY
    done
done
echo "done $(date)"
