#!/bin/bash
# Short test of stl-synth --gpu on the GPU nodes: setup time, seconds per
# evaluation and device memory (2 L-BFGS iterations from white noise).
#   sbatch --gres=gpu:a100:2 --ntasks=2 --export=ALL,TARGET=...,L=4 \
#          $HOME/stl-gpu/src/tools/hpc/gpu_test.slurm
#SBATCH --job-name=stl-gpu-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-$HOME/stl-gpu}
source "$G/src/tools/hpc/env_gpu.sh"
OUT="$G/runs/$SLURM_JOB_ID"
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks, target $TARGET, L ${L:-4}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
(while sleep 60; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done) > gpu_usage.log 2>&1 &
MON=$!
export STL_GPU_THREADS=$SLURM_CPUS_PER_TASK
time mpirun -np "$SLURM_NTASKS" --bind-to none \
    "$CARGO_TARGET_DIR/release/stl-synth" --gpu --target "$TARGET" \
    --L "${L:-4}" --weights ergodic --niter "${NITER:-2}" ${EXTRA:-} --out test 2> test.err | tee test.log
kill $MON 2>/dev/null || true
tail -5 test.err
rm -f test_*.fits test_*.npy
python3 - test.json <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print({k: r[k] for k in ("nside", "J", "L", "coefficients", "evaluations", "seconds_per_evaluation", "setup_seconds")})
print(r["gpu"])
print(r["timers_seconds"])
EOF
