#!/bin/bash
# Benchmark of the device harmonic terms (docs/harmonic_speed.md) on one
# A100, at most 30 min (each step under a timeout, 27 min in all), built
# by harmonic_bench_build.slurm:
# - device_profile (crates/stl-cli/tests/ensemble.rs): the transforms of
#   one map, then the forward and reverse passes of the C_l, PDF and
#   Minkowski terms for 5 and 10 members, at nside 1024 and 2048, with the
#   Legendre GEMMs (and the former kernels, STL_SHT=kernels, at 1024), the
#   phases timed (STL_SHT_PHASES);
# - three evaluations of stl-ensemble at nside 1024, ten members, the
#   production options (calibration copied to $G/cal1024.json).
# Results in $G/bench/JOB: prof_*.json, run1024.log, run1024.json.
#   sbatch --export=ALL,G=/scratch/jmdeloui/stl-campeti-hs $G/src/tools/hpc/harmonic_bench_a100.slurm
#SBATCH --job-name=stl-hbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -u
G=${G:-/scratch/jmdeloui/stl-campeti-hs}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
OUT=$G/bench/$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$OUT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export STL_THREADS=$SLURM_CPUS_PER_TASK STL_GPU_THREADS=$SLURM_CPUS_PER_TASK
T=$(ls -t "$CARGO_TARGET_DIR"/release/deps/ensemble-* | grep -v '\.d$' | head -1)
B=$CARGO_TARGET_DIR/release/stl-ensemble
(while sleep 30; do date +%T; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; done) > gpu_usage.log 2>&1 &
MON=$!
# tag, seconds allowed, then the environment of the profile
prof() {
    tag=$1
    secs=$2
    shift 2
    start=$(date +%s)
    env "$@" STL_SHT_PHASES=1 STL_PROFILE_REPS=2 timeout "$secs" "$T" device_profile --ignored --nocapture --exact 2> "prof_$tag.err" | grep '^{' > "prof_$tag.json"
    rc=${PIPESTATUS[0]}
    echo "prof $tag: $(( $(date +%s) - start )) s, exit $rc"
}
prof n1024_g10 240 STL_PROFILE_NSIDE=1024 STL_PROFILE_MEMBERS=10
start=$(date +%s)
STL_GPU_CACHE_MB=40960 timeout 480 mpirun -np 1 --bind-to none "$B" run --calibration "$G/cal1024.json" --ps-variance-upper 3071 \
    --niter 1 --max-eval 3 --gpu --gpu-fields recompute --out run1024 2> run1024.err | tee run1024.log
echo "run1024: $(( $(date +%s) - start )) s"
prof n2048_g10 420 STL_PROFILE_NSIDE=2048 STL_PROFILE_MEMBERS=10
prof n1024_g5_kernels 240 STL_PROFILE_NSIDE=1024 STL_PROFILE_MEMBERS=5 STL_SHT=kernels
prof n2048_g5 240 STL_PROFILE_NSIDE=2048 STL_PROFILE_MEMBERS=5
kill $MON 2>/dev/null || true
for f in prof_*.json; do echo "== $f"; cat "$f"; done
echo "done $(date)"
