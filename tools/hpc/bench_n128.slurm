#!/bin/bash
# docs/bench_n128.md: 4-member synthesis at nside 128, L = 4, on one A100,
# STL-Dev-MPI (stl-synth) against the legacy full-sky autograd path of
# STL-Dev (tools/bench_n128/legacy_synth.py), minimum-memory and
# minimum-time configurations, same target, plan, loss, L-BFGS and initial
# maps. One run after the other on the same GPU, nvidia-smi sampled every
# 200 ms per run.
#   sbatch --export=ALL,G=/scratch/jmdeloui/stl-bench128 $G/src/tools/hpc/bench_n128.slurm
# Options (environment): NITER (100), B (4), NSIDE (128, of TARGET), RUNS (the configurations below,
# space-separated), PY (python with torch + CUDA), STLDEV (directory with
# STL_main and healpix_analyse), TARGET, OUT.
#SBATCH --job-name=stl-b128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -u
G=${G:-/scratch/jmdeloui/stl-bench128}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
NITER=${NITER:-100}
B=${B:-4}
NSIDE=${NSIDE:-128}
PY=${PY:-$G/env/bin/python}
STLDEV=${STLDEV:-$G/py}
TARGET=${TARGET:-$G/data/takahashi_r000_zs1_nside128_nested.npy}
OUT=${OUT:-$G/runs/$SLURM_JOB_ID}
RUNS=${RUNS:-rust_time rust_mem rust_cpu legacy_batch legacy_sequential}
BIN=$CARGO_TARGET_DIR/release/stl-synth
T=$SLURM_CPUS_PER_TASK
# Build here when build_gpu.slurm has not run yet (no-op otherwise; crates
# fetched beforehand on the login node).
(cd "$G/src" && cargo build --release --offline -j "$T" -p stl-cli --features mpi,gpu 2>&1 | tail -2) || exit 1
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, GPU ${CUDA_VISIBLE_DEVICES:-?}, $T CPUs, niter $NITER, B $B, commit $(cat $G/src/COMMIT)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$PY" "$G/src/tools/bench_n128/make_inits.py" "$NSIDE" "$B" init
COMMON="--target $TARGET --L 4 --weights none --ensemble $B --niter $NITER --init init_m{m}.npy --init-units standardized --init-noise 0"
export STL_THREADS=$T STL_GPU_THREADS=$T OMP_NUM_THREADS=$T

run() {
    local name=$1; shift
    echo "=== $name $(date +%T): $*"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > "gpu_$name.csv" 2>&1 &
    local mon=$!
    /usr/bin/time -v "$@" > "$name.log" 2> "$name.err"
    echo "exit $?"
    kill $mon 2>/dev/null; wait $mon 2>/dev/null
    grep -E "Maximum resident|Elapsed \(wall" "$name.err"
    tail -2 "$name.log"
}

for r in $RUNS; do
    case $r in
    rust_time) run rust_time mpirun -np 1 --bind-to none "$BIN" $COMMON --gpu --gpu-fields keep --gpu-terms device --gpu-cache full --out rust_time ;;
    rust_mem) run rust_mem mpirun -np 1 --bind-to none "$BIN" $COMMON --gpu --gpu-fields recompute --gpu-cache none --gpu-block-rows 1024 --block-pixels 1024 --out rust_mem ;;
    rust_cpu) run rust_cpu mpirun -np "$T" --bind-to core "$BIN" $COMMON --block-pixels 1024 --out rust_cpu ;;
    legacy_batch) run legacy_batch "$PY" "$G/src/tools/bench_n128/legacy_synth.py" --stl-dev "$STLDEV" --target "$TARGET" --init 'init_m{m}.npy' --ensemble "$B" --L 4 --niter "$NITER" --mode batch --device cuda --threads "$T" --out legacy_batch.json ;;
    legacy_sequential) run legacy_sequential "$PY" "$G/src/tools/bench_n128/legacy_synth.py" --stl-dev "$STLDEV" --target "$TARGET" --init 'init_m{m}.npy' --ensemble "$B" --L 4 --niter "$NITER" --mode sequential --device cuda --threads "$T" --out legacy_sequential.json ;;
    esac
done
rm -f init_m*.npy *_synthesis.fits
echo "done $(date)"
