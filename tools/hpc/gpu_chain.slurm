#!/bin/bash
# Synthesis chain at nside N on the GPU nodes of the hpc cluster, one GPU
# per MPI rank (as tools/hpc/chain_up.slurm on CPUs):
#   1. gaussianise the target T_N (rank anamorphosis, table kept);
#   2. bilinear up to 2N;
#   3. synthesis at 2N (ergodic weights) with stl-synth --gpu;
#   4. cut back to N (mean of the 4 children);
#   5. rank-based histogram matching on T_N.
#   sbatch --gres=gpu:a100:2 --ntasks=2 \
#          --export=ALL,PREP=$HOME/stl-mpi/prep/takahashi_zs1_2048,L=4,NITER=150,INIT=... \
#          $HOME/stl-gpu/src/tools/hpc/gpu_chain.slurm
# TARGET (the NESTED target at N) when steps 1-2 are to be done here, or
# PREP (a directory holding target_gauss_up_nested.npy and gauss_table.npz).
# Optional: INIT (a NESTED map at nside <= 2N in the units of the
# gaussianised up target, e.g. up_synthesis_nested.npy of the chain at
# N / 2), EXTRA (more stl-synth options).  The number of ranks must equal
# the number of GPUs (--ntasks = GPUs of --gres).
#SBATCH --job-name=stl-gpu-chain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-$HOME/stl-gpu}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
PY=${PY:-$W/pyenv/bin/python}
T=$G/src/tools
OUT="${RUNS:-$W/runs}/$SLURM_JOB_ID"
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks, GPUs ${CUDA_VISIBLE_DEVICES:-?}, target ${TARGET:-$PREP}, L ${L:-4}, niter ${NITER:-400}, init ${INIT:-noise}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
if [ -n "${PREP:-}" ]; then
    ln -sf "$PREP/target_gauss_up_nested.npy" target_gauss_up_nested.npy
    cp "$PREP/gauss_table.npz" .
else
    cp "$TARGET" target_nested.npy
    time "$PY" "$T/gaussianize.py" forward target_nested.npy target_gauss_nested.npy gauss_table.npz
    time "$PY" "$T/resample_nested.py" up target_gauss_nested.npy target_gauss_up_nested.npy
fi
INIT_OPT=""
[ -n "${INIT:-}" ] && INIT_OPT="--init $INIT"
# Device memory every 10 minutes (the report holds the peak at the end).
(while sleep 600; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done) > gpu_usage.log 2>&1 &
MON=$!
export STL_GPU_THREADS=$SLURM_CPUS_PER_TASK
time mpirun -np "$SLURM_NTASKS" --bind-to none \
    "$CARGO_TARGET_DIR/release/stl-synth" --gpu --target target_gauss_up_nested.npy \
    --L "${L:-4}" --weights ergodic --niter "${NITER:-400}" $INIT_OPT ${EXTRA:-} \
    --out up 2> up.err | tee up.log
kill $MON 2>/dev/null || true
tail -3 up.err
time "$PY" "$T/resample_nested.py" down up_synthesis_nested.npy synthesis_gauss_nested.npy
time "$PY" "$T/gaussianize.py" match gauss_table.npz synthesis_gauss_nested.npy synthesis_nested.npy
rm -f up_target.fits
echo "done $(date)"
