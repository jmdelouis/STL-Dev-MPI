#!/bin/bash
# STL ensemble minimisation at full nside on the hpc GPU nodes, from one
# initial map per member (the STL + diffusion hybrid maps), one H200 per MPI
# rank, the members split in groups (one group per rank by default).
#   sbatch --export=ALL,TARGET=$SCRATCH/diff2048/g2048.npy,INIT='$SCRATCH/diff2048/hybrid_m{m}.npy',B=4,NITER=150,OUT=$SCRATCH/diff2048/polish \
#          diffusion/hpc_polish.slurm
# Keep inputs and outputs on $SCRATCH: every member writes two 0.4 GB maps at
# nside 2048, and the home (NFS) quota is too small for them.
# G: the stl-synth build with MPI and GPU (stl-cli --features mpi,gpu) of a
# commit with --ensemble, --scatter-weight and {m} in --init (c0ac285 or later).
# Also runs on the A100 80 GB nodes (--gres=gpu:a100:1; NVRTC compiles the
# kernels for the device): 4 members at nside 2048 take 125 s per evaluation
# and 71 GB there, against 58 s on an H200.  Always name the GPU type.
# 3 cores per rank: the GPU does the work, and the user CPU quota (192 cores,
# GPU jobs included) is mostly held by the ensemble production chain.
#SBATCH --job-name=stl-hybrid-polish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=3
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-$HOME/stl-diff-gpu}
source "$G/src/tools/hpc/env_gpu.sh"
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks, B ${B}, groups ${NGROUPS:-$SLURM_NTASKS}, niter ${NITER}, init ${INIT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
(while sleep 600; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done) > gpu_usage.log 2>&1 &
MON=$!
export STL_GPU_THREADS=$SLURM_CPUS_PER_TASK
time mpirun -np "$SLURM_NTASKS" --bind-to none "$CARGO_TARGET_DIR/release/stl-synth" --gpu --target "$TARGET" \
    --L 4 --weights ergodic --ensemble "$B" --groups "${NGROUPS:-$SLURM_NTASKS}" --scatter-weight 1 \
    --niter "$NITER" --init "$INIT" --init-noise 0 ${EXTRA:-} --out polished 2> polished.err | tee polished.log
kill $MON 2>/dev/null || true
tail -3 polished.err
echo "done $(date)"
