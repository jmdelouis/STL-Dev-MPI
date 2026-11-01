#!/bin/bash
# Ensemble synthesis of P. Campeti's method (stl-ensemble, docs/campeti_merge.md)
# on one A100 of the hpc cluster: ten members, one MPI rank, harmonic terms
# on the device; the calibration from the ten references when CAL does not
# exist; the maps of the lowest loss every CKPT evaluations
# (ens_checkpoint_*, restart with INIT=$OUT/ens_checkpoint_m{m}_gaussian_nested.npy).
#   sbatch --dependency=afterany:JOB --export=ALL,G=/scratch/jmdeloui/stl-campeti \
#          /scratch/jmdeloui/stl-campeti/src/tools/hpc/ensemble_a100.slurm
# Options (environment): NSIDE (1024), NITER (2500), CUTOFF (3071, the C_l
# variance cutoff), CKPT (100), REF (directory of r00?_nside${NSIDE}_nested.npy),
# CAL ($G/cal${NSIDE}.json), INIT, RUN (directory name under $G/runs), EXTRA.
#SBATCH --job-name=stl-ens1024
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-/scratch/jmdeloui/stl-campeti}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
NSIDE=${NSIDE:-1024}
NITER=${NITER:-2500}
CUTOFF=${CUTOFF:-3071}
CKPT=${CKPT:-100}
REF=${REF:-/scratch/jmdeloui/takahashi}
CAL=${CAL:-$G/cal${NSIDE}.json}
OUT=$G/runs/${RUN:-$SLURM_JOB_ID}
B=$CARGO_TARGET_DIR/release/stl-ensemble
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, GPU ${CUDA_VISIBLE_DEVICES:-?}, nside $NSIDE, niter $NITER, cutoff $CUTOFF, init ${INIT:-random fields}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export STL_THREADS=$SLURM_CPUS_PER_TASK STL_GPU_THREADS=$SLURM_CPUS_PER_TASK
# Geometry cache of the SC engine: 40 GB, the rest of the 80 GB for the
# harmonic terms (transforms and Minkowski states) next to the engine.
export STL_GPU_CACHE_MB=${STL_GPU_CACHE_MB:-40960}
if [ ! -f "$CAL" ]; then
    time mpirun -np 1 --bind-to none "$B" calibrate --references $(ls "$REF"/r00?_nside${NSIDE}_nested.npy) \
        --nside "$NSIDE" --gpu --out "$CAL" 2>&1 | grep -v Authorization
fi
INIT_OPT=""
[ -n "${INIT:-}" ] && INIT_OPT="--init $INIT"
(while sleep 600; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done) > gpu_usage.log 2>&1 &
MON=$!
time mpirun -np 1 --bind-to none "$B" run --calibration "$CAL" --ps-variance-upper "$CUTOFF" \
    --niter "$NITER" --max-eval 50000 --gpu --gpu-fields recompute --checkpoint "$CKPT" $INIT_OPT ${EXTRA:-} \
    --out ens 2> ens.err | tee ens.log
kill $MON 2>/dev/null || true
tail -3 ens.err
md5sum ens_m0*_nested.npy > md5.txt
touch READY
echo "done $(date)"
