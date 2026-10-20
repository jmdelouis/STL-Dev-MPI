#!/bin/bash
# Scaling of one synthesis evaluation (forward + gradient) on the hpc
# cluster: the runs of RUNS (rank counts) on the allocated nodes, NITER
# L-BFGS iterations each, target TARGET (white-noise start), L orientations
# (default 2, as the published scaling runs).  Every rank
# reads and writes its own slice of the maps.
#   sbatch --nodes=2 --ntasks-per-node=192 --export=ALL,W=$W,RUNS="384",TARGET=... \
#          $W/src/tools/hpc/synth_scaling.slurm
#SBATCH --job-name=stl-scaling
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -u
W=${W:-$HOME/stl-mpi}
source "$W/src/tools/hpc/env.sh"
OUT="$W/runs/$SLURM_JOB_ID"
mkdir -p "$OUT"
cd "$OUT"
TARGET=${TARGET:-$W/data/kap_nside1024_nested.npy}
NITER=${NITER:-3}
echo "job $SLURM_JOB_ID: $SLURM_JOB_NUM_NODES node(s) $SLURM_JOB_NODELIST, $SLURM_NTASKS tasks, target $TARGET, niter $NITER"
for P in ${RUNS:-$SLURM_NTASKS}; do
    echo "=== $P ranks"
    time mpirun -np "$P" --map-by core --bind-to core \
        "$CARGO_TARGET_DIR/release/stl-synth" --target "$TARGET" --L "${L:-2}" --niter "$NITER" ${EXTRA:-} \
        --out "$OUT/p$P" 2> "$OUT/p$P.err" | tee "$OUT/p$P.log"
    tail -3 "$OUT/p$P.err"
    # Keep the reports; the maps of the largest run only.
    [ "$P" = "${KEEP_MAPS:-}" ] || rm -f "$OUT/p${P}"_*.npy "$OUT/p${P}"_*.fits
done
echo "done $(date)"
