#!/bin/bash
# Complete synthesis chain at nside N on the hpc cluster:
#   1. gaussianise the target T_N (rank anamorphosis, table kept);
#   2. bilinear up to 2N (the new finest level carries no power of its own);
#   3. synthesis at 2N (ergodic weights);
#   4. cut back to N (mean of the 4 children);
#   5. rank-based histogram matching on T_N.
#   sbatch --nodes=1 --ntasks=96 --export=ALL,W=$W,TARGET=$W/data/x_nside256_nested.npy,L=4,NITER=400 \
#          $W/src/tools/hpc/chain_up.slurm
# Optional: INIT (a NESTED map at nside <= 2N in the units of the
# gaussianised up target, e.g. up_synthesis_nested.npy of the chain at
# N / 2, coarse-to-fine), EXTRA (more
# stl-synth options).
#SBATCH --job-name=stl-chain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out

set -eu
W=${W:-$HOME/stl-mpi}
source "$W/src/tools/hpc/env.sh"
PY=${PY:-$W/pyenv/bin/python}
T=$W/src/tools
OUT="$W/runs/$SLURM_JOB_ID"
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NUM_NODES node(s) $SLURM_JOB_NODELIST, $SLURM_NTASKS tasks, target $TARGET, L ${L:-4}, niter ${NITER:-400}, init ${INIT:-noise}"
if [ -n "${PREP:-}" ]; then
    # Steps 1-2 done elsewhere (healpix_geo needs Python >= 3.10):
    # PREP holds target_gauss_up_nested.npy and gauss_table.npz.
    ln -sf "$PREP/target_gauss_up_nested.npy" target_gauss_up_nested.npy
    cp "$PREP/gauss_table.npz" .
else
    cp "$TARGET" target_nested.npy
    time "$PY" "$T/gaussianize.py" forward target_nested.npy target_gauss_nested.npy gauss_table.npz
    time "$PY" "$T/resample_nested.py" up target_gauss_nested.npy target_gauss_up_nested.npy
fi
INIT_OPT=""
[ -n "${INIT:-}" ] && INIT_OPT="--init $INIT"
time mpirun -np "$SLURM_NTASKS" --map-by core --bind-to core \
    "$CARGO_TARGET_DIR/release/stl-synth" --target target_gauss_up_nested.npy \
    --L "${L:-4}" --weights ergodic --niter "${NITER:-400}" $INIT_OPT ${EXTRA:-} \
    --out up 2> up.err | tee up.log
tail -3 up.err
time "$PY" "$T/resample_nested.py" down up_synthesis_nested.npy synthesis_gauss_nested.npy
time "$PY" "$T/gaussianize.py" match gauss_table.npz synthesis_gauss_nested.npy synthesis_nested.npy
rm -f up_target.fits
echo "done $(date)"
