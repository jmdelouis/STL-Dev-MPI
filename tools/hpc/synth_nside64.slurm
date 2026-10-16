#!/bin/bash
# Synthesis of the WebSky kappa map at nside 64 on one node of the hpc
# cluster with 1 to 128 MPI ranks (OpenMPI 5), from the same initial map
# as the STL-Dev reference (fixtures/synthesis); per-rank peak memory in
# the JSON reports.
#   sbatch --export=ALL,W=$W $W/src/tools/hpc/synth_nside64.slurm
#SBATCH --job-name=stl-synth64
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

set -u
W=${W:-$HOME/stl-mpi}
source "$W/src/tools/hpc/env.sh"
OUT="$W/runs/$SLURM_JOB_ID"
mkdir -p "$OUT"
cd "$OUT"
FIX=$(ls -d "$W"/src/fixtures/synthesis/synthesis_kap_nside64*/ | head -1)
echo "job $SLURM_JOB_ID on $(hostname) ($(lscpu | sed -n 's/^Model name: *//p')), $SLURM_NTASKS tasks"
# Runs "ranks:block_pixels[:conv_chunk]".
for RUN in ${RUNS:-1:16384 4:16384 16:16384 64:16384 128:16384 1:4096 1:16384:2048 128:16384:2048}; do
    IFS=: read -r P B C <<< "$RUN"
    TAG=p${P}_b${B}${C:+_c$C}
    echo "=== $P ranks, blocks of $B pixels${C:+, geometry by sub-blocks of $C rows}"
    # No /usr/bin/time on the nodes: stl-synth reports the per-rank peak (VmHWM).
    time mpirun -np "$P" --bind-to core \
        "$CARGO_TARGET_DIR/release/stl-synth" \
        --target "$W/data/kap_nside64_nested.npy" \
        --init "$FIX/initial_map.npy" --init-units standardized --init-noise 0 \
        --block-pixels "$B" ${C:+--conv-chunk $C} --niter 20 --out "$OUT/$TAG" 2> "$OUT/$TAG.err" | tee "$OUT/$TAG.log"
    tail -3 "$OUT/$TAG.err"
done
echo "done $(date)"
