#!/bin/bash
# Build stl-synth (MPI) on a compute node of the hpc cluster.
#   sbatch --export=ALL,W=$W $W/src/tools/hpc/build.slurm
#SBATCH --job-name=stl-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -eu
W=${W:-$HOME/stl-mpi}
source "$W/src/tools/hpc/env.sh"
cd "$W/src"
cargo build --release --offline -j "$SLURM_CPUS_PER_TASK" -p stl-cli --features mpi
ls -la "$CARGO_TARGET_DIR/release/stl-synth"
