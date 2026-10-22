#!/bin/bash
# Build stl-synth with MPI and the GPU engine on a compute node (crates
# fetched beforehand on the login node: cd $G/src && cargo fetch).
#   sbatch $HOME/stl-gpu/src/tools/hpc/build_gpu.slurm
#SBATCH --job-name=stl-build-gpu
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-$HOME/stl-gpu}
source "$G/src/tools/hpc/env_gpu.sh"
cd "$G/src"
cargo build --release --offline -j "$SLURM_CPUS_PER_TASK" -p stl-cli --features mpi,gpu
ls -la "$CARGO_TARGET_DIR/release/stl-synth"
