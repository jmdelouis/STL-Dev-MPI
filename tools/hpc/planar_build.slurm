#!/bin/bash
# Build of stl-synth2d with the GPU engine (feature gpu, no MPI) on a CPU
# node, in $G (sources $G/src, build $G/target).  Crates fetched beforehand
# on the login node (cd $G/src && cargo fetch).
#   sbatch --export=ALL,G=/scratch/jmdeloui/stl-planar2d $G/src/tools/hpc/planar_build.slurm
#SBATCH --job-name=stl-planar-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-/scratch/jmdeloui/stl-planar2d}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
cd "$G/src"
cargo build --release --offline -j "$SLURM_CPUS_PER_TASK" -p stl-cli --features gpu --bin stl-synth2d
cargo test --release --offline -j "$SLURM_CPUS_PER_TASK" -p stl-gpu --test planar --no-run 2>&1 | tail -2
cargo test --release --offline -j "$SLURM_CPUS_PER_TASK" -p stl-cli --features gpu --test planar --no-run 2>&1 | tail -2
ls -la "$CARGO_TARGET_DIR/release/stl-synth2d"
