#!/bin/bash
# Build of the harmonic-terms benchmark (docs/harmonic_speed.md) on a CPU
# node: stl-ensemble and the test binary of crates/stl-cli/tests/ensemble.rs
# (device_profile), with MPI and the GPU engine, in $G (sources $G/src,
# build $G/target).  Crates fetched beforehand on the login node
# (cd $G/src && cargo fetch).
#   sbatch --export=ALL,G=/scratch/jmdeloui/stl-campeti-hs $G/src/tools/hpc/harmonic_bench_build.slurm
#SBATCH --job-name=stl-hbench-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-/scratch/jmdeloui/stl-campeti-hs}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
cd "$G/src"
cargo build --release --offline -j "$SLURM_CPUS_PER_TASK" -p stl-cli --features mpi,gpu --bin stl-ensemble
cargo test --release --offline -j "$SLURM_CPUS_PER_TASK" -p stl-cli --features mpi,gpu --test ensemble --no-run 2>&1 | tail -3
ls -la "$CARGO_TARGET_DIR/release/stl-ensemble" $(ls -t "$CARGO_TARGET_DIR"/release/deps/ensemble-* | grep -v '\.d$' | head -1)
