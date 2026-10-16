# Environment of the STL-Dev-MPI build and runs on the hpc cluster
# (Centre Datarmor, Slurm, RHEL 9; sourced by the Slurm scripts).
# W: the work directory (toolchain, sources, build, runs).
W=${W:-$HOME/stl-mpi}
export RUSTUP_HOME=$W/rustup CARGO_HOME=$W/cargo CARGO_TARGET_DIR=$W/target
source $CARGO_HOME/env
source /usr/share/Modules/init/bash 2>/dev/null || true
module load openmpi/5.0.11
# No Rust or clang module: libclang (build time only, for the MPI bindings)
# comes from the PyPI 'libclang' wheel installed under $W/libclang.
export LIBCLANG_PATH=$W/libclang/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
