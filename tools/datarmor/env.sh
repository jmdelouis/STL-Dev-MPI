# Environment of the STL-Dev-MPI build and runs on Datarmor (sourced by the
# PBS scripts).  W: the work directory (toolchain, sources, build, runs).
W=${W:-/home1/scratch/jmdeloui/stl-mpi}
export RUSTUP_HOME=$W/rustup CARGO_HOME=$W/cargo CARGO_TARGET_DIR=$W/target
# The login profile puts the 'vacumm' conda env first; its gcc cannot link
# position-independent executables (no Scrt1.o): use the system gcc.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/envs/vacumm' | paste -sd:)
source $CARGO_HOME/env
source /usr/share/Modules/3.2.10/init/bash 2>/dev/null || true
module load impi/2019.9.304
export CC=/usr/bin/gcc I_MPI_CC=/usr/bin/gcc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc
# No Rust/clang module on Datarmor: libclang (build time only, for the MPI
# bindings) comes from a shared conda environment.
export LIBCLANG_PATH=/appli/conda-env/jupyterhub-ai/lib
