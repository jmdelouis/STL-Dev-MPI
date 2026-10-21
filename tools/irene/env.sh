# Environment of the STL-Dev-MPI build and runs on Irene Rome (TGCC;
# Slurm behind ccc_msub, OpenMPI 4.1; sourced by the batch scripts).
# W: the work directory (sources, build); data and runs go to scratch.
# No internet on Irene: sources come with their crates vendored
# (cargo vendor, .cargo/config.toml), the build is --offline.
W=${W:-/ccc/work/cont005/gen17889/delouisj/stl-mpi}
S=${S:-/ccc/scratch/cont005/gen17889/delouisj/stl-mpi}
# MPI_MODULE selects the MPI library (default OpenMPI 4.1.6; e.g.
# mpi/intelmpi/20.0.4); each library gets its own build directory.
MPI_MODULE=${MPI_MODULE:-mpi/openmpi/4.1.6}
case $MPI_MODULE in
    *intelmpi*) export CARGO_TARGET_DIR=$W/target-impi ;;
    *) export CARGO_TARGET_DIR=$W/target ;;
esac
export CARGO_HOME=$W/cargo
module load rust/1.93.0 python3/3.12 > /dev/null 2>&1
module unload mpi > /dev/null 2>&1
module load $MPI_MODULE > /dev/null 2>&1
# libclang (build time only, bindgen of the mpi crate): the system's LLVM 17.
export LIBCLANG_PATH=/usr/lib64/llvm17/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
# python3/3.12 ships numpy and scipy.
PY=${PY:-python3}
