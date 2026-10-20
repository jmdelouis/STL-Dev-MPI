#!/bin/bash
# Degrade a full-sky map to NESTED targets on a compute node.
#   sbatch --export=ALL,W=$W,SRC=/path/map.npy,PREFIX=$W/data/name,NSIDES="64 256" \
#          $W/src/tools/hpc/prepare_target.slurm
#SBATCH --job-name=stl-prep
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out

set -eu
W=${W:-$HOME/stl-mpi}
# The compute nodes have no numpy: a venv under $W (python3 -m venv $W/pyenv;
# $W/pyenv/bin/pip install numpy scipy, from the login node).
PY=${PY:-$W/pyenv/bin/python}
"$PY" "$W/src/tools/prepare_healpix_target.py" "$SRC" "$PREFIX" $NSIDES --ordering=${ORDERING:-auto}
