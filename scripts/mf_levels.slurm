#!/bin/bash
# Minkowski functionals of kappa per pyramid level (mf_levels.py) for the
# nside 512 and 1024 ensembles against the 10 Takahashi realizations.
#
#   sbatch --export=ALL,P=/scratch/jmdeloui/paper text/scripts/mf_levels.slurm
#
# NSIDES selects the steps (default "512 1024"); for 2048 add the runs in
# RUNS2048 (as in paper_figures.slurm).
#SBATCH --job-name=stl-paper-mfl
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
set -eu
P=${P:-/scratch/jmdeloui/paper}
PY=${PY:-$HOME/stl-mpi/ens/py313/bin/python}
REAL=${REAL:-/scratch/jmdeloui/takahashi}
R=${RUNS_DIR:-$HOME/stl-mpi/ens/runs}
S=$P/src/text/scripts
export MPLCONFIGDIR=$P/.mpl
for N in ${NSIDES:-512 1024}; do
  echo "== $N $(date +%T)"
  case $N in
    512) $PY $S/mf_levels.py $P/cache 512 $P/figures $REAL "kurtosis=$R/n512_garoupe_kurt_c" "no kurtosis=$R/n512_garoupe" ;;
    1024) $PY $S/mf_levels.py $P/cache 1024 $P/figures $REAL "kappa terms=$R/n1024_1978119" "no kappa terms=$R/n1024_1935615" ;;
    2048) eval "set -- ${RUNS2048:?LABEL=RUN_DIR arguments of the 2048 ensembles}"
          $PY $S/mf_levels.py $P/cache 2048 $P/figures $REAL "$@" --procs 5 --threads 6 ;;
  esac
done
echo "all done $(date)"
