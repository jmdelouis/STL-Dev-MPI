#!/bin/bash
# Figures and tables of the paper on the hpc cluster: the nside 512 and 1024
# ensembles against the 10 Takahashi realizations, and the caches of the
# realizations at nside 2048 (so that only the members remain to measure
# when the 2048 ensembles arrive).
#
#   sbatch --export=ALL,P=/scratch/jmdeloui/paper text/scripts/paper_figures.slurm
#
# P holds src/ (a copy of text/scripts and tools/minkowski.py with the
# repository layout), cache/ and figures/.  STEPS selects what to run
# (default "test 512 1024 2048real"; test = the Gaussian test of
# tools/minkowski.py).  For the 2048 ensembles: STEPS=2048 and
# RUNS2048='LABEL=RUN_DIR ...' (see text/TODO_2048.md).
#SBATCH --job-name=stl-paper-figs
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
set -eu
P=${P:-/scratch/jmdeloui/paper}
export PY=${PY:-$HOME/stl-mpi/ens/py313/bin/python}
export REAL=${REAL:-/scratch/jmdeloui/takahashi}
export CACHE=$P/cache OUT=$P/figures
R=${RUNS_DIR:-$HOME/stl-mpi/ens/runs}
S=$P/src/text/scripts
export MPLCONFIGDIR=$P/.mpl
mkdir -p "$MPLCONFIGDIR"
for step in ${STEPS:-test 512 1024 2048real}; do
  echo "== $step $(date +%T)"
  case $step in
    test)
      $PY $P/src/tools/test_minkowski.py ;;
    512)
      PROCS=8 THREADS=4 bash $S/make_all.sh 512 "no kurtosis=$R/n512_garoupe" "kurtosis=$R/n512_garoupe_kurt_c" ;;
    1024)
      PROCS=8 THREADS=4 bash $S/make_all.sh 1024 "no kappa terms=$R/n1024_1935615" "kappa terms=$R/n1024_1978119" ;;
    2048real)
      PROCS=5 THREADS=6 bash $S/make_all.sh 2048 ;;
    2048)
      eval "set -- ${RUNS2048:?LABEL=RUN_DIR arguments of the 2048 ensembles}"
      PROCS=5 THREADS=6 bash $S/make_all.sh 2048 "$@" ;;
  esac
done
echo "all done $(date)"
