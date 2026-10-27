#!/bin/bash
# Map figures only (fig_maps.py) on the hpc cluster, for the nside listed in
# NSIDES (default "512 1024"), with the runs of paper_figures.slurm.
#
#   sbatch --export=ALL,P=/scratch/jmdeloui/paper text/scripts/maps.slurm
#SBATCH --job-name=stl-paper-maps
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
set -eu
P=${P:-/scratch/jmdeloui/paper}
PY=${PY:-$HOME/stl-mpi/ens/py313/bin/python}
REAL=${REAL:-/scratch/jmdeloui/takahashi}
R=${RUNS_DIR:-$HOME/stl-mpi/ens/runs}
S=$P/src/text/scripts
export MPLCONFIGDIR=$P/.mpl OMP_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR"
for n in ${NSIDES:-512 1024}; do
  case $n in
    512) run="kurtosis=$R/n512_garoupe_kurt_c" ;;
    1024) run="kappa terms=$R/n1024_1978119" ;;
    *) run=${MAPS_RUN:?LABEL=RUN_DIR of the run to show at nside $n} ;;
  esac
  echo "== $n $(date +%T)"
  $PY "$S/fig_maps.py" "$n" "$P/figures" "$REAL" "$run"
done
echo "all done $(date)"
