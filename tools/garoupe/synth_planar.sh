#!/bin/bash
# Planar syntheses of the STL-Dev example (10 images of 256^2, the first 10
# Turb_6 targets) on garoupe:
#  1. STL-Dev itself (torch, one T4), 100 L-BFGS iterations, seed 26;
#  2. stl-synth2d --mode stldev from STL-Dev's initial maps (same loss,
#     same L-BFGS): the two loss histories should agree;
#  3. stl-synth2d --mode ensemble (ScatCov + power spectrum + Minkowski,
#     batch mean and variance), $NITER iterations from white noise, and
#     again after a warm start of 100 iterations on the stldev loss;
# then the figures of tools/plot_planar.py.
#   R=/tmp/$USER-planar2d/runs SRC=... STLDEV=... PY=... BIN=.../stl-synth2d bash tools/garoupe/synth_planar.sh
set -u
: "${R:=/tmp/$USER-planar2d/runs}"
: "${NITER:=500}"
mkdir -p "$R/fig" && cd "$R"
TURB=$STLDEV/data/test/Turb_6.npy
export PYTHONPATH=$STLDEV
echo "== torch $(date +%T)"
[ -f torch256.json ] || CUDA_VISIBLE_DEVICES=0 $PY $SRC/tools/run_planar_torch_synthesis.py --target "$TURB" --out torch256 > torch256.log 2>&1
tail -1 torch256.log
echo "== rust stldev $(date +%T)"
[ -f rust256_stldev.json ] || STL_THREADS=12 $BIN --target "$TURB" --ntarget 10 --nbatch 10 --mode stldev --niter 100 --init torch256_init.npy --out rust256_stldev > rust256_stldev.log 2>&1
tail -1 rust256_stldev.log
echo "== rust ensemble $(date +%T)"
[ -f rust256_ens.json ] || STL_THREADS=12 $BIN --target "$TURB" --ntarget 10 --nbatch 10 --mode ensemble --niter "$NITER" --out rust256_ens > rust256_ens.log 2>&1
tail -1 rust256_ens.log
echo "== rust ensemble, warm start $(date +%T)"
[ -f rust256_ens_warm.json ] || STL_THREADS=12 $BIN --target "$TURB" --ntarget 10 --nbatch 10 --mode ensemble --warm 100 --niter "$NITER" --out rust256_ens_warm > rust256_ens_warm.log 2>&1
tail -1 rust256_ens_warm.log
for r in rust256_stldev rust256_ens rust256_ens_warm; do $PY $SRC/tools/plot_planar.py --run $r --fig fig > fig/$r.summary.json; done
echo "== done $(date +%T)"
