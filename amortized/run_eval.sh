#!/bin/bash
# Evaluate a trained generator against an L-BFGS ensemble at one nside, with
# a short L-BFGS polish of 8 generated maps and the Takahashi cosmic-variance
# test (tools/cv10.py), on garoupe.
#
#   bash run_eval.sh NSIDE RUN LBFGS_RUN [GPU] [POLISH_ITERS]
#
# RUN holds model.pt (train.py); LBFGS_RUN holds ens_m*_synthesis_nested.npy
# (stl-synth --ensemble 8 --weights ergodic --scatter-weight 1).
set -e
N=$1; RUN=$2; LB=$3; GPU=${4:-1}; IT=${5:-50}
source /tmp/jmdeloui-amort/env.sh
D=/tmp/jmdeloui-amort/data
STL=$HOME/stl-campeti/target/release/stl-synth
export CUDA_VISIBLE_DEVICES=$GPU
$PY evaluate.py --target $D/r000_n$N.npy --nside $N --model $RUN/model.pt --count 1000 --save 8 --out $RUN/eval > $RUN/eval.log 2>&1
$PY evaluate.py --target $D/r000_n$N.npy --nside $N --maps "$LB/ens_m*_synthesis_nested.npy" --out $LB/eval > $LB/eval.log 2>&1
mkdir -p $LB/cv && for m in $LB/ens_m*_synthesis_nested.npy; do k=$(basename $m | sed 's/ens_m\([0-9]*\)_.*/\1/'); ln -sf $m $LB/cv/synthesis_m${k}_nested.npy; done
# Polish: IT L-BFGS iterations of the ensemble loss from 8 generated maps.
mkdir -p $RUN/polish
(cd $RUN/polish && $STL --gpu --target $D/r000_n$N.npy --weights ergodic --ensemble 8 --scatter-weight 1 \
    --init "$RUN/eval/synthesis_m{m}_nested.npy" --init-noise 0 --niter $IT --out $RUN/polish/ens > synth.log 2>&1)
$PY evaluate.py --target $D/r000_n$N.npy --nside $N --maps "$RUN/polish/ens_m*_synthesis_nested.npy" --out $RUN/polish/eval > $RUN/polish/eval.log 2>&1
mkdir -p $RUN/polish/cv && for m in $RUN/polish/ens_m*_synthesis_nested.npy; do k=$(basename $m | sed 's/ens_m\([0-9]*\)_.*/\1/'); ln -sf $m $RUN/polish/cv/synthesis_m${k}_nested.npy; done
$PY ../tools/cv10.py $N $RUN/cv10_n$N $D/r00?_n$N.npy -- generator=$RUN/eval L-BFGS=$LB/cv polished=$RUN/polish/cv > $RUN/cv10.log 2>&1
echo EVAL-DONE
