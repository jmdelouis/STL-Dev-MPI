until grep -q ALLDONE r5.status && grep -q ALLDONE r6.status && grep -q ALLDONE r7.status; do sleep 30; done
W=$HOME/stl-gpu-work; source $W/src/tools/garoupe/env.sh > /dev/null 2>&1
export CARGO_TARGET_DIR=/tmp/jmdeloui-geomcache/target
(cd /tmp/jmdeloui-geomcache/src && cargo build --release -p stl-cli --features gpu -j 10 > /tmp/jmdeloui-geomcache/build2.log 2>&1) && echo built || { echo BUILD_FAIL; echo ALLDONE; exit; }
export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=0 STL_THREADS=4 STL_GPU_THREADS=4
run() { n=$1; shift; /usr/bin/time -v /tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/g512.npy --L 4 --weights none --ensemble 4 --init init512_m{m}.npy --init-units standardized --init-noise 0 --quiet --gpu "$@" --niter 10 --out s512_$n > s512_$n.log 2>&1; echo $n $?; }
run mem_auto   --gpu-fields recompute --gpu-block-rows 4096 --block-pixels 4096 --geometry-cache auto
run keep_auto  --gpu-fields keep --geometry-cache auto
echo ALLDONE
