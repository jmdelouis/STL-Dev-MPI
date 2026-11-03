export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=0 STL_THREADS=4 STL_GPU_THREADS=4
run() { n=$1; shift; /usr/bin/time -v /tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/g512.npy --L 4 --weights none --ensemble 4 --init init512_m{m}.npy --init-units standardized --init-noise 0 --quiet --gpu "$@" --niter 10 --out s512_$n > s512_$n.log 2>&1; echo $n $?; }
run mem_off    --gpu-fields recompute --gpu-block-rows 4096 --block-pixels 4096 --geometry-cache off
run mem_host   --gpu-fields recompute --gpu-block-rows 4096 --block-pixels 4096 --geometry-cache host
run mem_device --gpu-fields recompute --gpu-block-rows 4096 --block-pixels 4096 --geometry-cache device
run keep_off    --gpu-fields keep --geometry-cache off
run keep_host   --gpu-fields keep --geometry-cache host
run keep_device --gpu-fields keep --geometry-cache device
echo ALLDONE
