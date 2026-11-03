export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=0 STL_THREADS=4 STL_GPU_THREADS=4
run() { n=$1; shift; /usr/bin/time -v /tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --L 4 --weights none --ensemble 4 --init init_m{m}.npy --init-units standardized --init-noise 0 --quiet --gpu "$@" --out t_$n > t_$n.log 2>&1; echo $n $?; }
run keep_off     --niter 20 --gpu-fields keep --geometry-cache off
run keep_device  --niter 20 --gpu-fields keep --geometry-cache device
run keep_host    --niter 20 --gpu-fields keep --geometry-cache host
run keep_transp  --niter 20 --gpu-fields keep --gpu-cache transposition
run mem_off      --niter 20 --gpu-fields recompute --gpu-block-rows 1024 --block-pixels 1024 --geometry-cache off
run mem_transp   --niter 20 --gpu-fields recompute --gpu-block-rows 1024 --block-pixels 1024 --gpu-cache transposition
run mem_host     --niter 20 --gpu-fields recompute --gpu-block-rows 1024 --block-pixels 1024 --geometry-cache host
run mem_device   --niter 20 --gpu-fields recompute --gpu-block-rows 1024 --block-pixels 1024 --geometry-cache device
run mem_auto     --niter 20 --gpu-fields recompute --gpu-block-rows 1024 --block-pixels 1024 --geometry-cache auto
echo ALLDONE
