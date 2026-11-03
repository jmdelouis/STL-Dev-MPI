until grep -q ALLDONE r3.status; do sleep 30; done
export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=0 STL_THREADS=4 STL_GPU_THREADS=4
run() { n=$1; shift; /usr/bin/time -v /tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/g256.npy --L 4 --weights ergodic --ensemble 10 --scatter-weight 1 --level-power-weight 10 --kappa-table /tmp/jmdeloui-ens/data/knots.npy --kappa-weight 0.1 --marginal-weight 10000 --init /tmp/jmdeloui-ens/init/synthesis_gauss_m{m}_nested.npy --init-noise 0 --quiet --gpu "$@" --out e256_$n > e256_$n.log 2>&1; echo $n $?; }
run t_keep_off    --niter 5 --gpu-fields keep --geometry-cache off
run t_keep_device --niter 5 --gpu-fields keep --geometry-cache device
run t_mem_off     --niter 5 --gpu-fields recompute --gpu-block-rows 4096 --block-pixels 4096 --geometry-cache off
run t_mem_host    --niter 5 --gpu-fields recompute --gpu-block-rows 4096 --block-pixels 4096 --geometry-cache host
run t_mem_device  --niter 5 --gpu-fields recompute --gpu-block-rows 4096 --block-pixels 4096 --geometry-cache device
echo ALLDONE
