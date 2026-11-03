export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=2 STL_THREADS=4 STL_GPU_THREADS=4
run() { n=$1; shift; /tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --L 4 --weights none --ensemble 4 --init init_m{m}.npy --init-units standardized --init-noise 0 --quiet --gpu --gpu-fields keep --geometry-cache device "$@" --out p_$n > p_$n.log 2>&1; echo $n $?; }
run auto      --niter 100 --precondition auto
run power     --niter 100 --precondition power
run b1        --niter 100 --precondition bands:1
echo ALLDONE
