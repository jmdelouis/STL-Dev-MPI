until grep -q ALLDONE r4_b.status; do sleep 30; done
export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=1 STL_THREADS=4 STL_GPU_THREADS=4
run() { n=$1; shift; /tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/g256.npy --L 4 --weights ergodic --ensemble 10 --scatter-weight 1 --level-power-weight 10 --kappa-table /tmp/jmdeloui-ens/data/knots.npy --kappa-weight 0.1 --marginal-weight 10000 --init /tmp/jmdeloui-ens/init/synthesis_gauss_m{m}_nested.npy --init-noise 0 --quiet --gpu --gpu-fields keep --geometry-cache device "$@" --out e256_$n > e256_$n.log 2>&1; echo $n $?; }
run p_none  --niter 100
run p_auto  --niter 100 --precondition auto
echo ALLDONE
