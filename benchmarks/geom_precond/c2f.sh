until grep -q ALLDONE precond_a.status; do sleep 30; done
export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=1 STL_THREADS=4 STL_GPU_THREADS=4
B="/tmp/jmdeloui-geomcache/target/release/stl-synth --L 4 --weights none --ensemble 4 --quiet --gpu --gpu-fields keep --geometry-cache device"
$B --target /tmp/jmdeloui-geomcache/data/tk64.npy --init init64_m{m}.npy --init-units standardized --init-noise 0 --niter 100 --out c64 > c64.log 2>&1; echo c64 $?
T128="--target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --init c64_m{m}_synthesis_nested.npy --niter 100"
$B $T128 --out c_c2f > c_c2f.log 2>&1; echo c_c2f $?
$B $T128 --freeze-nside 32 --out c_c2f_f32 > c_c2f_f32.log 2>&1; echo c_c2f_f32 $?
$B $T128 --freeze-nside 16 --out c_c2f_f16 > c_c2f_f16.log 2>&1; echo c_c2f_f16 $?
echo ALLDONE
