export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=2 STL_THREADS=4 STL_GPU_THREADS=4
I="--init c64_m{m}_synthesis_nested.npy"
/tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --L 4 --weights none --ensemble 4 --quiet --gpu --gpu-fields keep --geometry-cache device --niter 100 $I --precondition auto --out c_c2f_auto > c_c2f_auto.log 2>&1; echo c2f_auto $?
/tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --L 4 --weights none --ensemble 4 --quiet --gpu --gpu-fields keep --geometry-cache device --niter 100 $I --precondition bands:-0.25 --out c_c2f_bm025 > c_c2f_bm025.log 2>&1; echo c2f_bm025 $?
echo ALLDONE
