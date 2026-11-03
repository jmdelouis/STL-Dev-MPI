export STL_GPU_DEVICE=0 CUDA_VISIBLE_DEVICES=1 STL_THREADS=4 STL_GPU_THREADS=4
I="--init init_m{m}.npy --init-units standardized --init-noise 0"
/tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --L 4 --weights none --ensemble 4 --quiet --gpu --gpu-fields keep --geometry-cache device --niter 100 $I --precondition bands:-0.25 --out p_bm025 > p_bm025.log 2>&1; echo bm025 $?
/tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --L 4 --weights none --ensemble 4 --quiet --gpu --gpu-fields keep --geometry-cache device --niter 100 $I --precondition bands:-0.5 --out p_bm05 > p_bm05.log 2>&1; echo bm05 $?
/tmp/jmdeloui-geomcache/target/release/stl-synth --target /tmp/jmdeloui-geomcache/data/takahashi_r000_zs1_nside128_nested.npy --L 4 --weights none --ensemble 4 --quiet --gpu --gpu-fields keep --geometry-cache device --niter 100 $I --precondition bands:-1 --out p_bm1 > p_bm1.log 2>&1; echo bm1 $?
echo ALLDONE
