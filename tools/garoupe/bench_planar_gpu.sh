#!/bin/bash
# Rust planar engine on one T4 of garoupe (stl-synth2d --gpu, feature gpu):
# seconds per evaluation and peak device memory, STL-Dev loss with spectrum
# (spectrum on the host threads) and ScatCov only (--bins 0), in two modes:
# resident (one tile per level, fastest) and tiled (bounded memory).
#   B=... STLDEV=... BIN=<target-gpu>/release/stl-synth2d GPU=1 bash tools/garoupe/bench_planar_gpu.sh
# Uses the synthetic targets of bench_planar.sh.
set -u
: "${B:=/tmp/$USER-planar2d/bench}"
: "${GPU:=1}"
cd "$B"
TURB=$STLDEV/data/test/Turb_6.npy
run() {  # name target ntarget nbatch tile bins mode
    local out=gpu_$1.json
    [ -f "$out" ] && return
    echo "== $1 $(date +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU")"
    STL_GPU_DEVICE=$GPU STL_THREADS=12 timeout 3600 $BIN --target "$2" --ntarget "$3" --nbatch "$4" --tile "$5" --bins "$6" --mode "$7" --gpu --bench 3 > "$out.log" 2>&1
    tail -1 "$out.log" > "$out"
}
for spec in "256 $TURB 10 10" "512 field_512.npy 10 10" "1024 field_1024.npy 2 1" "2048 field_2048.npy 2 1" "4096 field_4096.npy 2 1"; do
    set -- $spec
    N=$1; T=$2; NT=$3; NB=$4
    BINS=$(( 1 << ($(echo "l($N)/l(2)" | bc -l | cut -d. -f1) - 4) ))
    for mode in resident:100000 tiled64:64 tiled256:256; do
        name=${mode%%:*}; tile=${mode##*:}
        run ${N}_${name}_full "$T" $NT $NB $tile $BINS stldev
        run ${N}_${name}_sc "$T" $NT $NB $tile 0 stldev
    done
done
run 256_resident_ensemble "$TURB" 10 10 100000 16 ensemble
run 512_resident_ensemble field_512.npy 10 10 100000 32 ensemble
echo "== done $(date +%T)"
