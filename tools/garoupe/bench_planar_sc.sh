#!/bin/bash
# ScatCov-only loss (STL-Dev's synthesize_from_maps without the power
# spectrum: compute_PS=False / stl-synth2d --mode stldev --bins 0): seconds
# per evaluation and peak memory, torch (CPU 12 threads, one T4) and Rust
# (12 threads, tiles of 64).  Same environment as bench_planar.sh; reuses
# its synthetic targets field_<N>.npy.
set -u
: "${B:=/tmp/$USER-planar2d/bench}"
cd "$B"
TURB=$STLDEV/data/test/Turb_6.npy
TORCH="$PY $SRC/tools/bench_planar_torch.py --no-ps"
export PYTHONPATH=$STLDEV
rust() {  # name target ntarget nbatch
    local out=sc_rust_$1.json
    [ -f "$out" ] && return
    echo "== rust $1 $(date +%T)"
    STL_THREADS=12 $BIN --target "$2" --ntarget "$3" --nbatch "$4" --mode stldev --bins 0 --tile 64 --bench 3 > "$out.log" 2>&1
    tail -1 "$out.log" > "$out"
}
torch_run() {  # name device target ntarget nbatch
    local out=sc_torch_$1.json
    [ -f "$out" ] && return
    echo "== torch $1 $(date +%T)"
    if [ "$2" = cpu ]; then
        (ulimit -v $((56 * 1024 * 1024)); CUDA_VISIBLE_DEVICES="" timeout 3600 $TORCH --target "$3" --ntarget "$4" --nbatch "$5" --evals 3 --threads 12 --out "$out" > "$out.log" 2>&1)
    else
        CUDA_VISIBLE_DEVICES=${GPU:-0} timeout 3600 $TORCH --target "$3" --ntarget "$4" --nbatch "$5" --evals 3 --out "$out" > "$out.log" 2>&1
    fi
    [ -f "$out" ] || echo "{\"failed\": \"$(tail -1 "$out.log" | tr -d '"\' | cut -c1-200)\"}" > "$out"
}
rust 256 "$TURB" 10 10
torch_run 256_cpu cpu "$TURB" 10 10
torch_run 256_gpu gpu "$TURB" 10 10
rust 512 field_512.npy 10 10
torch_run 512_cpu cpu field_512.npy 10 10
torch_run 512_gpu gpu field_512.npy 10 10
rust 1024 field_1024.npy 2 1
torch_run 1024_cpu cpu field_1024.npy 2 1
torch_run 1024_gpu gpu field_1024.npy 2 1
rust 2048 field_2048.npy 2 1
rust 4096 field_4096.npy 2 1
echo "== done $(date +%T)"
