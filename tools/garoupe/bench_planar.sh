#!/bin/bash
# Planar benchmark on garoupe (12 cores, 3 x T4): seconds per evaluation
# and peak memory of the STL-Dev loss (synthesize_from_maps, L = 4, power
# spectrum) in torch (CPU, 12 threads; one T4) and in Rust
# (stl-synth2d --bench, 12 threads), plus the Rust ensemble objective.
#   B=/tmp/$USER-planar2d/bench SRC=/tmp/$USER-planar2d/src STLDEV=... PY=... \
#   BIN=.../stl-synth2d bash tools/garoupe/bench_planar.sh
# Results: $B/*.json (one per configuration), summarised by
# tools/plot_planar.py --bench.
set -u
: "${B:=/tmp/$USER-planar2d/bench}"
mkdir -p "$B" && cd "$B"
TURB=$STLDEV/data/test/Turb_6.npy
TORCH="$PY $SRC/tools/bench_planar_torch.py"
export PYTHONPATH=$STLDEV

# Synthetic targets (lognormal fields; their content does not change the cost).
for N in 512 1024 2048 4096; do
    [ -f field_$N.npy ] && continue
    $PY - "$N" <<'EOF'
import sys
import numpy as np
n = int(sys.argv[1])
count = 10 if n <= 512 else 2
rng = np.random.default_rng(n)
k2 = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2
filt = (k2 + (4.0 / n) ** 2) ** -0.8
x = np.fft.ifft2(np.fft.fft2(rng.standard_normal((count, n, n))) * filt).real
x /= x.std()
np.save(f"field_{n}.npy", np.exp(0.8 * x))
EOF
done

rust() {  # name target ntarget nbatch mode tile
    local out=rust_$1.json
    [ -f "$out" ] && return
    echo "== rust $1 $(date +%T)"
    STL_THREADS=12 $BIN --target "$2" --ntarget "$3" --nbatch "$4" --mode "$5" --tile "$6" --bench 3 > "$out.log" 2>&1
    tail -1 "$out.log" > "$out"
}
torch_run() {  # name device target ntarget nbatch
    local out=torch_$1.json
    [ -f "$out" ] && return
    echo "== torch $1 $(date +%T)"
    if [ "$2" = cpu ]; then
        (ulimit -v $((56 * 1024 * 1024)); CUDA_VISIBLE_DEVICES="" timeout 3600 $TORCH --target "$3" --ntarget "$4" --nbatch "$5" --evals 3 --threads 12 --out "$out" > "$out.log" 2>&1)
    else
        CUDA_VISIBLE_DEVICES=0 timeout 3600 $TORCH --target "$3" --ntarget "$4" --nbatch "$5" --evals 3 --out "$out" > "$out.log" 2>&1
    fi
    [ -f "$out" ] || echo "{\"failed\": \"$(tail -1 "$out.log" | tr -d '"\\' | cut -c1-200)\"}" > "$out"
}

# The STL-Dev example: 10 images of 256^2, 10 targets.
for T in 16 32 64 128 256; do rust 256_stldev_t$T "$TURB" 10 10 stldev $T; done
rust 256_ensemble_t64 "$TURB" 10 10 ensemble 64
torch_run 256_cpu cpu "$TURB" 10 10
torch_run 256_gpu gpu "$TURB" 10 10
# Larger images.
rust 512_stldev_t64 field_512.npy 10 10 stldev 64
rust 512_ensemble_t64 field_512.npy 10 10 ensemble 64
torch_run 512_cpu cpu field_512.npy 10 10
torch_run 512_gpu gpu field_512.npy 10 10
for N in 1024 2048 4096; do
    rust ${N}_stldev_t64 field_$N.npy 2 1 stldev 64
    rust ${N}_stldev_t256 field_$N.npy 2 1 stldev 256
    rust ${N}_stldev_whole field_$N.npy 2 1 stldev $N
done
torch_run 1024_cpu cpu field_1024.npy 2 1
torch_run 1024_gpu gpu field_1024.npy 2 1
torch_run 2048_gpu gpu field_2048.npy 2 1
torch_run 2048_cpu cpu field_2048.npy 2 1
echo "== done $(date +%T)"
