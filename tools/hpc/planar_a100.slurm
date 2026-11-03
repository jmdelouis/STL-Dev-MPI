#!/bin/bash
# Planar benchmark on one A100 of hpc (docs/planar_2d.md, section 5c):
# STL-Dev (torch, tools/bench_planar_torch.py) against stl-synth2d --gpu,
# STL-Dev loss with and without the spectrum, the sizes of the garoupe
# benchmark; first the device tests of the planar engine.  Every run under
# a timeout; at most 60 min in all.  Built by planar_build.slurm.
#   sbatch --export=ALL,G=/scratch/jmdeloui/stl-planar2d,PY=<python with torch> $G/src/tools/hpc/planar_a100.slurm
#SBATCH --job-name=stl-planar-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

set -u
G=${G:-/scratch/jmdeloui/stl-planar2d}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
OUT=$G/bench/$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$OUT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export STL_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$G/stl-dev
BIN=$CARGO_TARGET_DIR/release/stl-synth2d
TURB=$G/stl-dev/data/test/Turb_6.npy

echo "== device tests $(date +%T)"
for t in $(ls -t "$CARGO_TARGET_DIR"/release/deps/planar-* | grep -v '\.d$'); do
    timeout 300 "$t" 2>&1 | grep -E "^test |test result"
done

# Synthetic targets (lognormal fields, as tools/garoupe/bench_planar.sh).
for N in 512 1024 2048 4096; do
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

rust() {  # name target ntarget nbatch tile bins
    echo "== rust $1 $(date +%T)"
    timeout 600 "$BIN" --target "$2" --ntarget "$3" --nbatch "$4" --tile "$5" --bins "$6" --mode stldev --gpu --bench 3 > "rust_$1.log" 2>&1
    tail -1 "rust_$1.log" | tee "rust_$1.json"
}
torch_run() {  # name target ntarget nbatch [--no-ps]
    echo "== torch $1 $(date +%T)"
    timeout 900 $PY "$G/src/tools/bench_planar_torch.py" --target "$2" --ntarget "$3" --nbatch "$4" --evals 3 ${5:-} --out "torch_$1.json" > "torch_$1.log" 2>&1
    [ -f "torch_$1.json" ] || echo "{\"failed\": \"$(tail -1 "torch_$1.log" | tr -d '"\\' | cut -c1-200)\"}" > "torch_$1.json"
    tr -d '\n ' < "torch_$1.json" | cut -c1-300; echo
}
bins() { echo $(( 1 << ($(awk -v n="$1" 'BEGIN { printf "%d", log(n) / log(2) }') - 4) )); }

for spec in "256 $TURB 10 10" "512 field_512.npy 10 10" "1024 field_1024.npy 2 1" "2048 field_2048.npy 2 1" "4096 field_4096.npy 2 1"; do
    set -- $spec
    N=$1; T=$2; NT=$3; NB=$4
    for mode in resident:100000 tiled256:256 tiled64:64; do
        rust ${N}_${mode%%:*}_full "$T" "$NT" "$NB" "${mode##*:}" "$(bins "$N")"
        rust ${N}_${mode%%:*}_sc "$T" "$NT" "$NB" "${mode##*:}" 0
    done
    torch_run ${N}_full "$T" "$NT" "$NB"
    torch_run ${N}_sc "$T" "$NT" "$NB" --no-ps
done
echo "done $(date)"
