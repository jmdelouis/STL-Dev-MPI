# Sourced by the examples' run.sh.  Override any of these from the environment:
#   STL_BIN   directory of the binaries (default: the workspace's target/release)
#   PY        Python with numpy, matplotlib and healpix_geo (figures only)
#   OUT       directory of the outputs (default: <example>/out, not versioned)
set -euo pipefail
ROOT=$(cd "$HERE/../.." && pwd)
: "${STL_BIN:=${CARGO_TARGET_DIR:-$ROOT/target}/release}"
: "${PY:=python3}"
: "${OUT:=$HERE/out}"
mkdir -p "$OUT"
# The HEALPix target of examples 01, 03 and 05: the WebSky kappa map at
# nside 64 (NESTED, float64), shipped with the fixtures.
KAPPA64=$ROOT/fixtures/synthesis/synthesis_kap_nside64_nested_niter20/target_map.npy

# step NAME COMMAND...: runs COMMAND, prints its wall-clock time.
step() {
    local name=$1 t0
    shift
    t0=$(date +%s.%N)
    echo "== $name"
    "$@"
    awk -v a="$t0" -v b="$(date +%s.%N)" -v n="$name" 'BEGIN { printf "== %s: %.1f s\n", n, b - a }'
}
