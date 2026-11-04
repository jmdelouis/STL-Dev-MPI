#!/bin/bash
# Scattering-covariance statistics of maps with stl-stats: the Takahashi kappa
# target at nside 64, the synthesis of example 01 and a white-noise map,
# all standardised by the target's mean and std (--norm).
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../common.sh"
: "${SYNTH:=$HERE/../01_healpix_synthesis/out/cpu_synthesis_nested.npy}"

# White noise with the mean and the standard deviation of the target.
"$PY" -c "import numpy as np; t = np.load('$KAPPA64'); np.save('$OUT/white_noise_nested.npy', t.mean() + t.std() * np.random.default_rng(0).standard_normal(t.size))"

step "stl-stats, CPU" "$STL_BIN/stl-stats" --out "$OUT/stats.json" --norm "$KAPPA64" --L 4 --ranks 4 \
    -- "$KAPPA64" "$SYNTH" "$OUT/white_noise_nested.npy"
if [ "${GPU:-1}" = 1 ]; then
    step "stl-stats, GPU" "$STL_BIN/stl-stats" --out "$OUT/stats_gpu.json" --norm "$KAPPA64" --L 4 --gpu \
        -- "$KAPPA64" "$SYNTH" "$OUT/white_noise_nested.npy"
    "$PY" - "$OUT/stats.json" "$OUT/stats_gpu.json" <<'PY'
import json, sys
import numpy as np
a, b = (json.load(open(p)) for p in sys.argv[1:])
c = lambda m: np.array(m["re"]) + 1j * np.array(m["im"])
d = max(np.abs(c(x) - c(y)).max() / np.abs(c(x)).max() for x, y in zip(a["maps"], b["maps"]))
print(f"{len(a['plan'])} coefficients per map; CPU vs GPU max difference {d:.1e} (relative to the largest coefficient of each map)")
PY
fi
"$PY" "$HERE/../plotting.py" stats "$HERE/stats.png" "$OUT/stats.json"
