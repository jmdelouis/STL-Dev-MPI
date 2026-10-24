#!/bin/bash
# Diagnostics of a run_hybrid_ensemble.sh directory: pyramid / C_l /
# moments / independence of the members (evaluate.py) for the coarse STL
# ensemble, the hybrid maps before and after the minimisation and the
# control, the STL residuals of the minimisations, and their convergence.
#   diffusion/eval_hybrid_ensemble.sh OUT_DIR TARGET.npy B [LABEL=MAP.npy ...]
# PYTHON from the environment.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=$1 target=$2 B=$3
shift 3
py=${PYTHON:-python}
members() { local f=""; for m in $(seq 0 $((B - 1))); do f="$f,$out/$(printf "$1" "$m")"; done; echo "${f#,}"; }
$py -u "$here/evaluate.py" "$out/eval256" "$target" "polished=$(members polished_m%03d_synthesis_nested.npy)" \
    "control=$(members control_m%03d_synthesis_nested.npy)" "hybrid_init=$out/hybrid.npy" "$@" 2>&1 \
    | grep -v Warning | tee "$out/eval256.txt"
$py -u "$here/evaluate.py" --nside 64 "$out/eval64" "$target" "stl_coarse=$(members coarse_m%03d_synthesis_nested.npy)" \
    "polished=$(members polished_m%03d_synthesis_nested.npy)" "control=$(members control_m%03d_synthesis_nested.npy)" \
    "$@" 2>&1 | grep -v Warning | tee "$out/eval64.txt"
$py - "$out" <<'EOF' | tee "$out/residuals.txt"
import json
import sys

out = sys.argv[1]
for run in ("coarse", "polished", "control"):
    d = json.load(open(f"{out}/{run}.json"))
    h = d["loss_history"]
    first = lambda thr: next((i for i, v in enumerate(h) if v < thr), None)
    mean = {k: round(v["rms_sigma"], 2) for k, v in d["final_residual_in_ergodic_sigma"].items()}
    print(f"{run}: {d['evaluations']} evaluations, {d['seconds']:.0f} s, loss {h[0]:.4g} -> {h[-1]:.4g}, "
          f"below 1 / 0.3 after {first(1)} / {first(0.3)} evaluations")
    print(f"  mean of the members vs target (sigma / sqrt B): {mean}")
    for fam, v in d["ensemble"]["per_member_in_ergodic_sigma"].items():
        print(f"  {fam:5s} member residual {v['residual_rms_sigma']:.2f} sigma, scatter {v['scatter_rms_sigma']:.2f} sigma, "
              f"scatter / ergodic variance {v['scatter_over_ergodic_variance']:.2f}")
EOF
