"""Tables of docs/geometry_cache_preconditioner.md from the stl-synth reports of a run directory.

    python summarize.py RUNDIR [REFERENCE]

t_*.json, s512_*.json: geometry cache timings (s / evaluation, device phases, memory, identical
trajectories within a field mode); p_*.json and c_*.json: loss after the run and the
evaluations needed to reach REFERENCE's final loss (default p_none.json).
"""

import glob
import json
import os
import sys

run = sys.argv[1]
J = lambda f: json.load(open(f))
name = lambda f: os.path.basename(f)[:-5]

timing = sorted(glob.glob(f"{run}/t_*.json")) + sorted(glob.glob(f"{run}/s512_*.json"))
if timing:
    print("| run | s / eval | evals | geometry build s | weights recomputed s | weights uploaded s | to host s | device peak GB | device cache GB | host cache GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    hist = {}
    for f in timing:
        r = J(f)
        t, g = r["timers_seconds"], r["gpu"]
        n = r["evaluations"] if "evaluations" in r else len(r["loss_history"])
        gb = lambda v: f"{v / 1e9:.2f}"
        print(f"| {name(f)} | {r['seconds_per_evaluation']:.3f} | {n} | {t.get('gpu_geometry', 0):.2f} | {t.get('gpu_geometry_weights', 0):.2f} "
              f"| {t.get('gpu_geometry_upload', 0):.2f} | {t.get('gpu_geometry_to_host', 0):.2f} | {gb(g['memory_peak_bytes_per_rank'][0])} "
              f"| {gb(g['geometry_cache_used_bytes_per_rank'][0])} | {gb(g.get('geometry_host_used_bytes_per_rank', [0])[0])} |")
        hist[name(f)] = r["loss_history"]
    for mode in ("t_keep", "t_mem", "s512_keep", "s512_mem"):
        runs = {k: v for k, v in hist.items() if k.startswith(mode)}
        ref = runs.get(f"{mode}_off")
        if ref:
            print(mode, {k: (v == ref) for k, v in runs.items()})

pre = sorted(glob.glob(f"{run}/p_*.json")) + sorted(glob.glob(f"{run}/c_*.json"))
if pre:
    ref_file = sys.argv[2] if len(sys.argv) > 2 else f"{run}/p_none.json"
    level = J(ref_file)["loss_history"][-1]
    print(f"\nreference level {level:.4e} ({name(ref_file)}, {len(J(ref_file)['loss_history'])} evaluations)")
    print("| run | precondition | evaluations | final loss | final / reference | evaluations to reference | best loss at eval 50 |")
    print("|---|---|---|---|---|---|---|")
    for f in pre:
        r = J(f)
        h = r["loss_history"]
        best = [min(h[: i + 1]) for i in range(len(h))]
        reach = next((i + 1 for i, v in enumerate(best) if v <= level), None)
        spec = r.get("precondition")
        spec = spec if isinstance(spec, str) else spec.get("spec")
        print(f"| {name(f)} | {spec} | {len(h)} | {h[-1]:.3e} | {h[-1] / level:.2f} | {reach if reach else '-'} | {best[min(49, len(h) - 1)]:.3e} |")
