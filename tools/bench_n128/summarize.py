"""Table and loss figure of a run directory of tools/hpc/bench_n128.slurm.

    python summarize.py RUN_DIR [FIGURE.png]

Per configuration: evaluations, optimisation and setup times, peak device
memory (the code's own counter and the nvidia-smi maximum of the whole
device, CUDA context included), host peak RSS (GNU time, largest process),
GPU utilisation (nvidia-smi samples every 200 ms over the optimisation, the
last `seconds` of the run) and the loss trajectory against legacy_batch.
"""

import csv
import datetime
import json
import os
import re
import sys

RUNS = ("legacy_batch", "legacy_sequential", "rust_time", "rust_mem", "rust_cpu")


def gpu_samples(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.reader(f):
            try:
                t = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                rows.append((t, float(r[1]), float(r[2])))
            except (ValueError, IndexError):
                pass
    return rows


def max_rss_kb(path):
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", open(path).read()) if os.path.exists(path) else None
    return int(m.group(1)) if m else None


def main():
    d = sys.argv[1]
    out = {}
    for name in RUNS:
        path = os.path.join(d, name + ".json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        gpu = gpu_samples(os.path.join(d, f"gpu_{name}.csv"))
        busy = None
        if gpu:
            end = gpu[-1][0]
            window = [s for s in gpu if s[0] >= end - r["seconds"]]
            busy = sum(s[1] for s in window) / max(1, len(window))
        if name.startswith("legacy"):
            dev_peak = r["cuda_max_allocated_bytes"]
            rss = r["rss_peak_bytes"]
        else:
            dev_peak = (r.get("gpu") or {}).get("memory_peak_bytes_per_rank", [None])[0]
            rss = max(r["memory_peak_bytes_per_rank"])
        out[name] = dict(
            evaluations=r["evaluations"],
            seconds=r["seconds"],
            seconds_per_evaluation=r["seconds"] / r["evaluations"],
            setup_seconds=r["setup_seconds"],
            device_peak_bytes=dev_peak,
            nvidia_smi_peak_mib=max((s[2] for s in gpu), default=None),
            host_rss_peak_bytes=rss,
            time_max_rss_kb=max_rss_kb(os.path.join(d, name + ".err")),
            gpu_busy_percent=busy,
            loss_history=r["loss_history"],
        )
    ref = out.get("legacy_batch", {}).get("loss_history")
    for name, o in out.items():
        h = o["loss_history"]
        if ref:
            n = min(len(h), len(ref))
            rel = [abs(a - b) / abs(b) for a, b in zip(h[:n], ref[:n])]
            o["first_eval_rel_diff"] = rel[0]
            o["max_rel_diff_first10"] = max(rel[:10])
            o["first_eval_rel_diff_above_1e-6"] = next((i + 1 for i, x in enumerate(rel) if x > 1e-6), None)
        o["loss_first"], o["loss_last"] = h[0], h[-1]
    gb = lambda x: f"{x / 1e9:.2f}" if x else "-"
    print("| run | evals | s / eval | total s | setup s | device peak GB (code) | nvidia-smi peak GB | host RSS GB | GPU busy % | final loss |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, o in out.items():
        smi = "-" if o["nvidia_smi_peak_mib"] is None else f"{o['nvidia_smi_peak_mib'] * 1.048576e-3:.2f}"
        busy = "-" if o["gpu_busy_percent"] is None else f"{o['gpu_busy_percent']:.0f}"
        print(
            f"| {name} | {o['evaluations']} | {o['seconds_per_evaluation']:.4f} | {o['seconds']:.1f} | {o['setup_seconds']:.1f} | "
            f"{gb(o['device_peak_bytes'])} | {smi} | {gb(o['host_rss_peak_bytes'])} | {busy} | {o['loss_last']:.6e} |"
        )
    for name, o in out.items():
        if "first_eval_rel_diff" in o:
            print(f"{name}: rel diff to legacy_batch eval 1 {o['first_eval_rel_diff']:.1e}, max over evals 1-10 "
                  f"{o['max_rel_diff_first10']:.1e}, first eval above 1e-6: {o['first_eval_rel_diff_above_1e-6']}")
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    if len(sys.argv) > 2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        styles = dict(legacy_batch="-", legacy_sequential="--", rust_time="-", rust_mem="--", rust_cpu=":")
        for name, o in out.items():
            ax.semilogy(range(1, len(o["loss_history"]) + 1), o["loss_history"], styles.get(name, "-"), label=name, lw=1.2)
        ax.set_xlabel("evaluation")
        ax.set_ylabel("loss")
        ax.set_title("nside 128, 4 members, L = 4, one A100")
        ax.legend()
        fig.tight_layout()
        fig.savefig(sys.argv[2], dpi=130)


if __name__ == "__main__":
    main()
