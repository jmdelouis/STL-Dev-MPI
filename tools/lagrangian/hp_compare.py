"""HEALPix coarse-to-fine comparison (docs/lagrangian_synthesis.md): loss at
equal cost and maps.

    python hp_compare.py --dir RUNS --out PREFIX

Reads RUNS/{pix128,cond32,cond64,cond128,auto32,auto64,auto128}.json
(stl-synth reports, `loss_history`) and the synthesis maps; the cost of a
chain is sum evals (nside / 128)^2.  Writes PREFIX.json and PREFIX_maps.png
(gnomonic views, norm='hist', one colour scale for all maps).
"""
import argparse
import json

import numpy as np


def history(path):
    return np.minimum.accumulate(np.array(json.load(open(path))["loss_history"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = a.dir
    pix = history(f"{d}/pix128.json")
    rep = {"pixel": {"evaluations": len(pix), "final": float(pix[-1])}}
    for name in ["cond", "auto"]:
        coarse = sum(len(json.load(open(f"{d}/{name}{n}.json"))["loss_history"]) * (n / 128) ** 2 for n in (32, 64))
        h = history(f"{d}/{name}128.json")
        rep[name] = {"coarse_equivalent": coarse, "evaluations_128": len(h), "first_128": float(h[0]), "final": float(h[-1])}
        rep[name]["at_equal_cost"] = {}
        for e in [100, 200, 400, 600, 800, 1000]:
            i, j = int(e - coarse), e
            rep[name]["at_equal_cost"][e] = [float(h[min(i, len(h) - 1)]) if i >= 0 else None, float(pix[min(j, len(pix) - 1)])]
        reach = np.argmax(h < pix[-1]) if (h < pix[-1]).any() else None
        rep[name]["equivalent_to_pixel_final"] = None if reach is None else float(coarse + reach)
    json.dump(rep, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    try:
        import healpy as hp
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("no figure:", e)
        return
    maps = {"target": np.load(f"{d}/k128.npy"), "pixel L-BFGS": np.load(f"{d}/pix128_synthesis_nested.npy"),
            "c2f white": np.load(f"{d}/auto128_synthesis_nested.npy"), "c2f conditional": np.load(f"{d}/cond128_synthesis_nested.npy")}
    lo, hi = np.percentile(maps["target"], [0.5, 99.5])
    fig = plt.figure(figsize=(16, 4.5))
    for k, (t, m) in enumerate(maps.items()):
        hp.gnomview(m, nest=True, rot=(30, 30), xsize=400, reso=6, norm="hist", min=lo, max=hi, title=t, sub=(1, 4, k + 1), cmap="inferno", cbar=False, fig=fig.number)
    fig.savefig(a.out + "_maps.png", dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
