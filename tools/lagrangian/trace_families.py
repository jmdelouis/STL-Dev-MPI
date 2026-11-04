"""Per-family split of a traced stl-synth2d run (docs/lagrangian_synthesis.md, step 0).

    python trace_families.py --run RUNPREFIX --out FIGPREFIX

Reads RUNPREFIX.json (plan, trace_evaluations, loss_history),
RUNPREFIX_trace.npy [S, n, 2] (batch-mean feature residuals),
RUNPREFIX_features_target.npy [Bt, n, 2] and RUNPREFIX_trace_maps.npy.
Writes FIGPREFIX_families.png, FIGPREFIX_maps.png and FIGPREFIX_families.json.

Two measures per group of coefficients:
- `sq`: sum |residual|^2, the STL-Dev loss (stldev mode) split exactly;
- `z2`: sum |residual|^2 / var_targets, the residual in units of the
  target scatter (comparable between modes).
Groups: mean/var, S1, S2, PS, and S3/S4 by native scale j3 and by
scale gap j3 - j1 (large gaps are the cross-scale, morphological terms).
"""
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def groups(plan, n):
    g = {}

    def add(k, i):
        g.setdefault(k, []).append(i)

    for i, q in enumerate(plan):
        f = q["family"]
        if f in ("S3", "S4"):
            add(f, i)
            add(f"{f} j3={q['j3']}", i)
            add(f"{f} gap={q['j3'] - q['j1']}", i)
        elif f in ("S1", "S2"):
            add(f, i)
            add(f"{f} j={q['j3'] if q['j3'] is not None else q['j1']}", i)
        else:
            add("mean/var", i)
    for i in range(len(plan), n):
        add("PS", i)
    return {k: np.array(v) for k, v in g.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = json.load(open(a.run + ".json"))
    tr = np.load(a.run + "_trace.npy")
    tr = tr[..., 0] + 1j * tr[..., 1]
    ft = np.load(a.run + "_features_target.npy")
    ft = ft[..., 0] + 1j * ft[..., 1]
    var_t = np.maximum(np.var(ft, axis=0), 1e-30 * np.median(np.var(ft, axis=0)))
    ev = np.array(rep["trace_evaluations"])
    n = tr.shape[1]
    gr = groups(rep["plan"], n)
    sq = {k: (np.abs(tr[:, v]) ** 2).sum(1) for k, v in gr.items()}
    z2 = {k: (np.abs(tr[:, v]) ** 2 / var_t[v]).sum(1) for k, v in gr.items()}
    out = {"evaluations": ev.tolist(), "loss_history": rep["loss_history"], "sq": {k: v.tolist() for k, v in sq.items()},
           "z2": {k: v.tolist() for k, v in z2.items()}, "counts": {k: int(len(v)) for k, v in gr.items()}}
    json.dump(out, open(a.out + "_families.json", "w"))

    top = ["mean/var", "S1", "S2", "S3", "S4", "PS"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for m, (name, d) in enumerate([("|r|^2 (STL-Dev loss)", sq), ("|r|^2 / var_targets", z2)]):
        x = ax[m, 0]
        for k in top:
            if k in d:
                x.semilogy(ev, np.maximum(d[k], 1e-30), label=k)
        if m == 0:
            x.semilogy(np.arange(len(rep["loss_history"])), rep["loss_history"], "k:", lw=0.7, label="loss")
        x.set_title(f"by family, {name}")
        x.set_xlabel("evaluation")
        x.legend(fontsize=7)
        for c, fam in enumerate(["S3", "S4"]):
            x = ax[m, 1 + c]
            keys = sorted([k for k in d if k.startswith(fam + " gap=")], key=lambda s: int(s.split("=")[1]))
            cm = plt.cm.viridis(np.linspace(0, 1, max(len(keys), 1)))
            for k, col in zip(keys, cm):
                x.semilogy(ev, np.maximum(d[k], 1e-30), color=col, label=k)
            x.set_title(f"{fam} by scale gap j3 - j1, {name}")
            x.set_xlabel("evaluation")
            x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out + "_families.png", dpi=110)

    mp = np.load(a.run + "_trace_maps.npy")
    tg = np.load(a.run + "_target.npy")
    picks = [i for i in np.unique(np.searchsorted(ev, [0, 10, 30, 100, 300, 1000, ev[-1]])) if i < len(ev)]
    fig, ax = plt.subplots(1, len(picks) + 1, figsize=(3 * (len(picks) + 1), 3.2))
    t0 = (tg[0, 0] - tg[0, 0].mean()) / tg[0, 0].std()
    lim = np.percentile(np.abs(t0), 99.5)
    for x, i in zip(ax[:-1], picks):
        x.imshow(mp[i, 0], cmap="inferno", vmin=-lim, vmax=lim)
        x.set_title(f"eval {ev[i]}")
        x.axis("off")
    ax[-1].imshow(t0, cmap="inferno", vmin=-lim, vmax=lim)
    ax[-1].set_title("target 0")
    ax[-1].axis("off")
    fig.tight_layout()
    fig.savefig(a.out + "_maps.png", dpi=110)
    last = {k: sq[k][-1] for k in top if k in sq}
    print(json.dumps({"final_sq": last, "final_z2": {k: z2[k][-1] for k in top if k in z2}}))


if __name__ == "__main__":
    main()
