"""Maps and diversity of several stl-synth2d runs (docs/lagrangian_synthesis.md 6.7-6.8).

    python compare2d.py --target Turb_6.npy --runs pix1000,c2f1000,hlin --out FIG

For every run PREFIX (PREFIX_maps.npy, [B, H, W] or [B, 1, H, W], target units):
- maximum over periodic translations (FFT cross-correlation) of the Pearson
  correlation of every synthesis with every target (mean of the per-image
  maxima, overall max) and between distinct syntheses of the run;
- the same between distinct targets, for reference.
Prints a JSON table, writes FIG.json and FIG.png (image 0 of every run and
target 0, one colour scale: the targets' 0.5-99.5 percentiles).
"""
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path, n=None):
    x = np.load(path).astype(float)
    if x.ndim == 4:
        x = x[:, 0]
    return x[:n] if n else x


def unit(x):
    x = x - x.mean(axis=(-2, -1), keepdims=True)
    return x / np.sqrt((x ** 2).mean(axis=(-2, -1), keepdims=True))


def max_corr(a, b, same=False):
    """Per-image max over translations and pairs of corr(a_i, b_j); i != j when same."""
    fa, fb = np.fft.fft2(unit(a)), np.fft.fft2(unit(b))
    best = np.full(len(a), -1.0)
    for i in range(len(a)):
        for j in range(len(b)):
            if same and i == j:
                continue
            c = np.real(np.fft.ifft2(fa[i] * np.conj(fb[j]))).max() / a[i].size
            best[i] = max(best[i], c)
    return float(best.mean()), float(best.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--ntarget", type=int, default=10)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = load(a.target, a.ntarget)
    runs = a.runs.split(",")
    labels = a.labels.split(",") if a.labels else runs
    rep = {"targets_vs_targets": max_corr(t, t, same=True)}
    maps = {}
    for r, lab in zip(runs, labels):
        x = load(f"{r}_maps.npy")
        maps[lab] = x[0]
        rep[lab] = {"vs_targets": max_corr(x, t), "vs_each_other": max_corr(x, x, same=True)}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(a.out + ".json", "w"), indent=1)
    lo, hi = np.percentile(t, [0.5, 99.5])
    panels = [("target 0", t[0])] + list(maps.items())
    fig, ax = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for k, (name, img) in enumerate(panels):
        ax[k].imshow(img, vmin=lo, vmax=hi, cmap="inferno")
        ax[k].set_title(name, fontsize=9)
        ax[k].axis("off")
    fig.savefig(a.out + ".png", dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
