"""Are the syntheses of stl-synth2d --c2f different from each other and from the targets?

    python diversity.py --run PREFIX --run2 PREFIX2 --target Turb_6.npy --levels 32,64,128 --out FIG

For the final level (PREFIX_maps.npy) and every coarse level
(PREFIX_c2f_<n>_maps.npy, targets Fourier-truncated as in the driver):
- zero-shift Pearson correlation between the synthesised images (mean, max);
- maximum over periodic translations (FFT cross-correlation) of the
  correlation of every synthesis with every target, and between targets
  (mean of the per-image maxima, overall max);
- final level: the same between the syntheses of the two seeds.
Prints a JSON table and writes FIG.png (4 syntheses, 4 targets).
"""
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def nyquist(x):
    n = x.shape[-1]
    k = np.fft.fftfreq(n) * n
    m = (k[:, None] ** 2 + k[None, :] ** 2) <= (n / 2) ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(x) * m))


def resample(x, m):
    n = x.shape[-1]
    X = np.fft.fftshift(np.fft.fft2(x), axes=(-2, -1))
    a = (n - m) // 2
    Y = X[..., a:a + m, a:a + m].copy()
    Y[..., 0, :] = 0
    Y[..., :, 0] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(Y, axes=(-2, -1)))) * (m / n) ** 2


def unit(x):
    x = x.reshape(len(x), -1, *x.shape[-2:])[:, 0]
    x = x - x.mean(axis=(1, 2), keepdims=True)
    return x / np.sqrt((x ** 2).mean(axis=(1, 2), keepdims=True))


def pearson(a, b, same):
    c = np.einsum("ixy,jxy->ij", a, b) / a[0].size
    if same:
        c = c[~np.eye(len(a), dtype=bool)]
    return float(np.abs(c).mean()), float(np.abs(c).max())


def shifted(a, b, same):
    """Per pair, max over periodic shifts of the correlation; mean of per-row maxima and the overall max."""
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    m = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        cc = np.real(np.fft.ifft2(A[i][None] * np.conj(B))) / a[0].size
        m[i] = np.abs(cc).reshape(len(b), -1).max(1)
    if same:
        np.fill_diagonal(m, 0)
    return float(m.max(1).mean()), float(m.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run2", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ntarget", type=int, default=10)
    ap.add_argument("--levels", default="32,64,128")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = nyquist(np.load(a.target)[: a.ntarget].astype(float))
    rows = {}
    for n in [int(v) for v in a.levels.split(",")] + [t.shape[-1]]:
        s = np.load(f"{a.run}_maps.npy" if n == t.shape[-1] else f"{a.run}_c2f_{n}_maps.npy")
        tn = unit(t if n == t.shape[-1] else resample(t, n))
        s = unit(s)
        rows[n] = {"synth_synth_pearson": pearson(s, s, True), "synth_target_max_shift": shifted(s, tn, False),
                   "synth_synth_max_shift": shifted(s, s, True), "target_target_max_shift": shifted(tn, tn, True)}
    s1, s2 = unit(np.load(f"{a.run}_maps.npy")), unit(np.load(f"{a.run2}_maps.npy"))
    rows["seed_vs_seed"] = {"pearson": pearson(s1, s2, False), "max_shift": shifted(s1, s2, False)}
    print(json.dumps(rows, indent=1))
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    tt = unit(t)
    fig, ax = plt.subplots(2, 4, figsize=(14, 7.2))
    lim = np.percentile(np.abs(tt), 99.5)
    for k in range(4):
        ax[0, k].imshow(s1[k], cmap="inferno", vmin=-lim, vmax=lim)
        ax[0, k].set_title(f"synthesis {k}")
        ax[1, k].imshow(tt[k], cmap="inferno", vmin=-lim, vmax=lim)
        ax[1, k].set_title(f"target {k}")
    for x in ax.flat:
        x.axis("off")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=100)


if __name__ == "__main__":
    main()
