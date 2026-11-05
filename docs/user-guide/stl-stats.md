# `stl-stats`: statistics of maps

`stl-stats` computes the scattering-covariance statistics of full-sky HEALPix maps with the plan
and the engine of `stl-synth`, and writes them with the plan to one JSON file. Use it to look at
a target before a synthesis, to check a synthesis against its target, or to build a set of
statistics over realisations.

```bash
stl-stats --out stats.json --norm target.npy -- target.npy synthesis_nested.npy
```

For statistics inside a Python program, averaged over masks, on disc patches, cross statistics
of two maps, or with a gradient, use the Python engines ({doc}`native-engine`, {doc}`patch-engine`).

---

## Key parameters

| Option | Default | Description |
|---|---|---|
| `--out FILE` | required | the JSON output |
| `--norm MAP` | each map's own moments | standardise every map with the mean and std of `MAP`, so that all statistics are in the same units |
| `-- MAP ...` | required | the maps, after `--`, all at the same nside |
| `--J`, `--L`, `--kernel-size`, `--fewer` | `log2(nside) − 1`, 4, 5, off | the plan, as in `stl-synth` |
| `--ranks N`, `--gpu`, `--block-pixels`, `--gpu-*` | | the engine, as in `stl-synth` |

Every option: {doc}`../api/cli/stl-stats`.

## Output

```python
import json, numpy as np
r = json.load(open("stats.json"))
r["nside"], r["J"], r["L"], r["kernel_size"], r["fewer"], r["normalisation"]
plan = r["plan"]          # one dict per coefficient: family, j1, j2, j3, l1, l2, l3, native_dg
for m in r["maps"]:       # one entry per map: file, mean, std, seconds, re, im
    s = np.array(m["re"]) + 1j * np.array(m["im"])      # in the order of the plan
fam = np.array([c["family"] for c in plan])               # mean, var, S1, S2, S3, S4
```

At nside 64, `J = 5`, `L = 4`, the plan has 2522 coefficients: `mean`, `var`, 20 S1, 20 S2, 240
S3 and 2240 S4. The values are those of STL-Dev's `PyramidStatistics` to 1e-11.

---

## Minimal example

```bash
T=fixtures/takahashi/takahashi_r000_zs1_nside64_nested.npy
stl-stats --out stats.json --norm $T --ranks 4 -- $T
```

A comparison of a target, its synthesis and white noise, with the figure:
{doc}`../tutorials/statistics`. In Python with masks and figures: {doc}`../tutorials/map_statistics`.

---

## Pitfalls and costs

- The maps must be **full-sky NESTED** with finite pixels; all maps of one call have the same
  nside.
- Without `--norm`, each map is standardised by its own mean and std: the statistics of a map and
  of twice that map are then identical. Pass the target as `--norm` to compare amplitudes.
- **Cost**: at nside 64 about 2.5 s per map on 4 CPU ranks, below 0.1 s on a T4 after a setup of
  about 15 s (device kernels compiled by NVRTC). The CPU and GPU values agree to 5e-15.
- The statistics are pixel means at each coefficient's native level: compare maps at the same
  nside, `J`, `L` and kernel.
