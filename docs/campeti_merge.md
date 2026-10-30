# Merging P. Campeti's ensemble package with STL-Dev-MPI

Merge plan for `STL_ensemble_GH200_v2_variance_ratio.zip` (P. Campeti, version
`2026-10-29-v2-variance-ratio`, 142 files) and the current STL-Dev-MPI.
Status: plan only. Nothing of his package is in the repository yet.

## 1. His base and his patch

His package embeds a snapshot of the crates (`STL-Dev-MPI/`). The manifests
(`NATIVE_SOURCE_MANIFEST.json`, `native_build_report.json`) pin
**f083c4a** (main on 26 Oct, "docs/gpu.md: memory of the shared device
engine"). The snapshot was checked file by file against `git archive
f083c4a`. Ignoring line endings, it is identical except for the six files of
his `native.patch` (377 lines):

| File | Change |
|---|---|
| `Cargo.toml`, `Cargo.lock` | new workspace member `crates/stl-bridge` |
| `stl-core/src/engine.rs` | `LocalEngine::reload(x)`: loads a new map into a persistent CPU engine (decimations, halos) and zeroes the stored moduli |
| `stl-gpu/src/engine.rs` | `GpuEngine::load` copies into the existing map buffers instead of allocating a new pyramid per map; `HostForwardState`, `checkpoint_forward()` (host copy of `maps`, `m`, `z`, `|z|`) and `restore_forward()` |
| `crates/stl-bridge/` (new) | C ABI `cdylib`, 220 lines (section 2.1) |

f083c4a already has the shared device engine (badc63b), `--ensemble`,
`--scatter-weight`, `--level-power-weight`, `--kappa-table`,
`--marginal-weight`. It does not have the kurtosis term, the Minkowski
term, DIA or anything later.

## 2. What his package adds

### 2.1 `stl-bridge`

A C ABI over the engines, loaded from Python with `ctypes`. One handle owns
one engine (CPU `LocalEngine` or `GpuEngine`, batch fixed at creation), the
full plan (`Family::ALL`, one channel, `fewer = false`, `dg(j) = j`, gauge
cosmo, kernel 5, L = 4) and the geometry.

| Function | Calls |
|---|---|
| `stlp_new(json)` | `LocalGeometry::new`, `EngineConfig::new(j, false)`, `build_plan`, `LocalEngine::new` or `GpuEngine::new(.., Partition::serial, batch, 1, GpuOptions{block_rows, cache, cache_bytes, threads})` |
| `stlp_info` | plan as JSON; device memory, timers, geometry cache |
| `stlp_forward(x, gen, member, objective, moments)` | `reload` + `forward` / `forward_moments` (CPU), `load` + `forward(&Serial, plan, moments)` (GPU); values and within-map spatial variances |
| `stlp_vjp(cot, ...)` | `reverse` + `map_cotangent` (CPU), `reverse` + `analysis_vjp` + download (GPU) |
| `stlp_checkpoint` / `stlp_restore` | host copy of the forward state (CPU: `maps`, `m`; GPU: `HostForwardState`) |
| `stlp_lbfgs(x, options, callback)` | `stl_core::lbfgs::minimize` with a Python callback |

Every forward state carries an identity (handle, trial generation, member,
objective, layout); `vjp` and `restore` refuse a stale or foreign state.
Panics are caught at the ABI boundary.

### 2.2 The Python driver (`stl_port/`, 4.7 k lines with tests and tools)

| Module | Role |
|---|---|
| `native.py` | `ctypes` binding of the bridge: `NativeEngine`, `Snapshot`, `native_lbfgs` |
| `ensemble.py` | `EnsembleObjective`: the five-block loss and its gradient for the 10 members |
| `losses.py`, `sc_variance.py` | losses on the batch mean and the batch variance of each feature |
| `sc_normalization.py` | S0 = (mean, standard error of the mean) from the native `mean`, `var`; within-map sigma `sqrt(V / N_level)` |
| `lut.py` | pooled look-up table g <-> kappa (8192 knots, linear in Gaussian coordinates, linear extrapolation), `kappa = f(g) - mean(f(g))` |
| `harmonics.py` | full-sky SHT with DUCC (`ducc0`), its exact discrete adjoint, derivative maps |
| `pdf.py` | cloud-in-cell PDF of kappa, 100 bins |
| `minkowski.py` | V0, V1, V2 of kappa at the finest level, harmonic derivatives, 65 thresholds |
| `calibration.py` | targets from TRAIN54 (54 reference maps, bootstrap of 10 for the means), cache with hashes and guards |
| `driver.py`, `synthesize.py` | CLI: `calibrate`, `run`, `evaluate`, `benchmark` |
| `migrate_cache.py`, `diagnose.py`, `benchmark_memory.py`, `preflight.py` | v1 -> v2 cache migration, diagnostics, GH200 measurements |
| `tests/` (39 test functions; the "71 passed" of the v2 report are five of the files, parametrised), `validate_*.py`, `tools/gh200_v2.sh` | CPU tests, native checks, the GH200 runner (`adopt-v1`, `validate`, `migrate-v1`, `benchmark`, `pilot`) |

`reference/` holds the source scripts of the method (FOSCAT kernels,
`05-Multireso...py`, `08-minkowski_sht_loss.py`, 370 kB); `provenance/` holds
the v1 documents and reports.

### 2.3 The objective

`L = (L_GSC + L_PSC + L_PS + L_PDF + L_MF) / 5`. Each block compares the
**batch mean** over the 10 members with the bootstrap mean of 10 TRAIN54
maps, and the **batch variance** (ddof = 1) with the variance over the 54
maps. The pullback of the moments is `a / B + 2 c (t_b - mu) / (B - 1)`.

| Block | Feature | Mean term | Variance term |
|---|---|---|---|
| GSC | full plan of g (S0, S1-S4) | `mean |mu - t|^2 / sigma^2`, sigma = mean over TRAIN54 of the within-map sigma | `mean log(max(V/V_train, 1e-12))^2` (v2) |
| PSC | S1-S4 of kappa | sum of 4 family means, sigma equalised per orientation | same as GSC |
| PS | 16 log bands of the C_l of kappa (DUCC, lmax 3 nside - 1) | `10 mean (log mu - log t)^2` | `2.5 mean log((v+e)/(v_t+e))^2`, bands 20-383 at 128 |
| PDF | CIC density of kappa | `mean ((mu-t)/sigma)^2` | `mean log ratio^2` |
| MF | V0, V1, V2 of kappa (finest level) | `mean (mu-t)^2 / v_t` | `10 mean log ratio^2` |

v2 changed only the SC variance terms (v1 used an absolute floor 1e-12 on
the variances, which switched off the physical SC variance terms).

### 2.4 Execution modes

| Mode | Engine batch | Per SC block and evaluation |
|---|---|---|
| `resident` | 10 | one forward, one reverse; refused above nside 128 |
| `stream-replay` (default) | 1 | per member: forward, then forward again and reverse in the backward pass |
| `stream-checkpoint` | 1 | per member: forward + host copy of the forward state, then restore and reverse |

The members are run one after the other through one batch-1 GPU engine
(what the email calls running the members on the GPU; there is no rotation
of the maps in the code). The power, PDF and MF blocks run on the CPU with
DUCC. L-BFGS is the native one, over the 10 x 12 nside^2 vector, through
the callback.

### 2.5 Validation evidence in the zip

- v2: 71 CPU tests of the Python losses (`verification/local_cpu_tests.txt`);
  no native library, no GPU, no healpy/ducc0 in that environment.
- v1 (`provenance/v1_validation/`): native CPU, nside 8, B = 10:
  resident/replay/checkpoint identical to the bit, VJP vs finite
  differences 1.2e-10 absolute, one accepted L-BFGS step.
- The GH200 runs of the email (nside 128: 4.1 GB, 2500 iterations) have no
  logs in the zip. The 54 TRAIN54 maps are not in the zip either.

## 3. His losses and the terms that exist here

| His block | Term here (`stl-synth`) | Same / different |
|---|---|---|
| GSC mean | `--weights ergodic --ensemble B` | same estimator (pixel mean, sigma^2 = within-map variance / n). Here the target is one map and its sigma; his target is a bootstrap mean of 10 maps and sigma the mean over 54 maps. Here the weights are multiplied by B (reduced chi^2 of the mean); his are not. His S0 second entry is the standard error of the mean, ours the raw `var` |
| GSC variance | `--scatter-weight X`: `X/n sum (D_i / sigma_i^2 - 1)^2` | same quantity (variance of the members, ddof 1, complex = Var Re + Var Im). Reference: ergodic variance of the target here, population variance of 54 maps there; linear vs log ratio |
| PSC mean | `--kappa-table KNOTS --kappa-families F --kappa-weight X` | same pipeline (one more wavelet transform of kappa = f(g) per level, on the same engine). f: monotone cubic, 256 knots, no monopole removal here; linear in Gaussian coordinates, 8192 knots, `- mean` there. His physical sigma is equalised per family |
| PSC variance | none | new (the scatter term acts on g only) |
| PS (C_l of kappa) | `--level-power-weight` (variance of the pyramid details of g) | different quantity: pixel-space levels vs harmonic bands; no SHT in the Rust code |
| PDF (CIC of kappa) | `--marginal-weight` (sorted values of g vs the target, squared W2); tail: `--marginal-tail-weight` (branch `claude/kappa-minkowski`, e0496a1) | related, not the same: ours holds every rank of g, his the binned density of kappa, mean and variance over members |
| MF | `--minkowski-weight --minkowski-nside A-B` | ours: every pyramid level of kappa, 3x3 stencil derivatives, 36 thresholds, sigma from 10 realisations, mean only; his: finest level, harmonic derivatives, 65 thresholds, mean and variance. Our V1, V2 are 0.33-0.49 of the harmonic ones |
| none | `--kurtosis-weight` | not in his objective |
| init: Gaussian random field with the bootstrap mean C_l | `--init` coarse-to-fine | different |
| `native_lbfgs` | the same `stl_core::lbfgs::minimize` | same code; its vector operations are threaded on `claude/musing-elgamal-9897d9` |

## 4. What STL-Dev-MPI has gained since f083c4a

On main:

- **Kurtosis of the kappa details** and S3 of kappa (39d2f60, recipe
  `--kurtosis-weight 0.1 B --kurtosis-nside 32-N`).
- **Minkowski term per level** (9c5bb9d, f05ba66) and `stl-minkowski`
  (sigma files over r000-r009), merged in 97f3791.
- `--freeze-nside` (0a1b919), overlap mode (a1ff836), the diffusion
  prototype (python, `diffusion/`).
- **Masked DIA convolution** on the GPU, the default since 8120aec: 1.6x
  (nside 1024) to 2.6x (nside 2048) per evaluation on one A100; kept
  geometry 9608 -> 3152 bytes per row.

Not merged:

- `claude/musing-elgamal-9897d9` (head 22cefc6, merged with main e1deaf1):
  - wavelet fields recomputed per block (`--gpu-fields`, 13ae057): fields
    188 -> 60 bytes per pixel and member at L = 4;
  - the objective's terms on the device (`--gpu-terms device`, 96c6bf1,
    feb832c): kappa, kurtosis, level power, marginal, scatter;
  - threaded L-BFGS vector operations (2c51bd9, `STL_THREADS`; partial sums
    in chunk order, independent of the number of threads);
  - 2048 x 10 on one A100: 61.7 s per evaluation, peak 67.5 GB, production
    terms (preliminary).
- `claude/kappa-minkowski` (e0496a1): `--marginal-tail-weight` (the
  Minkowski term itself is already in main). It conflicts with
  musing-elgamal in `stl-cli/src/{lib,marginal}.rs` (the marginal moved to
  the device there).

Effect on his package: his bridge never calls `stl-cli` or `stl-mpi`; it uses
the engines directly. DIA, recompute and the threaded L-BFGS reach it through
`stl-core` and `stl-gpu` without change to the Python. The new `stl-cli` terms
do not reach it (his objective is in Python).

## 5. Conflicts and the bridge on the current API

The public calls of the bridge (`LocalEngine::new/forward/forward_moments/
reverse/map_cotangent`, `GpuEngine::new/load/forward/reverse/analysis_vjp/
device_memory/geometry_cache/timers`, `Partition::serial`, `minimize`) have
the same signatures on f083c4a, main and musing-elgamal. What breaks:

| Item | main | musing-elgamal | Port |
|---|---|---|---|
| `GpuOptions { block_rows, cache, cache_bytes, threads }` | new field `conv` | new fields `conv`, `fields` | build from `GpuOptions::default()` and set `conv`, `fields` from the JSON config (`"conv"`, `"fields"`) |
| `LocalEngine::reload` | absent | absent | add (his hunk applies as is) |
| `GpuEngine::load` buffer reuse | applies | `load` now calls `load_device`, which allocates the pyramid | redo the reuse in `load_device`, or drop it (allocation is cheap next to a sweep; measure) |
| `checkpoint_forward` (`z`, `|z|`) | applies | with `recompute`, `z` and `|z|` are not kept between the sweeps | store `maps`, `m` (and `z`, `|z|` when kept); restore recomputes, as the CPU path already does. At nside 2048, L = 4 that is ~1.3 GB of host memory per member instead of ~7.7 GB |
| summation order | as f083c4a | L-BFGS dot products by chunks of 65536 | trajectories equal to rounding, not to the bit; his `optimizer_unchanged_sha256` pin must move |
| version pins (`BASELINE_REVISION`, `upstream_revision`, `native_revision`) | f083c4a | f083c4a | new base revision; his cache guard then asks for a new calibration or an explicit compatibility record |

Default convolution: DIA. His trajectories were made with the stencils;
`"conv": "stencils"` keeps them reproducible.

## 6. Python and Rust

Stay in Python (his method, cheap per pixel, or no Rust equivalent):

- calibration, bootstrap, caches and guards; the GH200 runner;
- LUT, CIC PDF, the moment losses (a few `len(plan) x B` arrays);
- C_l and harmonic Minkowski functionals (DUCC SHT; at nside 2048 with
  lmax 6143 about 15 SHTs per member and evaluation, on the CPU: to be
  measured, likely the dominant cost at 2048).

Into Rust, in this order:

1. **Resident batch at nside 2048.** With `recompute`, ten nside-2048 maps
   fit on one 80 GB A100 (67.5 GB with every production term); the GH200
   has 96 GB. The resident mode then needs one forward and one reverse per
   SC block instead of 10 x (2 forwards + 1 reverse). Only his guard
   "resident <= nside 128" stands in the way. The memory figure is to be
   measured on the GH200 (or one A100 run with the main session's OK).
2. **Both SC blocks in one engine call.** g and kappa through the same
   engine already; a bridge call that takes g, the LUT knots and the two
   cotangent vectors would save the host round trips (as the kappa term of
   `stl-cli`).
3. **Population targets in `stl-synth`.** Target mean and variance per
   coefficient from a set of reference maps (`stl-stats` over TRAIN54 or
   r000-r009), sigma from the mean within-map sigma, a log-ratio variance
   term on g and on kappa. That makes his GSC/PSC blocks `stl-synth` terms
   with MPI groups and the device objective, and lets his runs and ours be
   compared on the same footing.
4. Later, if the SHT dominates: harmonic terms on the device (needs an SHT;
   not planned).

## 7. Base branch

| Option | Gains for his method | Port cost | Risk |
|---|---|---|---|
| A. main (4f651b1) | DIA, Minkowski/kurtosis available for comparison | small (`conv` field, his hunks apply) | none; but resident 2048 x 10 does not fit (fields kept: 94.5 GB at 2048 x 10) |
| **B. `claude/musing-elgamal-9897d9` (22cefc6)** | A + recomputed fields (resident 2048 x 10 fits), threaded L-BFGS, device-terms engine | medium (`fields`, checkpoint without `z`, `load_device`) | the branch is not merged yet; it is pushed and tested, and contains main up to DIA |
| C. B + `claude/kappa-minkowski` (e0496a1) | B + tail term in `stl-synth` | B + a conflict in `stl-cli` unrelated to his code | the tail is not in his objective |

Recommendation: **B**. The tail term does not touch the bridge; it can come
with the merge of musing-elgamal into main.

## 8. Work plan

1. Branch `claude/campeti-merge-19032c` from the chosen base.
2. `crates/stl-bridge`: his crate, ported (section 5), tests in Rust:
   CPU vs GPU, resident vs stream vs checkpoint, VJP vs finite differences,
   on fixtures at nside 8-32.
3. `python/stl_ensemble/`: his `stl_port`, `tests`, `configs`, the
   validation scripts and docs; `reference/` and `provenance/` left out
   unless wanted. Changes: options `conv` and `fields`, revision pins,
   resident allowed above 128 behind an explicit memory check.
4. garoupe (one T4, after `nvidia-smi`): his `validate` stage (pytest,
   `native_smoke`, `validate_sc_ratio_native`, `validate_modes` at nside 8,
   both physical scales) on his exact source and on the port; then a
   nside-128 benchmark (memory, s/eval per mode) and a short L-BFGS run,
   both sources, from the same synthetic 54-map population (TRAIN54 is not
   available here).
5. Report: differences to the bit or to rounding, memory and time per
   evaluation per mode, the nside-2048 estimate and what remains to measure
   on the GH200.

## 9. Open questions

- TRAIN54: which simulations (Takahashi zs?) and where; the nside-128
  comparison with his GH200 run needs them, or his cache.
- The 2048 C_l variance cutoff (`ps_variance_upper`) is still unresolved
  in his config.
