# Merging P. Campeti's ensemble package with STL-Dev-MPI

Merge plan for `STL_ensemble_GH200_v2_variance_ratio.zip` (P. Campeti, version
`2026-10-29-v2-variance-ratio`, 142 files) and the current STL-Dev-MPI.

## 0. Status (30 Oct 2026)

Decisions of J.-M. Delouis on the plan below: base branch
`claude/musing-elgamal-9897d9` (option B of section 7); then **everything in
Rust**, with a HEALPix SHT written in Rust for the harmonic terms (section 6
is superseded).  Branch `claude/campeti-merge-19032c`:

| Commit | What |
|---|---|
| ccc44f6 | merge of `claude/musing-elgamal-9897d9` (22cefc6) on main 4f651b1 |
| 79d2fee | his `native.patch` ported (section 5) and `crates/stl-bridge` |
| 244c99c | `python/stl_ensemble`: his Python as delivered (the reference) |
| 4f3668c | `stl-core::sht` and `fixtures/ensemble` (his Python on the CPU bridge) |
| 82189c2 | `stl-ensemble`: his objective in Rust (`crates/stl-cli/src/ensemble`) |
| 5c925f4 | Minkowski states kept between the passes; timings on garoupe |

Full run at nside 128 (ten members, 2500 iterations, 3 T4): 2 h 34 min,
loss 61.8 to 0.625 (section 8.5).

Tests on garoupe before the merge into main (revision 2b45ba4, release
profile, 3 T4, no device test skipped): `cargo test --workspace` (with the
doc-tests), `-p stl-bridge --features gpu`, `-p stl-mpi --features gpu`,
`-p stl-cli --features gpu,mpi`, `tools/mpi_check.sh 2 4` and `2 3`, and
the 20-iteration trajectory against STL-Dev (`--ignored`, 4 ranks): all
pass.

`stl-ensemble` (section 8 of this note: usage, validation, timings) runs his
method without Python: `calibrate` builds the LUT and the targets from any
list of reference maps, `run` minimises the ten members with the native
L-BFGS.  The Python package and the bridge stay in the repository as the
reference the Rust is tested against (`tools/dump_ensemble_reference.py`).

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

## 8. `stl-ensemble`: his method in Rust

### 8.1 Usage

    # targets from the reference maps (NESTED, any nside >= N; here the ten
    # Takahashi realisations r000-r009 at nside 2048, block-averaged to 128)
    stl-ensemble calibrate --references r000.npy ... r009.npy --nside 128 --out cal128.json [--gpu]
    # ten members, one device engine for the SC of g and of kappa
    stl-ensemble run --calibration cal128.json --niter 2500 --max-eval 50000 --gpu --out ens
    # two GPUs: two MPI groups of five members
    mpirun -n 2 stl-ensemble run --calibration cal128.json ... --gpu --groups 2 --out ens

Outputs: `ens.json` (loss and five blocks per evaluation, seconds per
evaluation, timers, peak memory), `ens_mMMM_gaussian_nested.npy` (the
variable g) and `ens_mMMM_physical_nested.npy` (kappa = f(g) - mean) per
member.  `stl-ensemble --help` lists the options.

### 8.2 Python module -> Rust

| `stl_port` | Rust |
|---|---|
| `lut.py` (`PooledLUT`, `probability_grid`, `source_ppf`) | `ensemble/lut.rs` (numpy `quantile` and `linspace` replicated; `erfinv` to double precision) |
| `losses.py`, `sc_variance.py`, `sc_normalization.py` | `ensemble/losses.rs` |
| `harmonics.py` (`FullSkySHT`, DUCC) | `stl-core/src/sht.rs` |
| `pdf.py`, `minkowski.py`, `lp_windows` | `ensemble/fields.rs` |
| `calibration.py` | `ensemble/calibration.rs` |
| `ensemble.py` (`EnsembleObjective`) | `ensemble/objective.rs` |
| `driver.py`, `native.py`, the bridge | `ensemble/cli.rs`, `bin/stl-ensemble.rs`; the engines and L-BFGS directly |

The execution modes of section 2.4 disappear: the ten members are one
batch of the engine (or B / G per MPI group), so each SC block is one
forward and one reverse sweep per evaluation, on the shared device engine
(DIA geometry, recomputed fields when they do not fit).

### 8.3 Validation against his Python

`tools/dump_ensemble_reference.py` runs his package (with ducc0, healpy and
the CPU bridge) on nside 16, 12 synthetic reference maps, 10 members, and
writes `fixtures/ensemble`; `crates/stl-cli/tests/ensemble.rs` compares:

| Quantity | Largest relative difference |
|---|---|
| SHT against DUCC: analysis, C_l, synthesis, analysis cotangent, five derivative maps, their cotangent | 7e-14 |
| LUT: probability grid, physical knots | 0 (bitwise) |
| LUT: Gaussian knots (`f32` of erfinv) | at most 2 knots differ by one `f32` ulp |
| calibration targets (SC of g and kappa: means, variances, sigmas; C_l, PDF, Minkowski) | 3e-14 |
| loss, each of the five blocks | 1.3e-13 |
| gradient of the ten members | 9e-14 |

Finite differences agree with the gradient (and with his) only to 1.5e-3:
the curvature floor of the Minkowski functionals is held fixed in the
cotangent, as in his code, and the CIC PDF has kinks.

### 8.4 What differs from his package

- References: any list of maps instead of the 54 TRAIN54 files; the target
  of the means is the mean of a given draw (`--bootstrap-ids`), all the
  references when there are as many as members (r000-r009 for ten members),
  else a seeded draw with replacement (not numpy's).  The population
  variances are over all the references (ten instead of 54: noisier).
- Preprocessing: the references are block-averaged in `f64` to nside 512,
  rounded to `f32`, then to the target nside, rounded to `f32` (his:
  `hp.ud_grade` then a `float32` mean); maps read as NESTED `.npy`/FITS.
- Initial fields: Gaussian random fields with the same C_l (mean C_l of the
  drawn Gaussian references, C_0 = 0, plus the first drawn reference's
  mean), drawn with the package's counter-based normal generator, not
  numpy's: the same law, other numbers.
- Sums: in other orders (chunked, threads) than numpy's pairwise sums:
  differences at the 1e-16 level, see 8.3.
- His guards (54 maps, nside 128 or 2048 only, pinned revisions, cache
  hashes) are not reproduced; the calibration records its references and
  draw.

### 8.5 Timings on garoupe (Tesla T4)

nside 128, ten members, targets from the ten Takahashi realisations
r000-r009 (zs1, nside 2048 block-averaged to 128; `cal128.json`, 10 min 43 s
on one CPU core, 0.9 GB), 3 L-BFGS iterations from the Gaussian random
fields; garoupe /tmp/jmdeloui-campeti/runs:

| Layout | s / evaluation | before keeping the Minkowski states |
|---|---|---|
| 1 T4, one batch of 10 (`STL_THREADS=12`) | 6.08 | 8.76 |
| 2 T4, `--groups 2` (5 members per GPU, `STL_THREADS=6`) | 3.26 | 5.30 |
| 3 T4, `--groups 5` (2 members per rank, 5 ranks, `STL_THREADS=2`) | 3.10 | |

The losses of the three layouts agree to 2e-13.  On one T4 before the
change: SC of g and kappa on the device 2.3 s, Minkowski functionals on the
host 5.7 s (forward, replay and cotangent; 65 % of the evaluation), C_l and
PDF 0.6 s.  The Minkowski states are now kept between the forward and the
reverse pass (6 maps per member, up to 4 GiB per group; nside 2048
recomputes them as the source) and the 65 Gaussians of a pixel come from
a recurrence (results unchanged to 1e-13, section 8.3).  Peak device memory
2 GB on one T4 (batch 10), host 0.65 GB.

Full run as his (`--niter 2500 --max-eval 50000`, 3 T4, `--groups 5`,
`STL_THREADS=2`; garoupe /tmp/jmdeloui-campeti/runs/full128, revision
5c925f4): **2 h 34 min** wall clock (30 Oct 19:23 to 21:57, setup 10 s),
2869 evaluations at 3.10 s, exit status 0, no device error.  It stopped on
the 2500-iteration limit (1.15 evaluations per iteration): the loss still
fell by about 4e-4 per evaluation at the end, so none of the tolerance tests
(`gtd`, gradient 1e-12, step or loss change 1e-15) can have fired.  The
loss went from 61.81 to 0.625; the last value is the lowest (the 279 rises
of `loss_history` are line-search trials).  Peak device memory 2.8 GB (max
over the ranks, two ranks on each of two T4), host 0.53 GB per rank.
Time per evaluation (rank 0, two members): Minkowski functionals 1.72 s
(forward 1.11, reverse 0.62; 56 %), Gaussian SC 0.56 s, physical SC
0.42 s, C_l and PDF 0.34 s, host chain and physical map 0.04 s.

Final blocks (`blocks_final` of `ens.json`):

| Block | Loss | Mean term | Variance term |
|---|---|---|---|
| Gaussian SC | 2.611 | 2.221 | 0.389 |
| Physical SC | 0.412 | 0.282 | 0.131 |
| C_l | 0.0080 | 0.0066 | 0.0014 |
| PDF | 0.0156 | 0.0104 | 0.0052 |
| Minkowski | 0.0801 | 0.0710 | 0.0091 |

The ten final physical maps against the ten references at nside 128 (plain
block means of r000-r009, the maps the targets come from: a training-set
check, not a validation; z is the difference of the two means over its
standard error):

| Statistic | Members (mean +- sd) | References (mean +- sd) | z | sd ratio |
|---|---|---|---|---|
| std | 3.315e-4 +- 2.2e-5 | 3.308e-4 +- 2.4e-5 | 0.07 | 0.93 |
| skewness | 12.9 +- 2.0 | 11.3 +- 1.1 | 2.1 | 1.8 |
| excess kurtosis | 452 +- 176 | 287 +- 75 | 2.7 | 2.3 |
| quantile 0.001 | -1.999e-4 +- 1.2e-5 | -1.918e-4 +- 1.2e-5 | -1.5 | 0.97 |
| quantile 0.01 | -1.831e-4 +- 9.9e-6 | -1.802e-4 +- 1.1e-5 | -0.6 | 0.86 |
| median | -8.30e-5 +- 6.5e-6 | -8.20e-5 +- 6.2e-6 | -0.4 | 1.05 |
| quantile 0.99 | 1.255e-3 +- 9.1e-5 | 1.252e-3 +- 9.8e-5 | 0.05 | 0.93 |
| quantile 0.999 | 3.77e-3 +- 2.1e-4 | 3.82e-3 +- 2.8e-4 | -0.4 | 0.76 |
| minimum | -2.334e-4 +- 7.4e-6 | -2.046e-4 +- 1.2e-5 | -6.6 | 0.64 |
| maximum | 2.52e-2 +- 3.3e-3 | 1.99e-2 +- 3.5e-3 | 3.4 | 0.92 |

The bulk (std, quantiles 0.001 to 0.999) is within the scatter of the
references.  The extremes are wider: the minima are lower than the
references' (beyond its extreme knots the LUT extrapolates linearly in
Gaussian coordinates, and his method adds no clipping, by design), and the
few brightest pixels push skewness and kurtosis up; a handful of pixels
per map weigh little in the five blocks.  His package has
no nside-128 result to compare with: his `VALIDATION_REPORT.md` states that
no native128 scientific run was made (the TRAIN54 maps were absent), and
his earlier FOSCAT runs used another engine (7x7 kernel, 8 orientations,
SciPy L-BFGS-B up to 8000 iterations).

### 8.6 What remains

- nside 2048: memory and time of a whole evaluation to measure on the
  GH200 or an A100 (section 9 extrapolates the harmonic terms).
- Several ranks per group compute the harmonic terms redundantly (no gain
  for them); with one rank per group (one GPU per group) nothing is redundant.
- The 2048 C_l variance cutoff (`--ps-variance-upper`) is still a choice
  to make (383 at 128 in his configuration).
- TRAIN54 itself (Takahashi zs1, r000-r053) for a like-for-like comparison
  with his GH200 runs.

## 9. Harmonic terms on the device (`--harmonic gpu`)

Branch `claude/ensemble-gpu-terms` (from the merge PR's head 5391d45).
Profile of the host terms first (`host_profile` in
`crates/stl-cli/tests/ensemble.rs`, one member, one evaluation, 4 threads
of garoupe; the Minkowski thresholds are the functionals minus their
transforms):

| nside | SHT (2 analyses, 2 syntheses, derivatives and their VJP) | Minkowski thresholds | CIC PDF | C_l products | total |
|---|---|---|---|---|---|
| 128 | 0.37 s (57 %) | 0.27 s | 0.006 s | 0.0003 s | 0.65 s |
| 512 | 15.9 s (79 %) | 4.1 s | 0.06 s | 0.05 s | 20.1 s |
| 1024 | 116 s (87 %) | 17.0 s | 0.26 s | 0.05 s | 133 s |

The transforms grow as `nside^3` and dominate from nside 128 on.

### 9.1 What runs where

- `stl_gpu::harmonic::GpuSht` (`crates/stl-gpu/src/harmonic.{rs,cu}`): the
  port of `stl_core::sht`, same conventions and packing, NESTED maps on
  the device.  Forward Legendre sums one thread per (m, ring pair), the
  recursion and the sums over l in the host's order; transposed sums one
  block per (m, chunk of 1024 ring pairs): each thread adds its four rings
  in order, the threads in a fixed tree (warp shuffles, then the warps in
  order), the chunks in order.  Recursion coefficients in a device table
  (IEEE division and square root: the host's values).  Ring FFTs by cuFFT,
  one batched plan per ring length (the polar pairs, the equatorial
  rings), loaded at run time (`crates/stl-gpu/src/cufft.rs`: cudarc only
  binds cuFFT for CUDA 12; garoupe uses the CUDA 10 toolkit's
  `libcufft.so.10` with its 11.4 driver, `LD_LIBRARY_PATH` or
  `STL_CUFFT`).
- `crates/stl-cli/src/ensemble/gpu_terms.rs`: per member, the map goes up
  once, its analysis gives the C_l (the host's sum over m), the CIC
  histogram (one thread per chunk of 2^14 pixels in pixel order: the
  host's sums), the Minkowski functionals (pixel sums per chunk of 2^16
  in a fixed tree, chunks added in order on the host; Gaussians of the
  thresholds evaluated directly, where the host uses a recurrence); the
  states (u and the five derivative maps) stay on the device between the
  passes; the map cotangent comes down once.  The losses, the LUT and the
  chain to the Gaussian maps stay on the host.
- `--harmonic cpu|gpu` (default `gpu` with `--gpu`): the host path stays
  for the tests and machines without cuFFT; the report says which ran.

### 9.2 Exactness (garoupe, Tesla T4)

| Check | Largest difference |
|---|---|
| `GpuSht` against `stl_core::sht`, nside 1 to 128, every transform and transpose (`crates/stl-gpu/tests/harmonic.rs`) | 2.9e-15 relative (C_l bitwise) |
| The five blocks of the fixture objective, device against host terms (`device_terms_match_the_host`) | 1.5e-13 (Minkowski; C_l 1e-15, PDF and SC bitwise) |
| Its gradient | 9.0e-14 |
| Loss after 5 L-BFGS iterations at nside 128 (full runs below) | 1.5e-11 relative (rounding carried by the trajectory) |

### 9.3 Gain

Harmonic terms of one member and one evaluation (`device_profile`
against `host_profile`, one T4 against 4 host threads):

| nside | device | host (4 threads) | gain | device SHT share |
|---|---|---|---|---|
| 128 | 0.092 s | 0.65 s | 7.1 | 0.097 s of transforms in isolation |
| 512 | 1.86 s | 20.1 s | 10.8 | 1.47 s |
| 1024 | 11.2 s | 133 s | 11.9 | 9.5 s |

Full evaluations of `stl-ensemble run` at nside 128, ten members, Takahashi
targets (`cal128.json`), 5 L-BFGS iterations (6 evaluations), garoupe:

| Layout | `--harmonic` | s / evaluation | host CPU (`time -v`) | mean GPU utilisation (whole run) |
|---|---|---|---|---|
| 1 T4, batch 10, 12 threads | cpu | 6.04 | 315 %, 183 s user | 20 % |
| 1 T4, batch 10, 12 threads | gpu | 3.17 | 106 %, 39 s user | 43 % |
| 3 T4, 5 groups of 2, 2 threads per rank | cpu | 3.10 | 663 %, 262 s user | 16 / 16 / 9 % |
| 3 T4, 5 groups of 2, 2 threads per rank | gpu | 1.24 | 502 %, 134 s user | 30 / 30 / 14 % |

On one T4 the evaluation is 1.9 x faster and uses 4.6 x less host CPU
time; on 3 T4, 2.5 x faster.  With five ranks the remaining host CPU is
mostly OpenMPI's busy polling in the collectives (`--mca
mpi_yield_when_idle 1` would release it), not computation.  The rest of
an evaluation is now the two SC objectives (1.27 + 1.20 s of 3.76 s per
evaluation on one T4).

### 9.4 Extrapolation to nside 2048 (preliminary)

The transforms scale as `nside^3` (x 7.3 from 512 to 1024 on the host, x 6.5
on the device): the harmonic terms of one member and evaluation would
take about 90 s on a T4, against about 1070 s on 4 host threads (360 s on
the 12 of garoupe).  The Legendre sums are bound by double-precision
arithmetic (about 85 GFLOP/s on the T4, a third of its peak): an A100
(9.7 TFLOP/s double precision, 2 TB/s) should bring them to 2-6 s per
member and evaluation, i.e. 20-60 s for ten members, next to the SC
objectives measured on one A100 at nside 2048 (one SC objective of ten
members with its device terms: 62 s per evaluation,
`benchmarks/terms/a100/n2048_B10_device.json`; the method has two, of g
and kappa).  Device memory of the harmonic terms at nside 2048: about
10 GB of transients for one member (five complex ring buffers, the
Legendre sums, the partial a_lm) plus 2.4 GB of Minkowski state per kept
member: it fits next to the engine on an 80 GB device, not on a T4.
To be measured; no A100 run was made for this.
