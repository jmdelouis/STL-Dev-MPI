# Cheaper syntheses: a host tier for the geometry cache, and per-scale L-BFGS preconditioners

Two independent attempts to make a synthesis cheaper, measured on garoupe
(Tesla T4, 2026-11-03):

1. keep the convolution geometry instead of rebuilding it at every
   evaluation, on the device when it fits and in page-locked host memory
   otherwise (`--geometry-cache`);
2. cut the number of L-BFGS iterations with a per-scale preconditioner,
   the map staying the variable (`--precondition`), and with a coarse-to-fine
   start (`--init` from a lower nside, `--freeze-nside`).

Summary:

- The geometry cache changes no bit of any result (trajectories identical
  in every policy, nside 32 to 512). Keeping the geometry, in any form,
  halves the time per evaluation against rebuilding it (`off`). The new host
  tier does the most good when the device cannot hold the whole geometry:
  `--geometry-cache auto` is the fastest policy at nside 512 × 4, at 7.0 s
  per evaluation against 8.0 s for the device cache alone and 12.3 s for
  no cache (−12 % and −43 %). In recompute mode it gives 7.8 s against
  10.2 s and 15.0 s (−24 % and −48 %). With `host` and small blocks, the
  device holds 4.4 GB instead of 12.5 GB.
- No band preconditioner beats plain L-BFGS from white noise. The power laws
  `4^(γ j)` are worse for every γ tried, from −1 to +1. The adaptive per-band
  scaling (`auto`) is a draw. From a coarse-to-fine start, `bands:-0.25`
  reaches the reference loss in 73 evaluations instead of 91, the same for
  two summation orders. On the 10-member recipe at nside 256 it loses
  (12 % higher loss after 100 iterations).
- For the nside 2048 × 10 production on A100s, the host tier will most
  probably not pay: the GPU recomputes faster than PCIe uploads, and
  `auto`'s calibration then drops the host copies. The 13.1 s of geometry
  per evaluation go only if the whole geometry fits on the devices, i.e.
  the sky shared by 4 to 5 GPUs or more (section 4).

## 1. Geometry cache

### 1.1 What is new

The convolution geometry of a block (DIA coefficients, or bound stencils
with their transposition) depends on the decomposition only.
`--gpu-cache full|transposition|none` already kept it on the device:

- `full`: whole, for the blocks that fit;
- `transposition`: the DIA masks or the transposition (16 bytes per row),
  with the weights recomputed at every use (`gpu_geometry_weights`);
- `none`: everything rebuilt at every evaluation (`gpu_geometry`).

At nside 2048 × 10 on an A100, the whole geometry (≈3.3 kB per row with
DIA at L = 4, ≈200 GB for the ≈70 M rows of all levels) does not fit, so
`auto` takes `transposition`. There the weights cost 13.1 s per evaluation.

New, `--geometry-cache auto|device|host|off` (`GpuOptions::host_cache_bytes`,
`GeometryCache::Host`):

| Policy | Device | Host | Per use |
|---|---|---|---|
| `device` | as `--gpu-cache full` | nothing | nothing, or the weights recomputed for the blocks that do not fit |
| `host` | masks / transposition only | the weights of every block (page-locked) | the weights uploaded |
| `auto` | as `--gpu-cache auto` | the weights of the blocks the device does not keep whole | uploaded if an upload beats a recomputation, else recomputed |
| `off` | nothing | nothing | everything rebuilt |

The host copy of a block is made once, when the block is first stripped.
The weights of a stripped block are then uploaded (`gpu_geometry_upload`)
instead of recomputed: they are the same values, so the results are equal
to the bit. The host budget is `STL_HOST_CACHE_MB`, otherwise half of
`MemAvailable` divided by the ranks of the node. If a page-locked allocation
fails, no more host copies are made. `auto` times one upload against one
recomputation of the same block, the first time a host copy is used
(`gpu.geometry_host_calibration` in the report), and drops the host copies
if recomputing is faster. `--gpu-cache` keeps its meaning when
`--geometry-cache` is not given.

Tests: `stl-gpu/tests/stats.rs::geometry_cache_policies_agree` (statistics
and gradients over three evaluations, for every policy, device budget, host
budget and the calibration, both convolution forms: equal to the bit).
At nside 32, the five configurations of `docs/bench_n128.md`'s check give
loss histories identical to those of the `main` binary.

### 1.2 Measurements

Same problem as `docs/bench_n128.md`: Takahashi r000 z_s = 1, 4 members,
L = 4, J = 6 (8 at nside 512), `--weights none`, white-noise initial
maps from `make_inits.py`. One T4 of garoupe, 4 host threads. Keep:
`--gpu-fields keep`, default blocks. Recompute ("mem"): `--gpu-fields
recompute` with blocks of 1024 cells at nside 128 and 4096 at nside 512.
nside 128: 20 iterations (25 evaluations). nside 512: the gaussianised
Takahashi map (nside 2048 averaged to 512), 10 iterations (13 evaluations).
`refresh` is `--gpu-cache transposition` with `STL_GPU_CACHE_MB=128`: the
transposition on the device and every weight recomputed, the state of the
2048 production. Seconds below are over the whole run.

nside 128 × 4:

| Fields | Policy | s / eval | build s | weights recomputed s | uploaded s | device peak GB | device cache GB | host cache GB | host RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| keep | off | 0.792 | 10.25 | 0 | 0 | 1.46 | 0 | 0 | 0.45 |
| keep | device (= transposition: all fits) | 0.393 | 0.29 | 0 | 0 | 1.69 | 0.84 | 0 | 0.45 |
| keep | host | 0.586 | 0.31 | 0 | 4.73 | 1.46 | 0.01 | 0.83 | 1.26 |
| mem | off | 1.633 | 19.67 | 0 | 0 | 0.25 | 0 | 0 | 0.44 |
| mem | device (= auto) | 0.858 | 0.43 | 0 | 0 | 1.76 | 1.52 | 0 | 0.45 |
| mem | host | 1.117 | 0.48 | 0 | 6.49 | **0.28** | 0.01 | 1.51 | 1.92 |

nside 512 × 4:

| Fields | Policy | s / eval | build s | weights recomputed s | uploaded s | device peak GB | device cache GB | host cache GB | host RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| keep | off | 12.32 | 94.6 | 0 | 0 | 4.54 | 0 | 0 | 2.8 |
| keep | refresh | 9.46 | 3.4 | 52.9 | 0 | 4.64 | 0.13 | 0 | 2.8 |
| keep | device | 7.97 | 3.3 | 31.0 | 0 | 11.05 | 6.45 | 0 | 2.8 |
| keep | host | 8.24 | 3.3 | 0 | 35.7 | 4.60 | 0.08 | 14.1 | 16.6 |
| keep | **auto** | **7.04** | 3.2 | 0 | 20.0 | 11.10 | 6.45 | 7.7 | 10.4 |
| mem | off | 14.97 | 116.2 | 0 | 0 | 1.83 | 0 | 0 | 2.9 |
| mem | refresh | 13.06 | 6.2 | 80.8 | 0 | 2.03 | 0.13 | 0 | 2.9 |
| mem | device | 10.24 | 44.5 | 7.3 | 0 | 12.53 | 10.70 | 0 | 2.9 |
| mem | host | 9.64 | 4.8 | 0 | 40.3 | 4.44 | 0.10 | 17.8 | 20.3 |
| mem | **auto** | **7.83** | 3.9 | 0 | 16.2 | 13.15 | 10.70 | 7.2 | 9.9 |

10 members at nside 256, the recipe of the nside-512 ensemble on garoupe
(`--weights ergodic --ensemble 10 --scatter-weight 1 --level-power-weight 10
--kappa-table knots.npy --kappa-weight 0.1 --marginal-weight 10000`, init
from the nside-256 Gaussian syntheses), 5 iterations:

| Fields | Policy | s / eval | device peak GB | device cache GB | host cache GB |
|---|---|---|---|---|---|
| keep | off | 7.62 | 2.97 | 0 | 0 |
| keep | device | 4.18 | 6.39 | 3.68 | 0 |
| mem | off | 8.77 | 0.89 | 0 | 0 |
| mem | device | 4.72 | 5.45 | 4.54 | 0 |
| mem | host | 6.21 | **1.06** | 0.03 | 4.5 |

Within each field mode, every policy gives the same loss history to the bit.

Reading:

- The rebuild is the expensive part, not the weights. Keeping the
  transposition alone (`refresh`) already removes 70-95 % of the geometry
  time, and keeping everything halves the time per evaluation at nside 128.
  At nside 128 × 4 the whole geometry (0.84 GB keep, 1.5 GB mem) fits on a
  T4, so `auto` and `device` are the same there. The 63 % of geometry time
  in `docs/bench_n128.md`'s memory-minimal run comes from its `--gpu-cache
  none`.
- On a T4 an upload is faster than a recomputation. A block's weights go
  up at ≈10 GB/s (PCIe gen3, page-locked; 14.1 GB × 2 uses × 13 evaluations
  in 35.7 s). Recomputing them runs at ≈4-6 GB/s equivalent, because fp64
  is 1/32 of fp32 on a T4. `auto`'s calibration measured 0.036 s to upload
  a block against 0.055 s to recompute it.
- `auto` combines the two. At nside 512, 6.5-10.7 GB of whole geometries
  stay on the device and the 7-8 GB of weights that do not fit come from
  the host. This beats both pure policies.
- `host` keeps the device at its memory floor: 0.28 GB instead of 1.76 GB
  at 128 (mem), 4.4 GB instead of 12.5 GB at 512, 1.06 GB instead of
  5.45 GB for the 10-member run. It costs 0.3-1.5 s per evaluation more than
  `device`, still 30-35 % less than `off`. The page-locked copy costs about
  1.2 times its size in host RSS.
- In `mem` at nside 512, `device` rebuilds 44 s of geometry. The level fields
  leave the cache budget short, so some blocks are not kept at all; the host
  tier avoids that.

## 2. Per-scale preconditioner

### 2.1 Construction

Let `Pi_j` be the mean over the groups of `4^j` consecutive NESTED pixels
(the children of a pixel of level `j`), broadcast back to the pixels. The
`Pi_j` are orthogonal projectors, nested as `I = Pi_0 >= Pi_1 >= ...`.
The Haar bands

    B_j = Pi_j - Pi_{j+1}  (j < K),   B_K = Pi_K

are therefore orthogonal projectors summing to the identity, and

    P = sum_j alpha_j B_j = alpha_0 I + sum_{j>=1} (alpha_j - alpha_{j-1}) Pi_j

is symmetric positive definite for any `alpha_j > 0`. It also has an
explicit square root. L-BFGS takes `H0 = h P` with `h = s.y / y.Py` of the
last pair, instead of torch's `H0 = (s.y / y.y) I`; that is L-BFGS in the
metric `P^-1`. The first direction is `-P g`, with step
`min(1, 1/|Pg|_1)`. The map stays the variable: STL-Dev's Laplacian bands as
variables stall L-BFGS. The strong Wolfe line search is unchanged.
`K` = J (the coarsest pyramid level), or less if a rank's range is not
aligned on groups of `4^K`. Groups never cross ranks, so `P` is local, and
only the band dot products of `auto` go through `Allreduce`
(`stl_core::precond`, `stl_core::lbfgs::minimize_preconditioned`).

Factors (`--precondition`):

- `bands:G`: `alpha_j = 4^(G j)`; G > 0 favours the large scales;
- `power`: `alpha_j = e_j / e_0`, `e_j` the target's energy per degree of
  freedom in band `j` (a prior-like inverse Hessian);
- `auto`: per-band Barzilai-Borwein, `alpha_j = (s.B_j y)/(y.B_j y)` from
  the last pair, bounded to `[h/1000, 1000 h]`; `H0 = P` (no extra
  scaling). This is torch's scalar per band.

`none` (the default) runs the same operations as before, and so does
`bands:0` (`P = I` exactly): the loss histories at nside 32 are identical to
the `main` binary's. Unit tests: `stl_core::precond` (symmetry, positivity,
eigenvalues `alpha_j` on each band), `stl_core::lbfgs::preconditioned_lbfgs`
(the identity reproduces the trajectory to the bit; on a quadratic whose
curvature is `4^(-2j)` on band `j`, `bands:2` converges in a few
iterations).

### 2.2 Measurements

Same nside 128 × 4 problem, keep, `--geometry-cache device`, 100
iterations. Reference: the plain run reaches 2.380e-6 after 100 iterations
(115 evaluations). "Evaluations to reference" counts the evaluations until
the best loss so far reaches that level. To show the noise, the plain run
was repeated in recompute mode, which has another summation order:
trajectories that differ only by rounding part at the 1e-6 level after
about 25 evaluations (`docs/bench_n128.md`).

From white noise:

| Precondition | Evaluations | Final loss | Final / reference | Evaluations to reference | Best loss at evaluation 50 |
|---|---|---|---|---|---|
| none | 115 | 2.38e-6 | 1.00 | 115 | 7.6e-5 |
| none (recompute: rounding only) | 115 | 2.52e-6 | 1.06 | - | 7.6e-5 |
| bands:1 | 125 | 5.8e-3 | 2429 | - | 9.6e-3 |
| bands:0.5 | 120 | 2.9e-5 | 12.4 | - | 4.8e-3 |
| bands:0.25 | 119 | 7.5e-6 | 3.2 | - | 1.4e-4 |
| bands:-0.25 | 122 | 4.7e-6 | 2.0 | - | 1.4e-4 |
| bands:-0.5 | 123 | 1.1e-5 | 4.5 | - | 2.9e-4 |
| bands:-1 | 125 | 5.9e-5 | 24.8 | - | 1.7e-3 |
| power | 124 | 1.2e-4 | 50.0 | - | 6.7e-3 |
| auto | 122 | 1.77e-6 | 0.74 | 113 | 6.4e-5 |
| auto (recompute) | 126 | 1.83e-6 | 0.77 | 116 | 6.4e-5 |

Coarse-to-fine. First 100 iterations at nside 64 (target = mean of the 4
children; 115 evaluations of 0.117 s, about 33 nside-128 evaluations). Then
100 iterations at nside 128 from those maps
(`--init c64_m{m}_synthesis_nested.npy`), with the noise of the missing
scales added (`--init-noise auto`):

| Fine stage | Evaluations | Final loss | Final / reference | Evaluations to reference |
|---|---|---|---|---|
| none | 113 | 1.30e-6 | 0.54 | 91 |
| none (recompute) | 112 | 1.45e-6 | 0.61 | 91 |
| `--freeze-nside 32` | 115 | 1.35e-6 | 0.57 | 98 |
| `--freeze-nside 16` | 119 | 8.9e-7 | 0.37 | 76 |
| auto | 119 | 9.2e-7 | 0.39 | 79 |
| bands:-0.25 | 118 | 8.0e-7 | 0.34 | 73 |
| bands:-0.25 (recompute) | 118 | 7.9e-7 | 0.33 | 73 |

10 members at nside 256, the recipe of section 1.2, warm start from the
nside-256 Gaussian syntheses, 100 iterations (reference: the plain run,
0.933):

| Precondition | Evaluations | Final loss | Final / reference |
|---|---|---|---|
| none | 106 | 0.933 | 1.00 |
| none (recompute) | 109 | 0.914 | 0.98 |
| auto | 110 | 1.339 | 1.43 |
| bands:-0.25 | 108 | 1.048 | 1.12 |

Reading:

- From white noise, every fixed power law is worse than none, and the
  further γ is from 0 the worse it gets, in both directions. torch's
  scalar `H0` already captures what a fixed per-scale scaling can give.
  Favouring the large scales (γ > 0) is the worst: their steps blow the
  line search up. `power` is no better. `auto` ends 25 % lower after 100
  iterations, but it reaches the reference level at the same evaluation,
  which is within the trajectory noise.
- Coarse-to-fine helps: 91 evaluations at nside 128 instead of 115, plus
  ≈33 equivalent evaluations at nside 64. That is no net gain in time to
  the reference level, but a loss 0.54 times lower after the same 100
  fine iterations. On top of it, `bands:-0.25` (73 evaluations, the same in
  both summation orders), `--freeze-nside 16` (76) and `auto` (79) each save
  another 15-20 %. Freezing at nside 32 does not help (98).
- On the 10-member ensemble recipe, a warm start with many terms (ergodic
  weights, level power, kappa, marginal), both preconditioners lose.

The preconditioner is not a lever for the 2500 iterations of the
production recipe. It stays available behind `--precondition`, default
`none`.

## 3. What was not done

- Caching the geometry per block shape (blocks related by the HEALPix
  symmetries share their weights up to rotation) would cut its size by a
  large factor. It would not be bit-identical: the weights would be
  computed at other angles.
- Preconditioners built on the Gaussian decimation `D` / `Dᵀ` instead of the
  Haar groups: `sum_j c_j (Dᵀ)^j D^j` is SPD only for non-decreasing
  factors. Since no Haar factor helped, it was not tried.

## 4. Projection for the nside 2048 × 10 production (A100)

Per evaluation (`benchmarks/fields/a100`, production timers): 56.5 s of GPU
time, of which `geometry_weights` 13.1 s, plus 15.1 s of host L-BFGS. The
geometry cache budget (13-22 GB) holds 6-10 % of the whole geometry. The
rest, ≈200 GB of weights, is recomputed twice per evaluation (forward and
reverse sweeps), which amounts to ≈30 GB/s equivalent.

- Host tier: 2 × 200 GB through PCIe gen4 (≈22 GB/s page-locked) is ≈18 s
  per evaluation, more than the 13.1 s of recomputation. It would also need
  ≈240 GB of page-locked host memory against the job's 128 GB. `auto`'s
  calibration should keep the recomputation (this is to check on the A100;
  `geometry_host_calibration` in the report). Expected gain: none.
  On a GPU with weak fp64 or a fast host link (T4, L4, Grace Hopper), the
  T4 numbers above apply.
- Device cache: the 13.1 s go entirely once the whole geometry fits on the
  devices. With `--gpu-fields recompute`, the fields and decimation rows
  take ≈42 GB of an A100, so the ≈200 GB of geometry need the sky shared by
  5 or more A100s (`mpirun -np 5`, one rank per GPU, the existing
  decomposition). Then per evaluation: 56.5 → 43.4 s of GPU work (−23 %),
  and with the host L-BFGS, 71.6 → 58.5 s (−18 %), plus the multi-GPU
  speed-up of the other phases.
- Iterations: no preconditioner shortens the production recipe. A
  coarse-to-fine start helps the first hundred fine iterations, not the
  2500.

The next lever after the geometry is the host L-BFGS (15.1 s per
evaluation, 21 %), whose vector operations would move to the device.

## 5. Reproduce

Reports, logs and scripts: `benchmarks/geom_precond/` (`t_*`: nside 128
timings; `s512_*`: nside 512; `p_*`: preconditioners; `c_*`: coarse-to-fine;
`e256_*`: 10 members at nside 256; `*.sh`: the runs, in the order
`timing`, `precond_a/b`, `c2f`, `r2_*` … `r8`). Tables:

```bash
python tools/geom_precond/summarize.py benchmarks/geom_precond
```

The initial maps come from `tools/bench_n128/make_inits.py` (PR #8):
`make_inits.py 128 4 init`, `make_inits.py 64 4 init64` and
`make_inits.py 512 4 init512`.
