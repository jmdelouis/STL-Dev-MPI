# Cost of an evaluation of the ensemble objective at nside 1024 and 2048

This note profiles one evaluation of `stl-ensemble run` (P. Campeti's
five-block objective, docs/campeti_merge.md) at nside 1024 with ten
members.  It then sizes the options that would bring nside 2048 within
reach on one A100.  The profile came first; the options are ranked by
gain, exactness and effort.  Sections 3 and 4 measure the exact options
(0, b, a) on a T4 and on an A100, and section 5 projects nside 2048 from
those measurements.

## 1. Profile at nside 1024

### 1.1 A100: timeline of the production run

The production run (job 2570602, one A100-SXM4-80GB, 16 cores of an EPYC
7742, ten members in one batch, `--gpu --gpu-fields recompute`, harmonic
terms on the device) was sampled from an overlapping Slurm step
(`srun --jobid=2570602 --overlap`), without stopping it.  The step
sampled every 100 ms the device utilisation and memory (`nvidia-smi
-lms 100`), and the CPU ticks and running threads of the `stl-ensemble`
process (`/proc/PID/stat`, `/proc/PID/task/*/stat`).  This covered
evaluations 117 to 129 on 31 Oct at 12:05 and 12:16.  Every evaluation
repeats the same pattern (67-68 s):

| Phase (order of `EnsembleObjective::loss_and_grad`) | Seconds | Device | Host cores |
|---|---|---|---|
| L-BFGS step (two-loop recursion, line search updates) | 4 | idle | 10-11 |
| whole maps, physical map (LUT), owned slices, layout of g | 17 | idle | 1 |
| SC of g (forward and reverse) | 10.5 | 96 % | 1 |
| layout of kappa | 11 | idle | 1 |
| SC of kappa (forward and reverse) | 10.5 | 96 % | 1 |
| harmonic terms (C_l, PDF, Minkowski; forward and reverse, ten members) | 14 | 99 % | 1 |
| **total** | **67-68** | 35 s busy | |

For 33 s of the 68 s the device waits, and for 28 of those seconds the
host runs a single thread.  The two 11-s stretches are
`Objective::to_layout` (crates/stl-mpi/src/synthesis.rs).  It places
the owned slice into the level-0 layout with `layout.pos(id)` for every
pixel.  The distributed partition stores level 0 as `Layout::Sparse`,
even on one rank (crates/stl-mpi/src/decomposition.rs), so every pixel
costs a binary search over 12.6 M ids: 126 M searches per SC block.  The
rest of the 17-s stretch is the LUT inverse (`EnsembleObjective::physical`,
one binary search over the LUT knots per pixel, single thread) and copies
of whole maps.

The device memory goes from 48-50 GB (geometry cache of 40 GB included)
up to 57.9 GB at the peaks of the SC sweeps.

### 1.2 T4: the same evaluation split by the objective's timers

On garoupe, rank 0 of `--groups 2` has five members on one T4 (12
threads, `STL_GPU_CACHE_MB=1536`, meas/grf.json of the nside-1024
measurement).  One evaluation takes 177.0 s:

| Timer | Seconds |
|---|---|
| gaussian_sc | 60.7 |
| physical_sc | 53.5 |
| mf_forward + mf_reverse | 21.6 + 23.6 |
| power_pdf_forward + power_pdf_reverse | 5.6 + 4.6 |
| host_physical_map | 5.7 |
| host_chain | 1.7 |

Each transform of the device terms on one T4 (`device_profile` test of
crates/stl-cli/tests/ensemble.rs, best of 3):

| nside (lmax) | analysis | synthesis | derivatives (5 maps) | derivatives_vjp | all terms, one member |
|---|---|---|---|---|---|
| 512 (1535) | 0.149 s | 0.134 s | 0.427 s | 0.474 s | 1.86 s |
| 1024 (3071) | 1.048 s | 0.901 s | 2.601 s | 3.019 s | 11.21 s |

From 512 to 1024 the transforms grow by 6.1-7.0 (lmax^3 gives 8).

### 1.3 Transforms per member and evaluation

| Pass | Transforms |
|---|---|
| forward | analysis of kappa (C_l); analysis of u = (kappa - mean)/sigma (Minkowski); derivatives of a(u) |
| reverse | synthesis of the C_l cotangent; derivatives_vjp; synthesis of the result (analysis_vjp) |
| reverse, when the Minkowski states do not fit | analysis of u and derivatives again |

The states (u and the five derivative maps, 48 bytes per pixel and member)
are kept when they take at most 4 GiB (`MF_STATE_BYTES`).  That covers five
members at 1024 (3 GB) but not ten (6 GB).  The A100 run therefore
recomputes them.  In T4 seconds, one member costs 9.5 s of transforms with
the states kept and 13.2 s with them recomputed.  The A100 spends 1.4 s per
member, 10.6 times faster than the T4, whereas its FP64 peak is 39 times
higher: the Legendre kernels do not reach the A100's FP64 throughput.

## 2. Options

All gains are for one evaluation with ten members on one A100.

**0. Host passes (exact, same bits).**
- Copy the owned slice into the layout row by row: the owned ids are
  contiguous in the sorted layout, so one `pos` per row is enough, as in
  `host_owned_gradient`.
- Run the LUT inverse and the chain on the host threads, keeping the
  per-member means summed in the current order.
- Drop the copies of whole maps on one rank.

Gain at 1024: 68 s -> about 41 s.  Effort: half a day.

**b. Fewer transforms (exact to rounding).**
- a(u) = (a(kappa) - mean a(1)) / sigma, by linearity, with a(1) computed
  once: one analysis less.
- The two syntheses of the reverse become one.  The cotangent through the
  standardisation is linear, so `Y c1 + P(g_u + Y c2)` equals
  `Y(c1 + c2 / sigma) + P(g_u)` minus two scalar corrections.  The
  corrections come from the coefficient-space products of `c2` with
  `Y* 1` and with `Y* u`, both of which are known.
- Keep the states when they fit a device budget.  Otherwise keep a(u)
  (75 MB per member at 1024, 300 MB at 2048) and recompute only the
  derivatives.

Per member that leaves one analysis, one derivatives (two when
recomputed), one derivatives_vjp and one synthesis.  In T4 seconds this is
7.6 s instead of 13.2 s, so 14 s -> about 8.5 s at 1024.  Effort: one day.

**a. Legendre sums as batched GEMMs (exact to rounding).**
- For each m and parity of l + m, build the table P_m[l, ring pair] once
  per stage, with the scaled recursion of the current kernels (values that
  underflow stay 0, as now).
- Share each table among the ten members and among the sums of the stage:
  - A with the a_lm;
  - C with l(l+1) a_lm;
  - B from the two sums of l a_lm and of c_{l+1} a_{l+1,m}, since
    dlambda_l = l cos(theta) lambda_l - c_l lambda_{l-1}.
- This gives 20 to 80 columns per GEMM; run it with cuBLAS DGEMM, loaded
  at run time as cuFFT is now (FP64 tensor cores on the A100).  The adjoint
  is the transposed GEMM; batch the ring FFTs.

Each stage then costs about 2 x 10^11 FP64 operations per sum and member
at 1024: about 4 x 10^12 per evaluation.  Today the recursion is redone
for every member, transform and sum, and runs far from peak.  Estimate:
the SHT part goes from 12 s to 1-1.5 s at 1024, and to about 10 s at
2048.  On a T4, which has no FP64 tensor cores, it is about 4 times
faster.

Memory: the sums for G members take 3 G x 0.8 GB at 2048, so members go
in groups of five (12 GB), and the per-m tables take at most 3 GB.  This
fits the 30 GB left by the SC engine at 2048.  Effort: three to five days
with the tests (agreement with the host to about 1e-13).

**c. Local variant (changes the loss).**  Derivatives by pixel stencils,
Minkowski functionals per level of the pyramid, and band powers from the
power per level, with a transform only on a degraded map for the low l.
This is O(N): about 5 s at 2048 instead of about 10 s with option a.
That gain does not justify changing P. Campeti's loss unless option a
fails.

## 3. Measured on garoupe (T4), branch `claude/harmonic-speed`

Options 0, b and a are implemented.  None of them changes the loss, and
the tests pass: device against host to 1.5e-13 on the blocks and 9e-14
on the gradient, and the transforms to 3e-14.

**Full evaluations.**  nside 1024, ten members on 2 T4 (`--groups 2`,
five members per rank, `--gpu-fields recompute`, 4 threads per rank).
These are 4 evaluations from the default Gaussian fields, and the losses
of the 4 evaluations equal those of the A100 production run to the
printed digits.  Times are seconds per evaluation, rank 0:

| | before (223ace8) | 0 (c45f92d) | 0 + b (7f1323c) | 0 + b + a (bc4b351) |
|---|---|---|---|---|
| one evaluation | 174 | 152.0 | 142.9 | 134.7 |
| SC of g + SC of kappa | 114.3 | 95.2 | 95.9 | 95.6 |
| harmonic terms (5 members) | 55.3 | 55.5 | 45.9 | 37.6 |
| host passes (physical map, chain) | 7.4 | 1.8 | 1.8 | 1.8 |
| peak device memory | | 13.7 GB | 9.7 GB | 10.9 GB |

**Harmonic terms per member and evaluation** (`device_profile`, one T4,
nside 1024, the states kept):

| | seconds | of which Legendre tables, GEMMs, ring FFTs |
|---|---|---|
| former kernels, six transforms | 11.2 | |
| b, kernels (`STL_SHT=kernels`), batches of 5 | 9.27 | FFTs 0.59 |
| b + a, GEMMs, batches of one | 7.53 | 2.00, 2.92, 0.58 |
| b + a, GEMMs, batches of five | 6.86 | 1.02, 3.25, 0.58 |
| the same, ring FFTs of the five derivative maps batched | 6.73 | 0.90, 3.68, 0.33 |

The batched ring FFTs use one cuFFT plan per polar ring length over the
five derivative maps of a member: 2 (nside - 1) + 5 launches instead of
5 nside.  The transforms take 1.5 GB of device memory at 1024 (the
1 GiB table of a block of m, the plans and their work area).  The device
transforms agree with DUCC (fixture `sht_n16`, one map and a batch of
two) to 6e-14, with the GEMMs and with the kernels.

The T4 does little FP64 (0.25 TFLOPS).  Its DGEMMs run at about half that
peak (3.9 x 10^11 operations per member at 1024 take 3.25 s), so sharing
the tables only saves their construction.  On the A100 the DGEMMs use the
FP64 tensor cores (19.5 TFLOPS), where the former kernels reached only a
quarter of their T4 efficiency (section 1.3).  Its gain is therefore
larger than the T4's 26 %: a factor 1.7 to 1.9 (section 4).

**Memory.**  `GpuTerms` sizes its batches from the free device memory
(80 %, or `STL_SHT_GROUP_MB`), at about 14 maps per member, which is
5.6 GB per member at 2048.  It keeps the Minkowski states only when a
batch of one member still fits.  The Legendre tables of a block of m take
at most `STL_SHT_TABLE_MB` (1 GiB).  With the 30 GB that the SC engine
leaves at 2048, that gives batches of 4 members and about 24 GB for the
harmonic terms.

## 4. Measured on one A100 (hpc, job 2592869)

Bench `tools/hpc/harmonic_bench_a100.slurm` at commit 00ea531, on 1 Nov 2026:
one A100-SXM4-80GB, 16 cores, built by `harmonic_bench_build.slurm`.

**Full evaluations at nside 1024.**  Ten members in one batch, with the
options of the production run: `--gpu --gpu-fields recompute`,
`STL_GPU_CACHE_MB=40960`, and its calibration.  These are 4 evaluations
from the default Gaussian fields.  Their losses equal those of the
production run (job 2570602, commit 223ace8) to the 11 printed digits.

| | production (223ace8) | 0 + b + a (00ea531) |
|---|---|---|
| evaluations 2, 3, 4 (wall time between the printed lines) | 65.2, 65.4, 65.7 s | 25.7, 26.2, 25.8 s |
| SC of g + SC of kappa | 21 s | 21.5 s |
| harmonic terms (10 members) | 14 s | 3.8 s |
| host passes (layout, physical map, chain) | 28 s | 1.0 s |
| peak device memory (40 GB geometry cache included) | 57.9 GB | 61.7 GB |

The production column comes from the timeline of section 1.1.  The
00ea531 column averages the objective's timers over the five calls of
the run: four evaluations and the final one, the first of which includes
4 s of warm-up.  The host peak is 19.5 GB.  Early evaluations have at
most three L-BFGS pairs.  With the full history the production run
spends about 2.5 s more per evaluation (65.4 s, then 68 s), so the steady
state of the new code is about 28 s per evaluation: 2.4 times faster.

**Harmonic terms per member and evaluation** (`device_profile`, the whole
device free, best of 2):

| nside | Legendre sums | members per batch | seconds | of which tables, GEMMs, columns, ring FFTs |
|---|---|---|---|---|
| 1024 | former kernels, six transforms (section 1.3) | 10 | 1.4 | |
| 1024 | kernels (`STL_SHT=kernels`), option b | 5 | 0.716 | FFTs 0.118 |
| 1024 | GEMMs, options b + a | 10 | 0.369 | 0.030, 0.065, 0.008, 0.118 |
| 2048 | GEMMs, options b + a | 5 | 2.96 | 0.95, 0.81, 0.06, 0.41 |
| 2048 | GEMMs, options b + a | 10 | 2.60 | 0.76, 0.65, 0.06, 0.41 |

- At 1024, options b and a divide the former 1.4 s by 3.8.  The GEMMs
  alone divide the time of the kernels by 1.9 (1.7 at equal batches,
  judging by the 12 % gap between batches of 5 and 10 at 2048).  Per
  member, the A100 is 18 times faster than the T4 (6.73 s).
- The DGEMMs run at about 6 TFLOPS at 1024 and 5 TFLOPS at 2048
  (3.9 x 10^11 and about 3.1 x 10^12 operations per member).  That is a
  quarter of the FP64 tensor-core peak.  These GEMMs are narrow (at most
  80 columns), so the reads of the tables bound them.
- At 2048, building the tables costs more than the GEMMs (0.76 s against
  0.65 s per member), because they are rebuilt at every stage and for
  every batch.  Generating them inside the GEMM would be the next gain.
- With nothing to share the tables, one map alone is slower with the
  GEMMs than with the kernels: the four transforms of one map take 0.94 s
  against 0.67 s at 1024.  In the ensemble, batches hold 5 to 10 members.
- At 2048 the reverse pass costs 2.3 times the forward pass.  The
  Minkowski states take 2.4 GB per member, which exceeds the 8 GiB of
  `STL_MF_STATE_MB` for 5 members, so the reverse recomputes the
  derivatives.

**Memory.**  The transforms take 1.46 GB at 1024 and 2.47 GB at 2048
(the 1 GiB table of a block of m, the cuFFT plans and their work areas).
Over a 2048 batch, `nvidia-smi` sampled every 30 s reached 45 GB with 10
members and 26 GB with 5.

## 5. Projection to nside 2048 x 10 members on one A100

The SC engine and the host passes are scaled from the measured 1024
evaluation by the number of pixels (x 4).  The harmonic terms are the
measured 2048 values: 26 s for one batch of ten, or about 30 s in
batches of four or five when the SC engine leaves about 30 GB.  The
L-BFGS step takes 4 s at 1024 with the full history, so about 16 s at
2048.

| | SC (g + kappa) | harmonic terms | host + L-BFGS | one evaluation |
|---|---|---|---|---|
| production code (223ace8), extrapolated | 84 s | 112 s | 132 s | 328 s (5.5 min) |
| 0 + b + a | 86 s | 26-30 s (measured) | 20 s | about 135 s (2.2 min) |

With the new code, 2500 iterations (about 2875 evaluations) take about
108 h on one A100, or about 55 h on two with `--groups 2`.  At nside
1024 the same run takes about 22 h instead of 54 h.  The SC engine is
now about 65 % of an evaluation at 2048 and three quarters at 1024.  This projection
uses no full evaluation at 2048, since there is no calibration at 2048
yet.
