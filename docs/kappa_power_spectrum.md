# Kappa power spectrum of gaussianised syntheses

Syntheses of the Takahashi r000 zs1 convergence map follow the recipe of
`USAGE.md` §4.2: gaussianise the target (rank to normal), synthesise the
gaussianised map `g`, map back to kappa by histogram matching.  The
10-member ensemble test at nside 256 (`--ensemble 10 --level-power-weight 10
--scatter-weight 1`, init from the nside-128 members, 200 iterations) gave a
kappa C_l 15-20 % low at l 135-284 and 26 % high at l 599-767, the same for
every member, while every family of the gaussianised statistics was at about
1 sigma per member and the variance of every pyramid level within 4 %.  This
note finds the cause and compares fixes at nside 256.

Summary: the deficit comes from the extreme tail of `g` (the halos), which
the gaussianised statistics barely weigh.  Two terms fix it:

- `--kappa-table KNOTS --kappa-families mean,var,S1,S2 --kappa-weight 0.1`:
  the first-order statistics of `kappa = f(g)` in the loss;
- `--marginal-weight 1000 B`: the pixel histogram of `g` held at the
  target's, so that `f(g)` is the map the histogram matching produces.

With both, the kappa C_l of a 4-member ensemble go from 12.5 % rms (24 % max)
to 6.6 % rms (14 % max) over l = 31-767.  The gaussianised statistics stay at
1 sigma per member.  The evaluation costs 1.8 times as much.  Cheaper ideas
were tested and fail, for reasons given in §4.

With these two terms the kurtosis of the kappa details overshoots at
intermediate scales (3 times the target's at nside 64).  A third term,
`--kurtosis-weight 0.1 B --kurtosis-nside 32-NSIDE`, holds it on every
member at 3 % of the cost, with the statistics unchanged (§7).

The Minkowski functionals of the kappa levels stay outside cosmic variance
with these three terms (V1, V2 maxima of 4-8 cosmic sigma at levels
64-256).  A fourth term, `--minkowski-weight 0.025 B --minkowski-standardise`,
brings V0 and the medians of V1, V2 back at nside 256 at no cost.  Used
from nside 256 on with the native level, it does the same at nside 512 and
brings V1 at 512 back too, without cost to the other statistics.  The
largest V1, V2 deviations stay at 4-6 at levels 64-256, and the kurtosis of
kappa at 512 rises to +4 cosmic sigma (control +1 to +2), as when the term
is added to a converged ensemble (§8).  That excess sits in the few largest
values of g, which the histogram term leaves free.  Holding them
(`--marginal-tail-weight 0.1 B --marginal-tail 30` at nside 512) brings the
kurtosis back to +0.1 cosmic sigma after 600 iterations (control +1.1).
Peaks and C_l are unchanged.  The level variances of g lag the control's by
about 150 iterations.  Ten times stronger, the term stalls the level power
(§8.6).

## 1. Diagnosis: a Gaussian copula at small scales

![diagnosis](figures/kappa_cl_diagnosis.png)

C_l of the synthesis over C_l of the target (mean of the 10 members of the
test run), for `g`, for Hermite polynomials of it, for kappa, and for a
Gaussian random field with exactly the target's C_l of `g` mapped through
the same table (mean of 10 realisations):

| l | g | He2(g) | He3(g) | kappa | GRF -> kappa |
|---|---|---|---|---|---|
| 31-38 | 0.91 | 0.96 | 0.89 | 0.89 | 1.23 |
| 82-105 | 1.10 | 0.95 | 0.87 | 0.92 | 0.90 |
| 135-172 | 1.00 | 0.94 | 0.93 | 0.85 | 0.85 |
| 173-221 | 1.05 | 0.97 | 0.99 | 0.83 | 0.79 |
| 222-284 | 0.81 | 0.93 | 1.01 | 0.80 | 0.83 |
| 285-364 | 0.92 | 1.02 | 1.08 | 0.88 | 0.89 |
| 468-598 | 1.03 | 1.13 | 1.24 | 1.11 | 1.12 |
| 599-767 | 1.07 | 1.16 | 1.26 | 1.26 | 1.19 |

- The deficit appears in the transform.  In `g` the ratios oscillate around
  1 from one octave to the next: the wavelet bands hold the power of each
  octave, not its shape within it.  The low-order Hermite polynomials are
  within about 10 %.
- Above l = 100 the kappa C_l of the synthesis are those of a Gaussian field
  with the right C_l of `g`.  The synthesis has a Gaussian copula at small
  scales.  (At l < 100 it does much better than the Gaussian field, which is
  20-140 % high there.)
- The kappa power at l > 100 comes from the extreme tail of `g`: 25 pixels
  above 4 sigma and 1062 above 3 sigma, out of 786432.  Their arrangement
  (extended, clustered halos) weighs almost nothing in the gaussianised
  statistics.

A single map from white noise (400 iterations, `--level-power-weight 1`,
variant G below) shows the same deficit, so screening was done on one map.

## 2. Single maps

Protocol: `stl-synth --L 4 --weights ergodic --niter 400
--level-power-weight 1` from white noise (seed 11), then `gaussianize.py
match`.  Every variant is judged on its final kappa map:

- C_l: rms and largest deviation of C_l / C_l(target) over the 13 log bands
  from l = 31 to 767;
- S residuals: rms over each family, in ergodic sigma, of the final map's
  statistics.  In gaussianised space, the kappa map is rank-gaussianised
  back, so these are the statistics of what the matching produced, not of
  the synthesis variable.  In kappa space, the target is kappa itself.  Both
  are `stl-synth --niter 0` evaluations;
- variance of the pyramid details of the rank-gaussianised map, levels 8 to
  256 (range over the levels);
- excess kurtosis of the kappa details at levels 32, 64, 128, 256 (target
  50, 100, 340, 513).

| Variant | C_l rms / max | S1-S4, g space | S1-S4, kappa space | var levels, g | kurtosis 32/64/128/256 |
|---|---|---|---|---|---|
| G (current recipe) | 0.152 / 0.256 | 0.46 0.67 3.36 1.30 | 4.96 3.39 3.70 3.67 | 1.00-1.01 | 36 / 82 / 381 / 668 |
| G + marginal | 0.116 / 0.190 | 0.39 0.35 0.60 0.41 | 5.68 2.76 2.98 2.96 | 1.00-1.01 | 45 / 55 / 220 / 957 |
| log(kappa + 3e-4) | 0.106 / 0.285 | 15.6 11.2 18.9 7.6 | 2.63 2.63 2.95 3.52 | 0.89-1.18 | 31 / 115 / 457 / 585 |
| log(kappa + 4e-3) | 0.078 / 0.146 | 64 43 31 35 | 7.0 2.1 1.7 2.1 | 0.65-1.81 | 28 / 105 / 320 / 623 |
| kappa directly | 0.088 / 0.254 | 107 72 34 43 | 15.1 2.7 1.6 1.9 | 0.51-2.19 | 30 / 94 / 265 / 615 |
| G + power per level of kappa | 0.068 / 0.128 | 0.65 0.43 1.39 0.69 | 7.84 2.40 2.61 2.54 | 1.00-1.01 | 41 / 114 / 377 / 796 |
| same + marginal | 0.109 / 0.173 | 0.40 0.36 0.62 0.41 | 6.62 2.39 2.80 2.81 | 1.00-1.01 | 24 / 59 / 206 / 985 |
| G + all families of kappa, weight 1 | 0.139 / 0.299 | 1.61 4.65 2.89 6.18 | 4.76 3.14 2.97 2.51 | 0.99-1.09 | 40 / 159 / 230 / 618 |
| G + all families of kappa, weight 0.1 | 0.126 / 0.249 | 0.80 1.25 2.66 2.35 | 4.52 3.14 3.22 2.73 | 1.00-1.02 | 41 / 124 / 288 / 622 |
| G + S1,S2 of kappa, weight 1 | 0.068 / 0.132 | 4.31 3.05 5.41 6.98 | 1.17 1.35 2.21 3.82 | 0.99-1.08 | 64 / 260 / 215 / 814 |
| G + S1,S2 of kappa, weight 0.1 | 0.066 / 0.139 | 1.89 1.87 2.37 3.81 | 1.63 1.40 2.10 3.14 | 1.00-1.02 | 78 / 167 / 261 / 847 |
| **G + S1,S2 of kappa, weight 0.1 + marginal** | **0.062 / 0.125** | **0.53 0.62 0.79 0.54** | **0.16 0.10 1.60 2.59** | **1.00-1.01** | 57 / 181 / 245 / 704 |

![single maps](figures/kappa_cl_single.png)

![gnomonic views](figures/kappa_cl_gnomonic.png)

Gnomonic views (12 x 12 degrees, two fixed centres, asinh scale): the
syntheses are independent realisations, so only the texture compares.

Cost per evaluation: on one T4, G 1.46 s, G + S1,S2 of kappa 2.65 s (x1.8),
G + all families of kappa 3.13 s (x2.1).  On 32 CPU ranks, G + marginal
5.15 s and G + marginal + S1,S2 of kappa 9.38 s (x1.8).  The marginal term
is one sample sort per member and evaluation.

## 3. Ensembles

Protocol of the test run, with 4 members: `--ensemble 4 --groups 4
--scatter-weight 1 --level-power-weight 4 --niter 200`, init from the first
4 nside-128 members of the production chain (`--init ..._m{m}...`, noise
`auto`).  Weights per member as in production: the level-power and marginal
weights are multiplied by B.

| | test, 10 members | control, 4 members | S1,S2 of kappa | S1,S2 of kappa + marginal |
|---|---|---|---|---|
| C_l kappa, l >= 31: rms / max | 0.133 / 0.260 | 0.125 / 0.241 | 0.110 / 0.173 | **0.066 / 0.138** |
| l 135-172 / 222-284 / 599-767 | 0.85 / 0.80 / 1.26 | 0.84 / 0.80 / 1.24 | 0.89 / 0.85 / 1.09 | 1.02 / 0.86 / 1.00 |
| per member S1 S2 S3 S4 (sigma) | 1.09 0.99 1.08 1.02 | 0.98 0.91 1.00 0.93 | 0.97 0.88 1.02 0.94 | 0.98 0.90 1.03 0.95 |
| scatter / ergodic variance, S1-S4 | 1.20 0.95 1.17 1.10 | 1.08 0.87 1.09 1.05 | 1.08 0.80 1.11 1.05 | 1.08 0.82 1.11 1.05 |
| final maps (after matching), S1-S4 per member, g space | | 1.0-2.1, 1.3-2.8, 3.0-3.4, 1.5-1.9 | | 0.9-1.4, 0.7-1.3, 1.0-1.1, 0.9-1.0 |
| final maps, S1-S4 per member, kappa space | | 5.3-7.2, 2.9-5.8, 3.1-5.9, 3.2-7.2 | | 0.9-1.5, 1.1-2.3, 1.9-2.8, 2.1-4.0 |
| var of the levels in g, 8 ... 256 | 0.96-1.04 | 0.96-1.06 | 0.94-1.07 | 0.92-1.07 |
| kappa S1, S2 of the mean (sigma) | | | 0.08 0.12 | 0.15 0.15 |
| kurtosis of kappa, levels 32 / 64 (target 50 / 100) | 28 / 95 | 30 / 114 | 42 / 124 | 85 / 248 |
| s / eval, ranks | 9.4, 170 | 14.6, 40 | 16.9, 72 | 16.6, 72 |

![ensembles](figures/kappa_cl_ensemble.png)

With `--kappa-weight 0.3` (and the marginal term) the C_l improve a little
(5.6 % rms, 13.1 % max).  The levels of `g` drift, though: 0.75 at the
finest, 1.17 at 128.  0.1 is the setting.

Without the marginal term the kappa term reaches its targets (S1, S2 of kappa
at 0.1 sigma) but the C_l move little.  It gets there by moving values of
`g` off the Gaussian marginal, and the histogram matching undoes that.  In
the single maps the squared 1-D Wasserstein distance to the target's values
grows from 1.3e-4 (G) to 3.4e-3, with pixels down to g = -26.

## 4. What fails, and why

- **Correcting the C_l afterwards.**  Alternating a C_l correction (the
  ensemble-mean transfer function, smoothed in l) with histogram matching:
  8 iterations bring the test ensemble to 2.7 % rms.  The finest level of
  `g` then holds only 72 % of its variance, and the kurtosis of its details
  is 1.5 instead of 0.37.
- **Synthesising a less gaussianised field** (log(kappa + lambda), or kappa
  itself).  The deficit at l 135-284 goes, but the bulk is lost: in
  gaussianised space the final maps are 8 to 107 sigma away, and the level
  variances are off by up to a factor 2.
- **Power per level of kappa = f(g)** (the `--level-power-weight` term on
  kappa).  On its own it is met by isolated spikes far in the extrapolated
  tail of `f`: kurtosis up to x10.  With the marginal held, the level
  variances of kappa are exactly 1, but the C_l barely move (0.84-0.86 at
  l 135-284).  The Haar details have too broad a window in l to set the
  shape of the C_l.
- **All the families of kappa** (the statistics of kappa as a second loss).
  With the ergodic weights the term is a mean over its coefficients, and the
  56 S1, S2 coefficients that carry the power are lost among 5824 S3, S4
  ones.  At weight 0.1 the C_l barely improve.  At weight 1 neither term
  converges in 400 iterations.  Restricting the term to mean, var, S1, S2
  gives them all the weight, at a slightly lower cost.
- **Constant extrapolation of f beyond the table.**  Pixels pushed beyond
  the last knot get no gradient from the kappa term, and they pile up at the
  maximum: 107 pixels above 4 sigma, var of kappa 24 sigma off.  `f` is
  extrapolated linearly.

## 5. Pitfalls of the marginal term

Without it, even the current recipe loses accuracy in the histogram
matching.  The synthesis has S3 at 0.6 sigma, the final map 3.4 sigma: the
synthesised `g` has 67 pixels above 4 sigma against 25, and the matching
moves them.  With `--marginal-weight 1000` (one map) the squared 1-D
Wasserstein distance drops from 1.3e-4 to 4e-8, the final map keeps the
synthesis's residuals, and nothing else changes.  In an ensemble it makes
the members' variances identical (scatter / ergodic variance of var 0.01):
the histogram is held exactly per member.

## 6. Recommended setting

```bash
python tools/gaussianize.py forward T.npy T_gauss.npy table.npz
python tools/gaussianize.py knots table.npz knots.npy              # 256 knots of g -> kappa
stl-synth --target T_gauss.npy --L 4 --weights ergodic \
    --ensemble 10 --groups 10 --scatter-weight 1 --level-power-weight 10 \
    --marginal-weight 10000 \
    --kappa-table knots.npy --kappa-families mean,var,S1,S2 --kappa-weight 0.1 \
    --kurtosis-weight 1 --kurtosis-nside 32-$NSIDE \
    --init prev_m{m}_synthesis_nested.npy --niter 200 --out ens
python tools/gaussianize.py match table.npz ens_m000_synthesis_nested.npy ens_m000_kappa.npy   # per member
```

- `--kappa-weight` is not multiplied by B: like the main term, the kappa term
  is the reduced chi^2 of the ensemble mean.
- `--marginal-weight` is multiplied by B, like `--level-power-weight`: the
  term is `w / B sum_m W2^2(member m, target)`.
- `--kurtosis-weight` is multiplied by B too (0.1 per member): the term is
  `w / B sum_m sum_n (K_n / K_n(target) - 1)^2`.  `--kurtosis-nside 32-NSIDE`
  takes every level from the details of nside 32 to the finest (§7).
- Optional, at every stage from nside 256 on: `--minkowski-weight 0.025 B
  --minkowski-standardise --minkowski-sigma sig.json --minkowski-nside
  64-NSIDE`, with `sig.json` from `stl-minkowski r000.npy r001.npy ...
  --nside 32-NSIDE --standardise --sigma-out sig.json`.  It holds V0 and the
  bulk of V1, V2 of the kappa levels and V1 at the native level (§8.3,
  §8.5).  It costs the tail of kappa at the finest level (kurtosis +4
  cosmic sigma at nside 512, peaks above 3 sigma -0.9), and 15 % per
  evaluation.  Not on a converged ensemble (§8.4).
- Optional, when the kurtosis of kappa at the finest level matters:
  `--marginal-tail-weight X --marginal-tail K` with X = 0.1 B (512 /
  NSIDE)^2 and K = 30 (NSIDE / 512)^2 (tested at 512 only).  It brings that
  kurtosis at 512 from +4 to +0.7 cosmic sigma after 450 iterations, and to
  +0.1 after 600 (control +2.3 and +1.1).  The level variances of g lag the
  control's by about 150 iterations (§8.6).  Ten times stronger, it stalls
  the level power.

Open points:

- The largest remaining deviation is 0.86 at l 222-284, and the octave
  oscillations of the C_l of `g` (0.81 to 1.13) are unchanged.  The kappa
  statistics are dyadic too.
- The kurtosis of the kappa details at levels 32-64 overshoots (85 / 248
  against 50 / 100) without the kurtosis term: §7.
- The V1, V2 of the kappa levels keep maxima of 4-6 cosmic sigma at 64-256
  with or without the Minkowski term; its 3 x 3 stencil does not see the
  pixel scale of each level (§8.2, §8.4, §8.5).
- Without the tail term, the kurtosis of kappa at the finest level is +1
  to +4 cosmic sigma high.  It comes from the few largest values of g.
  Holding them costs about 150 iterations of level-power descent at 512
  (§8.6).  The tail above nside 512 is untested.
- Not yet tested at nside 1024 and 2048.  The 256 knots hold the tail there.
  On r000 the excess kurtosis of the finest details through the knots is
  1325 against 1330 at 1024, and 1834 against 1832 at 2048.

## 7. Kurtosis of the kappa details at intermediate scales

With the setting of §6 the tails of the kappa details are too heavy at
intermediate scales.  `K_n = <d_n^4> / <d_n^2>^2` is the kurtosis of the
details of level n of the 4-children pyramid (as for `--level-power-weight`);
the tables give the excess `K_n - 3`.  The target's excess at the levels 32,
64, 128 and 256 is 50, 100, 340 and 513.

- In the 4-member test of §3, rerun on a T4, the medians over the members
  are at 2.3, 3.0, 0.7 and 1.3 times the target's.
- The nside-512 ensemble (10 members from the nside-256 maps) gives 63, 223,
  284, 479 and 1378 at 32 to 512 (targets 50, 100, 340, 513, 855).
- The nside-1024 test gives 86, 349 and 550 at 32, 64 and 128.

The current recipe without the kappa terms (control of §3) stays near the
target at 64 (83 to 218 over its members, median 114).  The S1, S2 of kappa
with the marginal term are what pushes the level-64 tails up.

Kurtosis is a noisy statistic, dominated by a few halos.  A jackknife over
the 12 base faces gives the target 50 ± 18, 100 ± 27, 340 ± 100 and
513 ± 64 at 32, 64, 128, 256.  Single maps are therefore a poor probe: one
white-noise realisation of the §6 setting sits at 0.8 of the target at 64,
another (§2) at 1.8.  The decisive comparison is the ensemble from the
nside-128 members, where the overshoot is systematic.

Two terms were tested:

- `--kappa-s3-weight X --kappa-s3-nside 64-128` adds the S3 of kappa (every
  `j1 <= j3`, with `j3` at nside 64 or 128: 80 coefficients) as a separate
  reduced chi^2 next to the S1, S2 one.  Without `--kappa-s3-nside`, all
  the S3 are kept.
- `--kurtosis-weight X --kurtosis-nside A-B` adds
  `X / B sum_m sum_n (K_n(f(g_m)) / K_n(target) - 1)^2` over the levels of
  nside A to B, with its exact gradient.  It costs +3 % per evaluation.

Single maps (protocol of §2 with `--marginal-weight 1000`, final kappa maps;
S residuals in sigma):

| Variant | kurtosis / target, 32 64 128 256 | C_l rms / max | S1-S4, g space | S1-S4, kappa space |
|---|---|---|---|---|
| §6 setting | 1.74 0.81 0.99 1.16 | 0.074 / 0.153 | 0.57 0.63 0.81 0.54 | 0.17 0.10 1.71 2.69 |
| + kurtosis 0.1, 64-128 | 1.30 1.01 1.02 1.33 | 0.070 / 0.131 | 0.47 0.55 0.76 0.53 | 0.08 0.10 1.59 2.37 |
| + kurtosis 0.1, 32-256 | 1.01 0.99 1.07 1.04 | 0.079 / 0.151 | 0.66 0.70 0.84 0.57 | 0.09 0.16 1.69 2.45 |
| + kurtosis 1, 64-128 | 1.93 0.97 0.92 1.53 | 0.079 / 0.164 | 0.87 0.85 0.85 0.67 | 0.23 0.24 1.65 2.41 |
| + kurtosis 1, 32-256 | 1.03 1.01 1.06 1.05 | 0.081 / 0.161 | 1.04 1.03 1.01 0.80 | 0.42 0.33 1.78 2.91 |
| + kurtosis 10, 64-128 | 1.92 0.96 0.96 1.23 | 0.096 / 0.193 | 2.17 1.94 1.42 1.19 | 2.02 1.27 2.63 3.66 |
| + S3 of kappa 0.1, 64-128 | 1.17 1.58 0.73 1.48 | 0.070 / 0.137 | 0.99 0.89 0.97 0.80 | 0.21 0.44 1.64 2.36 |
| + S3 of kappa 1, 64-128 | 0.87 1.03 0.88 1.27 | 0.169 / 0.336 | 7.32 6.20 2.78 1.80 | 5.92 3.13 4.33 6.31 |
| + S3 of kappa 0.1, all levels | 0.99 1.09 0.71 1.43 | **0.057 / 0.100** | 0.52 0.58 0.88 0.67 | 0.33 0.20 **0.46 1.72** |

Ensembles (protocol of §3 with the marginal term and the S1, S2 of kappa;
kurtosis weights per member, 0.4 for B = 4):

| | §6 setting | + kurtosis 0.1, 64-128 | **+ kurtosis 0.1, 32-256** | + S3 of kappa 0.1, 64-128 |
|---|---|---|---|---|
| kurtosis 32, members (target 50) | 157 76 28 167 | 67 66 47 51 | **52 50 52 52** | 72 37 103 117 |
| kurtosis 64 (target 100) | 276 362 142 317 | 100 98 97 100 | **103 100 102 101** | 213 151 449 98 |
| kurtosis 128 (target 340) | 122 237 270 229 | 328 286 318 308 | **351 335 329 344** | 271 166 171 160 |
| kurtosis 256 (target 513) | 871 648 706 641 | 700 802 690 702 | **559 594 527 547** | 637 1068 767 960 |
| median / target, 32 64 128 256 | 2.32 2.95 0.68 1.32 | 1.17 0.98 0.92 1.37 | **1.03 1.01 1.00 1.08** | 1.75 1.81 0.50 1.68 |
| C_l kappa, l >= 31: rms / max | 0.068 / 0.142 | 0.059 / 0.131 | 0.065 / 0.137 | 0.067 / 0.131 |
| per member S1 S2 S3 S4 (sigma) | 0.97 0.89 1.03 0.95 | 0.96 0.88 1.03 0.94 | 0.97 0.88 1.04 0.95 | 1.03 0.97 1.06 0.98 |
| scatter / ergodic variance, S1-S4 | 1.04 0.78 1.13 1.05 | 1.05 0.82 1.12 1.04 | 1.05 0.79 1.12 1.04 | 1.00 0.81 1.13 1.07 |
| final maps, S1-S4, g space | 1.04 0.97 1.05 0.95 | 1.15 1.08 1.03 0.95 | 1.02 0.94 1.04 0.95 | 1.07 1.03 1.06 0.98 |
| final maps, S1-S4, kappa space | 1.16 1.50 2.37 2.87 | 1.25 1.75 2.40 2.91 | 1.50 1.84 2.53 3.01 | 1.44 1.48 2.15 2.46 |
| var of the levels in g, 8 ... 256 | 0.87-1.09 | 0.89-1.08 | 0.85-1.09 | 0.68-1.27 |
| peaks above 8 sigma at 128 / 64, over target | 0.86 / 0.89 | 0.84 / 0.96 | 0.70 / 0.95 | 1.10 / 0.96 |
| s / eval (4 members, one T4) | 5.70 | 5.80 | 5.87 | 6.41 |

Peaks are the local maxima of the kappa map at the levels 256, 128 and 64
of the pyramid, counted above nu sigma (sigma of the target's map at that
level, nu = 2 to 8).  At 256 every variant is within 7 % of the target.
At 128 there are 15-20 % too many low peaks (nu 2 to 3), at 64 too few
(nu 4 to 6: 0.81-0.93).  The kurtosis term leaves them, except fewer
high peaks at 128 with 32-256.

![kurtosis per level](figures/kappa_kurtosis.png)

What works, and what does not:

- **The kurtosis term holds exactly the levels it is given, and only
  those.**  Restricted to 64-128 it fixes both levels on every member and
  pulls 32 down with them in the ensemble.  256 stays 1.4 times too high.
  In single maps it moves the excess to 32 and 256, more so at weight 1
  (1.9 and 1.5).  On 32-256 every level ends within 8 % of the target.
- **Weight 0.1 per member is the scale.**  At 1 the statistics of g rise to
  about 1 sigma in single maps, at 10 to 2 sigma.  At 0.1 they do not move
  in the ensemble.  Only the per-member S1, S2 of kappa of the final maps
  rise a little (1.16 / 1.50 to 1.50 / 1.84 sigma).
- **The targeted S3 of kappa does not hold the kurtosis.**  It is met
  (0.16 sigma at weight 0.1), but the level-64 kurtosis of that realisation
  goes to 1.6 times the target's.  At weight 1 it is met by distorting the
  bulk: S1, S2 of g at 7 sigma.  In the ensemble (weight 0.1) the
  kurtosis stays 1.8 times too high at 32 and 64 and halves at 128.  The
  levels of g drift (0.68 at the finest, 1.27 at 128), and there are 27 %
  too many peaks at 128.  S3 is a mean over the sky of the coupling between
  scales, which a few halos barely move.  The kurtosis is set by those
  halos.
- **The S3 of kappa at every level** (weight 0.1, separate chi^2) is the
  best single map for the C_l (5.7 % rms) and for the S3, S4 of kappa (0.46,
  1.72 sigma), without the kurtosis.  Not tried in an ensemble yet.

Recommended: add `--kurtosis-weight 0.1 B --kurtosis-nside 32-NSIDE` to §6
(every level from the details of nside 32 to the finest; levels of a range
beyond the map's nside are ignored).

Caveats:

- Each member is held at the target's own realisation of a noisy statistic
  (± 27 at 64), so the members keep no scatter of their kurtosis (32-256:
  50-52 at 32 for 4 members).  A term on the mean over the members would
  keep it, if that matters.
- Tested at nside 256 only.  At 512 the finest level is 1.6 times too high
  as well (1378 against 855), which 32-NSIDE covers.

## 8. Minkowski functionals per level

The paper's measure (`text/scripts/mf_levels.py` on branch
`claude/distracted-bohr-952341`) takes kappa at each level of the
4-children pyramid, standardises it, and computes V0, V1, V2 at
nu = -2 ... 5 with harmonic derivatives.  Deviations are in units of the
std of the 10 Takahashi realizations r000-r009.  r001-r009 against r000
give a median |z| of 0.9-2.0 and a maximum of 3.5-3.8.  The nside-512
ensemble with the setting of §6 and §7 (`n512_garoupe_kurt_c`) is outside
that range for V1 and V2: maxima of 4.4-6.4 at 64-256, and systematic
offsets at 128 and 256 (rms over nu of the members' median z 2.0-2.5,
against 1.1-1.3).  Thresholds where the std of the realizations is below
5 % of its maximum are left out; an earlier version of the script kept
them and gave maxima of 12-18 there.

### 8.1 The term

`--minkowski-weight X` adds

    X / B sum_m mean_{n, k, i} ((V_k,i(m) - V_k,i(target)) / sigma_k,i)^2

over the members m, the levels n of `--minkowski-nside` (default 64 to
nside / 2), the functionals k and the thresholds i.  It acts on
`kappa = f(g)` with `--kappa-table`, like the kurtosis term.  The
estimator (`crates/stl-cli/src/minkowski.rs`):

- the level map is the mean of the 4^n children (as in `mf_levels.py`);
  with `--minkowski-standardise` it is standardised by its own mean and
  std, so that the thresholds are nu_i for every map, and the gradient goes
  through the mean and the std;
- V0, V1, V2 follow Schmalzing & Gorski (1998), with the step replaced by
  a logistic function of width h = 0.2 (in units of the level's std) and
  `delta(x - nu)` by its derivative;
- the gradient and Hessian are those of the least-squares quadratic through
  the cell and its 8 (or 7) neighbours.  The fit is done in gnomonic
  coordinates of the tangent plane, where the Christoffel symbols vanish,
  so V1 and V2 do not depend on the frame.  `|grad x|^2 + eps` in V2 has
  eps = 0.01 times the target's mean squared gradient;
- sigma is the std over r000-r009 of the same estimator.  It is written by
  `stl-minkowski r000.npy r001.npy ... --nside 32-NSIDE --standardise
  --sigma-out sigma.json` and read with `--minkowski-sigma sigma.json`.
  The file fixes the thresholds, h, eps and the standardisation.  As in
  `mf_levels.py`, thresholds where sigma is below 5 % of its maximum are
  left out.

The ranks of a group gather each level (6 MB at nside 256 for one map) and
work on their own cells plus one ring of neighbours.  The (member, level)
pairs run on host threads (`STL_MINKOWSKI_THREADS`).  The work is on the
CPU, also in GPU runs.

Tests: exact caps of a linear field (V0 = (1 - t) / 2,
V1 = sqrt(1 - t^2) / 8, V2 = t / (4 pi), convolved with the logistic kernel,
within 1e-3); gradient against finite differences on 2 ranks for both
settings; equality of 1 and 4 ranks to 1e-12.

### 8.2 The estimator against `mf_levels.py`

On r000 degraded to 64, 128 and 256 (`stl-minkowski --json` against
`tools/minkowski.py`):

- V0 agrees: the ratio is 1.000 for hard thresholds (h = 0.02), 0.997-0.999
  for h = 0.2.
- V1 and V2 do not: the stencil gives 0.33-0.49 times the harmonic values.
  The least-squares quadratic over 3 x 3 cells smooths the pixel scale,
  which carries most of the gradient of a level map.  The harmonic
  derivatives of `alm2map_der1` keep everything up to l = 3 nside - 1.
- The z of r001-r009 against r000 correlate with `mf_levels.py`'s at 0.83 to
  0.92 for V0 with `--minkowski-standardise`, and 0.15 to 0.56 for V1, V2
  (largest at 256).  With fixed thresholds in kappa units (mean + nu std
  of the target) the correlation is 0.3 for V0 and about 0 for V1, V2.
  There V0 is dominated by the 13 % cosmic scatter of the level variance,
  which `--level-power-weight` already holds.

So the term uses `--minkowski-standardise`.  On the members of
`n512_garoupe_kurt_c` this estimator finds them outside cosmic variance
too: max |z| 7.5 for V1 and 23 for V2 at 256, and 6.5 / 11 at 128
(`mf_levels.py`: 5.4 / 6.4 and 6.1 / 5.8).  With the
same estimator, r001-r009 against r000 give a median |z| of 1.0-2.3 and a
maximum below 3.9.

### 8.3 Ensembles at nside 256

Protocol of §7: 4 members from the nside-128 maps, 200 iterations, the §6
setting with `--kurtosis-weight 0.4 --kurtosis-nside 32-256`.  The control
reproduces the §7 column bit for bit.  The term is added with
`--minkowski-sigma sig256s.json --minkowski-standardise` on the default
levels 64 and 128; "+ 256" adds the native level (`--minkowski-nside
64-256`).  Weights are given per member (`--minkowski-weight` = weight x B).
The judge is `mf_levels.py` (harmonic derivatives), median / max |z| over
members x thresholds:

| level, functional | r001-r009 | control | 0.025 | 0.025, + 256 | 0.1 | 1 |
|---|---|---|---|---|---|---|
| 64 V0 | 0.91 / 2.93 | 0.99 / 2.61 | 0.39 / 2.24 | 0.43 / 2.07 | 0.36 / 2.00 | 0.46 / 2.06 |
| 64 V1 | 1.05 / 3.45 | 1.32 / 4.99 | 0.99 / 5.22 | 1.01 / 5.27 | 1.04 / 3.84 | 1.06 / 4.07 |
| 64 V2 | 1.06 / 3.52 | 1.04 / 4.72 | 1.05 / 4.00 | 0.92 / 3.72 | 0.98 / 4.08 | 0.91 / 6.01 |
| 128 V0 | 1.49 / 3.61 | 1.92 / 4.14 | 0.50 / 2.51 | 0.64 / 2.39 | 0.79 / 2.11 | 0.98 / 2.77 |
| 128 V1 | 1.14 / 3.63 | 1.99 / **8.41** | 1.07 / 3.65 | 1.09 / 3.66 | 0.98 / 3.62 | 1.36 / 4.61 |
| 128 V2 | 0.94 / 3.55 | 1.58 / 5.74 | 0.93 / 4.58 | 1.17 / 4.10 | 1.07 / 4.96 | 1.53 / 4.83 |
| 256 V0 | 1.93 / 3.63 | 0.90 / 1.83 | 0.56 / 1.29 | 0.22 / 0.71 | 0.67 / 1.31 | 1.52 / 2.70 |
| 256 V1 | 1.20 / 3.76 | 1.23 / 3.92 | 0.73 / 3.57 | 0.80 / 3.77 | 0.93 / 3.19 | 1.78 / 4.94 |
| 256 V2 | 1.10 / 3.69 | 1.35 / **6.91** | 0.89 / 5.12 | 0.81 / 4.44 | 1.00 / 5.48 | 1.76 / 9.56 |

The other statistics (as in §7; cosmic sigma from `cv10.py`):

| | control | 0.025 | 0.025, + 256 | 0.1 | 1 |
|---|---|---|---|---|---|
| per member S1 S2 S3 S4 (sigma) | 0.97 0.88 1.04 0.95 | 0.97 0.88 1.04 0.95 | 0.96 0.87 1.04 0.95 | 0.97 0.87 1.04 0.96 | 1.74 1.54 1.11 1.01 |
| final maps, S1-S4, g space | 1.02 0.94 1.04 0.95 | 0.96 0.85 1.04 0.95 | 1.02 0.95 1.04 0.95 | 0.96 0.86 1.04 0.96 | 2.49 2.30 1.12 1.10 |
| final maps, S1-S4, kappa space | 1.50 1.84 2.53 3.01 | 1.07 1.26 2.15 2.70 | 1.04 1.24 2.13 2.67 | 1.15 1.28 2.16 2.63 | 3.06 1.69 2.40 2.57 |
| C_l kappa, l >= 31: rms / max | 0.065 / 0.137 | 0.064 / 0.134 | 0.060 / 0.128 | 0.066 / 0.135 | 0.095 / 0.170 |
| kurtosis / target, 32 64 128 256 | 1.03 1.01 1.00 1.08 | 1.02 1.01 0.99 1.13 | 1.03 1.02 1.06 1.08 | 1.01 1.03 1.05 1.17 | 1.04 1.05 1.13 1.21 |
| var of the levels in g against r000, cosmic sigma, 64 / 128 / 256 | +5.7 / +3.2 / -6.3 | +5.6 / +3.0 / -6.2 | +5.6 / +3.1 / -6.3 | +6.3 / +5.2 / -7.7 | +8.9 / +18.3 / -13.2 |
| peaks at 128 over target, nu 2 / 8 | 1.20 / 0.70 | 1.06 / 0.86 | 1.06 / 0.87 | 1.06 / 0.81 | - |
| s / eval (4 members, one T4) | 5.85 | 6.43 | 8.08 | 6.90 | 6.97 |

The s / eval with the term are upper bounds: those runs shared the host
cores with other runs, and the runs at 0.025, 0.1 and 1 used the
single-threaded term.

- **The term is met on its levels.**  With its own estimator the members
  at 0.025 are within |z| 1.9 at 64 and 128 (V0 at 128: 3.1), against 6.1
  for V1 and 12.8 for V2 at 128 in the control.
- **In the harmonic measure, V0 and the medians go back to cosmic
  variance.**  V0 is below the reference everywhere.  The medians of V1, V2
  go from 1.3-2.0 to 0.7-1.2, the reference's.  The worst maxima go from
  8.4 (V1, 128) and 6.9 (V2, 256) to 3.7 and 4.4-5.1.
- **Some maxima of V1, V2 stay at 4-5**, above the reference's 3.5-3.8,
  on the low-threshold side.  The stencil derivatives see only part of
  what the harmonic ones see (§8.2), so the term cannot hold those.
- **The native level helps at 256** (V0 0.22 / 0.71, V2 max 4.4 instead of
  5.1), for 25 % more time.  Without it, the soft V2 at 256 rises to a
  maximum of 9 (control 5.3), while the harmonic one drops from 6.9 to
  5.1.
- **The kappa statistics improve too**: S1, S2 of kappa of the final maps
  from 1.50 / 1.84 to 1.07 / 1.26 sigma per member, and the excess of low
  peaks at 128 (1.20) drops to 1.06.  Nothing else moves at 0.025.
- **Weight 0.1 per member starts to pull the levels of g** (+5.2 cosmic
  sigma at 128 instead of +3.2), and 1 per member degrades every
  statistic.  The deviations of the levels of g from r000 (+6, -6 cosmic
  sigma at 64 and 256) are those of the control: the level variances of a
  gaussianised map hardly vary between realizations.

### 8.4 Continuation at nside 512

`n512_garoupe_kurt_c` (10 members, the setting of §6 and §7, B = 10) is
continued by 150 iterations on two T4, without the term and with
`--minkowski-sigma sig512s.json` on the default levels 64-256, at 0.1 and
0.025 per member (`--minkowski-weight 1` and `0.25`).  `mf_levels.py`,
median / max |z| / rms over nu of the members' median z:

| level, functional | r001-r009 | start | +150 it | +150 it, 0.1 | +150 it, 0.025 |
|---|---|---|---|---|---|
| 64 V0 | 0.91 / 2.93 / 0.90 | 0.81 / 3.07 / 1.11 | 0.85 / 2.94 / 1.10 | 0.41 / 3.19 / 0.80 | 0.49 / 3.02 / 0.86 |
| 64 V1 | 1.05 / 3.45 / 1.09 | 1.36 / 5.20 / 1.55 | 1.27 / 5.26 / 1.53 | 1.17 / 5.18 / 1.38 | 1.28 / 4.99 / 1.42 |
| 64 V2 | 1.06 / 3.52 / 1.11 | 1.03 / 4.41 / 1.28 | 0.93 / 4.56 / 1.31 | 1.06 / 5.29 / 1.36 | 1.02 / 4.24 / 1.29 |
| 128 V0 | 1.49 / 3.61 / 1.63 | 1.52 / 3.32 / 1.83 | 1.67 / 3.53 / 1.97 | 0.68 / 2.78 / 0.90 | 0.74 / 2.90 / 1.12 |
| 128 V1 | 1.14 / 3.63 / 1.17 | 1.96 / 6.11 / 2.39 | 1.88 / 6.16 / 2.29 | 1.16 / 5.07 / 1.58 | 1.32 / 5.43 / 1.72 |
| 128 V2 | 0.94 / 3.55 / 1.11 | 1.70 / 5.75 / 2.03 | 1.73 / 5.41 / 1.97 | 1.32 / 5.22 / 1.57 | 1.26 / 5.39 / 1.63 |
| 256 V0 | 1.93 / 3.63 / 2.00 | 2.78 / 3.62 / 2.69 | 3.08 / 3.91 / 2.90 | 0.65 / 2.02 / 1.00 | 1.23 / 2.34 / 1.33 |
| 256 V1 | 1.20 / 3.76 / 1.31 | 2.01 / 5.40 / 2.52 | 1.55 / 5.29 / 2.21 | 1.91 / 4.88 / 1.97 | 1.92 / 4.93 / 1.91 |
| 256 V2 | 1.10 / 3.69 / 1.18 | 2.38 / 6.37 / 2.51 | 2.13 / 5.38 / 2.31 | 1.85 / 7.18 / 2.22 | 1.76 / 6.74 / 2.10 |
| 512 V1 | 1.15 / 3.70 / 1.18 | 1.53 / 3.45 / 1.60 | 1.33 / 3.08 / 1.41 | 0.60 / 2.58 / 0.74 | 0.72 / 2.02 / 0.85 |
| 512 V2 | 0.94 / 3.45 / 0.96 | 1.61 / 7.48 / 2.24 | 1.52 / 6.70 / 2.25 | 0.79 / 5.33 / 1.83 | 0.76 / 5.13 / 1.81 |

V0 at 512 is fixed by the histogram matching (median |z| below 0.5 for
every run).  The other statistics: the report of `stl-synth` (ensemble
mean, in ergodic sigma), `cv10.py` (members against r000, in cosmic sigma
of the realizations, median over the members) and the term's own
estimator:

| | start | +150 it | +150 it, 0.1 | +150 it, 0.025 |
|---|---|---|---|---|
| loss without the Minkowski term | 1.718 | 1.157 | 1.273 | 1.256 |
| mean, var of g | 2.54, 12.97 | 0.73, 1.01 | 2.94, 6.00 | 2.38, 6.03 |
| S1 S2 S3 S4, rms | 3.22 3.19 1.39 0.80 | 2.08 2.06 1.15 0.71 | 2.54 2.72 1.13 0.69 | 2.50 2.63 1.11 0.68 |
| kappa term; mean of kappa | 0.0195; 3.12 | 0.0029; 0.43 | 0.0107; 1.97 | 0.0058; 1.30 |
| marginal term | 0.040 | 0.0067 | 0.0134 | 0.0120 |
| level power term; finest level variance / target | 0.769; 0.78 | 0.496; 0.82 | 0.611; 0.80 | 0.610; 0.80 |
| var of the levels of g, 128 / 256 / 512 (cv10) | +6.0 / +4.8 / -7.8 | +4.7 / +2.6 / -6.4 | +5.2 / +3.8 / -7.1 | +5.2 / +3.7 / -7.1 |
| var of kappa, 256 / 512 (cv10) | -1.0 / +0.4 | -1.3 / +0.8 | +0.3 / -1.3 | +0.2 / -1.0 |
| kurtosis of kappa at 512 (cv10) | +2.3 | +1.1 | +3.8 | +4.0 |
| peaks above 3 / 5 / 10 sigma (cv10) | +0.1 / +0.1 / -1.1 | -0.1 / 0.0 / -0.9 | -1.2 / -0.7 / -1.2 | -0.9 / -1.0 / -1.2 |
| C_l, l >= 31: median / max abs z (cv10) | 0.3 / 0.6 | 0.3 / 0.6 | 0.2 / 0.6 | 0.2 / 0.5 |
| term's estimator, max abs z of V1 / V2 at 256 | 7.5 / 22.8 | 6.6 / 19.8 | 1.6 / 0.5 | 2.4 / 0.8 |
| s / eval (2 T4) | 41.3 | 40.8 | 43.2 | 42.8 |

- **The term is met on its levels at both weights.**  Its own estimator
  gives a maximum |z| of 1.7 (0.1) and 2.4 (0.025) at 64-256, against 4.0
  to 19.8 for the control.
- **In the harmonic measure V0 goes back to cosmic variance** at 64-256
  (rms of the members' median z 0.8-1.3, control 1.1-2.9).  The systematic
  offsets of V1, V2 drop at 128 (rms 2.3 / 2.0 to 1.6-1.7), and those of V1
  at 512, a level the term does not include (1.41 to 0.74-0.85).
- **V1, V2 at 256 and the maxima do not move**: rms 1.9-2.2 at 256
  (control 2.2-2.3), maxima 4.2-7.2 at 64-256 (control 4.6-6.2, reference
  3.5-3.8).  The term's estimator is within 2.4 there.  What is left is at
  the pixel scale of each level, which the 3 x 3 stencil smooths (§8.2).
- **The tail of kappa pays for it.**  The kurtosis of kappa at 512 goes from
  +1.1 to +3.8-4.0 cosmic sigma (start +2.3) and the peaks above 3 sigma
  from -0.1 to between -0.9 and -1.2.  In the report the mean and variance of g stop
  at 2.4-2.9 and 6.0 ergodic sigma (control 0.7 and 1.0, start 2.5 and 13),
  S1, S2 at 2.5-2.7 (control 2.1) and the mean of kappa at 1.3-2.0 (0.4).
  In cosmic sigma the level variances of g move by 1.2 at most, and the C_l
  do not move.
- **The weight does not set the cost.**  0.025 per member costs the same as
  0.1.  On a converged ensemble the term starts at 1.6 times the rest of
  the loss at 0.025 per member (6.3 times at 0.1) and ends 40 (100) times
  lower: LBFGS spends the 150 iterations on it in both cases.

So, added to a converged nside-512 ensemble, the term trades the tail of
kappa for V0 and the bulk of V1, V2, and leaves the largest V1, V2
deviations where they were.  It is not part of the §6 command.  §8.5 runs
the term from the start of the ensemble (members initialised from nside-256
members synthesised with it), with the native level at 512
(`--minkowski-nside 64-512`), which helped V2 at 256 in §8.3.  Untested:
derivatives that keep the pixel scale (a wider stencil or harmonic
derivatives), so that the term sees what the judge sees.

### 8.5 The term from nside 256 on

`n512_garoupe_kurt_c` comes from three stages: `n256_1961965` (10 members,
200 iterations from the nside-128 members, `--level-power-weight 10
--scatter-weight 1` only), `n512` (300 iterations, the setting of §6 and §7)
and `n512c` (150 more).  The chain is redone with the term at 0.025 per
member at every stage, native level included, from the same nside-128
members and the same seed, on two T4:

- 256: `--kappa-table knots256.npy --kappa-weight 0 --minkowski-weight 0.25
  --minkowski-nside 64-256 --minkowski-sigma sig256s.json`.  The table only
  gives the term `kappa = f(g)`; there are no kappa statistics, no marginal
  and no kurtosis term at this stage, as in the control;
- 512: `--minkowski-weight 0.25 --minkowski-nside 64-512 --minkowski-sigma
  sig512s.json` added to both runs.

At 256, against `n256_1961965` (`mf_levels.py`, median / max |z| / rms over
nu of the members' median z):

| level, functional | r001-r009 | `n256_1961965` | with the term |
|---|---|---|---|
| 64 V0 | 0.91 / 2.93 / 0.90 | 1.60 / 3.34 / 1.68 | 0.34 / 1.64 / 0.47 |
| 64 V1 | 1.05 / 3.45 / 1.09 | 1.26 / 5.45 / 1.44 | 0.90 / 5.64 / 1.09 |
| 64 V2 | 1.06 / 3.52 / 1.11 | 1.03 / 4.57 / 1.20 | 0.77 / 3.72 / 0.93 |
| 128 V0 | 1.49 / 3.61 / 1.63 | 2.00 / 6.17 / 2.12 | 1.26 / 2.79 / 1.24 |
| 128 V1 | 1.14 / 3.63 / 1.17 | 1.49 / 7.01 / 1.79 | 0.86 / 4.36 / 0.99 |
| 128 V2 | 0.94 / 3.55 / 1.11 | 1.16 / 5.10 / 1.41 | 0.90 / 4.31 / 1.11 |
| 256 V0 | 1.93 / 3.63 / 2.00 | 2.20 / 4.80 / 2.39 | 1.35 / 2.37 / 1.37 |
| 256 V1 | 1.20 / 3.76 / 1.31 | 2.03 / 9.69 / 2.52 | 2.47 / 5.58 / 2.76 |
| 256 V2 | 1.10 / 3.69 / 1.18 | 1.03 / 15.01 / 3.29 | 1.34 / 10.97 / 2.69 |

| | `n256_1961965` | with the term |
|---|---|---|
| mean, var of g; S1 S2 S3 S4, rms (ergodic sigma) | 0.06, 7.89; 1.06 1.15 1.12 0.70 | 0.36, 7.78; 1.09 1.19 1.13 0.71 |
| term's estimator, max abs z of V1 / V2 at 64 / 128 / 256 | 4.1 / 4.8, 6.9 / 8.4, 6.1 / 9.7 | 1.5 / 1.8, 2.7 / 2.9, 3.8 / 3.6 |
| kurtosis of kappa at 64 / 128 / 256 (cv10) | -0.2 / +0.2 / +1.8 | +4.4 / +10.8 / -1.4 |
| peaks above 3 / 5 / 10 sigma (cv10) | +0.2 / +0.3 / +0.4 | +0.1 / +1.0 / +0.5 |
| C_l, l >= 31: median / max abs z (cv10) | 0.7 / 2.3 | 0.7 / 2.0 |
| s / eval | 9.4 (160 CPU ranks) | 6.5 (2 T4) |

- **V0 goes back to cosmic variance at every level, and the medians of V1,
  V2 at 64 and 128 too**.  Their maxima drop to 3.7-5.6 (control 4.6-7.0,
  reference 3.5-3.6).  The term's own estimator is within 3.8 everywhere.
- **At the native level V1, V2 stay out**: maxima 5.6 and 11 (control 9.7
  and 15), and the median of V1 rises from 2.0 to 2.5.  The pixel scale of
  the native level is what the stencil smooths most (§8.2).
- **Without a kurtosis term the term fattens the tails at 64 and 128**:
  kurtosis of kappa +4.4 and +10.8 cosmic sigma, against -0.2 and +0.2.  In
  §8.3 the kurtosis term was there and nothing moved.  The 512 stage has
  it.

At 512, after the 300 + 150 iterations, against the control at the same
count (§8.4's start), the control 150 iterations later (§8.4's +150 it) and
§8.4's continuation at 0.025 per member (`mf_levels.py` as above; V0 at 512
is fixed by the histogram matching):

| level, functional | r001-r009 | control, 450 it | control, 600 it | §8.4, 0.025 | from 256, 450 it |
|---|---|---|---|---|---|
| 64 V0 | 0.91 / 2.93 / 0.90 | 0.81 / 3.07 / 1.11 | 0.85 / 2.94 / 1.10 | 0.49 / 3.02 / 0.86 | 0.43 / 3.01 / 0.85 |
| 64 V1 | 1.05 / 3.45 / 1.09 | 1.36 / 5.20 / 1.55 | 1.27 / 5.26 / 1.53 | 1.28 / 4.99 / 1.42 | 1.27 / 4.56 / 1.58 |
| 64 V2 | 1.06 / 3.52 / 1.11 | 1.03 / 4.41 / 1.28 | 0.93 / 4.56 / 1.31 | 1.02 / 4.24 / 1.29 | 1.06 / 5.63 / 1.32 |
| 128 V0 | 1.49 / 3.61 / 1.63 | 1.52 / 3.32 / 1.83 | 1.67 / 3.53 / 1.97 | 0.74 / 2.90 / 1.12 | 0.85 / 2.27 / 1.01 |
| 128 V1 | 1.14 / 3.63 / 1.17 | 1.96 / 6.11 / 2.39 | 1.88 / 6.16 / 2.29 | 1.32 / 5.43 / 1.72 | 1.28 / 4.85 / 1.61 |
| 128 V2 | 0.94 / 3.55 / 1.11 | 1.70 / 5.75 / 2.03 | 1.73 / 5.41 / 1.97 | 1.26 / 5.39 / 1.63 | 1.39 / 5.66 / 1.61 |
| 256 V0 | 1.93 / 3.63 / 2.00 | 2.78 / 3.62 / 2.69 | 3.08 / 3.91 / 2.90 | 1.23 / 2.34 / 1.33 | 0.72 / 1.76 / 0.90 |
| 256 V1 | 1.20 / 3.76 / 1.31 | 2.01 / 5.40 / 2.52 | 1.55 / 5.29 / 2.21 | 1.92 / 4.93 / 1.91 | 1.70 / 4.24 / 1.71 |
| 256 V2 | 1.10 / 3.69 / 1.18 | 2.38 / 6.37 / 2.51 | 2.13 / 5.38 / 2.31 | 1.76 / 6.74 / 2.10 | 1.73 / 5.75 / 2.02 |
| 512 V1 | 1.15 / 3.70 / 1.18 | 1.53 / 3.45 / 1.60 | 1.33 / 3.08 / 1.41 | 0.72 / 2.02 / 0.85 | 0.86 / 2.31 / 0.97 |
| 512 V2 | 0.94 / 3.45 / 0.96 | 1.61 / 7.48 / 2.24 | 1.52 / 6.70 / 2.25 | 0.76 / 5.13 / 1.81 | 0.79 / 4.81 / 1.67 |

| | control, 450 it | control, 600 it | §8.4, 0.025 | from 256, 450 it |
|---|---|---|---|---|
| loss without the Minkowski term | 1.718 | 1.157 | 1.256 | 1.540 |
| mean, var of g | 2.54, 12.97 | 0.73, 1.01 | 2.38, 6.03 | 2.93, 11.15 |
| S1 S2 S3 S4, rms | 3.22 3.19 1.39 0.80 | 2.08 2.06 1.15 0.71 | 2.50 2.63 1.11 0.68 | 3.28 3.26 1.28 0.76 |
| kappa term; mean of kappa | 0.0195; 3.12 | 0.0029; 0.43 | 0.0058; 1.30 | 0.0134; 2.58 |
| marginal term | 0.040 | 0.0067 | 0.0120 | 0.027 |
| level power term; finest level variance / target | 0.769; 0.78 | 0.496; 0.82 | 0.610; 0.80 | 0.691; 0.79 |
| var of the levels of g, 128 / 256 / 512 (cv10) | +6.0 / +4.8 / -7.8 | +4.7 / +2.6 / -6.4 | +5.2 / +3.7 / -7.1 | +5.7 / +4.1 / -7.4 |
| var of kappa, 256 / 512 (cv10) | -1.0 / +0.4 | -1.3 / +0.8 | +0.2 / -1.0 | +0.3 / -1.2 |
| kurtosis of kappa at 64 / 128 / 512 (cv10) | +0.0 / +0.1 / +2.3 | -0.0 / +0.1 / +1.1 | +0.1 / +0.3 / +4.0 | +0.0 / +0.2 / +4.0 |
| peaks above 3 / 5 / 10 sigma (cv10) | +0.1 / +0.1 / -1.1 | -0.1 / 0.0 / -0.9 | -0.9 / -1.0 / -1.2 | -0.9 / -0.5 / -0.6 |
| C_l, l >= 31: median / max abs z (cv10) | 0.3 / 0.6 | 0.3 / 0.6 | 0.2 / 0.5 | 0.2 / 0.6 |
| term's estimator, max abs z of V1 / V2 at 256; at 512 | 7.5 / 22.8; 3.5 / 10.5 | 6.6 / 19.8 | 2.4 / 0.8 | 1.5 / 1.2; 1.2 / 1.1 |
| s / eval (2 T4) | 41.3 | 40.8 | 42.8 | 47.3 |

- **The native level comes back.**  V1 at 512 is within cosmic variance
  (max 2.3) and the maximum of V2 drops to 4.8 (control 7.5 at 450 it, 6.7
  at 600 it; §8.4 5.1).  V0 is within cosmic variance at every level, with
  smaller offsets than in §8.4 at 256 (rms 0.9 against 1.3, control 2.7-2.9).
- **The other statistics do not pay for it.**  The loss without the term is
  1.54 against 1.72 for the control at the same count (in §8.4 the term
  raised it from 1.16 to 1.26).  S1-S4, the level variances of g and the
  C_l are those of the control.  Started at 256, the term never has to undo
  a converged state.
- **V1, V2 at 128 and 256 move closer, their maxima stay**: rms of the
  members' median z 1.6-2.0 (control 2.0-2.5 at 450 it, reference 1.1-1.3),
  maxima 4.2-5.8 at 64-256 (control 4.4-6.4, §8.4 4.2-6.7, reference
  3.5-3.8).  The maximum of V2 at 64 rises from 4.4 to 5.6.  The term's own
  estimator is within 1.9 at every level, against 22.8 for the control: what
  is left is what the 3 x 3 stencil does not see.
- **The tail of kappa at 512 pays as in §8.4**: kurtosis +4.0 cosmic sigma
  (control +2.3 at 450 it, +1.1 at 600 it), peaks above 3 sigma -0.9
  (control +0.1), variance of kappa -1.2 (+0.4).  Peaks above 10 sigma are
  closer (-0.6 against -1.1).  The kurtosis excursions of the 256 stage
  (+4.4 and +10.8 at 64 and 128) are gone: the kurtosis term of the 512
  stage holds them.  One member reaches +9 at nside 8 and 16 (control at
  most +3.2; `n256_1961965` had +32 and +51 there).
- 47.3 s per evaluation, 15 % more than the control (4 % without the
  native level in §8.4).  The 512 stage takes 6 h 30 on two T4.

So the term is best used from the first stage, with the native level: V0
at every level and V1 at the native level go back to cosmic variance, the
V1, V2 offsets at 128-256 shrink, and the rest of the loss does not suffer.
Two things it does not fix: V1, V2 maxima of 4-6 at 64-256 and of 4.8 for V2
at 512 (reference 3.5-3.8), and the kurtosis of kappa at the finest level,
4 cosmic sigma high whether the term starts early or late.  §8.6 finds
where that excess comes from and tests a term on the extreme ranks of g.
Untested: derivatives that keep the pixel scale.

### 8.6 The extreme ranks of g

The kurtosis term acts on `kappa = f(g)` through the 256 knots.  cv10 and
`mf_levels.py` judge the histogram-matched kappa.  On the members of §8.5
the two disagree.  The excess kurtosis of the finest details at 512 is
793-890 through the knots (r000: 855) but 1069-1598 after matching (control
at 450 it: 814-929 and 975-1450).  The members compress the top of g: their
three largest values are at 4.77-4.82, against 4.80, 4.90 and 5.11 for the
target.  The matching still gives those pixels the target's largest kappa
(0.078, 0.069, ...), often next to low siblings.  Putting the knot values
back on those three pixels alone brings the matched excess to 842-957.  A
larger kurtosis weight would act on a quantity already at its target.  The
1/N of the histogram term is what leaves those ranks free: a maximum of g
0.3 below the target's adds 3e-8 to the squared distance at nside 512.

`--marginal-tail-weight X --marginal-tail K` (USAGE.md) adds
`X / B sum_m sum_i (x_m,(i) - t_(i))^2` over the K smallest and K largest
ranks of each member, without the 1/N.  The 512 stage of §8.5 is redone
twice from the same nside-256 members with the term, nothing else changed.
The strong tail is `--marginal-tail-weight 10 --marginal-tail 100` (1 per
member).  The soft tail is `--marginal-tail-weight 1 --marginal-tail 30`
(0.1 per member; the top three ranks carried the whole excess).  Against
the control and §8.5 at 450 iterations (`mf_levels.py` as above):

| level, functional | r001-r009 | control, 450 it | from 256 (§8.5) | strong tail | soft tail |
|---|---|---|---|---|---|
| 64 V0 | 0.91 / 2.93 / 0.90 | 0.81 / 3.07 / 1.11 | 0.43 / 3.01 / 0.85 | 0.59 / 3.82 / 0.97 | 0.46 / 3.36 / 0.87 |
| 64 V1 | 1.05 / 3.45 / 1.09 | 1.36 / 5.20 / 1.55 | 1.27 / 4.56 / 1.58 | 1.22 / 3.87 / 1.38 | 1.38 / 5.17 / 1.58 |
| 64 V2 | 1.06 / 3.52 / 1.11 | 1.03 / 4.41 / 1.28 | 1.06 / 5.63 / 1.32 | 1.01 / 5.34 / 1.16 | 1.12 / 5.40 / 1.36 |
| 128 V0 | 1.49 / 3.61 / 1.63 | 1.52 / 3.32 / 1.83 | 0.85 / 2.27 / 1.01 | 1.01 / 2.95 / 1.42 | 0.70 / 2.65 / 0.99 |
| 128 V1 | 1.14 / 3.63 / 1.17 | 1.96 / 6.11 / 2.39 | 1.28 / 4.85 / 1.61 | 1.53 / 5.15 / 1.83 | 1.33 / 5.51 / 1.72 |
| 128 V2 | 0.94 / 3.55 / 1.11 | 1.70 / 5.75 / 2.03 | 1.39 / 5.66 / 1.61 | 1.41 / 4.84 / 1.66 | 1.35 / 6.19 / 1.73 |
| 256 V0 | 1.93 / 3.63 / 2.00 | 2.78 / 3.62 / 2.69 | 0.72 / 1.76 / 0.90 | 1.40 / 3.47 / 1.72 | 0.58 / 1.95 / 0.96 |
| 256 V1 | 1.20 / 3.76 / 1.31 | 2.01 / 5.40 / 2.52 | 1.70 / 4.24 / 1.71 | 2.83 / 5.69 / 3.09 | 1.90 / 5.07 / 1.99 |
| 256 V2 | 1.10 / 3.69 / 1.18 | 2.38 / 6.37 / 2.51 | 1.73 / 5.75 / 2.02 | 3.11 / 10.06 / 3.67 | 1.96 / 6.74 / 2.17 |
| 512 V1 | 1.15 / 3.70 / 1.18 | 1.53 / 3.45 / 1.60 | 0.86 / 2.31 / 0.97 | 2.99 / 4.81 / 2.93 | 1.06 / 2.34 / 1.21 |
| 512 V2 | 0.94 / 3.45 / 0.96 | 1.61 / 7.48 / 2.24 | 0.79 / 4.81 / 1.67 | 1.99 / 8.53 / 2.77 | 0.90 / 5.19 / 1.78 |

| | control, 450 it | from 256 (§8.5) | strong tail | soft tail |
|---|---|---|---|---|
| loss after 300 it; after 450 it | 2.725; 1.718 | 2.563; 1.574 | 6.692; 4.929 | 3.104; 2.025 |
| loss at evaluation 120 of the 512 stage | - | 6.88 | 23.3 | 8.79 |
| sum over the 30 / 100 extreme ranks at each end, per member | - | 0.15-2.70 / 0.17-2.74 | 0.0001-0.0006 / 0.0003-0.0014 | 0.003-0.008 / 0.016-0.175 |
| largest deviation of the top 3 ranks (standardised) | - | 0.34 | 0.02 | 0.05 |
| excess kurtosis of the finest kappa details, knots / matched (r000 855) | 814-929 / 975-1450 | 793-890 / 1069-1598 | 863-914 / 917-1013 | 838-926 / 907-1016 |
| mean, var of g | 2.54, 12.97 | 2.93, 11.15 | 12.75, 24.55 | 3.09, 10.79 |
| S1 S2 S3 S4, rms | 3.22 3.19 1.39 0.80 | 3.28 3.26 1.28 0.76 | 5.37 4.99 1.85 1.10 | 3.39 3.23 1.39 0.82 |
| kappa term; mean of kappa | 0.0195; 3.12 | 0.0134; 2.58 | 0.173; 9.27 | 0.0171; 2.77 |
| level power term; finest level / 256 variance over the target's | 0.769; 0.78 / 1.10 | 0.691; 0.79 / 1.09 | 2.716; 0.62 / 1.32 | 1.004; 0.75 / 1.12 |
| var of the levels of g, 128 / 256 / 512 (cv10) | +6.0 / +4.8 / -7.8 | +5.7 / +4.1 / -7.4 | +8.0 / +15.6 / -13.4 | +6.6 / +6.0 / -8.9 |
| var of kappa, 256 / 512 (cv10) | -1.0 / +0.4 | +0.3 / -1.2 | +1.4 / -2.0 | +0.4 / -1.3 |
| kurtosis of kappa at 64 / 128 / 512 (cv10) | +0.0 / +0.1 / +2.3 | +0.0 / +0.2 / +4.0 | +0.4 / +0.5 / +0.9 | +0.1 / +0.2 / +0.7 |
| peaks above 3 / 5 / 10 sigma (cv10) | +0.1 / +0.1 / -1.1 | -0.9 / -0.5 / -0.6 | +0.8 / +1.2 / -0.5 | -0.8 / -0.3 / -0.5 |
| C_l, l >= 31: median / max abs z (cv10) | 0.3 / 0.6 | 0.2 / 0.6 | 0.3 / 1.5 | 0.3 / 0.6 |
| term's estimator, max abs z of V1 / V2 at 256; at 512 | 7.5 / 22.8; 3.5 / 10.5 | 1.5 / 1.2; 1.2 / 1.1 | 3.3 / 1.8; 2.6 / 1.1 | 1.7 / 1.2; 1.3 / 1.6 |
| s / eval (2 T4) | 41.3 | 47.3 | 48.2 | 48.2 |

- **Both tails hold the extreme ranks and bring the kurtosis back.**  The
  top three ranks sit within 0.02 (strong) and 0.05 (soft) standardised
  units of the target's.  The matched excess of the finest details falls to
  907-1016.  The kurtosis of kappa at 512 falls to +0.9 and +0.7 cosmic
  sigma, every member within +1.4.
- **The strong tail stalls the level power.**  From evaluation 40 on, its
  loss is about three times that of §8.5 at the same evaluation.  The
  level-power term carries most of it and does not move over the last 150
  iterations: the finest level of g holds 0.62 of the target's variance at
  300 and at 450 iterations, and 256 holds 1.32.  cv10 sees the same shift:
  level variances of g +15.6 / -13.4 cosmic sigma at 256 / 512.  The V1, V2
  of the two finest levels go out with it (medians 2.0-3.1, maxima up to
  10).  Peaks above 5 sigma rise to +1.2 and the largest C_l deviation to
  1.5.
- **The soft tail only slows it.**  Its loss is 10-30 % above that of §8.5
  at the same evaluation (2.03 against 1.57 at the end), and the excess is
  again in the level power (1.00 against 0.69).  The level variances of g are
  about 1 cosmic sigma further from r000 than the control's at 256 and 512.
  Mean, variance, S1-S4, the kappa term, the peaks and the C_l are those of
  §8.5.  V0 stays within cosmic variance at every level, and V1, V2 at 512
  too.  The V1, V2 maxima at 64-256 are 0.5-1 above §8.5's (5.1-6.7 against
  4.2-5.8, control 4.4-6.4), with the medians at 256 at 1.9-2.0.
- Per pixel, the tail's curvature is 2 X / B.  Every other term is a mean
  over pixels, with a curvature per pixel of order 1 / N; for the histogram
  term it is 2 w / (B N).  At 1 per member the tail is about 3000 times
  stiffer than the histogram term, and the L-BFGS step scaling follows the
  stiffest directions.  Ten times softer, the descent of the rest is
  delayed, not stopped.  The same balance at another nside needs X N fixed:
  X = 0.1 B (512 / NSIDE)^2.  K should scale as N (K = 30 (NSIDE / 512)^2)
  to hold the same fraction of the histogram.  That scaling is not tested.

So the soft tail fixes the kurtosis of kappa at the finest level without
cost to the peaks, the C_l or the statistics of the loss.  It costs a slower
descent of the level power, worth about 1 cosmic sigma in the level
variances of g after 450 iterations, and slightly larger V1, V2 maxima at
64-256.  The strong tail is not to be used.

The soft tail is continued for 150 more iterations with the same flags, as
the control was in §8.4 (`--init-noise 0`, a fresh L-BFGS history).  At 600
iterations:

| level, functional | r001-r009 | control, 600 it | soft tail, 450 it | soft tail, 600 it |
|---|---|---|---|---|
| 64 V0 | 0.91 / 2.93 / 0.90 | 0.85 / 2.94 / 1.10 | 0.46 / 3.36 / 0.87 | 0.47 / 3.00 / 0.82 |
| 64 V1 | 1.05 / 3.45 / 1.09 | 1.27 / 5.26 / 1.53 | 1.38 / 5.17 / 1.58 | 1.35 / 4.92 / 1.56 |
| 64 V2 | 1.06 / 3.52 / 1.11 | 0.93 / 4.56 / 1.31 | 1.12 / 5.40 / 1.36 | 1.07 / 4.68 / 1.44 |
| 128 V0 | 1.49 / 3.61 / 1.63 | 1.67 / 3.53 / 1.97 | 0.70 / 2.65 / 0.99 | 0.79 / 2.47 / 1.03 |
| 128 V1 | 1.14 / 3.63 / 1.17 | 1.88 / 6.16 / 2.29 | 1.33 / 5.51 / 1.72 | 1.42 / 5.08 / 1.70 |
| 128 V2 | 0.94 / 3.55 / 1.11 | 1.73 / 5.41 / 1.97 | 1.35 / 6.19 / 1.73 | 1.34 / 5.64 / 1.68 |
| 256 V0 | 1.93 / 3.63 / 2.00 | 3.08 / 3.91 / 2.90 | 0.58 / 1.95 / 0.96 | 0.88 / 1.83 / 0.97 |
| 256 V1 | 1.20 / 3.76 / 1.31 | 1.55 / 5.29 / 2.21 | 1.90 / 5.07 / 1.99 | 1.72 / 4.29 / 1.75 |
| 256 V2 | 1.10 / 3.69 / 1.18 | 2.13 / 5.38 / 2.31 | 1.96 / 6.74 / 2.17 | 1.79 / 5.76 / 1.96 |
| 512 V1 | 1.15 / 3.70 / 1.18 | 1.33 / 3.08 / 1.41 | 1.06 / 2.34 / 1.21 | 1.12 / 2.66 / 1.23 |
| 512 V2 | 0.94 / 3.45 / 0.96 | 1.52 / 6.70 / 2.25 | 0.90 / 5.19 / 1.78 | 0.95 / 5.04 / 1.78 |

| | control, 600 it | soft tail, 600 it |
|---|---|---|
| loss at evaluation 1 / 80 / 150 of the +150 | 1.718 / 1.365 / 1.194 | 2.025 / 1.687 / 1.494 |
| loss after 600 it; Minkowski and tail terms | 1.157; - | 1.484; 0.024, 0.0024 |
| largest deviation of the top 3 ranks (standardised) | 0.29 | 0.02 |
| excess kurtosis of the finest kappa details, knots / matched (r000 855) | 816-898 / 901-1133 | 847-882 / 847-903 |
| mean, var of g | 0.73, 1.01 | 0.44, 3.90 |
| S1 S2 S3 S4, rms | 2.08 2.06 1.15 0.71 | 2.36 2.48 1.15 0.71 |
| kappa term; mean of kappa | 0.0029; 0.43 | 0.0032; 0.65 |
| level power term; finest level / 256 variance over the target's | 0.496; 0.82 / 1.05 | 0.779; 0.77 / 1.09 |
| var of the levels of g, 128 / 256 / 512 (cv10) | +4.7 / +2.6 / -6.4 | +5.8 / +4.4 / -7.9 |
| var of kappa, 256 / 512 (cv10) | -1.3 / +0.8 | +0.2 / -1.2 |
| kurtosis of kappa at 64 / 128 / 256 / 512 (cv10) | -0.0 / +0.1 / +0.6 / +1.1 | +0.0 / +0.1 / +0.1 / +0.1 |
| peaks above 3 / 5 / 10 sigma (cv10) | -0.1 / 0.0 / -0.9 | -0.6 / -0.2 / -0.4 |
| C_l, l >= 31: median / max abs z (cv10) | 0.3 / 0.6 | 0.2 / 0.6 |
| term's estimator, max abs z of V1 / V2 at 256; at 512 | 6.6 / 19.8; 3.3 / 13.4 | 1.4 / 0.6; 0.9 / 0.6 |
| s / eval (2 T4) | 40.8 | 48.2 |

- **The kurtosis of kappa reaches the reference.**  It is +0.1 cosmic
  sigma at 256 and at 512, every member within +0.4, against +0.6 and +1.1
  for the control.  The matched excess of the finest details is 847-903
  (r000 855), and the top three ranks are within 0.02 of the target's.  The
  control's 150 iterations lower its kurtosis too (from +2.3 to +1.1), but
  its top ranks stay up to 0.3 off.
- **The level-power lag does not close.**  Over the 150 iterations both
  losses fall by about the same amount (0.54 and 0.56), and the gap stays
  at about 0.3.
  The level power is 0.28 of it (0.78 against 0.50).  The level variances of
  g at 600 iterations are those of the control at 450 (+5.8 / +4.4 / -7.9
  against +6.0 / +4.8 / -7.8).  That is 1.1-1.8 cosmic sigma further from
  r000 than the control at 600.  The variance of g is 3.9 ergodic sigma off
  (control 1.0), and S1, S2 are 2.4, 2.5 (control 2.1).
- **The Minkowski functionals follow the iterations.**  V0 stays within
  cosmic variance at every level, and V1 at 512 too.  The V1, V2 maxima at
  64-256 fall from 5.1-6.7 to 4.3-5.8 (control at 600: 4.6-6.2).  The V1
  maxima are below the control's.  The V2 maxima are 0.1-0.4 above.  The
  V1, V2 medians at 256 are 1.7-1.8, against 1.6-2.1 for the control
  (reference 1.1-1.2), and V0 at 256 is 0.9 (control 3.1).  The maximum of V2
  at 512 is 5.0 (control 6.7).  The term's own
  estimator is within 1.5 at every level (control 19.8).
- Peaks and C_l stay within cosmic variance (-0.6 / -0.2 / -0.4; 0.2 /
  0.6).  The variance of kappa at 512 is -1.2 (control +0.8).

So after 600 iterations the soft tail passes the kurtosis of kappa, the
peaks and the C_l.  Its V0, V1, V2 are as close as the control's or closer,
apart from the V1, V2 maxima of 4-6 at 64-256 that no recipe brings down
(reference 3.5-3.8).  It costs about 150 iterations of level-power descent:
its level variances of g at 600 iterations are the control's at 450.
Untested: the tail above nside 512.
