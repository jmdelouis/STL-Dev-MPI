# Amortized synthesis: generators instead of one L-BFGS run per map

Design note, for discussion before any heavy implementation.  The aim:
accept a large one-off cost (training, or the first few syntheses), so
that the next 1 000 or 10 000 maps that match a scattering-covariance
target cost almost nothing each.

## 1. Where we start

What one map costs today (L = 4, measured; see `docs/gpu.md`,
`docs/harmonic_speed.md`, `docs/dia_convolution.md` and the session notes):

| nside | one evaluation (statistics + gradient), per map | iterations to converge | one map |
|---|---|---|---|
| 128 | 0.027 s on an A100 (batch 4: 0.106 s); ~0.3 s on a T4 with the harmonic terms | 300 to 2 500 | 0.2 to 12 min |
| 256 | 0.5 s on a T4 | 300 to 1 000 | 3 to 8 min (T4) |
| 1024 | 2.6 s on an A100 (batch 10: 25.7 s); 23 s on a T4 | 1 000 to 2 500 | 1 to 2 h (A100) |
| 2048 | 18 s on an A100 (DIA, single map); 6 to 14 s per member with batch 10 | 1 000 to 2 500 | 5 to 10 h (A100) |

An ensemble of 10 maps at nside 2048 is about 100 A100 hours.  10 000 maps
by L-BFGS are out of reach above nside 256.

What the repository already has for an amortized approach:

- **The loss and its exact gradient for a batch of maps, callable from
  Python.**  `crates/stl-bridge` (C ABI) exposes `stlp_forward` (statistics
  of `batch` maps, optionally their spatial variances) and `stlp_vjp` (map
  cotangent from coefficient cotangents) on the CPU or the GPU engine;
  `python/stl_ensemble/stl_port/native.py` wraps it (`NativeEngine.forward`,
  `.vjp`).  A `torch.autograd.Function` around these two calls makes the
  Rust pyramidal gradient a PyTorch layer.  Memory stays bounded by the
  Rust engine, not by an autograd graph (STL-Dev's autograd graph would
  need ~380 GB at nside 2048; the sweep needs ~50 B per pixel).
- **The right loss for a population.**  `stl-synth --ensemble B
  --scatter-weight X` matches the mean of the statistics of B maps *and*
  holds their scatter at the ergodic variance (X = 1: every member about
  1 sigma from the target, `docs/USAGE.md` §5.4).  Paolo Campeti's
  `sc_loss` (`python/stl_ensemble`) matches a target mean and a target
  variance of the population.  Either loss, applied to the outputs of a
  generator instead of to free maps, is the training objective of an
  amortized generator.
- **A multi-scale conditional diffusion model** (`diffusion/`,
  `docs/diffusion_design.md`): face tiles with re-oriented halos, or a
  sphere-native network with the STL stencils; trained on one Takahashi
  sky.  It learns the two or three finest levels well (right variance,
  kurtosis, isotropy) but not the coarse levels (too few pixels in one
  sky), and its samples are 3 to 70 ergodic sigma off in the STL
  statistics.  As an initialisation of `stl-synth` it halves the number of
  iterations and keeps the fine-level non-Gaussianity.

FOSCAT's fast synthesis (`foscat.SynthHalfUNet2D`, 2D only) is the
template for direction 1: a decoder-only U-Net, input Gaussian noise
`[N, 1, H, W]`, skips that are the frozen FOSCAT wavelets applied to the
pooled noise at every scale, blocks upsample + concat + 2 x (Conv3x3,
BatchNorm, LeakyReLU); Adam, cosine schedule, lr 1e-3, 2 000 to 3 000
steps, a fresh noise batch of N = 4 to 8 per step.  The default
"microcanonical" loss is

    sum_k (mean_batch Phi_k - Phi*_k)^2 / (std_batch Phi_k * |Phi*_k| + eps)

so it matches the batch mean, and the batch std in the denominator makes
collapse expensive, but it does not hold the scatter at any particular
value.  The per-sample loss (`microcanonical=False`) is documented as
prone to mode collapse.  No measured quality or cost is given.

## 2. What "matching the target" should mean for 10 000 maps

Not that every map reproduces `Phi*` exactly.  A map drawn from the true
process sits about 1 ergodic sigma from the expected statistics, and
10 000 maps must reproduce that scatter, or the ensemble is too narrow
for any covariance or likelihood study.  The acceptance tests, per
generated population:

1. **Mean**: the mean of the generated statistics within `sigma / sqrt(N)`
   of the target, per family (S1 to S4, mean, var).
2. **Scatter**: per-map residuals of about 1 sigma,
   `scatter_over_ergodic_variance` about 1 (`stl-synth` report).
3. **Cosmic variance (Takahashi)**: with `tools/cv10.py`, the per-level
   detail variance and kurtosis, peak counts and C_l bands of the
   generated maps inside the envelope of the 10 realisations r000 to r009;
   Minkowski functionals, C_l and scattering populations plotted as in
   `benchmarks/campeti_n1024_a100/plots/`.
4. **Beyond the diagonal**: the scatter term holds only the variance of
   each coefficient.  With thousands of maps, compare the full covariance
   of the statistics (and of the C_l) with the Takahashi realisations;
   this is where a generator can be wrong while passing tests 1 to 3.
5. **Diversity and memorisation**: pairwise correlation of generated maps
   (should be ~0 above the scales the target fixes), nearest-neighbour
   distance to the training sky (`diffusion/memorisation.py`), and a
   per-map check that the large-scale phases of r000 are not reproduced.

Tests 1 and 2 cost one batched forward of the Rust engine.  An L-BFGS
"polish" (section 4.6) is only needed if per-map residuals are well
above 1 sigma.

## 3. Cost model of training through the statistics loss

A training step on a batch of B generated maps costs B map evaluations
of the Rust engine (one forward + one VJP each) plus the generator's
forward and backward on B maps (small next to the Rust sweeps for a
U-Net of a few M parameters; ~10 % at nside 256 on a T4).  With S steps,
the training costs about `S * B` map evaluations.

An L-BFGS ensemble of 10 members with 2 500 iterations is 25 000 map
evaluations.  A FOSCAT-like training (S = 3 000 to 10 000 steps, B = 8)
is 24 000 to 80 000 map evaluations: **one generator costs about as much
as 1 to 3 L-BFGS ensembles of 10 maps**, and every map after that costs
one forward pass of the generator.

| nside | map eval (GPU) | training, S = 5 000, B = 8 | per map after training | break-even vs L-BFGS |
|---|---|---|---|---|
| 64 | ~0.05 s (T4) | ~0.6 T4 h | ms | ~10 maps |
| 128 | ~0.12 s (T4), 0.03 s (A100) | ~1.5 T4 h | ms | ~10 maps |
| 256 | 0.5 s (T4) | ~6 T4 h | ~0.1 s | ~15 maps |
| 1024 | 2.6 s (A100) | ~30 A100 h | ~1 s | ~20 maps |
| 2048 | 18 s (A100), ~8 s in batch | ~100 to 200 A100 h | ~5 s | ~20 maps |

These are estimates from the measured map evaluations; the unknowns are
S (Adam on network weights may need more steps than L-BFGS on pixels)
and how well a generator matches at all, which is what the prototype
measures.  Section 5 shows how the hierarchical generator cuts the
nside 1024 to 2048 numbers by training most of the network at low nside.

## 4. Options

### 4.1 U-Net from noise, trained through the ScatCov loss (FOSCAT style)

Generator `G_theta(z)`, `z` white noise on the sphere; loss = the ensemble
loss of `stl-synth` (mean + scatter term, ergodic weights) or Campeti's
moment loss on a batch of B = 8 to 16 outputs, through the Rust VJP.  On
HEALPix the FOSCAT architecture maps directly onto the NESTED pyramid:
pool the noise to every level (mean of the 4 children), wavelet skips =
the STL wavelet of the pooled noise (fixed, `stl-core` convolution or the
torch transcription in `diffusion/stl_torch_geometry.py`), decoder blocks
= upsample to the 4 children + local convolutions.  Local convolutions:
the face tiles with re-oriented halos of `diffusion/model.py` (cuDNN
conv2d, fast) or the sphere-native STL stencils
(`diffusion/sphere_conv.py`, 5 x slower to sample, smaller seam).

- **Training**: no data, only the target statistics (or the population
  mean and variance from the 10 Takahashi skies).  Cost: section 3.
- **Per map**: one forward pass (milliseconds at nside 128, seconds at
  2048).
- **Exactness**: matches the population mean by construction of the loss;
  the scatter term (X = 1) sets the per-map residual at about 1 sigma.
  Whether a network of limited capacity reaches the L-BFGS level (0.3 to
  0.5 sigma for the mean) is the open question.
- **Diversity**: the scatter term, not FOSCAT's anti-collapse
  denominator, holds the diversity at the ergodic value.  No training
  maps, so nothing to memorise; but the generator can learn a narrow
  family of maps that all have the right statistics (the analogue of
  microcanonical synthesis reproducing the target's phases).  Test 5.
- **High nside**: feasible because the Rust sweep bounds the memory of
  the loss; the generator itself is local, its activations can be tiled
  like the diffusion sampler.  Cost at 2048 is the issue (section 5).
- **Fit with the code**: `torch.autograd.Function` over `NativeEngine`;
  the Rust engine in f64 on the GPU, the generator in f32 on the same GPU
  (T4: 15 GB is enough at nside 256 with batch 8).

**This is the cheapest direction to try and the one I recommend first.**

### 4.2 Diffusion or flow matching trained on maps

The existing multi-scale diffusion (one conditional denoiser per level).
Flow matching / rectified flow is the same network with a different
training target and a sampler of 5 to 20 steps instead of 63.

- **Training data is the bottleneck**: the target is a set of statistics,
  not a set of maps.  One sky is enough for the finest levels only
  (section 8.8 of the diffusion note).  The Takahashi suite has 108
  full-sky realisations; 10 are on hpc.  Or distil: train on maps made by
  4.1 or by L-BFGS (then diffusion adds only what those maps contain).
- **Per map**: 63 denoiser steps per level: 12 to 15 s at nside 256 and
  ~13 min at nside 2048 on a T4; rectified flow could cut that 3 to 10 x.
- **Exactness**: no statistic is enforced; 3 to 70 sigma off on one-sky
  training.  Needs a polish or guidance.
- **Diversity**: learned from the data; memorises on small coarse levels.
- **Use**: the learned prior for the finest levels where the statistics
  do not pin the detail variance and kurtosis (section 1 of the diffusion
  note).  It is a complement to 4.1, not a replacement.

### 4.3 Diffusion with ScatCov guidance

Sample the diffusion model and add, at every denoising step, the gradient
of the ScatCov loss of the denoised estimate (classifier-free style
guidance, or "diffusion posterior sampling").  One Rust VJP per step:
~60 map evaluations per map, against 300 to 2 500 for L-BFGS: 5 to 40 x
cheaper per map, but not free.  The guided samples keep the learned
fine-level non-Gaussianity and land on the statistics.  Worth it only
once 4.2 is trained on enough skies.

### 4.4 Pretrained generator for "all" statistics, fine-tuned per target

A generator conditioned on the statistics vector (FiLM per level, or
cross-attention from the coefficients, compressed by family and
orientation: a few hundred numbers), trained on a broad family of fields
(Gaussian with varied spectra, lognormal, kappa-like, turbulence,
multifractal cascades) with the statistics loss against the conditioning
vector.  For a new target: zero-shot, then a short fine-tune of FiLM
parameters or LoRA adapters through the same loss (a few hundred steps).

- **Training**: 10^5 to 10^6 steps over thousands of targets: hundreds of
  GPU hours at nside 128; impractical at nside >= 1024 except with the
  hierarchical scheme below.  Research-level risk: the network must
  generalise over a family of non-Gaussian processes.
- **Per target**: fine-tune ~ a few hundred steps, i.e. 1/10 of 4.1.
- **Pay-off**: when many targets are needed (several redshifts, several
  cosmologies, components of a foreground model), not for 10 000 maps of
  one target.  Defer until 4.1 works for one target.

### 4.5 Hierarchical coarse-to-fine generator (the pyramid)

One small network per level `n`: `x_n = up(x_{n+1}) + g_n(up(x_{n+1}), z_n)`,
the deterministic decoder of 4.1 split by level, mirroring the
coarse-to-fine initialisation of `stl-synth` and the levels of the
diffusion model.  Train the coarse levels at low nside (cheap), freeze
them, and train only the next level at the next nside with the statistics
loss at that nside (progressive growing).  The statistics coupling a
coarse scale to the new fine scale (S3, S4 with j1 < j3) still need the
full sky at the new nside, so every new level costs map evaluations at
its own nside, but for far fewer steps (a few hundred: the coarse part is
already right; the new level is local and close to white).

At nside 2048, 500 steps x B = 4 x 8 s = ~5 A100 hours for the finest
level, against 100 to 200 for training 4.1 end to end at 2048.  The
networks `g_n` for the finest levels can also be the conditional
diffusion denoisers of 4.2 (stochastic decoder) or, with the same
weights, plain local generators.

### 4.6 Warm start + short polish

Any generator gives the init of `stl-synth`; 20 to 50 L-BFGS iterations
bring each map from ~1 to ~0.3 sigma.  At nside 128 to 256 this is a few
seconds per map; at nside 2048 it is 6 to 15 A100 minutes per map, so
10 000 polished maps are again thousands of GPU hours.  Polish is a
diagnostic and a fallback (for the few maps that will be analysed in
depth), not the production path at high nside.  The diffusion hybrid
(section 8.5 of the diffusion note) is this option with ~300 iterations.

### 4.7 Other ideas, briefly

- **Amortized inverse (encoder stats -> latent -> decoder)**: the same as
  4.4 with an explicit encoder; no benefit for one target.
- **Neural operators / attention on the pyramid**: global receptive field
  for the coarse levels, where convolutions see few pixels.  Only the
  coarsest levels (nside <= 32, < 12 000 pixels) need it, and there a
  plain MLP or a small transformer on the 12 x nside^2 pixels is cheap.
- **Symmetries of the grid**: rotations by pi/2 about the pole and the
  north-south reflection permute HEALPix pixels exactly (16 copies of a
  map, no interpolation).  The copies are not independent realisations
  (same structures), so they do not help cosmic variance studies; useful
  only as data augmentation for training 4.2.
- **Gaussian latent trick**: generate the coarse levels as Gaussian
  random fields with the target C_l (free, exact spectrum, correct
  cosmic variance of the large scales) and let the network generate only
  the non-Gaussian finer levels.  Cheap and robust; a good baseline for
  the coarse part of 4.5.
- **Rust/candle instead of PyTorch**: possible later (one binary, MPI
  decomposition of the generator like the engine), but the PyTorch +
  bridge path costs nothing to set up and keeps FOSCAT's code at hand.

## 5. Recommendation and staged plan

Recommendation: **4.1 trained with the ensemble loss of `stl-synth`
(mean + scatter, ergodic weights), with the 4.5 split by level so that it
can grow to nside 2048, and 4.6 only as a check.**  It needs no training
data, it reuses the Rust engine as is, its cost is that of one or two
L-BFGS ensembles, and its quality can be measured with the tests we
already have.  4.2/4.3 stay the tools for the fine-level non-Gaussianity
once more skies are available; 4.4 only if many targets are needed.

Stages (garoupe unless said otherwise):

0. **Bridge layer** (half a day).  `torch.autograd.Function` over
   `NativeEngine` (GPU backend, batch B), the ensemble loss (mean +
   scatter term) in torch on the returned coefficients, gradient check
   against finite differences and against `stl-synth`'s loss on the same
   maps.  Build `stl-bridge --features gpu` on garoupe (the 2026-10-20
   build of `stl-synth` shows the toolchain is there).
1. **Prototype at nside 64, then 128** (a few T4 hours).  Target:
   Takahashi r000 zs1 gaussianised (as in the diffusion study), L = 4,
   J = 6/7.  Generator: pyramid decoder with STL-wavelet skips of the
   pooled noise, face tiles with halos.  B = 8, Adam, 5 000 steps.
   Measure against an L-BFGS ensemble of 10 at the same nside (same
   loss): tests 1, 2 and 3 of section 2 on 1 000 generated maps, and the
   cost per map.  Ablations: scatter weight X = 0 / 1 (collapse),
   FOSCAT's microcanonical loss, with and without the wavelet skips.
2. **Polish check** (an hour): 20 / 50 L-BFGS iterations from 10
   generated maps; how far they move and whether the C_l / Minkowski
   envelope changes.
3. **Grow to nside 256** (4.5): freeze the nside-128 generator, train one
   more level (500 to 1 000 steps), compare with training end to end.
   Gives the per-level step counts used to cost 1024 and 2048.
4. **Decision point** with the user: if stage 3 holds, nside 1024 and
   2048 on hpc (needs explicit OK for the A100; ~5 to 30 A100 hours by
   the estimate of section 4.5) and the Takahashi full-covariance test
   (section 2, test 4) on 10 000 maps.

## 6. Prototype at nside 64 (stages 0 to 2, garoupe, 2026-11-03)

Code: `amortized/` (`stl_torch.py` the Rust engine as a torch layer and
the ensemble loss, `generator.py`, `train.py`, `evaluate.py`,
`check_grad.py`, `run_eval.sh`).  `stl-bridge --features gpu` of this
branch's base (9c52ff4) on one T4; generator in float32 on the same GPU.
Target: Takahashi r000 zs1 kappa (not gaussianised) at nside 64, J = 5,
L = 4, standardised; ergodic weights, scatter weight 1, batch B = 8, Adam
(lr 1e-3, cosine), 3 000 steps.

**Stage 0.**  The directional derivative of the ensemble loss through
`StatsFunction` matches a central finite difference to 1e-8 relative
(error falling as h^2, h = 1e-3 to 1e-5), on the maps and through the
generator's weights: the bridge's complex convention is the one of torch.

**Cost.**  0.45 s per step (B = 8: 0.06 s per map evaluation, forward +
reverse sweep; the generator ~10 %), 3 000 steps in 22 to 29 minutes.
Generation: 4.8 ms per map.  L-BFGS ensemble of 8 (same loss, 500
iterations): 136 s, 17 s per map.  A batch of 16 with wider channels
(96 to 64) costs 1.28 s per step and reached the same loss at the same
step: neither gradient noise nor width is the limit.

**Quality** (1 000 generated maps against the 8 L-BFGS members; per-map
residual in ergodic sigma, rms over each family; scatter over the ergodic
variance):

| | S1 | S2 | S3 | S4 | var | scatter / ergodic | skewness (target 8.4) |
|---|---|---|---|---|---|---|---|
| L-BFGS, 8 maps | 0.96 | 0.93 | 0.95 | 0.94 | 0.82 | 0.97 to 1.03 | 5.0 |
| generator `c` | 3.46 | 2.86 | 3.15 | 3.06 | 3.87 | 3 to 8 (var 0.2) | 3.7 |
| generator `c` + 50 L-BFGS iterations | 1.12 | 0.90 | 1.04 | 0.99 | 2.92 | 0.6 to 1.1 (var 0.05) | 4.2 |

- The population mean of the generator is 2 to 2.5 sigma off per family;
  the maps are too different from each other in S1 to S4 and too alike in
  their variance (scatter 0.2 of the ergodic one).
- **Grid pattern** (first generator, `a`): with nearest-neighbour
  up-sampling the generator adds a z-independent pattern, periodic on the
  pixel grid, uniform over the faces, 7.7 % of the variance; the maps are
  correlated with each other (0.07, up to 0.17; L-BFGS 0.007).  Smooth
  up-sampling (nearest then [1, 2, 1]^2 / 16) removes it (0.011), and
  does not change the residuals.
- **Finest level and tail** (`tools/cv10.py`,
  `figures/amortized_cv10_n64.png`): the generator's finest level has 0.4
  of r000's detail variance (-4.2 cosmic sigma), the C_l fall above l ~
  100, the kurtosis of the details is 3 cosmic sigma low at every level,
  and the peaks above 10 sigma are 4.3 cosmic sigma short.  The 50-iteration
  polish brings S1 to S4 to 1 sigma but keeps all of this: **the
  statistics at 1 sigma do not certify the map**.  L-BFGS from white noise
  gets the level variances right (within 0.7 cosmic sigma) and the C_l
  (max 0.7 sigma), with the known kurtosis deficit of STL syntheses.
- **Output anamorphosis** (`--marginal`, runs `e`/`f`): generating a
  Gaussian-like field and mapping it through the target's quantile
  function gives the tail by construction, but the linear extrapolation
  beyond the knots makes the scatter term explode (10^4 to 5 x 10^4) and
  training does not converge.  Its 50-iteration polish gives the best
  skewness (5.5) but 1.2 to 1.7 sigma per member.

**Reading.**  The pipeline (Rust loss and exact gradient inside a torch
training loop) works and is cheap.  The plain convolutional generator
trained through the strict ergodic loss stalls around 3 sigma per map
in 3 000 steps, and its failures are structural (finest scale, tail, map
variance) rather than a matter of training time.  Polish from it costs
1.5 s per map against 17 s from white noise (11 x), but inherits the
defects that the statistics do not see.  Break-even of generator +
polish against L-BFGS at nside 64: about 85 maps; for 10 000 maps,
4.5 T4 hours against 47.

### 6.1 The three fixes (runs `g`, `h`, `i`, 3 000 steps each)

- `i`: map path up-sampled by nearest neighbour (features still smooth)
  and `--level-power-weight 10` (the `stl-synth` term, transcribed in
  `train.py`).
- `g`: `i` plus a per-map latent amplitude: every block also sees one
  constant channel per map, the global mean of the map's noise times
  sqrt(npix) (a N(0, 1) scalar, so a function of the noise and
  reproducible).
- `h`: `g` with the generator making the gaussianised field g (main loss
  on the statistics of g) and the statistics of kappa = f(g) as a second
  reduced chi^2 of factor 0.1, f the target's monotone table (256 knots,
  piecewise linear, linear beyond the ends): the `--kappa-table`
  objective.

Per-map residuals (ergodic sigma) and the cosmic-variance test
(`tools/cv10.py`, members against the 10 Takahashi skies, in cosmic
sigma; "finest" is the detail variance of level 64):

| | S1 | S2 | S3 | S4 | var | finest | max \|z\| C_l, l >= 31 | peaks > 3 / 5 / 10 sigma |
|---|---|---|---|---|---|---|---|---|
| L-BFGS from white noise | 0.96 | 0.93 | 0.95 | 0.94 | 0.82 | +0.1 | 0.7 | -1.0 / -1.2 / -1.8 |
| `c` (section 6) | 3.46 | 2.86 | 3.15 | 3.06 | 3.87 | -4.2 | 4.9 | -1.4 / -2.7 / -4.3 |
| `i` | 3.28 | 2.67 | 3.10 | 2.94 | 2.00 | -2.1 | 1.9 | -2.6 / -1.2 / -2.3 |
| `g` | 3.60 | 2.94 | 3.23 | 3.08 | 1.34 | -1.5 | 2.1 | -1.0 / -0.7 / -2.3 |
| `g` + 50 L-BFGS iterations | 1.12 | 0.89 | 1.02 | 0.99 | 0.92 | -1.5 | 1.4 | -1.4 / -0.8 / -1.3 |
| `h` | 57 | 25 | 17 | 15 | 15 | -6.6 | 6.8 | -7.4 / -6.3 / -4.3 |
| `h` + 50 L-BFGS iterations | 1.21 | 0.97 | 1.09 | 1.03 | 3.82 | -5.4 | 5.3 | -3.9 / -3.8 / -3.4 |

- **Fix 1 works**: the finest level goes from 0.4 to 0.78 of r000's
  detail variance (-4.2 to -2.1 cosmic sigma), the coarser levels are
  within 0.5 cosmic sigma, the C_l within 2 sigma
  (`figures/amortized_cv10_n64_g.png`).
- **Fix 2 works on the variance**: the per-map `var` residual goes from
  2.0 to 1.34 sigma, and after the polish to 0.92 (it stayed at 2.9 sigma
  without the latent: the generator's maps all had the same variance).
- **`g` + 50 L-BFGS iterations matches L-BFGS from white noise on every
  test** within about 1 cosmic sigma, except the finest level (-1.5) and
  the kurtosis of the details, 2 to 3 cosmic sigma low as for L-BFGS
  itself (a known limit of the STL statistics, not of the generator):
  every family at 0.9 to 1.1 sigma per map, C_l max 1.4 sigma, peaks
  above 10 sigma -1.3.  Cost: 1.5 s per map against 17 s, after 23
  minutes of training.
- **The generator alone** stays at about 3 sigma per map in S1 to S4,
  with 3 to 9 times the ergodic scatter between maps: its maps are more
  different from each other than real skies are.  The scatter term of
  the loss (X = 1) does not hold that scatter down within 3 000 steps.
- **Fix 3 fails as implemented**: the scatter term of g explodes during
  training (10^4), the kappa term never comes down, and the maps are 15
  to 57 sigma off.  Its polished maps have the best kurtosis of all runs
  (within 1 cosmic sigma at levels 8 to 32), so the idea is worth another
  try, with a lower learning rate or a first phase on the g statistics
  alone.

Maps (`figures/amortized_maps_n64.png`, mollview and a 20-degree
gnomview with `norm='hist'` and the target's scale): the generator's
maps have filaments, voids and peaks like the target, a little less
filamentary; after the polish they are close to the L-BFGS ones.

**Status at nside 64.**  The amortized path that works today is
"generator + short polish": 23 T4 minutes once, then 1.5 s per map
for maps as good as L-BFGS from white noise (17 s per map).  Break-even
~85 maps; 10 000 maps: 4.5 T4 hours instead of 47.  A generator good
enough without the polish (about 1 sigma per map) is not reached yet;
the gap is the excess scatter between maps.

**Next steps:**

1. Excess scatter of the generator alone: a larger scatter weight (X = 10)
   or a longer schedule; check whether the per-map residual comes down
   towards 1 sigma.
2. Fix 3 again, with a phase on the g statistics alone before adding the
   kappa term, for the kurtosis.
3. nside 128 by progressive growing from `g` (stage 3), measuring the
   steps the new level needs: that number sets the cost at nside 1024
   and 2048.

**Earlier list of fixes (section 6, tried in 6.1):**

1. Finest level: no blur on the map path (smooth up for the features
   only, nearest + learned detail for the map), and a level-power term
   (`--level-power-weight` of `stl-synth`) in the training loss so that
   every level's variance is held directly.
2. Per-map variance: a global latent amplitude per map (one random
   scalar per map modulating the heads), so that the variance of the maps
   can scatter as much as the target's.
3. Tail: generate the gaussianised field g and train on the statistics
   of g and of kappa = f(g) (the `--kappa-table` objective of `stl-synth`),
   with the extrapolation of f clamped.
4. Then nside 128 and progressive growing (stage 3).
