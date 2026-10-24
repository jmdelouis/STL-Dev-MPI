# Multi-scale conditional diffusion on the HEALPix pyramid

Design note and prototype report for a learned generative model of
full-sky convergence maps, complementary to the scattering-covariance (STL)
synthesis of this repository.  Code: `diffusion/` (Python, PyTorch).

## 1. Why

STL syntheses fit every statistic (mean, var, S1 to S4) to about 0.5
ergodic sigma, yet the finest pyramid level of the result holds only about
0.6 of the target's detail variance (nside 256, Takahashi r000 zs1,
gaussianised: 0.58 without the level-power term, 0.76 when the coarse
levels are frozen; see section 8).  The scattering statistics do not pin the
finest scale.  A model that learns the conditional law of every level of the
pyramid given the coarser one sets the variance, the non-Gaussianity and
the coupling of the finest details by construction, and its samples are
cheap once trained.

## 2. Principle

Wavelet score-based generative modelling (Guth, Coste, De Bortoli, Mallat
2022) and multi-scale local conditional models (Kadkhodaie, Guth, Mallat,
Simoncelli 2023): with an orthogonal multi-scale decomposition
`x_n -> (x_{n+1}, d_n)`,

    p(x_0) = p(x_J) * prod_{n < J} p(d_n | x_{n+1}),

learn the coarsest level `x_J` and, at every finer level, the conditional
law of the details given the coarse map with a score-based diffusion model
whose denoiser takes `x_{n+1}` as an extra input.  Sampling goes top-down:
draw `x_J`, then `d_{J-1} | x_J`, rebuild `x_{J-1}`, and so on.  Two
properties make this work where a single-scale diffusion model struggles:

- whitened conditional details are close to white noise, so every
  conditional score is well conditioned and a few tens of denoiser steps
  suffice (Guth et al.);
- the conditional law of `d_n` given `x_{n+1}` is local: a denoiser with
  a receptive field of a few tens of pixels of level `n` is enough, because
  the long-range dependence is carried by `x_{n+1}` (Kadkhodaie et al.).
  Local models need far less training data than global ones, which is the
  only reason this can be trained on one sky (section 5).

## 3. Representation

### Pyramid and Haar details

The NESTED pyramid of STL-Dev-MPI: `x_{n+1}` is the mean of the 4 children
of `x_n` (`tools/resample_nested.py down`).  Its details `d_n = x_n -
up(x_{n+1})` have 3 degrees of freedom per parent; the model works with the
orthonormal Haar coordinates on the parent grid (`diffusion/healpix_faces.py`):

    hx = (c1 + c3 - c0 - c2) / 2,  hy = (c2 + c3 - c0 - c1) / 2,  hd = (c0 + c3 - c1 - c2) / 2

(`c_k` the NESTED child `k`), so the model input and output live on one grid
of `12 nside_{n+1}^2` pixels with 3 channels, the constraint "children
average to the parent" holds exactly, and `mean(d_n^2) = (hx^2 + hy^2 +
hd^2) / 4` per parent.  Every level is whitened: the coarse map by its
standard deviation, every Haar channel by its RMS over the sky (stored with
the model).

### Face images

HEALPix NESTED is the Morton order of 12 square base faces: the pixel
`(ix, iy, face)` has in-face index `interleave(ix, iy)`.  So a level is 12
square images `nside x nside`, the NESTED parent is the 2 x 2 average
pooling of the image, and any 2D convolutional network applies with no
interpolation and no geometry tables.  Pixel shapes vary across a face (the
lattice is a rhombus lattice whose angles change towards the 3-face
vertices and the poles), which the network sees as a mild
position-dependent anisotropy; statistical isotropy of the field and the
symmetry augmentations make it learn approximately rotation-invariant
conditional laws.

### Boundaries between faces

Every 3 x 3 convolution pads its input by the pixels of the neighbouring
faces (`healpix_faces.pad_nest`), with the base-face adjacency of the
HEALPix C++ library (`T_Healpix_Base::neighbors`: offset of the face frame,
wrap, flips and transposition per face row).  It is exact on the 4 edges of
every face and at the corners where 4 faces meet (equatorial vertices and
the poles, where the corner is the opposite polar face); at the 8 vertices
where only 3 faces meet, the missing corner takes the nearest edge-neighbour
pixel (one duplicated pixel per conv layer).  Tested against
`healpy.get_all_neighbours` at nside 2 to 32 (`diffusion/test_healpix_faces.py`),
and the torch gather against the numpy table (`diffusion/test_model.py`).
The padding costs one gather per conv, negligible next to the convolution.

Alternatives considered: local gnomonic patches with interpolation (break
the exact pyramid and the 4-children structure); graph convolutions on the
HEALPix mesh (DeepSphere: isotropic kernels, slower, no U-Net pooling
mismatch but no standard tooling); patches of NESTED ids at a coarse level
(`kth_neighbourhood` halos, as in the Rust engine: the right choice for an
MPI version, see section 7, but no gain on one GPU).

## 4. Networks

One denoiser per level `n` (`diffusion/model.py`):

- input: the noisy whitened details `[3, n', n']` (`n' = nside_{n+1}`)
  scaled by the EDM `c_in`, concatenated with the whitened coarse map
  `[1, n', n']`; output: the 3 detail channels;
- U-Net with average-pooling down-sampling (the NESTED parent again) and
  nearest up-sampling, 2 residual blocks per resolution, 3 resolutions
  (channels 64-128-128, 5.5 M parameters, dropout 0.1 in every residual
  block), noise level injected by FiLM (Fourier
  features of `ln(sigma) / 4`), per-pixel channel normalisation, no
  attention: the network is local and translation-equivariant, so a model
  trained on crops applies to whole faces (theoretical receptive-field
  radius 41 parent pixels with 2 resolutions, 89 with 3; the effective one
  is much smaller);
- EDM preconditioning and loss (Karras et al. 2022) with `sigma_data = 1`
  (whitened details), `ln sigma ~ N(-1.2, 1.2)`; the loss equals 1 for the
  white-Gaussian denoiser `D = x / (1 + sigma^2)`, so the training curve
  reads directly as the information the coarse map and the non-Gaussianity
  add;
- sampling: EDM Heun sampler, 32 steps (63 denoiser evaluations), on the 12
  faces of a sample at once with HEALPix padding, fp16 autocast on the T4
  (bf16 on Ampere and later).

Coarsest level: a Gaussian random field with the angular power of the
target (healpy `synfast` at 8 times the coarse nside, then averaged down,
so that the NESTED means carry the pixel window and the power above
`3 nside`; nside 16 in the prototype).  A learned unconditional model of
the coarsest map needs many independent skies; with one realisation it
would memorise it (section 5).  This is an approximation: the gaussianised
target averaged to nside 16 has a skewness of -0.17 and an excess kurtosis
of -0.06, while GRFs at nside 16 give 0.00 +- 0.06 and 0.00 +- 0.09 (200
draws): the large-scale skewness of the target is a 2.6 sigma feature that
the coarse GRF does not reproduce (the gaussianisation at nside 256 pushes
the large-scale averages to negative skewness).

## 5. Training data

### One realisation

Only Takahashi r000 at zs1 is available.  Training uses ergodicity:

- random crops of the face images of the level (64 parent pixels at most,
  the whole face at the coarse levels), taken from faces padded by their
  neighbours so that crops straddle face edges;
- the 8 symmetries of the square on every crop: the transposition
  `ix <-> iy` is the mirror of the sky about the face's central meridian,
  exact on the HEALPix grid of every face; the half-turn is the point
  reflection through the face centre, exact for the equatorial faces; the
  other symmetries swap the north-east and north-west lattice directions,
  which are close to, not exactly, equivalent on the sphere (the field
  itself is statistically isotropic and parity-invariant);
- faces 5 (equatorial) and 10 (south polar) held out: the validation loss
  on them, against the loss on training faces with the same noise, measures
  over-fitting, and conditional samples on them measure memorisation.

The HEALPix grid has exact symmetries beyond these (rotations by 90 degrees
about the polar axis permute the faces, the north-south mirror), but they
only permute crops that the random crops already draw.

### What one realisation limits

- The coarse levels: nside 8 has 768 pixels, nside 16 has 3072; a
  conditional model there sees a few hundred independent patches and
  over-fits (section 8.1); the coarsest map is therefore Gaussian, and the
  large-scale non-Gaussianity (skewness -0.17 at nside 16, section 4) is
  not learned.
- The low multipoles: the GRF uses the power of r000 itself, so `C_l` at
  `l < 20` carries the cosmic variance of that one sky (40 % at `l = 4`)
  and the samples scatter around it, not around the ensemble mean.
- The tails: at the finest level of nside 256 there are 196 608 parent
  pixels, about 10^4 independent receptive fields.  The bulk of the
  conditional law is well sampled, the rare peaks (massive haloes) appear a
  handful of times: the extreme tail of the pixel histogram and the high-
  order moments are the least constrained part of the model.
- No external check of the sample variance: with one sky we can test that
  samples are independent and match the target's statistics, not that their
  scatter matches the scatter of the true ensemble.  Independent
  realisations are needed both to train the coarse levels and to validate
  ensemble statistics.

### More realisations

The Takahashi et al. (2017) full-sky suite has 108 realisations (r000 to
r107, nside 4096 binary files of about 3 GB per source redshift), public on
the Hirosaki server.  Downloaded on hpc or datarmor directly (never through
the PC), averaged to nside 2048 / 256 and gaussianised with the r000 table,
they would turn the per-level crops into 108 times more independent
patches, make a learned conditional model of the levels down to nside 16
possible, and give the reference scatter for the sample-variance test.
Status: question asked to the user (2026-10-24).

## 6. Evaluation

`diffusion/evaluate.py` (target, then ensembles of maps), as for the STL
syntheses:

- variance of the pyramid details per level over the target's;
- `C_l` from healpy `anafast` in logarithmic bands (two per octave) over
  the target's; the ratio of two independent skies scatters by
  `2 / sqrt(N_modes)` per band (90 % at `l = 2`, 10 % at `l = 20`, 3 % at
  `l = 64`), so the 10 % criterion applies above `l ~ 20`;
- pixel histogram of the finest map;
- skewness and excess kurtosis of `x_n` and of `d_n` per level;
- independence: correlation of the details of every level between members
  of an ensemble and with the target.

## 7. Compute and memory

Measured on garoupe (Tesla T4, fp16 autocast, PyTorch 2.7.1 + CUDA 11.8,
the newest build that runs on its 470 driver), then extrapolated.

Training cost does not depend on the nside of the sky, only on the crop
size and the number of steps.  U-Net 64-128-128, 5.5 M parameters: crops of
64 x 64 parent pixels, batch 32, 0.27 s per step (about 6 TFLOP/s), 12 000
steps in 55 minutes; crops of 16 x 16, batch 64: 0.11 s per step (launch
overhead dominates).  GPU memory: 5.2 GB for the 64 x 64 crops.  Every level
is an independent job: the 5 levels of nside 256 train on the 3 T4s of
garoupe in about one hour (an H200 should be several times faster; not
measured).

Sampling cost is the denoiser on the whole parent grid, 63 evaluations per
level.  Measured on one T4 for 8 samples of nside 256: 73 s for level 256,
22 s for level 128, 13 s for level 64, 15 s for level 32 (small grids are
launch-bound), so 9 s per sample for the finest level and 12 to 15 s per
sample in all.  The current sampler runs the 12 faces of a sample at once.
Activations of the U-Net at full resolution: `64 n'^2` values per tensor
for a face of `n' x n'` parent pixels, about 15 such tensors alive in
inference (skips included).

| nside | finest parent grid | per face (n') | activations, one face, fp16 | finest level, one T4 (extrapolated) |
|---|---|---|---|---|
| 256 | 196 608 | 128 | ~30 MB | 9 s per sample (measured) |
| 2048 | 12.6 M | 1024 | ~2 GB | ~10 min per sample |

The finest level is 3/4 of the work.  At nside 2048 the 12 faces at once
need ~24 GB of activations: the sampler must go face by face with a halo
(not implemented yet).  The network is local, so a face can also be cut
into tiles with a halo of the receptive field, exactly the block + halo
scheme of the Rust engine; the maps themselves take `12 * 2048^2 * 8 B =
400 MB` per level.  About 13 minutes per nside 2048 sample on one T4, so
the 3 T4s of garoupe give ~14 samples per hour; an H200 should be several
times faster (not measured).  An MPI version follows the STL-Dev-MPI
decomposition: contiguous NESTED ranges (whole faces or quarter faces),
halo of the receptive field exchanged once per denoiser evaluation, no
global reduction.

## 8. Prototype results

Target: `g256.npy`, the gaussianised Takahashi r000 zs1 map at nside 256
(the target of the STL runs in `~/frz-exp`); its pyramid gives the
nside 64 prototype (levels 16 to 64) and the nside 256 one (levels 16 to
256).  Runs on garoupe: first models in `/tmp/jmdeloui-diffusion/runs/r1`,
final models in `runs/r2`, samples and diagnostics in `eval/r2*`
(`diffusion/run_eval.sh`).

### 8.1 Over-fitting of one sky

The EDM loss on the held-out faces against the training faces (same crops,
noise levels and noise at every evaluation) shows that every level
over-fits with the first network (64-128-128, 5.5 M parameters, no
dropout):

| level (parent pixels in training) | best held-out loss (step) | training loss then | last evaluation (train / held-out) |
|---|---|---|---|
| 16 (640) | 0.932 (1 000) | 0.886 | step 5 000: 0.585 / 0.997 |
| 32 (2 560) | 0.883 (2 000) | 0.556 | step 12 000: 0.230 / 1.497 |
| 128 (40 960) | 0.867 (4 000, still decreasing) | 0.752 | stopped at 4 000 |
| 256 (163 840) | 0.906 (3 500) | 0.914 | step 6 500: 0.890 / 0.913 |

(1 is the loss of the white-Gaussian denoiser.)  The coarse levels learn
their training faces by heart within a few thousand steps; even the finest
level, with 164 k parent pixels and the 8 symmetries, starts to memorise
after about 4 000 steps of 32 crops.  A smaller network does not help: at
level 128, 32-64-64 (1.5 M parameters) with dropout 0.1 reaches only 0.899
on the held-out faces at step 6 500 (0.876 on the training faces), worse
than the large network at step 4 000.  The models used below keep the large
network, add dropout 0.1, and keep the EMA weights of the lowest held-out
loss (early stopping on faces 5 and 10); level 16 is not learned (the
coarsest map is a GRF at nside 16).  The gap between training and held-out
losses at the selected step is the memorisation that a single realisation
allows; section 8.3 measures it on samples.

Models of `runs/r2` (64-128-128, dropout 0.1, early stopping): best
held-out losses 0.838 (level 32, step 4 000 of 4 000; training faces
0.544), 0.793 (level 64, step 6 000 of 6 000; 0.593), 0.859 (level 128,
step 8 000 of 8 000; 0.768) and 0.905 (level 256, step 4 500 of 8 000;
0.914).  With dropout the held-out loss of levels 32 to 128 still
decreased at the end of the runs.

Longer runs of the coarse levels (`runs/r3`: level 32, 12 000 steps, best
held-out loss 0.743 at step 8 500; level 64, 16 000 steps, 0.686 at the
end) lower the held-out loss but make the samples worse: with the chain
started at nside 16, the detail variance of levels 32 and 64 drops to 0.54
and 0.67 (0.62 and 0.70 with the r2 models), the `C_l` lose up to 48 %,
and draws conditioned on the true parent become nearly deterministic (two
draws correlate at 0.91 on the training faces, 0.66 on the held-out
ones).  At these levels the held-out faces share their edges with the
training crops, and the loss rewards learning the conditional mean of this
one sky: more training is more memorisation, not a better conditional law.
These levels lack data, not training.

### 8.2 Samples at nside 256 and 64

Ensembles of 8 samples, Heun 32 steps, against the target and the STL
syntheses of the same target (one map each: `B_none`, `B_128` = coarse
levels frozen at nside 128, `B_lp10` = with the level-power term of weight
10).  Detail variance per level over the target's (fine nside 256, 128, 64,
32, 16), largest deviation of the `C_l` band power for `l >= 16` (where one
sky scatters by less than 12 % per band), excess kurtosis of the details at
levels 256 / 128 / 64, small-scale anisotropy (`tools/compare_syntheses.py`,
1 = isotropic), and mean |correlation| of the finest details between
members:

| run | coarsest map | learned levels | detail variance / target (256, 128, 64, 32, 16) | max \|C_l / target - 1\|, l >= 16 | detail kurtosis 256 / 128 / 64 | anisotropy | corr. members |
|---|---|---|---|---|---|---|---|
| target | | | 1 | 0 | 0.37 / 0.27 / 0.22 | 1.022 +- 0.034 | |
| diffusion | GRF nside 16 | 32 to 256 | 0.97, 0.85, 0.70, 0.62, 1.00 | 0.39 | 0.37 / 0.12 / 0.11 | 1.056 +- 0.042 | 0.001 |
| diffusion, churn 40 | GRF nside 16 | 32 to 256 | 1.04, 0.89, 0.71, 0.61, 1.00 | 0.40 | 0.33 / 0.12 / 0.12 | 1.066 +- 0.052 | 0.001 |
| diffusion | GRF nside 32 | 64 to 256 | 0.95, 0.88, 0.76, 1.00, 1.01 | 0.24 | 0.38 / 0.14 / 0.11 | 1.084 +- 0.041 | 0.001 |
| **diffusion** | **GRF nside 64** | **128, 256** | **0.97, 0.93, 0.99, 1.00, 1.01** | **0.091** | **0.36 / 0.16 / 0.02** | **1.052 +- 0.028** | **0.001** |
| **hybrid** | **STL `B_lp10` at nside 64** | **128, 256** | **0.98, 0.93, 1.00, 1.00, 1.00** | **0.091** | **0.35 / 0.14 / 0.28** | **1.018 +- 0.047** | 0.097 |
| hybrid | STL `B_none` at nside 64 | 128, 256 | 1.00, 0.96, 1.14, 1.05, 1.00 | 0.14 | 0.33 / 0.15 / 0.28 | 1.021 +- 0.050 | 0.111 |
| hybrid | STL `B_none` at nside 128 | 256 | 1.19, 1.36, 1.14, 1.05, 1.00 | 0.39 | 0.23 / 0.15 / 0.28 | 1.083 +- 0.053 | 0.203 |
| STL `B_none` | | | 0.58, 1.36, 1.14, 1.05, 1.00 | 0.10 | 0.07 / 0.15 / 0.28 | 1.149 +- 0.057 | |
| STL `B_128` | | | 0.76, 1.26, 1.06, 1.03, 0.99 | 0.13 | 0.53 / 0.01 / 0.27 | 1.138 +- 0.049 | |
| STL `B_lp10` | | | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.08 | 0.13 / 0.25 / 0.28 | 1.118 +- 0.052 | |

Figures: `docs/figures/diffusion_eval256_grf16.png`, `..._grf64.png`,
`..._stl64.png` (detail variance, `C_l` ratio with the one-sky scatter,
histogram, skewness and kurtosis per level).

- The finest level is right.  Its detail variance is 0.97 to 1.00 of the
  target's (STL without the level-power term: 0.58), its skewness and
  kurtosis match (0.25 / 0.36 against 0.23 / 0.37; STL: 0.04 to 0.09 /
  0.07 to 0.53), and the small-scale texture is isotropic (1.02 to 1.05
  against 1.12 to 1.15 for STL: the STL syntheses keep an east-west
  elongation of the smallest structures that the diffusion samples do not
  have).  The `C_l` above `l = 362` are within 7 %.
- The learned coarse levels are under-dispersed.  With the chain started
  at nside 16, levels 32 and 64 hold only 0.62 and 0.70 of the target's
  detail variance, and the `C_l` lose 20 to 40 % between `l = 45` and
  `l = 181`; the stochastic sampler does not change it (churn 40).
  Starting from a GRF at nside 64 and learning only levels 128 and 256
  meets the prototype criterion (per-level variance and `C_l` within 10 %)
  with independent samples, but the GRF lacks the level-64 kurtosis (0.02
  against 0.22) and the large-scale skewness.
- STL for the large scales, diffusion for the fine ones.  Conditioning the
  diffusion levels 128 and 256 on the STL synthesis with the level-power
  term (`B_lp10`, averaged to nside 64) keeps the criterion (variance 0.93
  to 1.00, `C_l` within 9.1 %) and the non-Gaussian statistics of both: the
  large-scale skewness and level-64 kurtosis of STL, the finest-level
  kurtosis and isotropy of the diffusion model.  The members share their
  coarse map (one STL synthesis), hence the 0.1 correlation of their
  finest details; an ensemble of STL syntheses (`--ensemble`) gives
  independent members.  Errors of the coarse map propagate and grow:
  `B_none` at nside 128, with 1.36 times the target's level-128 power,
  drives level 256 to 1.19.
- Every pair of samples is independent of the target (|correlation| of the
  details at most 0.024, at nside 16 where 3072 pixels give a noise of
  0.018) and, with a GRF start, of each other.

### 8.3 Memorisation and calibration

`diffusion/memorisation.py`: 4 draws of every level conditioned on the
target's own parent map, correlation with the target's details (`r_truth`)
and between draws (`r_pair`), variance over the target's:

| level | training faces: r_truth / r_pair / variance | held-out faces 5, 10: r_truth / r_pair / variance |
|---|---|---|
| 32 | 0.77 / 0.73 / 0.77 | 0.42 / 0.38 / 0.82 |
| 64 | 0.74 / 0.69 / 0.89 | 0.35 / 0.31 / 0.94 |
| 128 | 0.45 / 0.41 / 0.96 | 0.28 / 0.26 / 0.98 |
| 256 | 0.19 / 0.18 / 1.00 | 0.18 / 0.18 / 1.02 |

At level 256 the model is calibrated and does not memorise: the same
correlations and variance on training and held-out faces, `r_truth =
r_pair`.  At levels 32 and 64 the draws on training faces are nearly
deterministic (two draws correlate at 0.7) and closer to the truth than to
each other: the model has learned these faces; on new coarse maps it is
under-dispersed (0.77 to 0.94 given the true parent, 0.62 to 0.70 in the
full chain, where the coarse maps differ from anything seen in training).
The held-out faces are not fully independent at the coarse levels: crops
straddle face edges by up to half a face there, so the edges of faces 5
and 10 are seen in training.

### 8.4 Against the target statistics, and as the start of the STL minimisation

The STL statistics of the samples themselves (`stl-synth --niter 0
--init MAP --init-noise 0`, L = 4, ergodic weights, residual in ergodic
sigma, rms per family):

| map | var | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| STL `B_lp10` | 0.32 | 0.39 | 0.31 | 0.48 | 0.37 |
| diffusion, GRF nside 16 (2 members) | 67 to 78 | 67 | 65 | 9 | 17 |
| diffusion, GRF nside 64 (2 members) | 0.5 to 21 | 14 to 18 | 17 to 21 | 9 | 9 |
| hybrid `B_lp10` at 64 + diffusion (2 members) | 10.5 | 13 | 15 | 2.8 | 5.6 |

The full-sky ergodic sigmas are tiny (0.1 to 1 % of the coefficients), so
the few-percent agreement of the diffusion samples is tens of sigma: a
learned model does not reach the precision of the minimisation.  Used as
the initial map of the minimisation (`stl-synth --gpu --weights ergodic
--niter 300 --init SAMPLE --init-noise 0`, the garoupe build of 2026-10-20,
one T4), it does better than either alone:

| run (300 iterations) | residual S1 / S2 / S3 / S4 | detail variance 256 / 128 / 64 | detail skew / kurt 256 | max \|C_l ratio - 1\|, l >= 16 | loss below 0.3 / 0.15 after | final loss |
|---|---|---|---|---|---|---|
| target | | 1 / 1 / 1 | 0.23 / 0.37 | | | |
| **hybrid init, then STL** | **0.35 / 0.27 / 0.45 / 0.34** | **1.01 / 0.96 / 1.02** | **0.24 / 0.36** | **0.068** | **32 / 108 evaluations** | **0.122** |
| control: `B_lp10` at 64 + white noise, then STL | 0.35 / 0.29 / 0.46 / 0.35 | 0.87 / 0.53 / 1.55 | 0.07 / 0.07 | 0.081 | 70 / 167 | 0.127 |
| diffusion GRF 64 init, then STL | 0.63 / 0.74 / 0.69 / 0.49 | 1.02 / 1.00 / 1.02 | 0.24 / 0.36 | 0.074 | 266 / - | 0.258 |
| STL `B_lp10` (reference) | 0.39 / 0.31 / 0.48 / 0.37 | 1.00 / 1.00 / 1.00 | 0.05 / 0.13 | 0.076 | | |

The minimisation brings the statistics below 0.5 sigma, as well as STL
alone, and keeps the small scales that the diffusion model set: the
variance, skewness and kurtosis of the finest details stay those of the
target.  From white noise the same minimisation falls into the usual
degeneracy (level 128 at 0.53 of the target's variance, finest-level
kurtosis 0.07).  The statistics constrain what the diffusion model gets
wrong (a few percent everywhere), the diffusion model sets what the
statistics leave free (the finest level).  The diffusion start also
converges about twice as fast.  Starting from the GRF-64 diffusion sample
(large scales Gaussian) keeps the finest level as well but converges more
slowly (residuals 0.5 to 0.7 sigma, loss 0.258 after the same 300
iterations): the minimisation has to build the large-scale
non-Gaussianity that the STL coarse map brings for free.

### 8.5 Conclusion of the prototype

On one sky, the conditional diffusion model learns the two finest levels
of nside 256 (40 000 and 160 000 training parent pixels) well: right
variance, non-Gaussianity and isotropy, where STL is weakest.  It does not
learn the levels with fewer than ~10 000 parent pixels well enough: those
are better taken from an STL synthesis (with the level-power term).  The
best pipeline is: STL up to nside 64, diffusion for the finer levels, then
the STL minimisation at full resolution from that map.

## 9. Next steps

- More realisations (section 5) before scaling up: they are what would let
  the coarse levels be learned too.
- nside 2048: train levels 512, 1024, 2048 (and 256, 128) on the
  Takahashi map at nside 2048 (12.6 M parent pixels at the finest level,
  64 times more crops than at nside 256, so less over-fitting), condition
  on STL syntheses at nside 128 to 256; sample face by face with a halo
  (section 7).
- Non-Gaussianised kappa: train on the gaussianised map and invert the
  anamorphosis (`tools/gaussianize.py inverse`), or train on kappa itself
  with a per-level whitening (heavier tail, harder).
- An STL ensemble as the coarse maps of the hybrid, for independent
  members, each polished by the minimisation (`--ensemble` with `{m}` in
  `--init`).
- Check that the polished hybrid stays put over more iterations (300 here)
  and with the level-power term, on several members.
