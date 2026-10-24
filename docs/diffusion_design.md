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
  nearest up-sampling, 2 residual blocks per resolution, 2 or 3 resolutions
  (channels 32-64-64 to 64-128-128), noise level injected by FiLM (Fourier
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
  faces of a sample at once with HEALPix padding, bf16 autocast.

Coarsest level: a Gaussian random field with the angular power of the
target's coarse map (healpy `synfast`, nside 8 by default).  A learned
unconditional model of the coarsest map needs many independent skies; with
one realisation it would memorise it (section 5).  The convergence field
is close to Gaussian at the scales of nside 8 to 16 (skewness of the
gaussianised target averaged to nside 16: -0.17, within the scatter of one
realisation of 3072 pixels).

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
  over-fits; the coarsest map is therefore Gaussian, and the large-scale
  non-Gaussianity of convergence (weak at zs = 1 above a few degrees) is
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
level.  Activations of a U-Net with 64 channels at the full resolution,
per face image of `n' x n'` parent pixels: `64 n'^2` floats per tensor,
about 15 such tensors alive in inference (skips included).

| nside | finest parent grid | per face (n') | activations per face, fp32 | denoiser evaluations, all levels |
|---|---|---|---|---|
| 256 | 196 608 | 128 | ~60 MB | 63 per level, 5 levels |
| 2048 | 12.6 M | 1024 | ~4 GB (one face at a time; bf16 halves it) | 63 per level, 8 levels |

The work is dominated by the finest level (3/4 of the total).  At nside
2048 the finest level is 64 times the nside 256 one; on an H200 it is a
matter of minutes per sample, on the T4s of garoupe tens of minutes (face
by face).  The memory of the sampler is bounded by one face (or one crop
with a halo: the network is local, so a face can be cut into tiles with a
halo of the receptive field, exactly the block + halo scheme of the Rust
engine), and the maps themselves (`12 * 2048^2 * 8 B = 400 MB` per level).
An MPI version follows the STL-Dev-MPI decomposition: contiguous NESTED
ranges (whole faces or quarter faces), halo of the receptive field
exchanged once per denoiser evaluation, no global reduction.

## 8. Prototype results

Filled in section by section as the runs finish.

## 9. Next steps

- More realisations (section 5) before scaling up.
- nside 2048: the same models per level, trained on the Takahashi map at
  nside 2048 (12.6 M parent pixels at the finest level, 64 times more
  crops than at nside 256), on the hpc H200 or the garoupe T4s.
- Non-Gaussianised kappa: train on the gaussianised map and invert the
  anamorphosis (`tools/gaussianize.py inverse`), or train on kappa itself
  with a per-level whitening (heavier tail, harder).
- Hybrid with STL: diffusion for the finest levels, STL statistics as a
  check or as a guidance term.
