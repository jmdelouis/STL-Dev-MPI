# Native spherical U-Net and its training by scattering losses

The U-Net of Healpix-STL-IA (`stlia.unet.UNet`, `conv="sphere"`, `norm="group"`) in Rust, on
the CPU or one GPU, with its exact reverse pass, Adam and L-BFGS, and its training by the
scattering losses of `stlia/train_denoise.py` (`--real`, frozen noise biases) with everything on
one device: network, statistics engines, observations and noise pools. Only coefficient vectors
(values, variances, cotangents) and the losses go through the host. Branch `claude/native-unet`
(from `claude/spectrum`), bridge version `2026-11-07-v9`, Python `stl_port.native`
`2026-11-07-v8`.

## 1. The network

Exactly the torch reference, with the same parameter names, shapes and order as its
`state_dict`:

- **Convolution** `SphereConv(cin, cout)`, `stl_core::nn::SphereStencil`:
  - For every pixel, the `ksz x ksz` taps of the STL stencil (`local_kernel_grid`, pixel-size
    spacing) are rotated to the pixel and its gauge (`rotation_matrices`).
  - Each tap is the bilinear interpolation of 4 NESTED pixels (`interp_weights_nested`): the
    geometry of the STL wavelet convolution.
  - Then `y = W taps + b`, with `W` `[cout, ksz^2 cin]` tap-major.
- **Gauge**: one fixed gauge (`G = 1`), `phi` (local meridian) or `cosmo`. The network is
  therefore not gauge-equivariant. The stencil of orientation `g` of `G` is available
  (`SphereStencil::new(nside, ksz, gauge, g, G)`) for a gauge-equivariant variant: orientation
  channels with cyclic weight sharing (not implemented).
- **Graph** (levels `k = 0..L-1` at `nside >> k`):
  - encoder: mean of the 4 NESTED children, then `SphereBlock`;
  - decoder: `cat[parent copied into its 4 children, skip_k, cond_k]`, then `SphereBlock`;
  - `SphereBlock` = (conv, GroupNorm(min(8, c)), LeakyReLU(0.2)) twice;
  - head: `1 x 1` conv to one channel, initialised to zero;
  - output: `u = d - scale r(d / scale)`, with `scale = std(d)` (unbiased) unless given.
- **Precision**: `f32` activations and parameters, `f64` maps and statistics. The host
  convolution and every pixel reduction sum in double.
- **Device** (`stl_gpu::nn`):
  - A convolution is, tap by tap, a gather of the 4 interpolated pixels (`nn_gather`) and one
    SGEMM.
  - Its reverse recomputes the taps for the weight cotangent (SGEMM) and sends the tap cotangent
    back through the transposed stencil of the tap (`nn_scatter`, no atomics).
  - GroupNorm, bias and head reductions are per-channel double sums over chunks of 4096 pixels,
    in a fixed order.
  - Every run gives the same result.

## 2. Training

Loss terms of a field (`d`, half missions `d1`, `d2`, the 857 GHz map `T`, `u = f(d)`), each a
reduced chi^2 with ergodic weights (`DenoiseLoss`: `w_c = N_c / V_c / n`):

| term | model | target |
|---|---|---|
| `data` | `Phi(u) + B_data` | `Phi(d)` |
| `l1` | `Phi(u, u) + B_l1` | `Phi(d1, d2)` |
| `l2` | `Phi(u, u) + B_l2` | `Phi(d, u)` (moves with `u`: gradient through both) |
| `l3` | `Phi(T, u) + B_l3` | `Phi(T, d)` |

The frozen biases are means over `M` draws `(n, n1, n2)` of one simulation with a sign, at the
current `u`:
- `B_data = <Phi(u + n)> - Phi(u)`;
- `B_l1 = <Phi(u + n1, u + n2)> - Phi(u, u)`;
- `B_l2 = <Phi(u + n, u)> - Phi(u, u)`;
- `B_l3 = <Phi(T, u + n)> - Phi(T, u)`.

They are recomputed when the driver asks (every 50 steps in `train_denoise.py`).

As in `train_denoise.py` from Healpix-STL-IA a2141c1, the biases are zero on the coefficients
without noise bias, unless `bias_all=True` (`--bias-all`). These are the coefficients whose noise
average is the noise-free statistic, because the noise enters them only linearly:
- `data`: the mean;
- `l1`: S2;
- `l2`: S2, and S3 with the modulus of `u` (`c1 = 1`);
- `l3`: S2, and S3 with the modulus of `T` (`c1 = 0`).

The rule repeats for every mask. The
statistics may be averaged over masks (`set_masks`, as `NativeEngine`).

The targets that do not depend on `u` (`Phi(d)`, `Phi(d1, d2)`, `Phi(T, d)` and their
variances) are computed once per field. A step evaluates `Phi(u)` (one map) and the three
pairs `(d, u)`, `(u, u)`, `(T, u)` (one cross call). The torch pipeline instead pads its
batches to `B` maps and `3 + 3B` pairs.

The optimisers:
- **Adam**: torch's `Adam` (betas 0.9, 0.999, eps 1e-8, same update order) on the device. The
  cosine schedule is the driver's: one learning rate per step.
- **L-BFGS** on the parameters (`lbfgs(fields, options)`): the native strong-Wolfe minimizer of
  `stl_core::lbfgs`, with the options of `stlp_lbfgs`.
  - With frozen biases the objective is deterministic within a round. The driver refreshes the
    biases at the start of a round and chooses the field or fields of the round.
  - The parameter vector (and its gradient) crosses the host once per evaluation.

## 3. Python API (`stl_port.native.NativeUNet`)

```python
from stl_port.native import NativeUNet
net = NativeUNet(256, [32, 64, 96, 128, 128, 128], cond=1, gauge="phi", backend="gpu", device=0,
                 j=7, loss=dict(data=1, l1=1, l2=1, l3=1))
net.load_state_dict(torch_model.state_dict())        # or net.state_dict() -> {name: array}
net.set_cond(log857_standardised[None])              # [cond, npix] float32
u = net.apply(d)                                      # inference, [npix] or [B, npix]; scale = std(d)

net.set_common(T857)                                  # training data, uploaded once
for f in ("Q", "U"):
    net.set_field(f, d[f], d1[f], d2[f], noise=n[f], noise1=n1[f], noise2=n2[f])   # pools [S, npix]
for step in range(1, steps + 1):
    f = ("Q", "U")[step % 2]
    if refresh_due:
        net.estimate_bias(f, sims, signs)             # M draws (simulation index, sign)
    terms = net.step(f, lr0 * 0.5 * (1 + cos(pi * (step - 1) / steps)))   # dict total/data/l1/l2/l3

# L-BFGS by rounds: fixed field and biases within a round
net.estimate_bias("Q", sims, signs)
report = net.lbfgs("Q", {"max_iter": 20, "history_size": 10})   # evaluations, loss_history, terms_history
```

Driver: `python/stl_ensemble/train_native_unet.py`, the `--real --bias frozen` run of
`train_denoise.py`, with its option names and meaning.
- Options: `--optimizer adam` (cosine) or `lbfgs` (rounds, `--lbfgs-fields alternate|joint`),
  `--bias-all`, `--seconds`, and:
  - `--loss data=1,l1=1,l2=1,l3=1,cl=1`. `cl` is the binned half-mission cross spectrum
    `C_b(m d1, m d2)` against `C_b(m u, m u)` under the apodised galactic masks `--cl-masks`
    (`--cl-lmin 10`, `--cl-width 30`, `--cl-lmax 2 nside - 1`, `--cl-apod 5`): pseudo-spectra
    divided by `mean(m^2)`, a reduced chi^2 with the Gaussian variance of the target bins, no
    noise bias (the normalisation of `stl_port.denoise.SpectrumLoss`). By default the trainer
    computes it (`set_cl`, `stlp_train_set_cl`) on the output it already has, with the other
    terms, and adds its cotangent to the same backward pass: one network forward and one
    backward per step, the masked anafast, `C_l` and their VJP on the device (`GpuSht`, `GpuEb`
    for spin 2). `--cl-engine python` keeps the former driver path (`SpectrumLoss` from Python,
    its cotangent through `apply(d, cot=...)`: one more network forward and backward per step).
  - `--stat-masks 0,20,30,45`: the statistics averaged under `|b| > b0` (`set_masks`), every
    coefficient once per mask with its ergodic weight and its own frozen bias.
  - `--cond none|log857|sigma|log857+sigma`: `sigma` is `0.5 log` of the mean over the fields
    and the training simulations of the noise squared, standardised.
  - `--bias-warmup N`: live biases (`live`, `--realizations` distinct simulations per step for
    the data term and for the `l` terms) for the first `N` steps; frozen estimates every
    `--bias-every` steps from step `N + 1`.
- Writes `train.json`, `eval_{Q,U}.npz` (`sky = d`, `d`, `u`) and `model.pt` as `train_denoise.py`
  does. `stlia.unet.load` reads that `model.pt`; its output agrees with the native one to 1e-6.

Other calls:
- `live(field, data_draws, hm_draws, lr=0, backward=False)`: the live-bias loss
  (`train_denoise.py --bias live`), with the models as means over fresh draws and the gradient
  through them. `data_draws` and `hm_draws` are `(sims, signs)` of the data term and of the
  `l` terms; `lr > 0` makes it an Adam step.
- `engine={"cache_bytes": 0}` (or `cache`, `threads`, `conv`, `fields`): the options of the GPU
  statistics engines, as `NativeEngine`.
- `loss(field, backward=False)`: the terms without an update.
- `grads()`, `grads_dict()`, `zero_grad()`, `adam_step(lr)`.
- `apply(d, cot=...)`: parameter cotangents of `sum(cot u)`, for comparisons.

C ABI (`crates/stl-bridge/src/unet.rs`):
- `stlp_unet_new` / `_free` / `_info` / `_get_params` / `_set_params` / `_get_grads` /
  `_zero_grad` / `_adam` / `_set_cond` / `_apply`;
- `stlp_train_set_masks` / `_set_common` / `_set_field` / `_set_joint` / `_front` / `_set_cl` /
  `_bias` / `_loss` / `_live` / `_step` / `_lbfgs`; the terms are `[total, data, l1, l2, l3, cl]`
  (`cl` is 0 without `_set_cl`).

## 4. Validation

Fixtures from `tools/dump_unet_reference.py`: torch `stlia.unet.UNet` with random weights; with
`--double`, the same run in float64.

| check | output (`d - u`) | gradients | Adam step |
|---|---|---|---|
| host vs torch, nside 16, 3 levels, cond, gauges phi and cosmo, batch 2 | 9e-7 | 4e-5 | 7e-9 |
| device (T4) vs torch, same fixtures | 9e-7 | 2e-5 | 7e-9 |
| device (T4) vs torch float32, nside 64, 4 levels (32-128 channels) | 1.5e-6 | 7e-5 | 4e-9 |
| every float32 implementation vs torch float64, nside 64 | 1e-6 | 1.5e-3 | |

At nside 64 the gradients of the host, of the device and of torch float32 are all about 1.5e-3
from the float64 truth: the conditioning of the GroupNorm reverse in single precision.

Training step (`python/stl_ensemble/tests/test_native_unet.py`, nside 16, J = 3):
- Native trainer against the torch pipeline of `train_denoise.py` (`frozen_losses` with
  `StatsFunction` / `CrossStats` over `NativeEngine`, the same draws for the biases):
  - loss terms to 1.3e-7 (biases masked as Healpix-STL-IA a2141c1, and with `bias_all`);
  - parameter gradients to 1.1e-5;
  - an Adam step from the same gradients to 7e-9.
- An L-BFGS round lowers the loss (20.0 to 8.4 in 6 evaluations).
- Live biases (`live`), against `train_denoise.py` `losses()` + `delouis()` with the same draws:
  terms to 1.8e-7, gradients to 7.7e-6.
- On a garoupe T4 with the native side on the GPU: all the above pass. The training-step
  gradients are 3.7e-4 from the torch CPU reference, the float32 conditioning.

Best configuration of Healpix-STL-IA (`python/stl_ensemble/tests/test_native_denoise.py`: terms
`data`, `l1`, `l2`, `l3`, `cl`, statistics under the cuts 0, 20, 30, 45 deg, `cl` under the
apodised cuts 0, 30, 45, 60 deg, two conditioning maps `log857`, `sigma`), against
`train_denoise.py` rebuilt with torch autograd over `NativeEngine`, with the same weights and the
same draws:

| case | terms (max rel.) | gradients (max rel. / norm) |
|---|---|---|
| nside 16 synthetic, CPU, frozen / live | 2.1e-7 / 2.3e-7 | 3.4e-6 / 4.0e-6 |
| nside 16 synthetic, T4, frozen / live | 1.6e-7 / 1.8e-7 | 2.2e-6 / 3.9e-6 |
| nside 64 PR3 Q (FFP10 mc10-15), T4, frozen | 6.5e-7 | 1.6e-3 / 5.9e-4 |
| nside 64 PR3 Q, T4, live (warmup) | 6.2e-7 | 1.6e-3 / 6.0e-4 |

- The `cl` term agrees to 3.8e-7 with `stlia.spectra` on both of its backends (`SpectrumEngine`
  and the torch ring-FFT transform).
- The conditioning maps are identical to those of `train_denoise.py`, and so are the mask ids.
- The nside 64 gradients carry the float32 GroupNorm conditioning described above.
- Reports: `docs/runs/native_denoise/`.
- A 40-step run of the driver with the full configuration (warmup 10, biases every 10 steps) is
  in `smoke_n64.log`. `stlia.unet.load` reads its `model.pt` (`cond = 2`), and the output agrees
  with the native one to 1.4e-6.

## 5. Timings and memory

`python/stl_ensemble/bench_native_unet.py`, one garoupe Tesla T4.
- Setup: one frozen-bias step (data + l1 + l2 + l3) at the paper's settings: cond 1,
  `J = log2(nside) - 1`, 40 bias draws. Median of 5 steps after one.
- Peak: the device memory of the process seen by `nvidia-smi`.
- Torch: the pipeline of `train_denoise.py` (torch U-Net on the GPU, `NativeEngine` statistics
  through the host, batches padded to `B = 4` maps and `3 + 3B` pairs).

| nside, levels, parameters | implementation | step | bias estimate (40 draws) | peak |
|---|---|---|---|---|
| 64, 4, 908 577 | native | 0.30 s | 5.6 s | 1.2 GB |
| 64, 4, 908 577 | torch | 1.05 s | 3.9 s | 3.9 GB |
| 256, 6, 2 388 513 | native, 1 GiB geometry cache (default) | 6.4 s | 57 s | 8.5 GB |
| 256, 6, 2 388 513 | native, cache from free memory (`engine={"cache_bytes": 0}`) | 4.2 s | 38 s | 15.5 GB |
| 256, 6, 2 388 513 | torch | out of memory (15 GB T4) | | |

Breakdown at nside 256 (free-memory cache), per step:
- network forward 0.34 s, forward and reverse 1.0 s;
- statistics forward (auto, then the three cross pairs) about 0.95 s;
- statistics reverse about 2.1 s.

On a T4 the float64 statistics dominate: the T4 has 1/32 of its float32 rate in float64, an
A100 1/2. The torch reference runs at about 3.75 s/step on an A100.

The native bias estimate includes the creation of the statistics engines (they are built at
their first call); torch's excludes it. Bias draws are batched as in `train_denoise.py` (5
maps, 15 pairs per call): the nside 64 estimate is the same, 5.7 s.

Best configuration, nside 256 on a T4 (6 levels, J = 7; seconds per call):

| call | full sky | 4 statistics masks |
|---|---|---|
| frozen step (`step`) | 6.3 | 7.7 |
| live step, B = 4 (warmup) | 14.2 | 20.3 |
| bias estimate, 40 draws | 37.8 | 51.2 |
| `cl` (spectra on the GPU, value and gradient) | 0.16 | |
| extra network passes of `cl` (`apply`, then `apply(cot=...)`) | 1.4 | |

With `--steps 3000 --bias-warmup 500 --bias-every 50`, the best configuration costs about 1.7
times the run without masks, `cl` or warmup (`r256_native`).

## 6. Adam against L-BFGS (nside 64, PR3 maps)

`python/stl_ensemble/train_native_unet.py`; logs in `docs/runs/native_unet_n64/`.
- Data: the PR3 353 GHz Q and U maps at nside 64 with their half missions and the 857 GHz map
  (`/tmp/jmdeloui-stlia/data` on garoupe).
- Noise: the 6 simulations with full- and half-mission noise among 10-49.
- Loss: data + l1 + l2 + l3, frozen biases from 40 draws; U-Net with 4 levels (32 to 128
  channels, cond log I857).
- Budget: 900 s of training on one T4 per run, evaluations not counted.
- Metric: every 60 s, the frozen-bias loss of both fields with biases from one fixed set of
  draws at the current `u` (the same draws for every run).

The three optimisers:
- **Adam**: lr 1e-3, cosine over 3000 steps; one field per step, alternating; biases refreshed
  every 50 steps. 2340 steps in 900 s (0.38 s/step).
- **L-BFGS, alternate**: rounds of 20 iterations (history 10) on one field, alternating Q and
  U, biases refreshed at the start of each round. 77 rounds, about 22 evaluations each.
- **L-BFGS, joint**: the same rounds on the sum of both fields (`--lbfgs-fields joint`). 44
  rounds.

| training time | Adam | L-BFGS alternate | L-BFGS joint |
|---|---|---|---|
| 0 s | 0.170 | 0.170 | 0.170 |
| about 75 s | 0.126 (60 s) | 0.151 (71 s) | 0.092 (79 s) |
| about 150 s | 0.122 (120 s) | 0.158 (135 s) | 0.094 (157 s) |
| about 300 s | 0.076 (301 s) | 0.122 (331 s) | 0.125 (284 s) |
| about 600 s | 0.115 (602 s) | 0.097 (593 s) | 0.147 (622 s) |
| 900 s | 0.108 | 0.124 | 0.119 |
| mean over 450-900 s | 0.106 | 0.118 | 0.129 |

- **Training loss**: every optimiser lowers its own one. L-BFGS joint goes from 0.116 to 0.028
  per round; Adam's per-field loss goes from about 0.03 to 0.009 (Q).
- **Early progress**: joint L-BFGS reaches the plateau first, within about 80 s against 300 s
  for Adam.
- **Alternate L-BFGS**: each round fits its field and degrades the other one. U at the start of
  a round is up to 3 times its last value. It is the most erratic.
- **Plateaus**: after about 5 minutes all three sit at 0.10-0.13. The evaluation itself
  fluctuates by about ±0.03, mostly on U, through the biases estimated at the current `u` with
  draws from only 6 simulations. The plateaus are not distinguishable at this noise.

A sharper comparison needs the 40 training simulations of the paper and longer runs.

## 7. Joint network on (Q, U): `maps=2`, `--pol eb`, `ebqu` and `spin`

Steps 3 to 5 of the Rust-port plan of Healpix-STL-IA (`docs/rust_port.md` there): one network
denoises the two maps of a polarisation pair together.

- **Network.** `NativeUNet(maps=2)` (`stlia.unet.UNet(maps=2)`): two input and two output
  channels, one input scale per map (`--input-scale map`, the default) or one scale for both
  (`--input-scale shared`, torch's `d2.std()`). The `state_dict` is exchanged both ways with
  torch.
- **Joint step.** `set_joint(d2, fields, front, scales, weight)`, then the calls on the name
  `"joint"`. One step runs the network once, sums the terms of both fields (times `weight`),
  and sends the cotangent `[2, npix]` back through the network once.
- **Front end.** Optional `front="eth"`: `chi_E + i chi_B = conj-eth^2 (Q + iU)` at level 0
  (`NativeEngine(spin=2)`'s scalar pyramid), with the statistics masks eroded as there.
  `front(x)` brings the observations and noise pools of (Q, U) to the chi domain once.
- **Debiasing.** The frozen biases of both fields are estimated together from one set of draws.
  `n`, `n1`, `n2` come from the same FFP10 simulation and sign, for every term and both fields.
  `l2` compares `Phi(d, u)` with `<Phi(u + n, u)>`, with the gradient through both sides.
- **`--pol eb`.** The network works on the E and B maps (healpy `map2alm(pol=True, iter=3)`,
  `lmax = 3 nside - 1`, transformed once). The terms are those of E and B, summed, with `cl`
  the scalar spectra of E and B. The output in (Q, U) is `u = d - QU(d_EB - u_EB)`, so the part
  of `d` above the band limit stays in `u`.
- **`--pol spin`.** The network works on (Q, U). The terms are those of chi_E and chi_B,
  averaged as in `train_denoise.py --pol-stats chi`, and `cl` is the EE/BB spectra of (Q, U)
  (`BinnedSpectra(spin=2)`).
- **Caveat.** chi is the ell^2-weighted local E/B, not the harmonic E/B, so these statistics
  differ from those of `r1k_eb`. Its spectrum carries `(l+2)!/(l-2)!`, about `l^4`, so at
  nside 256 the statistics hardly see the largest scales and the network over-removes there
  (`n256_spin`, Healpix-STL-IA): prefer `eb` or `ebqu` at high nside.
- **`--pol ebqu`.** The network works on (Q, U) and its output is `u` in (Q, U). The terms are
  those of the harmonic E and B maps of the output, `EB(u)`, summed, with `cl` the scalar
  spectra of `EB(u)`, as `train_denoise.py --pol-stats eb`. `EB` is the front end
  `front="eb"` (`"eb:K"` for `iter=K`): the operator of `eb_maps` (spin-2 `map2alm` with
  `iter=3`, `lmax = 3 nside - 1`, then the scalar `alm2map` of `a_E` and `a_B`), with its
  exact transpose for the reverse pass. With the GPU backend it runs on the device
  (`stl_gpu::GpuEb`, batched over pairs): the spin-2 synthesis comes from the covariant
  Hessians of the scalar fields `Y(N_l a_E)`, `Y(N_l a_B)`, `N_l = sqrt((l-2)!/(l+2)!)`
  (`Q = -(H_tt - H_pp)[psi_E] + 2 H_tp[psi_B]`, `U = -2 H_tp[psi_E] - (H_tt - H_pp)[psi_B]`),
  through the device derivative transforms, and the analysis is its exact transpose; the CPU
  backend uses the host `stl_core::sht::EbTransform` (spin-weighted recursions), the test
  reference. The front end is linear, so the observations and noise pools are transformed
  once at set-up (`front(x)` on all the draws in one call) and a bias estimate applies it
  once, to `u`. Unlike `eb`, the part of `d` outside the E/B round trip
  (`P(d) = d - QU(EB(d))`, about half of an FFP10 noise map's rms) is not passed through: the
  network output replaces it.
- **Outputs.** `eval_{Q,U}.npz` is always in (Q, U), plus `eval_{E,B}.npz` with `eb` and `ebqu`.
  `model.pt` declares `maps`, `pol`, the layers and the input scales.

### Validation (garoupe, T4 and CPU, nside 16, against torch)

`tests/test_native_joint.py`; the log is in `docs/runs/native_joint/tests_t4.log`.

| Check | Terms | Parameter gradients |
|---|---|---|
| `maps=2` network vs `UNet(maps=2)` (outputs) | 1.1e-6 | 1.8e-5 (norm 8e-7) |
| eth front end vs `NativeEngine(spin=2)`; its transpose | 0; 7e-16 | |
| `eb` joint step vs `train_denoise.py` per-field terms, frozen / live | 2.1e-7 / 1.4e-7 | 5.8e-6 / 4.9e-6 |
| `spin` joint step vs `ChiTerms` + `BinnedSpectra(spin=2)`, frozen / live | 1.6e-7 / 1.2e-7 | 5.3e-6 / 3.6e-6 |
| eb front end vs healpy `eb_maps` (host / device); its transpose | 2.5e-14 / 2.3e-14; 1.7e-13 / 8e-13 | |
| device `GpuEb` vs host `EbTransform`, nside 8-32, forward / transpose / dot test | 8e-14 / 1.3e-13 / 3e-14 | |
| `ebqu` joint step vs `train_denoise.py` per-field terms of `eb_maps(u)`, frozen / live | 1.3e-7 / 1.3e-7 | 2.7e-6 / 3.1e-6 |
| `ebqu` loss of `eb_maps(u)` (float64) vs central difference in `u` (without `l2`) | | 6.5e-8 |

The finite difference leaves out `l2`: its chi2 weights, the variances of `Phi(d, u)`, depend
on `u` but are held fixed in the gradient, as in `train_denoise.py`. A difference over the
float32 network parameters is dominated by rounding, hence the difference in `u`.

The single-map tests (`test_native_unet`, `test_native_denoise`, the Rust U-Net fixtures) are
unchanged.

### Profile (one T4)

One frozen-bias joint step with `data + l1 + l2 + l3 + cl` (`cl` under 4 masks), no
statistics masks, cond log857; seconds (`tests/profile_native_joint.py`,
`docs/runs/native_joint/prof_*.json`):

| | eb n64 | spin n64 | eb n128 | spin n128 |
|---|---|---|---|---|
| whole step | 0.65 | 0.66 | 2.43 | 2.47 |
| statistics forward | 0.12 | 0.12 | 0.53 | 0.53 |
| statistics reverse | 0.28 | 0.29 | 1.13 | 1.16 |
| network forward + reverse | 0.10 | 0.10 | 0.29 | 0.30 |
| cl spectra and gradient (host transforms) | 0.03 | 0.02 | 0.09 | 0.09 |
| eth front end + transpose | | 0.004 | | 0.013 |
| bias estimate (8 draws, both fields) | 0.82 | 0.82 | 3.40 | 3.45 |

- **Against PyTorch.** torch takes 2.15 s (t64_ebdom) and 2.25 s (t64_jchi) per step at
  nside 64, and about 9.6 s (`r1k_eb`) at nside 128: the native step is 3.3 to 4 times faster.
- **Remaining time.** The `cl` term costs, beyond its spectra, one more network forward and one
  forward + reverse (0.1 s at nside 64, 0.4 s at nside 128). The host spin-2 SHT of the EE/BB
  spectra is 4 % of the step, so a device spin-2 SHT is not needed at these sizes.

### nside 64 comparison with the PyTorch runs

All runs: 1000 Adam steps (lr 1e-3, cosine), frozen biases from 8 draws refreshed every 50
steps, cond log857, `data + l1 + l2 + l3 + cl`, cl masks 0, 30, 45, 60°, simulations 10-15,
seed 0. Torch references: `t64_jeb1k` (harmonic E/B statistics in the loop), `t64_ebdom` (E/B
domain, brought back to (Q, U)) and `t64_jchi` (chi statistics, EE/BB cl). Each torch reference
matches a native run with `--input-scale shared` (as torch); the `map` runs differ by the input
scale only. Metrics by `/tmp/jmd_r.py`; full table in `docs/runs/native_joint/comparison_n64.txt`.

| run | s/step | removed rms / noise rms (Q, U) |
|---|---|---|
| torch t64_jeb1k | 2.7 | 0.47, 0.27 |
| torch t64_ebdom | 2.15 | 0.34, 0.31 |
| torch t64_jchi | 2.25 | 0.39, 0.31 |
| native eb, shared scale | 0.68 | 0.35, 0.31 |
| native spin, shared scale | 0.70 | 0.39, 0.45 |
| native eb, per-map scales | 1.31 (two runs on one T4) | 0.37, 0.30 |
| native spin, per-map scales | 1.32 (two runs on one T4) | 0.51, 0.38 |

The two per-map runs shared one T4, hence their doubled step time.

`(d - u)` EE over the half-mission noise at |b| > 30°, l in (10, 30), (30, 60), (60, 100),
(100, 128):

| run | 10-30 | 30-60 | 60-100 | 100-128 |
|---|---|---|---|---|
| torch t64_jeb1k | 0.46 | 0.79 | 1.19 | 0.73 |
| torch t64_ebdom | 0.66 | 0.87 | 0.61 | 0.39 |
| torch t64_jchi | 1.65 | 0.46 | 0.49 | 0.28 |
| native eb, shared | 0.26 | 0.43 | 0.55 | 0.37 |
| native spin, shared | 0.67 | 0.61 | 1.00 | 0.65 |
| native eb, per-map | 0.27 | 0.62 | 0.80 | 0.48 |
| native spin, per-map | 0.55 | 0.60 | 0.89 | 0.57 |

- **Spectra.** Every run keeps the half-mission cross spectrum to about 1 % over |b| > 30°
  (`u/x` 0.99-1.01 in every bin).
- **Large scales.** The native spin runs do not remove more than the noise at l 10-30 (0.67 and
  0.55, 1.16 at |b| > 60°), where torch `t64_jchi` removes 1.65 (3.8 at |b| > 60°).
- **Spread.** These are single trainings, and two trainings differ by a large fraction of the
  noise (Healpix-STL-IA paper). Differences below about 0.2 are within the spread.

### `--pol ebqu` against `--pol eb`

Logs, profiles and `train.json` in `docs/runs/native_ebqu/`.

**Cost per step** (one T4, `tests/profile_native_joint.py`: frozen biases, `data + l1 + l2 + l3
+ cl`, channels 32-128, no statistics masks), seconds:

| | eb n128 | ebqu n128 | eb n256 | ebqu n256 |
|---|---|---|---|---|
| whole step | 2.42 | 3.22 | 13.4 | 16.4 |
| E/B front end + transpose (device) | | 0.20 + 0.19 | | 0.71 + 0.67 |
| cl spectra and gradient | 0.10 | 0.49 | 0.32 | 1.71 |
| bias estimate (8 draws, both fields) | 3.40 | 3.60 | 19.6 | 20.6 |
| set-up transforms of the data and pools | 19.3 (healpy) | 3.3 (device) | 46.7 | 11.2 |

The `cl` term of the driver applies the front end and its transpose once more (from Python):
moving `cl` into the trainer would save one front pass per step. The T4 runs float64 at 1/32
of its float32 rate, so the transforms weigh less on an A100.

**nside 64** (1000 Adam steps, the options of `n256_eb_c16`: warmup 200 steps of live biases,
40 draws every 50 steps, stat masks 0, 20, 30, 45°, cl masks 0, 30, 45, 60°, cond
log857+sigma, simulations 10-49, channels 16-64, per-map input scales, seed 0). `d - u` over
the noise rms, split into the E/B round trip `QU(EB(d - u))` and `P(d - u)`, `P(x) = x -
QU(EB(x))` (`P` holds 0.43 and 0.39 of the rms of an FFP10 noise map in Q and U):

| run | d - u (Q, U) | inside E/B | P part | eval err/noise E, B |
|---|---|---|---|---|
| eb | 0.494, 0.424 | 0.494, 0.423 | 0.010, 0.011 | 0.545, 0.438 |
| ebqu | 0.484, 0.421 | 0.472, 0.416 | 0.106, 0.067 | 0.489, 0.446 |

With `ebqu` the network is free to change `P(u)`, which `eb` passes through; the loss does not
constrain `P(u)` (its terms read `EB(u)`), so how much of `P(d)` the output removes depends on
the network, and at nside 64 it removes a small part of it (0.11 and 0.07 of the noise rms).

### The cl term in the trainer (`set_cl`)

The trainer computes `cl` with the other terms (`stlp_train_set_cl`): the masked anafast of the
output after the front end (spin 0: the E and B maps of `eb` and `ebqu`, the field of `qu`,
both maps in one batch of transforms; spin 2: the EE and BB of the `(Q, U)` output of `spin`),
`C_l`, binning, chi^2 and the VJP on the device, the map cotangent added before the front end's
transpose (spin 0) or after it (spin 2). A step runs one network forward and one backward
instead of two; the live warm-up and the frozen steps alike. Logs and profiles in
`docs/runs/native_cl/`.

**Against the Python path** (`tests/test_native_cl.py`, nside 16, both backends; same
parameters, biases and draws; the Python path is `SpectrumLoss` with an extra network pass):

| | cl value (rel) | total (rel) | parameter gradients, max rel (T4 / CPU) |
|---|---|---|---|
| qu, frozen and live | 2e-15 | 2e-16 | 1.1e-7 / 2.1e-6 |
| eb | 2e-15 | 3e-16 | 8.9e-8 / 4.9e-6 |
| ebqu | 2e-15 | 1e-16 | 9.9e-8 / 3.0e-6 |
| spin | 3e-15 | 2e-16 | 3.5e-8 / 3.0e-6 |

The gradients differ by float32 accumulation only (two network backward passes against one;
the host network sums in another order). The device spin-2 analysis without iteration and the
spin-2 synthesis match the host to 1e-13 (`spin2_analysis_matches_host`, lmax below `3
nside - 1`). `test_native_joint`, `test_native_unet` and `test_native_denoise` are unchanged.

**Cost per step** (one T4, frozen biases, `data + l1 + l2 + l3 + cl`, cl masks 0, 30, 45,
60°), seconds:

| | eb n128 | ebqu n128 | eb n256 | ebqu n256 |
|---|---|---|---|---|
| step, cl from Python | 2.41 | 3.22 | 13.55 | 16.45 |
| step, cl in the trainer | 2.01 | 2.40 | 12.00 | 13.36 |
| cl in the trainer (spectra and VJP) | 0.06 | 0.06 | 0.22 | 0.14 |

`ebqu` costs 2.40 s against 2.01 s for `eb` at nside 128 and 13.4 s against 12.0 s at nside 256:
the E/B front end and its transpose only.

**nside 64 trajectories** (300 Adam steps, warm-up 100, otherwise the options of
`n256_eb_c16`; seed 0). Both paths are deterministic (two runs of each are identical step by
step). The loss is the same to 6 digits for the first 3 steps; from step 4 the float32
differences of the gradient are amplified by the early Adam steps (`m / sqrt(v)` close to
`sign(g)`), and the trajectories part by a few percent. The same happens to either path when
its cl cotangent is multiplied by `1 + 1e-7` (`pert`, `natpert`), which gives the spread of the
trajectories at step 300:

| pol | run | total (step 300) | cl | eval err/noise E, B, Q, U |
|---|---|---|---|---|
| eb | Python | 0.5244 | 0.0012 | 0.526, 0.415, 0.469, 0.385 |
| eb | Python, `1 + 1e-7` | 0.5242 | 0.0020 | 0.511, 0.412, 0.460, 0.375 |
| eb | trainer | 0.5419 | 0.0033 | 0.524, 0.410, 0.473, 0.370 |
| eb | trainer, `1 + 1e-7` | 0.5201 | 0.0025 | 0.513, 0.402, 0.461, 0.368 |
| ebqu | Python | 0.5146 | 0.0013 | 0.461, 0.434, 0.435, 0.402 |
| ebqu | Python, `1 + 1e-7` | 0.5232 | 0.0020 | 0.464, 0.424, 0.435, 0.395 |
| ebqu | trainer | 0.5326 | 0.0021 | 0.453, 0.414, 0.412, 0.400 |
| ebqu | trainer, `1 + 1e-7` | 0.5255 | 0.0018 | 0.460, 0.442, 0.431, 0.411 |

The two paths lie within the spread that a relative 1e-7 perturbation produces.
