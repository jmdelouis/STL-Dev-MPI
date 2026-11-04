# Native spherical U-Net and its training by scattering losses

The U-Net of Healpix-STL-IA (`stlia.unet.UNet`, `conv="sphere"`, `norm="group"`) in Rust, on
the CPU or one GPU, with its exact reverse pass, Adam and L-BFGS, and its training by the
scattering losses of `stlia/train_denoise.py` (`--real`, frozen noise biases) with everything on
one device: network, statistics engines, observations and noise pools. Only coefficient vectors
(values, variances, cotangents) and the losses go through the host. Branch `claude/native-unet`
(from `claude/spectrum`), bridge version `2026-11-07-v8`, Python `stl_port.native`
`2026-11-07-v7`.

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

They are recomputed when the driver asks (every 50 steps in `train_denoise.py`). The
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
- `stlp_train_set_masks` / `_set_common` / `_set_field` / `_bias` / `_loss` / `_step` /
  `_lbfgs`.

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
  - loss terms to 1.3e-7;
  - parameter gradients to 1.1e-5;
  - an Adam step from the same gradients to 7e-9.
- An L-BFGS round lowers the loss (20.0 to 8.4 in 6 evaluations).
- Live biases (`live`), against `train_denoise.py` `losses()` + `delouis()` with the same draws:
  terms to 1.8e-7, gradients to 7.7e-6.
- On a garoupe T4 with the native side on the GPU: all the above pass. The training-step
  gradients are 3.7e-4 from the torch CPU reference, the float32 conditioning.

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
