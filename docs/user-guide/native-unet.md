# `NativeUNet`: a spherical U-Net trained by scattering losses

`stl_port.native.NativeUNet` is the spherical U-Net of Healpix-STL-IA (`stlia.unet.UNet`,
`conv="sphere"`, `norm="group"`) in Rust, with its exact reverse pass, Adam and L-BFGS, on the CPU
or on one GPU. It is a denoiser `u = d − scale · r(d / scale)` of NESTED maps, trained by the
scattering losses of Healpix-STL-IA's `train_denoise.py` (Delouis et al. 2022): the network, the
statistics engines, the observations and the noise pools all stay on one device, and only
coefficient vectors and loss values cross the host.

- **Network**: the same parameter names, shapes and order as the torch `state_dict`
  (`load_state_dict` / `state_dict` exchange weights with `stlia.unet.UNet`); `float32`
  activations, `float64` maps and statistics; deterministic on the GPU.
- **Training**: the loss terms `data`, `l1`, `l2`, `l3` with frozen noise biases (or live biases
  with their gradient), statistics optionally averaged over galactic masks, one Adam step or one
  L-BFGS round per call.
- **Driver**: `python/stl_ensemble/train_native_unet.py`, the `--real --bias frozen` run of
  `train_denoise.py` with its option names, plus the `cl` term (`stl_port.denoise`).

```python
from stl_port.native import NativeUNet

net = NativeUNet(64, [32, 64, 96, 128], cond=1, gauge="phi", backend="gpu", j=5,
                 loss=dict(data=1, l1=1, l2=1, l3=1))
net.load_state_dict(torch_model.state_dict())     # optional: start from torch weights
net.set_cond(log857[None])                        # [cond, npix] float32, standardised
u = net.apply(d)                                  # inference: [npix] or [B, npix]
```

---

## Key parameters

| Argument | Default | Description |
|---|---|---|
| `nside` | required | resolution of the maps |
| `channels` | required | channels per level, e.g. `[32, 64, 96, 128, 128, 128]`; its length is the number of levels |
| `ksz`, `gauge` | 3, `"phi"` | stencil of the convolution (the STL stencil, bilinear taps) and its fixed gauge (`"phi"` or `"cosmo"`) |
| `slope` | 0.2 | LeakyReLU slope |
| `cond` | 0 | number of conditioning maps (`set_cond`), concatenated at every decoder level |
| `scale` | `None` | input scale; default `std(d)` (unbiased) per map |
| `backend`, `device` | `"cpu"`, 0 | network and training on the host or on one GPU |
| `j` | `None` | scales of the statistics of the losses (needed for training) |
| `loss` | `None` | weights of the terms, e.g. `dict(data=1, l1=1, l2=1, l3=1)` |
| `bias_batch` | 5 | maps per call of the bias estimate (pairs: 3 per map) |
| `bias_all` | `False` | estimate the bias of every coefficient, not only those with a noise bias |
| `engine` | `None` | options of the statistics engines (`cache`, `cache_bytes`, `threads`, `conv`, `fields`), as `NativeEngine` |

| Method | Does |
|---|---|
| `apply(d, scale=None, cot=None)` | `u` of `d`; with `cot`, adds the parameter cotangents of `sum(cot · u)` |
| `state_dict()`, `load_state_dict(state)`, `get_params()`, `set_params(w)` | weights as `{name: array}` or one `float32` vector |
| `set_cond(cond)`, `set_masks(masks)` | conditioning maps; statistics averaged over masks (as `NativeEngine`) |
| `set_common(T)`, `set_field(name, d, d1, d2, noise, noise1, noise2)` | upload the 857 GHz template, and per field the data, half missions and noise pools `[S, npix]`, once |
| `estimate_bias(name, sims, signs)` | freeze the noise biases at the current `u`, from draws (simulation, sign) |
| `step(name, lr)` | one Adam step (torch's `Adam`); returns the terms `total`, `data`, `l1`, `l2`, `l3` before the update |
| `lbfgs(names, options)` | one round of the native strong-Wolfe L-BFGS on the frozen-bias loss of one or several fields |
| `live(name, data_draws, hm_draws, lr=0)` | the live-bias loss, the gradient through the draws |
| `loss(name, backward=False)`, `grads()`, `zero_grad()`, `adam_step(lr)` | lower-level pieces |

Full signatures: {doc}`../api/python`.

---

## The loss

Each term is a reduced χ² with ergodic weights between a model and a target, `Phi` the scattering
statistics (auto or cross), `d` the data, `d1`, `d2` its half missions, `T` the 857 GHz map:

| Term | Model | Target |
|---|---|---|
| `data` | `Phi(u) + B_data` | `Phi(d)` |
| `l1` | `Phi(u, u) + B_l1` | `Phi(d1, d2)` |
| `l2` | `Phi(u, u) + B_l2` | `Phi(d, u)` (gradient through both sides) |
| `l3` | `Phi(T, u) + B_l3` | `Phi(T, d)` |

The biases are means over noise draws at the current `u`, e.g. `B_data = <Phi(u + n)> − Phi(u)`.
They are zero on the coefficients the noise enters only linearly (the mean of `data`, S2 of the
`l` terms, S3 with the noise-free modulus), unless `bias_all=True`.

---

## Minimal example

Adam with a cosine schedule, alternating Q and U, biases refreshed every 50 steps:

```python
import math
net.set_common(T857)
for f in ("Q", "U"):
    net.set_field(f, d[f], d1[f], d2[f], noise=n[f], noise1=n1[f], noise2=n2[f])
for step in range(1, steps + 1):
    f = ("Q", "U")[step % 2]
    if step % 50 == 1:
        net.estimate_bias(f, sims, signs)              # e.g. 40 draws
    terms = net.step(f, 1e-3 * 0.5 * (1 + math.cos(math.pi * (step - 1) / steps)))
u = net.apply(d["Q"])
```

L-BFGS by rounds (fixed field and biases within a round):

```python
net.estimate_bias("Q", sims, signs)
report = net.lbfgs("Q", {"max_iter": 20, "history_size": 10})   # evaluations, loss_history, terms_history
```

The full training run, on the PR3 353 GHz maps with the best Healpix-STL-IA configuration
(statistics under the cuts 0, 20, 30, 45°, the `cl` term, conditioning `log857+sigma`, a live-bias
warm-up):

```bash
python python/stl_ensemble/train_native_unet.py --data DIR --out RUN --nside 256 \
    --optimizer adam --steps 3000 --loss data=1,l1=1,l2=1,l3=1,cl=1 \
    --stat-masks 0,20,30,45 --cond log857+sigma --bias-warmup 500 --bias-every 50
```

It writes `train.json`, `eval_{Q,U}.npz` and a `model.pt` that `stlia.unet.load` reads.

---

## Pitfalls and costs

- **Single precision.** The network is `float32`: outputs agree with torch to 1e-6, parameter
  gradients to about 1e-5 at nside 16 and 1.5e-3 at nside 64 (the GroupNorm reverse in single
  precision; torch `float32` has the same error against `float64`).
- **One fixed gauge**: the network is not gauge-equivariant, as the torch reference.
- **Frozen biases** make the objective deterministic within a round: refresh them before each
  L-BFGS round, and every few tens of Adam steps.
- **Memory**: the geometry cache of the statistics engines is 1 GiB by default;
  `engine={"cache_bytes": 0}` sizes it from the free device memory (faster, more memory).
- **Cost** (one T4, frozen-bias step with `data + l1 + l2 + l3`): nside 64, 4 levels: 0.30 s per
  step and 1.2 GB, against 1.05 s and 3.9 GB for the torch pipeline; nside 256, 6 levels: 4.2 to
  6.4 s per step and 8.5 to 15.5 GB, where the torch pipeline runs out of memory. The float64
  statistics dominate on a T4.
- **Adam or L-BFGS**: at nside 64 joint L-BFGS rounds reach the plateau first (about 80 s against
  300 s for Adam); after 5 minutes all optimisers sit at the same level, within the noise of the
  evaluation.

Network, trainer, validation against torch, timings and the Adam / L-BFGS comparison:
{doc}`../native_unet`.
