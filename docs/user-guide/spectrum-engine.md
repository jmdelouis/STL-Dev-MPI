# `SpectrumEngine`: differentiable pseudo-spectra

`stl_port.native.SpectrumEngine` computes angular power spectra of full-sky HEALPix maps with the
conventions of healpy's `anafast` (`iter=0`, no ring weights), and their gradient with respect to
the maps, on the CPU or on a GPU. It covers:

- spin 0 (`TT`), auto and cross spectra;
- spin `s ≥ 1` fields given as `(Q, U)`: `EE`, `BB`, `EB` (spin 1 for vector fields such as
  currents or winds, spin 2 for polarisation or shear), as healpy's `map2alm_spin`;
- `K` masks or apodised weights in one call (pseudo-C_ℓ: the maps are multiplied by each weight;
  not divided by `f_sky`).

Use it next to the scattering statistics in a loss that must also hold the power spectrum.

```python
from stl_port.native import SpectrumEngine

eng = SpectrumEngine(nside=256, lmax=511, batch=1, backend="gpu")
cl = eng.forward(x)          # x [batch, npix] NESTED -> C_l [batch, lmax + 1]
grad = eng.vjp(dL_dcl)       # -> [batch, npix]
```

---

## Key parameters

| Argument | Default | Description |
|---|---|---|
| `nside` | required | resolution |
| `lmax` | `3 nside − 1` | highest multipole (at most `4 nside`) |
| `batch` | 1 | maps per call |
| `backend`, `device` | `"cpu"`, 0 | transforms on the host or on one GPU (spin 0) |
| `cross` | `False` | cross spectra of map pairs `[batch, 2, npix]` |
| `spin` | 0 | 0, or `s ≥ 1` with `(Q, U)` maps `[batch, 2, npix]` |
| `masks` | `None` | `[K, npix]` weights (any finite values) or NESTED id lists; C_ℓ become `[batch, K, ...]` |

| Method / attribute | Does |
|---|---|
| `forward(maps, generation=0, ...)` | C_ℓ `[batch, (K,) (components,) lmax + 1]` |
| `vjp(cotangents, generation=0, ...)` | gradient shaped like `maps` |
| `forward_cross(a, b)`, `vjp_cross(cot)` | cross spectra; `(grad_a, grad_b)` |
| `set_masks(masks)`, `fsky` | change the masks; `mean(w²)` per mask |
| `components` | `["EE", "BB", "EB"]` for spin `s` (4 with `cross`) |

Full signatures: {doc}`../api/python`.

---

## Minimal example

```python
import numpy as np
from stl_port.native import SpectrumEngine

eng = SpectrumEngine(nside=64, batch=1)              # lmax = 191
cl_target = eng.forward(t)
cl = eng.forward(x)
r = np.log(cl[0, 2:]) - np.log(cl_target[0, 2:])     # log-spectrum residual from l = 2
loss = np.sum(r ** 2)
cot = np.zeros_like(cl); cot[0, 2:] = 2 * r / cl[0, 2:]
grad = eng.vjp(cot)[0]                                # d loss / d x
```

With PyTorch, next to the scattering statistics: {doc}`../tutorials/torch_loss`.

---

## Pitfalls and costs

- **Exactness**: the spectra are those of `healpy.anafast(x, lmax, iter=0)` (spin `s`:
  `healpy.map2alm_spin`) to 1e-11; with `cross=True` and spin `s` the components are `E1E2`,
  `B1B2`, `E1B2`, `B1E2`.
- **Pseudo-spectra**: with masks, C_ℓ is that of the weighted map, not corrected for the mask
  (no `f_sky` division, no mode-coupling inversion).
- `vjp` must follow the `forward` of the same identity; `set_masks` drops the forward state; one
  caller thread per engine. Host transforms use `STL_THREADS` threads.
- Spin `s ≥ 1` always runs on the host transforms, even with `backend="gpu"`.
- **Cost** (nside 256, `lmax = 511`): one map, forward 0.035 s and forward + VJP 0.064 s on 20
  threads (healpy's `anafast` 0.021 s, without gradient); five masks, forward 0.099 s and VJP
  0.092 s on a T4 after a 15 s setup.

Definitions, validation against healpy and timings: {doc}`../spectrum`.
