# `PatchEngine`: statistics on disc patches

`stl_port.native.PatchEngine` computes the scattering-covariance statistics of a map known only on
a **disc** of the sphere, with their exact VJP, for batches of discs at once, on the CPU (patches
on host threads) or on one GPU. It was built for networks that see patches of the sky (for
example denoisers trained with STL losses, Delouis et al. 2022), where the full-sky engine would
waste most of its work.

The support of a disc is every NESTED pixel of `nside` whose centre lies within `radius` of the
disc centre; coarser levels keep the parents; the wavelet is bound to the support with
renormalised weights and the decimation is restricted to it. With `core < radius`, the
coefficients are means over the core pixels only, the ring between `core` and `radius` acting as
a halo. A disc of radius 180° gives the full-sky statistics. The plan is that of
{doc}`native-engine` (`cross=True`: the cross terms of two maps).

```python
from stl_port.native import PatchEngine

pe = PatchEngine(nside=256, j=4, backend="gpu")
ids = pe.select(centres, radius=10.0, core=4.0)   # NESTED pixels of each disc (list)
values, var = pe.forward(pe.gather(x))            # x [npix] or [B, npix] -> complex [B, ncoeff]
grads = pe.vjp(cotangents)                         # one gradient per patch, shaped like the patches
```

---

## Key parameters

| Argument | Default | Description |
|---|---|---|
| `nside`, `j` | required | resolution and scales |
| `kernel`, `orientations` | 5, 4 | stencil and `L` |
| `cross` | `False` | cross statistics of patch pairs `[2, n_i]` |
| `backend`, `device` | `"cpu"`, 0 | CPU (host threads) or one GPU |
| `threads` | 0 | host threads (0: `STL_THREADS`, else every CPU) |
| `cache` | 256 | disc supports kept on the host |
| `cache_bytes`, `reserve_bytes` | free memory, 2 GiB | GPU disc-geometry cache: cap, and the free memory it leaves (to share the GPU with a network) |

| Method | Does |
|---|---|
| `select(centres, radius, core=None)` | choose the discs: centres as NESTED pixels `[B]` or unit vectors `[B, 3]`, radii in **degrees**; returns the sorted pixel ids of each disc |
| `gather(maps)` | the patches `x[ids[i]]` from full-sky maps |
| `forward(maps, centres=None, radius=None, core=None, moments=True, generation=0)` | `(values [B, ncoeff], variances [B, ncoeff])` |
| `vjp(cotangents, generation=0)` | gradients shaped like `maps` (list, or array when all discs have the same size) |
| `forward_cross(a, b)`, `vjp_cross(cot)` | cross statistics of patch pairs |
| `forward_device(x)`, `vjp_device(cot)` | the same on a CUDA `torch` tensor holding the patches concatenated (no host round trip) |
| `native_pixels` | `[B, ncoeff]` pixels averaged by each coefficient |

Full signatures: {doc}`../api/python`.

---

## Minimal example

```python
import numpy as np
from stl_port.native import PatchEngine

pe = PatchEngine(nside=64, j=4, threads=8)
centres = np.arange(0, 768, 48) * 4**3            # 16 centres on the NESTED pixels of nside 8
ids = pe.select(centres, radius=10.0, core=7.0)
patches = pe.gather(x)                             # x: full-sky NESTED map [npix]
values, var = pe.forward(patches)                  # [16, ncoeff]
grads = pe.vjp(2 * (values - target))              # list of 16 arrays [n_i]
g = np.zeros_like(x)
for i, gi in zip(ids, grads):
    np.add.at(g, i, gi)                            # back to the full sky
```

A `torch.autograd.Function` around `forward` / `vjp`: {doc}`../tutorials/torch_loss`.

---

## Pitfalls and costs

- **Disc sizes differ** with the centre: patches are lists in general; draw centres from a finite
  set (e.g. the pixel centres of a coarse nside) so that their geometry is reused.
- **Caches.** On the host, `cache` disc geometries are kept, keyed by `(centre, R, r)`. On the
  GPU, the disc operators are cached least recently used first out: before a new disc is built,
  cached discs are evicted until `reserve_bytes` stay free on the device; the free memory is read
  at that time, so a network training on the same GPU is accounted for.
- **Geometry dominates** the first use of a disc: 0.06 s for a new disc against 0.2 ms cached
  (CPU, nside 64, 10°); on the GPU about 43 ms per new disc at nside 256 and 91 MB of device
  memory each.
- **Steady-state cost** (T4, nside 256, 240 cross pairs of 10° discs with a 4° core, 2912
  coefficients): forward 0.78 s, VJP 0.40 s per step; CPU forward 0.086 s and VJP 0.096 s for 16
  discs at nside 256 on 20 threads.
- One forward state per engine (as {doc}`native-engine`); radii are in degrees.

Definitions, validation and timings: {doc}`../disc_cross_statistics`.
