# Differentiable pseudo-spectra (anafast)

Auto and cross angular power spectra of full-sky HEALPix maps, with their gradient with respect to
the maps, on the CPU or on a GPU, through the Python bridge. Several masks in one call, with the
mask convention of the masked statistics (`docs/masked_statistics.md`). Branch `claude/spectrum`
(from `claude/masked-stats`), bridge version `2026-11-06-v6`, Python `stl_port.native`
`2026-11-06-v5`.

## 1. Definition

The spectra are those of `healpy.anafast(map, lmax=lmax, iter=0)` (no ring weights, no
iteration), for NESTED maps:

    a_lm = Omega sum_p x_p conj(Y_lm(p)),                Omega = 4 pi / N_pix
    C_l  = sum_m mult_m |a_lm|^2 / (2 l + 1)             (auto)
    C_l  = sum_m mult_m Re(a_lm conj(b_lm)) / (2 l + 1)  (cross)

with `mult_m = 1` for `m = 0` and 2 otherwise (the `a_lm` of a real field, `m >= 0`).

Masks: `K` weights `w_k` (`[K, npix]`, any finite values: binary or apodised). The maps are
multiplied by each weight before the analysis, `x -> w_k x`, so the output is the pseudo-spectrum of
each masked map, as `anafast(w_k * x)`. It is not divided by `f_sky`; the engine gives
`fsky = mean(w_k^2)` per mask for that. In the cross case both maps are multiplied by the same
weight. Without masks `K = 1` and the mask axis is dropped in Python.

Gradient: the VJP of `sum cot * C_l` with respect to the maps,

    d/da_lm   (auto)  = 2 mult_m cot_l a_lm / (2 l + 1)
    d/da_lm   (cross) = mult_m cot_l b_lm / (2 l + 1),   d/db_lm = mult_m cot_l a_lm / (2 l + 1)
    d/dx      = sum_k w_k Omega Y(c_k / mult)

(the transpose of the analysis, `Sht::analysis_vjp`), summed over the masks in mask order.

## 2. Python API (`stl_port.native.SpectrumEngine`)

```python
from stl_port.native import SpectrumEngine
eng = SpectrumEngine(256, lmax=511, batch=B, backend="gpu", device=0, masks=w)  # w: [K, npix] weights
cl = eng.forward(x, generation=g)           # x [B, npix] NESTED -> C_l [B, K, lmax + 1]
grad = eng.vjp(cot, generation=g)           # cot [B, K, lmax + 1] -> [B, npix]
eng.set_masks(None)                         # whole sky: C_l [B, lmax + 1]
eng.fsky                                    # mean(w^2) per mask, or None
# cross=True: maps [B, 2, npix] (or forward_cross(a, b) / vjp_cross(cot)); C_l = cross spectra only
```

- `lmax` defaults to `3 nside - 1` (healpy's default); at most `4 nside`.
- `forward` / `vjp` take `(generation, member=0, objective="cl")`; `vjp` refuses a state that is
  not the last forward of the same identity, and `set_masks` drops the forward state (as
  `NativeEngine`).
- `masks` may also be a list of NESTED id arrays (weight 1 on the ids), like `NativeEngine`.
- One caller thread per engine. The CPU transforms use `STL_THREADS` host threads (default: every
  CPU).

C ABI (`crates/stl-bridge`, `SpectrumHandle`):
- `stlp_spec_new(json)`: `nside`, `lmax`, `batch`, `cross`, `backend` (`cpu` | `gpu`), `device`;
- `stlp_spec_set_masks(handle, weights [K * npix], K)`, `K = 0` for no mask;
- `stlp_spec_forward(handle, x, n = B C npix, generation, member, objective, out [B K (lmax + 1)])`;
- `stlp_spec_vjp(handle, cot, n = B K (lmax + 1), generation, member, objective, out [B C npix])`;
- `stlp_spec_info` (JSON: version, sizes, backend, masks, `fsky`), `stlp_spec_free`.

## 3. Implementation

- CPU: `stl_core::sht::Sht` (ring FFTs, Legendre recursion on the fly in `l` at fixed `m`, no
  stored tables), one analysis per masked map; `Sht::cross_power` / `cross_power_vjp` are the
  cross additions.
- GPU: the transforms of `stl_gpu::GpuSht` (`analysis_many`, `synthesis_many`, cuBLAS Legendre
  sums in blocks of `m` whose tables take at most `STL_SHT_TABLE_MB`, 1 GiB by default), in groups
  of 16 maps. The masking, the spectra and their cotangents run on the host: the `a_lm` are
  `16 (lmax + 1)(lmax + 2) / 2` bytes per map (2.1 MB at `lmax = 511`).
- Memory: no Legendre table proportional to `N_pix lmax` (the torch SHT of Healpix-STL-IA stores
  2 GB at nside 256).

## 4. Validation

- `crates/stl-bridge/tests/spectrum.rs`: the handle against `Sht` directly (auto and cross, no
  mask and two masks, batch of two, nside 8) to 1e-14; the VJP against central differences
  (relative 1e-8); stale-identity and shape errors. With `--features gpu`: the device against the
  CPU at nside 32, `lmax = 63`, two masks, auto and cross, to 1e-12.
- `python/stl_ensemble/tests/test_spectrum.py`: against `healpy.anafast(..., iter=0)` (healpy
  1.20.1, `~/stl-deps/venv-cpu`), nside 32, `lmax = 63`, auto and cross, no mask and two masks
  (smooth and binary), to 1e-11 relative; gradients against finite differences; with
  `STL_TEST_GPU=1` the GPU against the CPU.

Results: all tests pass on WSL (CPU) and on a garoupe Tesla T4 (`--features gpu`). Device against
CPU: 3e-16 (values) and 7e-16 (gradient) at nside 32; 2e-15 at nside 128 (`lmax = 255`, two masks,
batch of two, auto and cross).

## 5. Timings

`python/stl_ensemble/bench_spectrum.py`, nside 256, `lmax = 511`, batch 1, 5 masks (5 analyses per
forward, 5 syntheses per VJP; 10 each with `--cross`), median of 5 evaluations after one warm-up:

| backend | forward | VJP | setup |
|---|---|---|---|
| T4 (garoupe), auto | 0.099 s | 0.092 s | 15 s |
| T4 (garoupe), cross | 0.243 s | 0.189 s | 15 s |
| garoupe CPU, 12 threads, auto | 0.381 s | 0.320 s | 0.3 s |
| WSL CPU, 20 threads, auto | 0.204 s | 0.178 s | 0.1 s |

The GPU setup is the CUDA context and the NVRTC compilation of the kernels, once per process.
Host transfers (maps up, `a_lm` down, cotangents up, maps down) are included.
