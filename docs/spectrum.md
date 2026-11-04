# Differentiable pseudo-spectra (anafast)

Auto and cross angular power spectra of full-sky HEALPix maps, with their gradient with respect to
the maps, on the CPU or on a GPU, through the Python bridge: spin 0 (`TT`) and spin `s >= 1`
fields given as `(Q, U)` (`EE`, `BB`, `EB`; spin 1 for vector fields such as currents or winds,
spin 2 for polarisation or shear). Several masks in one call, with the mask convention of the
masked statistics (`docs/masked_statistics.md`). Branch `claude/spectrum` (from
`claude/masked-stats`), bridge version `2026-11-06-v7`, Python `stl_port.native` `2026-11-06-v6`.

## 1. Definition

### 1.1 Spin 0

The spectra are those of `healpy.anafast(map, lmax=lmax, iter=0)` (no ring weights, no
iteration), for NESTED maps:

    a_lm = Omega sum_p x_p conj(Y_lm(p)),                Omega = 4 pi / N_pix
    C_l  = sum_m mult_m |a_lm|^2 / (2 l + 1)             (auto)
    C_l  = sum_m mult_m Re(a_lm conj(b_lm)) / (2 l + 1)  (cross)

with `mult_m = 1` for `m = 0` and 2 otherwise (the `a_lm` of a real field, `m >= 0`).

### 1.2 Spin s

A spin-`s` field is the pair of real maps `(Q, U)`, `Q + iU` of spin `s`. Its `E` and `B`
coefficients are those of `healpy.map2alm_spin([Q, U], s, lmax)` (spin 2: those of
`anafast(pol=True, iter=0)`):

    a_{+-s,lm} = Omega sum_p (Q_p +- i U_p) conj(_{+-s}Y_lm(p))
    E_lm = -(a_{s,lm} + a_{-s,lm}) / 2,      B_lm = i (a_{s,lm} - a_{-s,lm}) / 2

with `_{s}lambda_lm = (-1)^s sqrt((2l+1)/4pi) d^l_{m,-s}` and `_{-s}lambda_lm = sqrt((2l+1)/4pi)
d^l_{m,s}` (Wigner `d` functions; this fixes healpy's sign for odd spins); `E_lm = B_lm = 0` for
`l < s`. The components are, with the cross formula above:

- one field: `EE`, `BB`, `EB` (`EB = sum mult Re(E conj B) / (2l+1)`);
- two fields (`cross`): `E1E2`, `B1B2`, `E1B2`, `B1E2`.

### 1.3 Masks and gradient

Masks: `K` weights `w_k` (`[K, npix]`, any finite values: binary or apodised). The maps (`Q` and
`U` alike) are multiplied by each weight before the analysis, `x -> w_k x`, so the output is the
pseudo-spectrum of each masked map, as `anafast(w_k * x)`. It is not divided by `f_sky`; the engine
gives `fsky = mean(w_k^2)` per mask for that. In the cross case both fields are multiplied by the
same weight. Without masks `K = 1` and the mask axis is dropped in Python.

Gradient: the VJP of `sum cot * C_l` with respect to the maps. For a component `(a, b)`:

    d/da_lm = mult_m cot_l b_lm / (2 l + 1),   d/db_lm = mult_m cot_l a_lm / (2 l + 1)

(added when `a = b`), then the transpose of the analysis: spin 0 `Omega Y(c / mult)`
(`Sht::analysis_vjp`), spin `s` `Sht::analysis_spin_vjp`; then `sum_k w_k (...)` in mask order.

## 2. Python API (`stl_port.native.SpectrumEngine`)

```python
from stl_port.native import SpectrumEngine
eng = SpectrumEngine(256, lmax=511, batch=B, backend="gpu", device=0, masks=w)  # w: [K, npix] weights
cl = eng.forward(x, generation=g)           # x [B, npix] NESTED -> C_l [B, K, lmax + 1]
grad = eng.vjp(cot, generation=g)           # cot [B, K, lmax + 1] -> [B, npix]
eng.set_masks(None)                         # whole sky: C_l [B, lmax + 1]
eng.fsky                                    # mean(w^2) per mask, or None
# cross=True: maps [B, 2, npix] (or forward_cross(a, b) / vjp_cross(cot)); C_l = cross spectra only

pol = SpectrumEngine(256, lmax=511, batch=B, spin=2, masks=w)
cl = pol.forward(qu)                        # qu [B, 2, npix] (Q, U) -> [B, K, 3, lmax + 1]: EE, BB, EB
grad = pol.vjp(cot)                         # -> [B, 2, npix]
pol.components                              # ["EE", "BB", "EB"]; cross=True: [B, 2, 2, npix] -> 4 components
```

- `lmax` defaults to `3 nside - 1` (healpy's default); at most `4 nside`; `1 <= spin <= lmax`.
- `forward` / `vjp` take `(generation, member=0, objective="cl")`; `vjp` refuses a state that is
  not the last forward of the same identity, and `set_masks` drops the forward state (as
  `NativeEngine`).
- `masks` may also be a list of NESTED id arrays (weight 1 on the ids), like `NativeEngine`.
- One caller thread per engine. The host transforms use `STL_THREADS` threads (default: every
  CPU).
- Spin `s >= 1` runs on the host transforms with either backend (`info()["transforms"]` says
  where); there is no device spin transform yet.

C ABI (`crates/stl-bridge`, `SpectrumHandle`):
- `stlp_spec_new(json)`: `nside`, `lmax`, `batch`, `cross`, `spin`, `backend` (`cpu` | `gpu`),
  `device`;
- `stlp_spec_set_masks(handle, weights [K * npix], K)`, `K = 0` for no mask;
- `stlp_spec_forward(handle, x, n = B F P npix, generation, member, objective, out [B K C (lmax + 1)])`
  with `F` fields (1, or 2 with `cross`), `P` maps per field (1, or 2 for spin `s`), `C`
  components;
- `stlp_spec_vjp(handle, cot, n = B K C (lmax + 1), generation, member, objective, out [B F P npix])`;
- `stlp_spec_info` (JSON: version, sizes, spin, components, transforms, masks, `fsky`),
  `stlp_spec_free`.

## 3. Implementation

- Spin 0, CPU: `stl_core::sht::Sht` (ring FFTs, Legendre recursion on the fly in `l` at fixed
  `m`, no stored tables), one analysis per masked map; `Sht::cross_power` / `cross_power_vjp`.
- Spin 0, GPU: the transforms of `stl_gpu::GpuSht` (`analysis_many`, `synthesis_many`, cuBLAS
  Legendre sums in blocks of `m` whose tables take at most `STL_SHT_TABLE_MB`, 1 GiB by default),
  in groups of 16 maps. The masking, the spectra and their cotangents run on the host: the `a_lm`
  are `16 (lmax + 1)(lmax + 2) / 2` bytes per map (2.1 MB at `lmax = 511`).
- Spin `s`: `Sht::analysis_spin` / `analysis_spin_vjp`. Per `m` and per ring pair, the two
  Wigner recursions in `l` (`d^l_{m,-s}` and `d^l_{m,s}`, three-term, started at `l = max(m, s)`
  from the closed form, scaled by powers of `2^600` below double range as the spin-0 one) give
  `lambda+- = (_s lambda +- _{-s} lambda) / 2`; the south ring follows from `lambda+(pi - theta) =
  (-1)^(l+m+s) lambda+(theta)`, `lambda-(pi - theta) = -(-1)^(l+m+s) lambda-(theta)`. The
  recursions are shared by every map of the call (members and masks).
- Memory: no Legendre table proportional to `N_pix lmax` (the torch SHT of Healpix-STL-IA stores
  2 GB at nside 256).

## 4. Validation

- `crates/stl-core/src/sht.rs` (`spin_adjoint`): `analysis_spin_vjp` is the transpose of
  `analysis_spin` (1e-12, spins 1 to 3, nside 4 and 8); `E = B = 0` below `l = s`.
- `crates/stl-bridge/tests/spectrum.rs`: the handle against `Sht` directly (auto and cross, no
  mask and two masks, batch of two, nside 8) to 1e-14; the VJP against central differences
  (relative 1e-8); stale-identity and shape errors. With `--features gpu`: the device against the
  CPU at nside 32, `lmax = 63`, two masks, auto and cross, to 1e-12.
- `python/stl_ensemble/tests/test_spectrum.py` (healpy 1.20.1, `~/stl-deps/venv-cpu`):
  - spin 0 against `healpy.anafast(..., iter=0)`, nside 32, `lmax = 63`, auto and cross, no mask
    and two masks (smooth and binary), to 1e-11 relative;
  - spins 1, 2, 3 against `healpy.map2alm_spin` + `alm2cl`, nside 16, `lmax = 40`, two masks,
    auto (`EE, BB, EB`) and cross (4 components), to 1e-12; spin 2 against
    `anafast([0, Q, U], pol=True, iter=0)` (`EE, BB, EB`) to 1e-11;
  - gradients against finite differences (spins 0, 1, 2; auto and cross; with and without
    masks); with `STL_TEST_GPU=1` the GPU against the CPU.

Results: all tests pass on WSL (CPU) and on a garoupe Tesla T4 (`--features gpu`). Device against
CPU: 3e-16 (values) and 7e-16 (gradient) at nside 32; 2e-15 at nside 128 (`lmax = 255`, two masks,
batch of two, auto and cross).

## 5. Timings

### 5.1 Against healpy

`python/stl_ensemble/bench_spectrum_healpy.py`, one map (spin 0) or one `(Q, U)` pair, no mask,
nside 256, `lmax = 511`, WSL (20 CPUs, i7-13800H), `OMP_NUM_THREADS = STL_THREADS`, median of 7
after one warm-up, best of two rounds (healpy's 20-thread times vary by up to 3 times between runs);
the one-thread runs are pinned to one core (`taskset`). healpy has no gradient: its time is the
forward only (`anafast`; spin `s`: `map2alm_spin` + three `alm2cl`). "before" is the one-ring
scalar recursion of the first version (`claude/spectrum`), measured in the same session.

| spin | threads | healpy forward | STL forward | STL forward + VJP | before: forward | before: forward + VJP |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.026 s | 0.022 s | 0.038 s | 0.037 s | 0.069 s |
| 1 | 20 | 0.049 s | 0.034 s | 0.073 s | 0.121 s | 0.245 s |
| 2 | 20 | 0.043 s | 0.030 s | 0.077 s | 0.124 s | 0.239 s |
| 0 | 1 | 0.036 s | 0.073 s | 0.143 s | 0.247 s | 0.474 s |
| 1 | 1 | 0.109 s | 0.140 s | 0.332 s | 0.819 s | 1.592 s |
| 2 | 1 | 0.112 s | 0.140 s | 0.318 s | 0.833 s | 1.619 s |

The host transforms run as libsharp/DUCC do: the Legendre and Wigner recursions on blocks of ring
pairs (16 lanes for spin 0, 8 for spin `s`, vectorised, AVX2 when available), the rings where the
functions of `m` stay negligible skipped, per-lane accumulators for the transposed sums, and real
FFTs (section "Speed" of `crates/stl-core/src/sht.rs`). The results agree with the first version to
2e-15 (relative to the largest value) at nside 16, 64 and 256. On 20 threads the STL forward is now
faster than healpy's (1.2 times for spin 0, 1.4 times for spin `s`); on one thread it is 2 times
slower for spin 0 and 1.3 times for spin `s` (3.4 and 6 times faster than before). The VJP costs
about one more forward (1.3 forwards for spin `s` on one thread).

### 5.2 Masks and GPU

`python/stl_ensemble/bench_spectrum.py`, nside 256, `lmax = 511`, batch 1, 5 masks (5 analyses per
forward, 5 syntheses per VJP; 10 each with `--cross`), median of 5 evaluations after one warm-up:

| backend | forward | VJP | setup |
|---|---|---|---|
| T4 (garoupe), auto | 0.099 s | 0.092 s | 15 s |
| T4 (garoupe), cross | 0.243 s | 0.189 s | 15 s |
| garoupe CPU, 12 threads, auto | 0.381 s | 0.320 s | 0.3 s |
| WSL CPU, 20 threads, auto | 0.097 s | 0.077 s | 0.1 s |

The WSL row is the blocked recursions of 5.1 (0.168 s and 0.148 s before, same session); the garoupe
rows and the spin-`s` times below predate them.
The GPU setup is the CUDA context and the NVRTC compilation of the kernels, once per process.
Host transfers (maps up, `a_lm` down, cotangents up, maps down) are included.

Spin `s` with 5 masks (one `(Q, U)` pair, nside 256, `lmax = 511`), host transforms, one run:
garoupe 12 threads, spin 1 forward 0.42 s, VJP 0.64 s; spin 2 forward 0.42 s, VJP 0.61 s. Those are
the times of a `backend="gpu"` engine, since spin `s` has no device transform yet.
