# Masked full-sky statistics

Scattering statistics averaged over masks of the sky (for example galactic cuts `|b| > 0°, 20°,
30°, 45°`), with exact gradients, on the CPU or on a GPU, in one sweep for several masks. Branch
`claude/masked-stats` (from `claude/disc-cross-stats`), bridge version `2026-11-05-v5`, Python
`stl_port.native` `2026-11-05-v4`.

## 1. Definition

The fields are those of the full sky: wavelets `z_n`, moduli, stored decimated moduli `m[d][n]`,
the pyramid. Only the averages change. A coefficient is a pixel mean of its terms at its native
level `n`. With a mask, that mean is taken over the pixels of the mask at level `n` instead of the
whole sky:

    S_k = (1 / |M_k,n|) sum_{p in M_k,n} f_p,    V_k = (1 / |M_k,n|) sum_{p in M_k,n} |f_p|^2 - |S_k|^2

- `M_k,0` is the set of NESTED pixels of mask `k` at the full resolution.
- `M_k,n+1` holds the parents of `M_k,n`: a parent is kept when at least one of its children is.

This is the definition of the core of a disc patch (`docs/disc_cross_statistics.md`, §1.1). A disc
patch whose support is the whole sky (`R = 180°`) and whose core is the mask therefore has the
same statistics, and it serves as the reference in the tests. The terms near the mask edge read
pixels outside it (the wavelet is not truncated); the mask only selects which terms are averaged.

Several masks are evaluated in one call: values, variances and cotangents are
`[batch, K, ncoeff]`. The gradient of `Re sum(conj(cot) S)` is the sum over the masks. The full-sky
mask gives the unmasked statistics (to 1e-14). Auto and cross plans (`cross=True`) are both
supported.

## 2. Python API (`stl_port.native.NativeEngine`)

```python
from stl_port.native import NativeEngine
# masks: list of K NESTED id arrays at nside, or indicators [K, npix] (nonzero: in the mask)
eng = NativeEngine(256, 4, batch=B, backend="gpu", masks=masks)   # or eng.set_masks(masks)
values, var = eng.forward(x, moments=True)    # complex [B, K, ncoeff], float [B, K, ncoeff]
grad = eng.vjp(cot)                           # cot complex [B, K, ncoeff] -> [B, npix]
pixels = eng.native_pixels()                  # int [K, ncoeff]: pixels averaged by each coefficient
eng.set_masks(None)                           # back to the whole sky: [B, ncoeff]
# cross=True: forward_cross(a, b) / vjp_cross(cot), same [B, K, ncoeff] layout
```

- Under ergodicity, `var / pixels` estimates the variance of a coefficient for its mask, as for
  the full sky.
- Changing the masks invalidates the forward state: call `forward` again before `vjp`.

C ABI (`crates/stl-bridge`):
- `stlp_set_masks(handle, weights [K * npix], K)`, with `K = 0` for the whole sky;
- `stlp_native_pixels(handle, out [K * ncoeff])`;
- `stlp_forward` / `stlp_vjp` with sizes `B K ncoeff`;
- the info JSON gains `"masks": K`.

## 3. Implementation

- **CPU** (`stl_core::mask::Masks`, `LocalEngine::set_masks`): in every block, the operands of a
  coefficient are computed once. Then, for each mask, the partial sums go over the block pixels in
  the mask, divided by the mask's pixel count at that level. In the reverse sweep, the operand
  cotangents of the masks are added on the block before the field cotangents, so the fields are
  reversed once whatever `K` is.
- **GPU** (`GpuEngine::set_masks`): the indicators of every level are uploaded once (8 bytes per
  pixel per mask and level, `8 K 4N/3` bytes: 42 MB for 5 masks at nside 256). The coefficient
  kernels (`coef_partial`, `coef_reverse`) gain one indicator pointer per item and skip the pixels
  outside the mask. The items and contributions are those of the full sky, repeated per mask.
  Wavelets, moduli, decimations and their adjoints are unchanged and computed once.
- With MPI ranks, the counts are global (the masks are full-sky arrays) and the partial sums are
  reduced as for the full sky.

## 4. Validation

| Test | Bound | Measured |
|---|---|---|
| `stl-core` `tests/mask.rs`: each of 3 masks against a full-sky disc with the mask as core (values, variances, map gradient; C = 1, 2, both S3/S4 definitions, batch 2) | 1e-11 abs + 1e-10 rel; gradient 1e-10 | pass |
| `stl-core` `tests/mask.rs`: the full-sky mask against the unmasked engine | 1e-13 | pass |
| bridge `cpu_masks_match_single_masks`: 3 masks against handles with one mask each (values and variances bit-identical; gradient against the sum), auto and cross | 1e-13 | 8e-16, 1.4e-15 |
| Python `test_masks.py`: shapes, ids against indicators, native pixels, finite differences | 1e-7 (FD) | pass |
| bridge `gpu::device_masks_match_cpu`, Python `test_masked_gpu_matches_cpu`: T4 against the CPU, auto and cross | values 1e-13, variances 1e-12, gradient 1e-11 | GPU_RESULTS |

## 5. Speed

BENCH_RESULTS
