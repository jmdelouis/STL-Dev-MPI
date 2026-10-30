# Native adapter, version 2026-10-28-v1

The adapter calls the pinned `stl-core` and optional `stl-gpu` Rust implementations. It contains no replacement scattering engine. The native coefficient plan is complete `Family::ALL`, one channel, `fewer=false`, scale-to-level map `dg(j)=j`, and the existing Cosmo gauge. No gradient preconditioner or per-level variables are introduced.

`stl-bridge` is a small C ABI `cdylib`. Python loads it with `ctypes`; an environment variable `STL_NATIVE_LIBRARY` can specify its path. The serial CPU reference uses `LocalEngine`. The GPU implementation constructs one `GpuEngine` with the requested immutable batch (one for production streaming; ten only for numerical resident-reference tests). Both Gaussian and physical calls reuse that same engine, geometry and plan. CUDA state is not rebuilt per member.

## Changes to the native engine

- `LocalEngine::reload` reuses its geometry/partition and resets accumulated modulus fields before a new forward. This reset is essential: the CPU forward accumulates into the fields.
- GPU `load` copies into existing map buffers instead of allocating a fresh full pyramid per map.
- GPU `HostForwardState` stores all native forward fields consumed by reverse: maps, stored decimated moduli, complex wavelet fields, and their absolute values. Restore copies into the same device buffers and leaves geometry/cache intact.
- The bridge enforces snapshot identity using its engine owner, complete coefficient plan, trial generation, member, objective, resolution, complete per-level layout lengths, batch, channel count and fixed float64/complex128 dtype. The public adapter has no operation for mutating these structural properties. It rejects wrong-owner, wrong-member, wrong-objective and mismatched-generation states. The caller must assign a new generation for every changed trial ensemble.
- The CPU checkpoint stores maps and stored moduli. CPU reverse reconstructs block-local wavelets from these maps and moduli, as the existing implementation does; there are no omitted persistent wavelet fields in `LocalEngine`.

The coefficient VJP remains: native `reverse` -> per-level cotangents -> native `map_cotangent` (CPU) or `analysis_vjp` (GPU). For complex values the derivative convention is `Re(sum(conj(cotangent)*dS))`.

## L-BFGS

`native_lbfgs` directly invokes `stl_core::lbfgs::minimize`, including its established strong-Wolfe line search. It optimizes the full contiguous ten-member vector in one call. Its callback supplies one scalar and one equally normalized gradient; the adapter introduces no ensemble or block division. Defaults are lr=1, history_size=10, tolerance_grad=1e-12, tolerance_change=1e-15, with explicit max_iter/max_eval configuration. No optimizer state is restarted per member. A whole invocation retains one joint history for its full iteration budget.

A float64 contiguous initial input is updated in place. The callback sees a read-only ephemeral trial view and returns a same-shaped gradient. Python exceptions stop the optimizer through the guarded native call and are reraised. Native panics are contained before returning across the C ABI. Handles are single-caller-thread objects; the wrapper serializes operations with a lock.

## Build and validation

From the patched pinned repository:

```bash
cargo build --release -p stl-bridge
# For CUDA; drivers and NVRTC are loaded at runtime:
cargo build --release -p stl-bridge --features gpu
```

Set `STL_NATIVE_LIBRARY` to the resulting `target/release/libstl_bridge.so`. Run `python tests/native_smoke.py --backend cpu --output native_cpu_report.json` from this package. Run the corresponding `--backend gpu` test on the GH200 using the GPU build. This validates real native values, spatial variances, complete map gradients, forward-state restoration, native VJPs and one joint L-BFGS step. Tests do not certify TRAIN54 scientific reproduction.

No GPU hardware, peak memory or throughput result is implied by a successful build. `NativeEngine.info()` exposes native phase-end device-memory peaks, geometry cache budget/use and accumulated phase timers. Phase-end peaks are not guaranteed to capture every transient allocation; an external device-memory sampler is still needed for complete peak accounting.
