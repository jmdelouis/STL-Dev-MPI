# v2 validation report

Version: `2026-10-29-v2-variance-ratio`.

## Executed here

**71 selected CPU-side tests passed.** The command and output are retained in
`verification/local_cpu_tests.txt` and `verification/local_cpu_tests.xml`.

Command (from the package root, with the original v1 source location supplied):

```bash
python -m pytest -q \
  tests/test_variance_ratio_v2.py \
  tests/test_losses.py \
  tests/test_sc_normalization.py \
  tests/test_cache_migration_v2.py \
  tests/test_scope_v2.py
```

Coverage includes:

* Ratios 0.1, 1 and 10 at target variances from 1e-28 to 1, including the
  user's measured physical S2/S3/S4 medians. Losses reproduce `(log ratio)^2`.
* Real and complex coefficient-space variance-only directional derivatives
  with per-coordinate amplitudes down to 1e-14 and amplitude-scaled FD steps.
* Per-coefficient unit-rescaling invariance for the variance loss and cotangent.
* The specified hard-floor behavior, zero synthesized variance, invalid target
  rejection, and a case where a direct ratio would overflow but log differences do not.
* Nonzero variance gradients for the physical-scale case that v1 clamped flat.
* Mean loss and mean cotangent equality against the archived v1 implementation,
  with unchanged physical sigma equalization.
* New relative guard coverage, including Gaussian mean/S0-second entries; no
  silent dropping of nonpositive or nonfinite registered TRAIN54 targets.
* Read-only effective-sigma reporting and plan-order invariance.
* Synthetic cache migration: old manifest intact, all five array files
  byte-identical, new loss/code/cache identity, and successful strict v2 load.
* Rejection of changed target bytes, inconsistent feature tables, altered
  manifest/code identity, wrong native binary, wrong bootstrap seed and existing
  migration destination. No mock native engine is used in these tests. Synthetic
  migration tests isolate the source-verification operation; a separate test
  verifies the actual original 111-file package.
* Source-scope verification: all **56 native workspace files** are unchanged,
  including the GPU engine, bridge and L-BFGS implementation. The feature map,
  harmonic/LUT/PDF/MF modules are byte-identical. Mean-sigma and non-SC loss
  functions are AST-identical, as is the SC mean calculation before its variance term.

All delivered Python sources compile. Both GH200 shell launchers pass `bash -n`.

Environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2. This is not
the user's Python 3.12/GH200 runtime; the same suite is required there.

## Not executed here

No GH200/CUDA hardware, compiled STL bridge, Rust toolchain, healpy or ducc0 was
available in this execution environment. Optional dependency retrieval failed
because network name resolution was unavailable. Therefore the following are
**provided but not claimed as passed here**:

* Native variance-only map VJP finite differences on CPU/GPU.
* The revised full five-block resident/replay/checkpoint comparisons and accepted
  joint L-BFGS step, including the small physical-amplitude fixture.
* Migration of the user's actual TRAIN54 cache (only its diagnostic output is
  available here).
* GH200 performance measurements, map-quality validation or native2048 capacity.

The runner's `validate` stage executes the real native GPU checks and refuses to
record acceptance if any fail. It does not reuse v1 acceptance markers. Its
`migrate-v1` stage independently validates the actual saved targets and source
identity before creating a v2 cache. Source/target mismatches stop the workflow.

## Scope boundaries

The approved scientific change is the SC variance regularization. Mean weights,
other losses, B=10, float64, native grids/filters and native joint L-BFGS remain
unchanged. The absolute mean-sigma floor is only diagnosed, not retuned.

The native2048 power-variance upper cutoff is still explicit/unresolved. The
previously documented inclusive6143 endpoint issue is not changed in this
SC-only update; address it before selecting that full-resolution configuration.

v1 reports are archived under `provenance/v1_validation/`; they are historical
evidence and do not certify v2 GPU or scientific correctness.
