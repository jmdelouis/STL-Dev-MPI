# v2 scope and changes

Version: 2026-10-29-v2-variance-ratio.

## Scientific change

Both SC variance penalties operate on `V_syn / V_TRAIN54` with a dimensionless
hard lower bound of 1e-12. The scalar and exact variance cotangent are computed
in float64, with log differences to avoid overflowing the ratio. Existing
complex moments, ddof=1, coefficient-count reduction, physical mean family
reductions, global division by five and all weights remain unchanged.

Any selected nonpositive/nonfinite target variance is rejected, including
Gaussian S0. A zero synthesized variance receives the finite clamped loss and
zero derivative below the hard floor. No data-dependent masking/rescaling,
new optimizer, smoothing, preconditioning or per-member target is introduced.

## Reporting and provenance

* New explicit contract and implementation/cache identities; v1 caches cannot
  be loaded silently by v2.
* Explicit migration verifies the approved v1 implementation, cached target/LUT
  hashes, native binary, plan/configuration, bootstrap and stored feature moments.
  Arrays are copied, never recalculated or overwritten in the old cache.
* New target reports cover both variance sensitivity and unchanged raw/processed
  mean sigmas; full evaluation logs split SC mean and variance contributions.
* Original native engine, C bridge and L-BFGS source are byte-identical. Existing
  calibrated library is reused, avoiding recompilation/cache invalidation.

## Execution/test changes

* Separate GH200 v2 runner adopts the existing Torch-free `.venv`, using a symlink;
  no Python packages are installed or upgraded.
* CPU-only NUMA binding is explicit and recorded, including benchmark children.
  Strict binding remains an explicit alternate policy; there is no silent fallback.
* Unit regressions test the observed physical amplitudes, unit invariance,
  complex variance VJPs, hard-floor behavior and rejected targets.
* Read-only mean-loss equality and original native/other-loss scope are tested.
* Synthetic cache migration and tamper/rejection tests are included.
* Native variance-only map finite differences and complete three-mode accepted
  L-BFGS-step comparisons are required on the GH200, also at small physical amplitudes.

## Deliberately not changed

`stl_port/sc_normalization.py`, native Rust/CUDA sources, native optimizer,
harmonics, LUT, PDF and Minkowski implementations are unchanged. Physical
power/PDF/MF loss functions and mean-sigma function are AST-identical to v1.
The original `reference/` scripts remain untouched.

The full-resolution PS variance cutoff and inclusive6143 endpoint issue have not
been resolved in this SC-only update. No native2048 performance/quality claim is
made. Existing v1 benchmark and GPU validation results do not certify v2.
