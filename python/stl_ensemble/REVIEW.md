# Independent source-to-port review

Review version: 2026-10-28-v1. This is a source review and small numerical-test
record; it does not establish TRAIN54 scientific agreement or GH200 capacity.

## Reviewed authority

- Supplied Step-1 source SHA256:
  `1c507030ab95145c84405ac7e01426f0c9828a402d92d3a56ea7887b3f09b447`.
- Native STL baseline: `f083c4a753a244d1edf69fe4f944fa23fc5cdb76`, plus the
  supplied adapter/checkpoint changes.
- FOSCAT optimizer inspection uses
  [`Synthesis.py` at `17baec97672097d68fd4ef1e337e8ec362ef4326`](https://github.com/pcampeti/FOSCAT/blob/17baec97672097d68fd4ef1e337e8ec362ef4326/src/foscat/Synthesis.py).
  This identifies the inspected source, not the unrecorded historical FOSCAT
  installation that produced earlier maps.

## Optimizer requirement and exact source settings

The implementation keeps one joint L-BFGS optimizer for all ten members.
Streaming and checkpointing only change evaluation scheduling. They never
update one member independently or introduce a stochastic optimizer.

The supplied main script calls
`sy.run(imap, EVAL_FREQUENCY=100, NUM_EPOCHS=8000)`. In the inspected FOSCAT
`Synthesis.run`, lines 304–325 define `factr=10.0` and `NUM_STEP_BIAS=1`.
Lines 430–437 invoke SciPy `fmin_l_bfgs_b` with the complete joint gradient,
`pgtol=1e-32`, `factr=10.0`, and `maxiter=8000` for this caller. No bounds are
passed. `LEARNING_RATE=0.03` and `EPSILON=1e-7` are run-signature defaults but
are not passed to this SciPy optimization call.

SciPy 1.17.0 in the available verification environment gives the remaining
defaults `m=10`, `maxfun=15000`, and `maxls=20`. These are locally inspected
dependency defaults; the user's historical SciPy version was not supplied.
With double precision, `factr=10` corresponds to a relative function-reduction
threshold of approximately `2.22e-15`.

The port reuses the native STL L-BFGS implementation and its strong-Wolfe line
search. Its configured values are learning-rate parameter 1, history size 10,
gradient tolerance `1e-12`, change tolerance `1e-15`, and an explicit requested
iteration count. These stopping criteria are not algebraically equivalent to
SciPy's `factr`/`pgtol`; neither equal iteration counts nor equal history sizes
imply the same trajectory. The native default evaluation limit also differs
from SciPy's default and must appear in run records. Numerical mode comparisons
must use the same native optimizer options, including the evaluation limit.

## Confirmed population and preprocessing contract

- Calibration consumes r000–r053 only. The held-out r054–r107 population does
  not enter LUT, normalizers, support, mean targets, or population variances.
- One ordered draw of ten IDs with replacement controls every bootstrap mean.
  All 54 maps separately determine every unbiased population variance.
- At 128, preprocessing is raw 2048 RING float64 → `hp.ud_grade` to 512 → NEST
  float32 → contiguous 16-child means. Reference maps are not silently
  recentered. At 2048 the source maps are read at their native resolution.
- The GRF initializer reproduces the source RNG ordering: one advancing
  `default_rng`, member-major loop, then m-major packed-alm loop; for m>0 the
  complete real normal draw precedes the complete imaginary draw. It zeros
  C0 and adds the mean of the first bootstrap Gaussian reference, as the
  executable source does. There is no exact-shell rescaling or mean/std match.
- The LUT uses all TRAIN54 pixels in exact quantiles. The temporary memmap
  avoids a second pooled in-memory copy; no sampling/sketching approximation
  is substituted. Physical knots and Gaussian quantile knots retain the
  reference float32 rounding. Native map arithmetic is float64, including
  physical monopole removal and its pullback; this precision difference is
  explicit and is not a bitwise float32 reproduction claim.

## Loss and gradient review

The global objective and gradient are divided by five exactly once. Ensemble
size ten enters the derivatives of means and ddof=1 variances separately.
Gaussian SC mean uses coefficient counting; physical SC mean sums four
family means. SC variance counts all selected coefficients together and uses
the fixed `1e-12` clamp. Complex features use Var(Re)+Var(Im) and count once.

Physical sigma equalization is performed per family using the native final
orientation axis: l1 for S1/S2 and l3 for S3/S4. Grouping uses plan metadata,
so stable sorting by native level cannot scramble the orientation operation.
Complete orientation counts are checked. This is a documented native-axis
analogue; it is not an assertion that FOSCAT/native arrays are interchangeable.

The SC mapping/normalizer audit is in `SC_MAPPING.md`. It explicitly records
S0 recombination, S3P correspondence, inspected `norm='self'` behavior, and
the positive total-complex-variance native sigma analogue. No automatic
coefficient rescaling or variance-floor retuning is applied. Calibration saves
per-family floor counts and sensitivity diagnostics; scientific synthesis
rejects a wholly inactive S1–S4 variance family.

LP power uses per-member spectra and source unnormalized cosine-squared
windows; all mean and variance weights and the per-band variance epsilon are
preserved. The 2048 upper variance cutoff must be supplied explicitly.

CIC uses per-member densities with partial edge contributions, dropped
out-of-range mass, and no mass renormalization. MF thresholding uses each
member's own standardized field, with differentiation through its mean/std,
and a detached input-dependent curvature floor. The MF finite-difference
convention must freeze that floor; source-VJP equivalence is a separate test.

The independent `test_losses.py` run passed all ten tests. They cover complex
population cotangents; both SC mean reductions and clamp sensitivity; power,
PDF, and MF loss directional derivatives; the global five-block factor;
orientation-plan permutation invariance; exact pooled LUT quantiles; linear
tail extrapolation and per-member monopole pullback; and CIC boundary mass
and VJP. This reviewer did not rerun the native/GPU/MF-reference suites;
their executed evidence belongs in the aggregate validation report.

## Cache identity and allocation review

Cache identity includes the input file hashes, ordered bootstrap IDs, native
revision and binary hash, coefficient plan, implementation hash, LUT hash,
target hash, preprocessing, resolutions, and loss definitions. Execution
mode, CPU threads, and GPU cache size are deliberately excluded from the
scientific configuration so resident/replay/checkpoint modes consume the
same target bytes. A changed native binary currently requires recalibration
or an explicit compatibility review rather than silent cache reuse.

The streamed engine is constructed with batch one. Gaussian and physical SC
are consecutive passes over the same native engine; their checkpoint banks
do not coexist. Power alms remain on the CPU for the current trial; MF replay
stores only one member's derivative fields. Other scientific branches use
CPU arrays, not an additional GPU SC workspace.

The 82.5 GiB variables/gradient/ten-history-pair figure in the prompt is not
a complete process-memory prediction. The Python objective temporarily
holds another complete ten-member gradient (3.75 GiB at 2048), while native
L-BFGS also needs line-search/vector temporaries. CPU SHT geometry, ordering
indices, alms, derivative fields, and NumPy boolean/reduction temporaries
add memory. Exact pooled calibration has a float64 disk workspace of about
20.25 GiB at 2048, besides the float32 physical/Gaussian calibration arrays.
Mapped-file pages may be resident during quantile construction. None of
these observations certifies fit within the actual job cgroup allowance.

The preflight and bounded complete-evaluation benchmark must be run on the
GH200. Increasing CPU thread count, engine cache size, or resolution is not
evidence of speedup or capacity until measured.

## Remaining limitations

TRAIN54 raw maps and validated corresponding FOSCAT outputs were not provided
for this implementation environment, and no GH200 is available here. The
example `synth2048_L4_hpcgpu480_nested.npy` is now available as diagnostic
context (50,331,648 float64 pixels; file size 402,653,312 bytes). It is not a
calibration substitute. Scientific reproduction, high-resolution variance
activity, GPU checkpoint equality, and complete GH200 timings/memory remain
user-side acceptance tests. No optimizer termination can establish those
claims by itself.
