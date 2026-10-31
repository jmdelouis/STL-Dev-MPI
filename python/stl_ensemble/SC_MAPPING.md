# v2 addendum — authoritative exception to the historical v1 contract

Approved update `2026-10-29-v2-variance-ratio`: BOTH Gaussian and physical SC
population-variance penalties now use
`mean(log(max(V_syn/V_TRAIN54,1e-12))**2)`. The floor is dimensionless; all selected
TRAIN54 targets must be finite and positive. The old absolute variance floor
and its whole-family guard below describe historical v1, NOT active v2 behavior.
Mean sigma processing is unchanged and separately reported. No native feature,
map, kernel/orientation, PSD/PDF/MF, or optimizer change is made. Explicit cache
migration preserves old target-array bytes with a new identity and guard.

The original unedited document is in `provenance/v1_SC_MAPPING.md`.

---

# Native SC feature and normalization audit

Audit version: 2026-10-28-v1. This is a mapping of definitions, not a claim of
coefficient equality between libraries. The calibration and synthesis use the
same native plan and the same `SCNormalization` object.

## Sources and limits of historical attribution

The attached main script SHA256 is
`1c507030ab95145c84405ac7e01426f0c9828a402d92d3a56ea7887b3f09b447`.
Its `_SC_COMPONENT_NAMES`, target builder, Gaussian and physical losses are
the executable references. Its installed FOSCAT revision was not supplied.
The external definitions below were inspected from the current
[pcampeti/FOSCAT revision 17baec97672097d68fd4ef1e337e8ec362ef4326](https://github.com/pcampeti/FOSCAT/tree/17baec97672097d68fd4ef1e337e8ec362ef4326):

- `src/foscat/scat_cov.py`: blob `56f0964ef557ff4236ee737f86f162ae1faa1e17`,
  SHA256 `6cc01672dccd37b42e947112cf96b7467946be469edbbdfea5a9e3969d66c510`.
- `src/foscat/FoCUS.py`: SHA256
  `1352ead40fa204bd0cd5fe4c87572900d361a1812dd18d5f177da08404c16bf7`.

The native reference is
[STL-Dev-MPI f083c4a753a244d1edf69fe4f944fa23fc5cdb76](https://github.com/pcampeti/STL-Dev-MPI/tree/f083c4a753a244d1edf69fe4f944fa23fc5cdb76),
particularly `crates/stl-core/src/plan.rs` and `engine.rs`. The adapter uses
the complete `fewer=false` native definition. No additional pyramid operation
or gradient rescaling is inserted.

## Named family and index correspondence

Let `z[j,l]` be the native first wavelet of the analysis map at level `j`.
Let `a[j1,j3,l1,l3]` be the wavelet at level `j3` of the decimated modulus
of `z[j1,l1]`. Angle labels refer to native filters and are not assertions of
identical FOSCAT filter responses.

| Source component | Native plan entry and port feature | Complex convention / axes |
|---|---|---|
| `S0[...,0]` | `family=mean`: mean of level-zero map | One real coefficient |
| `S0[...,1]` | `family=var` slot recombined as `sqrt(max(q-m*m,0)/N0)`, with `q=native var=mean(x*x)` and `m=native mean` | One real coefficient; native `var` alone is an uncentered second moment and is **not** the source S0 component |
| `S1[j,l]` | `S1`, `j1=j`, `l1=l`: `mean(abs(z[j,l]))` | Real; complete scale and angle axes |
| `S2[j,l]` | `S2`, `j1=j`, `l1=l`: `mean(abs(z[j,l])**2)` | Real; complete scale and angle axes |
| `S3[j2,j3,...]` | `S3`, native `j1=source j2`, `j3=source j3`, `l1` on the earlier modulus, `l3` on the outer wavelet: `mean(a*conj(z[j3,l3]))` | Native is the conjugate of the source product `mean(z*conj(a))`, for otherwise identical fields; keep native complex convention and recompute targets in that convention |
| `S3P` | No independent component for this scalar, single-input auto-correlation | Source `eval(image1)` returns `S3P=None`, and the main script skips unavailable components. In source cross-correlation it exchanges the two input channels. Adding a duplicate auto-channel component would change coefficient weighting and is not done |
| `S4[j1,j2,j3,...]` | `S4`, complete `j1<=j2<=j3`, `l1,l2,l3`: `mean(a[j1]*conj(a[j2]))` | Complex, including real self-pairs; retain every orientation tuple and every intended scale triple |

The native plan is sorted stably by `native_dg`, with family enumeration and
orientation loops inherited from `build_plan`. The serialized full plan is
the definitive mapping from array index to family, channel, scale and angle.
There is no assumption of family-contiguous flattened arrays. The source's
packed scale indices are replaced by these named native indices.

The second S0 feature's pullback is explicit. For `s=sqrt((q-m*m)/N0)>0`,
`ds/dq=1/(2*N0*s)` and `ds/dm=-m/(N0*s)`. The adapter combines both raw native
cotangents before the engine reverse sweep. At exactly constant maps this
feature is nonsmooth; its selected pullback is zero. Finite-difference tests
avoid that point. The remaining native features have an identity adapter.

For the physical sigma equalization, the final angle axis is native `l1`
for S1/S2 and `l3` for S3/S4. FOSCAT array comments are not always consistent
with axis order, but its actual `convol`/`expand_dims` operations place the
outer-wavelet angle last. Grouping the flattened native plan by those labels
reproduces the required average over all other axes.

## `norm='self'`: executable behavior, not inferred intent

In the inspected `scat_cov.py`, `div_norm` at line 2876 constructs and
returns a **new** complex tensor. The auto-channel S1, S3 and S4 call sites
invoke `self.div_norm(...)` without assigning its result (around lines 3310,
3537 and 3747). Therefore those values are not divided by the proposed S2
denominators. S2 is divided only under `norm=='auto'`; the reference requests
`norm='self'`. Both the reference mean losses and population variances thus
operate on raw auto-channel coefficients in the inspected implementation.

The port follows that executable behavior. It does not introduce an intended
but unexecuted normalization ratio, its derivative, a fitted target scale,
or a global standardization. The mapping is recorded in every calibration
manifest. If the historical installed FOSCAT source had different call sites,
that is a different reference implementation and requires a fresh comparison;
this package does not claim the unavailable historical source was identical.

## Within-map sigma versus population variance

For a real pixel field `f`, source `masked_mean(calc_var=True)` returns
`S=mean(f)` and `sigma=sqrt((mean(f*f)-S*S)/N)`. Thus the native analogue
for real S1/S2 is exactly `sqrt(native_spatial_variance/N_native_level)`.
It is a within-map estimator quantity. It is **not** the standard deviation
across TRAIN54 maps, and it is not a fitted uncertainty of the ten-member
ensemble mean. Sigma calibration averages the individual TRAIN54 sigmas
arithmetically, rather than taking the square root of their average variance.

Source S0 is unusual: the returned feature is `[mean(x),sigma(mean(x))]`, and
its sigma object is `[sigma(mean(x)),sigma(mean(x))]`. The adapter preserves
that duplicated sigma convention; it does not propagate a new error bar for
the second S0 entry.

For complex fields, the source and attached large-array patch use `f*f`
rather than `abs(f)**2`, then add the real and imaginary parts of that second
moment before subtracting squared real and imaginary means. For `f=a+i*b`
the numerator is `mean(a*a-b*b+2*a*b)-mean(a)**2-mean(b)**2`, which is not
generally nonnegative and is not invariant under a constant phase rotation.
The native engine instead returns
`V=mean(abs(f)**2)-abs(mean(f))**2 = Var(a)+Var(b)`.

The explicitly adopted native-statistics analogue is `sigma=sqrt(V/N)` for
complex S3/S4. It is a mathematically defined total complex estimator sigma,
has the same real-field limit, and uses actual native within-map moments.
It is **not exact reproduction of that source complex-expression bug** and
must be listed among approved native-statistic differences in comparisons.
No native CLI loss weights are imported: the Gaussian mean remains a single
coefficient-count average, and the physical mean remains the sum of four
separate S1--S4 family means with source sigma equalization. The population
variance penalties remain a separate coefficient-count average, ddof=1,
weight 1 and absolute clamp `1e-12`.

## Scale range and amplitude-floor checks

The full native scale range is `j=0..log2(nside)`, down to nside 1. There are
8 native levels at 128 and 12 at 2048. No native resolution doubling occurs.
The inspected FOSCAT HEALPix code sets `J=log2(input_nside)+1`, then its
KERNELSZ>3 full-sky branch doubles the grid and increments Jmax. For the
reference nside128 this gives nine grids 256,128,...,1. Removing its extra
finest grid is an explicit difference required by the native-grid boundary;
discarding additional coarse native levels is not authorized.

With four angles, the full native plan contains 2 S0 entries, `4*J` S1,
`4*J` S2, `16*J*(J+1)/2` S3 and `64*J*(J+1)*(J+2)/6` S4. These are 8,322
coefficients at128 and 24,642 at2048. Each complex coefficient counts once
in a squared-modulus loss and once in `Var(Re)+Var(Im)`.

No arbitrary scaling is fitted to push target variances above `1e-12`.
Calibration reports per-family counts below the absolute clamp and its
remaining informative coefficient count. A wholly floor-limited physical
S1--S4 family is a scientific incompatibility to resolve explicitly, not
permission to disable its variance term or silently change its scale/floor.
Real TRAIN54 data are required to establish this. Small random numerical
fixtures only test arithmetic and execution-mode equivalence.

## Executed adapter checks

`tests/test_sc_normalization.py` checks S0 recombination, the duplicated S0
sigma, native-level pixel counts, separation from across-map variance,
the adapter VJP, metadata validation and complex sigma phase invariance.
All three tests passed in the local isolated environment. A separate check
using actual native CPU statistics at nside4/J2 gave normalization-VJP
absolute error `5.36e-11`, relative error `3.14e-11` for a central finite
difference with step `1e-6`. This checks the small coordinate adapter, not
scientific reproduction or GH200 performance.
