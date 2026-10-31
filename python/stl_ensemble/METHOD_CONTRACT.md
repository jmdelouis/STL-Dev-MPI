# v2 addendum — authoritative exception to the historical v1 contract

Approved update `2026-10-29-v2-variance-ratio`: BOTH Gaussian and physical SC
population-variance penalties now use
`mean(log(max(V_syn/V_TRAIN54,1e-12))**2)`. The floor is dimensionless; all selected
TRAIN54 targets must be finite and positive. The old absolute variance floor
and its whole-family guard below describe historical v1, NOT active v2 behavior.
Mean sigma processing is unchanged and separately reported. No native feature,
map, kernel/orientation, PSD/PDF/MF, or optimizer change is made. Explicit cache
migration preserves old target-array bytes with a new identity and guard.

The original unedited document is in `provenance/v1_METHOD_CONTRACT.md`.

---

# Source-derived Step-1 method contract

Audit version: 2026-10-27-v1. This document describes the supplied executable reference and the explicit authorized port differences. It does not certify that the native implementation has passed validation.

## Authority and input inventory

The supplied `05-Multireso_v41p23z1_STEP1ONLY_mean_gradient_v2.py` has SHA256 `1c507030ab95145c84405ac7e01426f0c9828a402d92d3a56ea7887b3f09b447`, matching the handoff. The source references below are line numbers in that unchanged attachment.

The main script imports and uses `foscat_fast_kernels`, `LocalizedAlm_optimized_v13_fixed`, `littlewood_paley_pcl.build_lp_binning_matrix`, and `minkowski_sht_loss.build_mf_sht_evaluator`. `minkowski_patch_loss.plot_mf_compare` is also imported for diagnostics, but that helper was not supplied. `projected_synthesis.py` is not imported or used. Its presence does not authorize projected optimization.

The notebook contains 20 cells and reports an old single-target Nside=4096 free synthesis followed by downsampling and exact histogram matching. The accompanying notes describe the same workflow. The attached example map has shape `(50331648,)`, dtype float64, Nside=2048, and 402,653,312 bytes including its header. The filename and notebook identify it as NESTED. These three attachments are contextual diagnostics, not calibration data and not the approved method. TRAIN54 raw FITS maps are absent from the attachment set.

## Exact populations, grids and initialization

- Statistical ensemble B=10. TRAIN54 IDs are 0 through 53; held-out IDs are 54 through 107. The validation split is never used for calibration.
- The bootstrap uses `np.random.default_rng(seed).choice(np.arange(54), size=10, replace=True)`. Default seed `20260914`, overridden by `V41_BOOTSTRAP_TARGET_SEED`, gives ordered IDs `[35,11,48,24,0,41,31,49,16,34]`. Preserve duplicates and order if an override produces them. Every mean target uses the same selected rows; population variances and common normalizers use all 54 rows.
- The source preprocessing is raw Nside=2048 RING float64 FITS -> `hp.ud_grade(...,512)` -> RING-to-NEST -> float32 -> reshape into contiguous 4x4 NEST children and mean to 128 (lines 1546–1600). Reference physical arrays are **not recentered**. No reference dipole removal is performed.
- Native 2048 calibration must read native maps directly. The 512 intermediate is a 128-reference convention, not a path for 2048.
- Original scientific engine settings are FOSCAT 7x7/eight orientations, float32 with CUDA TF32 enabled (lines 88–94, 1113–1118). Port settings are native STL 5x5/four orientations. Precision differences must be reported; streaming equivalence tests use identical precision in every mode.
- Original optimizer call is `sy.run(imap, EVAL_FREQUENCY=100, NUM_EPOCHS=8000)` without explicit optimizer overrides (lines 3915–3918). The joint L-BFGS family is a fixed requirement. Exact underlying FOSCAT optimizer defaults require source inspection; do not infer them from this call.
- Initialization seed is `61022026`, overridden by `V41_LARGE_INIT_SEED`. `V41_SPECTRAL_INIT_SEED` is retained only in old filename tags and has no Step-1 numerical role. Gaussian initialization power is the bootstrap mean of individual reference `hp.anafast(...,iter=0)` spectra. Set C0=0; no additional spectrum floor (`LARGE_INIT_CL_FLOOR_REL=0`). For each member, loop packed m-major alms: m=0 real N(0,C_l); m>0 real/imag independent N(0,C_l/2). Use the same advancing Generator across ten members. No exact shell rescaling or mean/std matching.
- A source nuance must be preserved or explicitly identified: with `LARGE_INIT_REMOVE_MONOPOLE=True`, the initializer zeros harmonic C0 then **adds the mean of the first bootstrap Gaussian reference map** after synthesis (lines 2895–2897, 2953–2966). It does not explicitly subtract the finite-pixel residual mean at that point. The first selected reference is representative only; it does not replace bootstrap spectral targets.

## Pooled LUT and physical convention

The LUT is built from every pixel of TRAIN54 at the actual calibration resolution (lines 1483–1542), using 8192 knots and `pmin=max(0.5/(Npixels_total+1),1e-15)`, `pmax=1-pmin`. The tail-rich grid has 25% left-log knots up to p=1e-3, 50% linearly spaced core knots, and a mirrored right tail. The code falls back to a uniform grid if uniqueness/filtering does not produce exactly 8192 knots; this fallback is part of source behavior.

Physical quantiles are computed in float64 with linear quantile interpolation then cast to float32. Strictly increasing physical knots are enforced sequentially with epsilon `max((max-min)*1e-7,1e-12)`. The Gaussianizing map interpolates physical value to p, clamps p to `[1e-15,1-1e-15]`, then applies the inverse normal CDF (whose source helper returns float32).

The inverse map converts p knots to Gaussian quantiles and interpolates directly in Gaussian coordinates, linearly extrapolating beyond both end knots. It is **not** `interp(Phi(g))` and does not clip g to the knots. The source Torch interpolation operates in float64 then casts its physical result to float32. Every synthesized physical member is `kappa=f(g)-mean_pixels(f(g))`; the derivative includes the same projection: `grad_g=f'(g)*(grad_kappa-mean(grad_kappa))`. No dipole removal, angular smoothing, exact histogram matching, or output power correction is authorized. Reference physical arrays retain their supplied preprocessing convention.

## Objective aggregation

The reference registers six blocks and consistently averages both their scalar values and gradients through `_Step1MeanLossSynthesis.calc_grad` (lines 550–710). The approved port removes only the Gaussian LP mean **optimization** block, leaving five blocks. Therefore `L=(L_GSC+L_PSC+L_PS+L_PDF+L_MF)/5`, and the full map gradient is divided by 5 exactly once. B=10 enters the statistical moment derivatives, not this block denominator.

For any real feature t[b,k], `mu=mean_b(t)`, `v=sum_b((t-mu)^2)/(B-1)`. Given cotangents a=dL/dmu and c=dL/dv, the exact member cotangent is `a/B + 2*c*(t_b-mu)/(B-1)`. Complex SC uses a complex mean and `Var(Re)+Var(Im)` for one variance feature per complex coefficient. Coefficient counting counts a complex coefficient once; squared mismatch is `Re(diff)^2+Im(diff)^2`.

## SC blocks

SC targets call `scat_operator.eval(map, calc_var=True, norm='self')` individually (lines 1899–2000). The arithmetic mean of the returned coefficients is the mean target. The arithmetic mean of returned **within-map sigma estimates** across TRAIN54 is the mean-loss normalizer. This is neither the across-map standard deviation nor the RMS of per-map sigma. The across-map population variance is a separate ddof=1 statistic, accumulated float64 and stored float32 in the source.

Gaussian SC (lines 2548–2653): families available in the source are S0, S1, S2, S3, S3P, S4. For each family, divide squared complex modulus/real mismatch by `max(sigma,1e-12)^2`, sum coefficients, then divide once by the number of coefficients across all families. Add an equally coefficient-count-weighted population variance penalty with weight 1: `(log(max(v,1e-12))-log(max(v_train,1e-12)))^2`. This is a clamp, not an additive epsilon. The derivative vanishes when synthesized variance is below the clamp.

Physical SC (lines 2790–2863): selected families are S1,S2,S3,S4 only. The mean term is the **sum of four separately averaged family mismatches**, all family weights 1, overall block weight 1. Each sigma is `process_sigma(mode='equalize',clamp_min_rel=1e-6,clamp_max_rel=1e6)`: clamp to median(sigma)*[1e-6,1e6], average over every axis except the last axis, divide this last-axis mean by its median+1e-12, divide sigma by that scale+1e-12, then clamp at 1e-12. It is an orientation-axis operation for the source layouts; flattening first would change it. Physical variance is a count-weighted reduction across the selected S1–S4 coefficients, weight 1, clamp 1e-12, not a sum of four variance means.

The local memory patch shows a source-level caution (lines 139–225): its generic complex `masked_mean(calc_var=True)` builds the second moment from `x*x`, then combines real and imaginary parts in an unusual formula. It is not safe to call that formula `E|x|²` without tracing the actual coefficient-family input. The patch only activates above 64 million elements, and the fast S4 contraction deliberately bypasses itself for `calc_var=True`. Native mean normalizers need a documented coefficient-specific analogue derived from actual raw fields and `norm='self'`; native CLI default ergodic weights cannot simply be adopted under the same label. Native/FOSCAT family ordering and normalization require repository source inspection in the implementation manifest.

Every native target cache must carry the coefficient plan and numerical normalization. Report TRAIN54 variance counts below 1e-12 by family and prove informative variance derivatives remain nonzero. A numerical scaling incompatibility is a scientific choice to resolve, not permission to drop the variance block or silently retune its floor.

## Physical LP power block

Per-map C_l uses full-sky quadrature with no angular mask, no MASTER, and no pixel-window deconvolution. Take moments only after computing each individual map's spectrum. The replacement transform must have a matching discrete adjoint, not assume an inverse SHT is its adjoint.

`build_lp_binning_matrix` uses 16 centers logarithmically spaced from 2 to `3*nside-1`, and delta=`log(lmax/2)/15`. At every integer ell, x=`(log(ell+0.5)-log(center))/delta`; weight is `cos(pi*x/2)^2` for |x|<=1 and zero elsewhere, explicitly zero ell<2. The source evaluates **ell+0.5**, although centers are ell. Matrix rows are not normalized: bandpowers are weighted sums `B@C_l`.

Mean loss (lines 2656–2742) is weight 10 times mean squared difference of log bandpowers. Epsilon=`1e-12*max(torch.median(abs(bootstrap_mean_bandpower)),1e-30)`, and both means are clamped to epsilon before logs. The variance penalty is weight `10*0.25=2.5`, mean squared log ratio `(v_syn+eps_b)/(max(v_train,0)+eps_b)`, with `eps_b=0.01*max(max(v_train,0),1e-30)`. Variance bands are selected by center in [20,383] at 128, all inclusive. The 2048 upper variance cutoff remains an explicit required resolution decision.

## Physical CIC PDF block

Support is global TRAIN54 min/max expanded by 2% of their span; there are 100 equally spaced **centers** including both endpoints, so dx=(max-min)/99 (lines 1634–1693,1726–1871). Each pixel contributes weights (1-frac),frac to floor(position) and floor(position)+1 only if those destination indices lie in [0,99]. There is no endpoint clipping of pixel mass and no renormalization of in-range mass; each map divides by Npixels*dx. A pixel just outside the center interval can contribute partially to the edge bin. Record outside fraction diagnostically.

Mean target is the bootstrap average of those per-map density vectors. Variance target is TRAIN54 ddof=1. The mean mask is `isfinite(bootstrap_mean) & bootstrap_mean>max(max(bootstrap_mean,0))*5e-6`. Mean sigma=`sqrt(max(train_var,0))`; its floor is 0.05 times the Torch median of positive finite sigma within the mean mask (lower middle for an even count), fallback dtype epsilon if none, and final clamp to dtype tiny. Mean loss has weight 1 and averages squared sigma-normalized mismatch on the mean mask.

Variance floor is 1e-4 times the Torch median of positive finite TRAIN54 variance within the same mean mask, with the same fallback and tiny clamp. Variance mask additionally requires finite variance strictly above this floor. Epsilon is 0.01*floor, clamped to dtype tiny. Variance loss weight 1 is the masked mean squared log ratio `(max(v_syn,0)+epsilon)/(v_train+epsilon)`. An uninformative mask contributes zero, matching reference behavior. Do not pool pixels across members before constructing per-member densities.

## Physical SHT Minkowski block

Full-sky evaluator has radius pi, core fraction 1, unit core/taper and native lmax `3*nside-1` (lines 3832–3874). Its source helper normalizes each member separately to `u=(kappa-mean(kappa))/sqrt(mean((kappa-mean(kappa))²)+1e-12)`, differentiating through both moments. Use raw u for thresholding, and harmonic derivatives of u; there is no bandlimited synthesis round trip of the threshold field. Use 65 thresholds linearly spaced [-4,4] and epsilon=0.125. H(z)=0.5*(1+erf(z/sqrt(2))); delta_epsilon(z)=exp(-z²/2)/(sqrt(2*pi)*epsilon), where z=(u-nu)/epsilon. Epsilon is threshold smoothing, not an angular beam.

For orthonormal covariant derivatives d1,d2,d11,d22,d12, q=d1²+d2², gradient magnitude=sqrt(q+1e-30), detached floor=`1e-4*mean_pixels(q)+1e-12`, curvature numerator=`2*d1*d2*d12-d1²*d22-d2²*d11`. V0=mean(H), V1=mean(delta*gradient magnitude)/4, V2=mean(delta*curvature numerator/(q+detached floor))/(2*pi). The floor is recomputed for each input but held constant in its VJP. A finite-difference test must freeze that floor or distinguish reference-VJP equivalence from the derivative of a recomputed scalar.

Each component mean is bootstrap mean and variance is TRAIN54 ddof=1. For each component independently, floor=`max(1e-4*max(max(train_var,0)),dtype.tiny)`. Both mean/variance masks require finite train_var strictly greater than floor. Mean loss weight 1 is the masked mean squared difference divided by train_var (sigma=sqrt(max(train_var,floor))). Variance loss weight 10 is masked mean square log ratio `(max(v_syn,0)+eps)/(train_var+eps)`, eps=`max(0.01*floor,dtype.tiny)`. Sum all three means and three variance terms into one MF block.

## Approved differences and acceptance boundaries

Approved differences are the native STL engine and recomputed native statistics; 5x5/four orientations; one-member persistent device residency with exact joint ten-member moments and L-BFGS; five blocks after removing Gaussian LP optimization; native-resolution outputs; CPU DUCC full-sky analysis and matching adjoint; explicit native-scattering grids equal to optimization grids. FOSCAT internally oversamples its K>3 input; identical 5x5/four settings do not establish identical scale response. No hidden 4096 workspace, free 4096 optimization, projection, second pyramid preconditioner, patch method, histogram matching, additional losses, or long unattended run is authorized.

Execution equivalence, native-VJP correctness, scientific reproduction against TRAIN54/held-out/FOSCAT, and GH200 capacity/performance are separate claims and require separate evidence. A lower loss or successful optimizer exit proves none of the latter three by itself.
