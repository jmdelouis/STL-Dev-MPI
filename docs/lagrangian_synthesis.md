# Lagrangian (deformation-based) synthesis: assessment, literature, design

Status: **design, plan and step 0 diagnostic** (3 November 2026). Sections 1–4
are the assessment and the design; section 5 is the experimental plan to be
agreed with the user before any implementation; results will go in section 6.

## 0. The question

Syntheses from scattering covariances start from white noise and run L-BFGS
on the pixel values. Campeti's ensemble needed about 2500 iterations at
nside 1024, and neither a per-band preconditioner nor coarse-to-fine starts
helped much (PR #11, `docs/geometry_cache_preconditioner.md`). The user's
hypothesis: the cost is dominated by *building structures* (clumps,
filaments) out of randomly placed values. In pixel space this means slowly
draining signal at one place so that it grows at another. A deformation of
the field, which packs or spreads the existing values, could reach the same
morphology in far fewer steps. So: optimise the best deformation instead of
the best pixel change.

## 1. Assessment

**The intuition is sound, and it has a precise mathematical form.** L-BFGS on
pixels is a quasi-Newton method in the Euclidean (L2) geometry of the pixel
values. In that geometry, moving a bump of mass `m` by a distance `d` costs
`||x - x(. - d)||^2 ≈ 2 m^2 / width` whenever `d` exceeds the bump width.
The cost does not depend on `d`, and the straight L2 path goes through
"half here, half there" states that the loss sees as two bumps. In the
Wasserstein (W2) geometry the same move costs `m d^2` and the path is a
translation. Gradient flows in W2 (Otto calculus) move mass along
displacements: the update is `∂x/∂t = div(x ∇ δL/δx)`, and the pixel
gradient is smoothed by an inverse Laplacian weighted by `x`. That is exactly
"packing or spreading" rather than "draining and growing".

**Two corrections to the intuition matter for the design.**

1. *The loss is translation invariant.* No structure has to reach a target
   position, so the cost is not distance travelled. What the synthesis must
   do is **gather**: turn the scattered noise into coherent objects that are
   correlated across scales (the S3/S4 terms) and keep the one-point PDF
   (S1/S2 per scale, and the kurtosis-like tails of κ). A deformation does
   this coherently across scales: one smooth displacement compresses a whole
   region into a filament at every scale at once. In pixel space, L-BFGS has
   to discover that coherence through its small history (10–50 vectors in
   10^5–10^7 dimensions). The expected gain therefore comes from
   **conditioning and coherence**, not from distance.
2. *The band preconditioner of PR #11 was linear and did not depend on the
   field.* It rescaled frequencies, which is what a W2 metric linearised
   around a *uniform* density does (W2 ≈ H^-1 there). That gave nothing, so a
   transport metric is only interesting if it **depends on the current field**
   (`∇x K ∇x^T`, section 3.4). A deformation parameterisation gives this
   naturally. The PR #11 result is evidence that the bottleneck is
   non-linear (morphological), which is what this proposal targets.

**Histogram preservation is an asset (the user's remark, 3 November).** A
pure warp keeps the one-point histogram of `x0`. So start from a field that
*already has the target's histogram*: a random permutation of the target's
pixels (white spectrum, exact PDF). This removes any Gaussianisation step,
which matters most for data with badly non-Gaussian histograms (κ, LSS,
dust): "we have the points, we only need to know where to place them".
The theory says this search space is exactly right:

- **Brenier's polar factorisation** (Brenier 1991, CPAM 44): every field
  with the same histogram as `x0` is `x0 ∘ s` for a measure-preserving map
  `s`. The set of syntheses with the right histogram is therefore the orbit
  of `x0` under measure-preserving rearrangements.
- **Shnirelman** (1987, Math. USSR Sb. 56) and Brenier: smooth
  volume-preserving diffeomorphisms (incompressible flows) are dense in the
  measure-preserving maps. A divergence-free displacement, `u = ∇^⊥ ω` (a
  stream function `ω`), therefore spans the right set in the continuum, and
  keeps the histogram exactly.

Two discrete effects break exact preservation, and the design must control
them:

1. a compressive component (`div u ≠ 0`) re-weights the areas, and thus the
   histogram, unless the Jacobian factor compensates in the mass form;
2. interpolation mixes neighbouring values (next item), which shrinks the
   tails.

The cheap, exact fix for both is a **histogram re-projection** at every
re-anchoring (section 3.5): replace the values of `x0 ∘ φ` by the target's
sorted values, in the rank order of `x0 ∘ φ`. This is O(N log N), it is the
projection step of Portilla & Simoncelli, and it is the discrete Brenier
rearrangement. The heavy tails then come for free, and the Eulerian residual
`r` is needed only for what the histogram constraint cannot reach. The limit
case of fine displacements is a discrete permutation of pixels (pair swaps).
It is not differentiable, but it is a useful mental model: the synthesis is a
*sorting problem in space*.

**Risks, stated plainly.**

- A pure warp `x0 ∘ φ` cannot create values that are not in `x0`. That is
  the point when `x0` carries the target's histogram (above). From Gaussian
  noise, heavy positive tails (κ peaks) would need the mass-conserving form
  (compression raises amplitude, as in Zel'dovich), an amplitude field `A` or
  a residual `r`.
- **Warping white noise with sub-pixel displacements smooths it.** Bilinear
  interpolation at offset `t` multiplies the variance of white noise by
  `(1 - 2t(1 - t))` per axis, so by 1/4 at half a pixel in 2D. A smooth
  displacement of white noise therefore imprints its own fractional-part
  pattern on the local variance. This is an artefact the loss will see and
  then fight. Mitigations: warp a field with a red spectrum (a coarse
  synthesis, or the current iterate after some Eulerian iterations, rather
  than raw white noise); use higher-order interpolation; or let `r` carry
  the pixel scale. The re-anchored scheme of section 3.5 avoids most of it.
- **Folds.** A pull-back with a non-injective `φ` duplicates structures
  rather than creating caustics. Push-forward (particle deposition) creates
  real caustics but can concentrate mass without limit. Diffeomorphic
  parameterisations (stationary velocity field with scaling-and-squaring)
  prevent folds, at the price of 6–8 compositions per evaluation.
- **L-BFGS and a moving parameterisation.** If `x0` is re-anchored to the
  current iterate, the variable changes meaning and the L-BFGS history must be
  reset. With restarts every K iterations, part of the curvature information
  is lost; K must be tuned.
- **Cost per evaluation is not the issue.** A warp and its adjoint are O(N)
  (4 taps per pixel), negligible next to one scattering evaluation. The
  question is only the number of iterations, and whether the optimiser ends
  at a lower loss.

**Verdict.** The idea is worth a prototype. It is cheap to implement on top
of the existing loss and gradient, the literature (section 2) shows the same
mechanism working in neighbouring fields, and the risks are identifiable and
testable. The plausible outcome is an Eulerian–Lagrangian hybrid that reaches
the reference loss in fewer iterations on filamentary fields (turbulence, κ).
The outcome is not guaranteed: if most of the 2500 iterations are spent on
fine pixel-scale details of the PDF, deformation will not help there.

## 2. Literature

The idea combines elements that exist separately. As far as we could find,
**no published work optimises a deformation field to match scattering /
wavelet-moment statistics**. The pieces exist:

### 2.1 Wasserstein gradient flows versus L2 flows

- Jordan, Kinderlehrer & Otto (1998), *The variational formulation of the
  Fokker–Planck equation*, SIAM J. Math. Anal. 29: the JKO scheme,
  `x_{k+1} = argmin L(x) + W2^2(x, x_k) / 2τ`. It is the W2 analogue of the
  proximal (implicit) gradient step that L2 descent approximates.
- Benamou & Brenier (2000), *A computational fluid mechanics solution to the
  Monge–Kantorovich mass transfer problem*, Numer. Math. 84: W2 as a minimal
  kinetic energy of a velocity field under the continuity equation, the
  Lagrangian/Eulerian dictionary used below.
- Otto (2001), *The geometry of dissipative evolution equations: the porous
  medium equation*, Comm. PDE 26: the Riemannian metric of W2. Its
  linearisation is the weighted H^-1 norm, so the W2 gradient is
  `div(ρ ∇ δL/δρ)`.
- Peyré (2015), *Entropic approximation of Wasserstein gradient flows*,
  SIAM J. Imaging Sci. 8: Sinkhorn-based JKO steps on grids. Cuturi (2013),
  *Sinkhorn distances*, NeurIPS. Peyré & Cuturi (2019), *Computational
  optimal transport*, Found. Trends ML; Santambrogio (2015), *Optimal
  Transport for Applied Mathematicians*, ch. 8.
- Brenier (1991), *Polar factorization and monotone rearrangement of
  vector-valued functions*, CPAM 44; Shnirelman (1987), *On the geometry of
  the group of diffeomorphisms and the dynamics of an ideal incompressible
  fluid*, Math. USSR Sb. 56. Every field with a given histogram is a
  measure-preserving rearrangement of any other, and incompressible flows
  are dense in these rearrangements: the basis of the histogram-exact
  Lagrangian synthesis (section 1).
- Li & Montúfar (2018), *Natural gradient via optimal transport*, Inf. Geom.
  1; Chen & Li (2020), *Optimal transport natural gradient for statistical
  manifolds with continuous sample space*, Inf. Geom. 3: the Wasserstein
  natural gradient, i.e. the transport-metric preconditioner of section 3.4.
- **Closest analogue in practice:** full-waveform inversion in seismology.
  There, the L2 misfit cannot move a wavelet by more than half a period
  ("cycle skipping"). Replacing it by a W2 misfit fixes this: Engquist &
  Froese (2014), *Application of the Wasserstein metric to seismic signals*,
  Commun. Math. Sci. 12; Métivier, Brossier, Mérigot, Oudet & Virieux (2016),
  *Measuring the misfit between seismograms using an optimal transport
  distance*, Geophys. J. Int. 205. This changes the *loss*, not the
  parameterisation, but it documents the same failure of L2 geometry.
- W2 needs non-negative densities. κ and turbulence maps are signed, which
  is one more reason to use a deformation (defined for any field) rather than
  a literal W2 flow on `x`.

### 2.2 Lagrangian methods in cosmology

- Zel'dovich (1970), A&A 5: `x = q + D(t) ∇ψ(q)`, density
  `1 + δ = 1 / |det(I + D ∇∇ψ)|`. Pancakes and filaments come from a
  *Gaussian* displacement potential. **This is the strongest physical argument
  for the user's idea:** the cosmic web is simple, nearly Gaussian, in
  displacement space, and complex in density space. Higher orders (LPT,
  2LPT): Bouchet, Colombi, Hivon & Juszkiewicz (1995), A&A 296; Bernardeau et
  al. (2002), Phys. Rep. 367.
- Particle-mesh solvers on displacements: COLA (Tassev, Zaldarriaga &
  Eisenstein 2013, JCAP 06, 036) and FastPM (Feng, Chu, Seljak & McDonald
  2016, MNRAS 463). FastPM is differentiable, with adjoint gradients with
  respect to the initial field (Seljak et al. 2017, JCAP 12, 009; Feng et al.
  2018, JCAP 07, 043).
- **Field-level inference** optimises the Gaussian initial field pushed
  through a Lagrangian forward model rather than the final density. It
  converges far better than optimising the density directly: BORG (Jasche &
  Wandelt 2013, MNRAS 432); Seljak et al. 2017; Modi, Feng & Seljak (2018),
  JCAP 10, 028. This is the closest precedent for "optimise in displacement
  space", with a likelihood loss instead of a statistics loss.
- Lagrangian Deep Learning: Dai & Seljak (2021), *Learning effective physical
  laws for generating cosmological hydrodynamics with Lagrangian Deep
  Learning*, PNAS 118 (arXiv:2010.02926). Each layer moves particles along
  the gradient of a learned potential (a Fourier filter of the density); a
  few layers reproduce hydrodynamic fields. **It is the most direct template
  for our parameterisation:** displacement = `∇` of a filtered potential, few
  parameters, mass deposited by CIC.
- Reconstruction by displacement fields: Eisenstein et al. (2007), ApJ 664
  (BAO reconstruction); Schmittfull, Baldauf & Zaldarriaga (2017), PRD 96
  (iterative reconstruction).
- **On the sphere: CMB lensing is exactly a remapping by a displacement
  field `T(n + ∇φ(n))`.** Lewis (2005), PRD 71 (lenspix); Reinecke, Belkner &
  Carron (2023), A&A 678 (lenspyx/ducc, exact remapping on the sphere and its
  adjoint). Gradients with respect to the lensing potential, `g ∇T`, are what
  iterative lensing reconstruction uses: Hirata & Seljak (2003), PRD 68;
  Carron & Lewis (2017), PRD 96. LenseFlow (Millea, Anderes & Wandelt 2019,
  PRD 100) integrates the remapping as a flow ODE with exact Jacobians and
  adjoints. This is the HEALPix machinery we need, already worked out.

### 2.3 Image registration with diffeomorphic deformations

- Demons: Thirion (1998), Med. Image Anal. 2. Diffeomorphic demons:
  Vercauteren, Pennec, Perchant & Ayache (2009), NeuroImage 45. Both
  alternate a *smoothed* force `g ∇x` with a Gaussian regularisation, which is
  exactly the Lagrangian step of section 3.5.
- LDDMM: Beg, Miller, Trouvé & Younes (2005), IJCV 61. Stationary velocity
  fields and scaling-and-squaring: Arsigny, Commowick, Pennec & Ayache
  (2006), MICCAI; Ashburner (2007), DARTEL, NeuroImage 38. Learned version:
  Balakrishnan et al. (2019), VoxelMorph, IEEE TMI 38.
- **Metamorphosis** (deformation plus intensity change, with a metric that
  mixes both): Miller & Younes (2001), IJCV 41; Trouvé & Younes (2005),
  Found. Comput. Math. 5. This is the mathematically clean form of the
  Eulerian–Lagrangian hybrid `x = A (x0 ∘ φ) + r`. Its metric is the
  preconditioner `σ^2 I + ∇x K ∇x^T` of section 3.4.

### 2.4 Texture synthesis, scattering and transport

- Statistics-matching syntheses all optimise pixels: Portilla & Simoncelli
  (2000), IJCV 40; Gatys, Ecker & Bethge (2015), NeurIPS. With scattering:
  Bruna & Mallat (2019), *Multiscale sparse microcanonical models*, Math.
  Stat. Learn. 1 (microcanonical gradient descent from white noise,
  interpreted as a *transport of the noise measure*, not of pixel mass);
  Allys et al. (2019, 2020: RWST, A&A 629; WPH for LSS, PRD 102); Zhang &
  Mallat (2021), ACHA 53 (phase harmonics); Cheng & Ménard (2021),
  arXiv:2112.01288 (guide); Cheng, Morel, Allys, Ménard & Mallat (2024), *Scattering spectra models
  for physics*, PNAS Nexus 3; Marchand et al. (2022), *Wavelet conditional
  renormalization group*, arXiv:2207.04941 (coarse-to-fine). **None of
  these optimises a deformation.**
- OT as the **loss** on feature distributions: Tartavel, Peyré & Gousseau
  (2016), *Wasserstein loss for image synthesis and restoration*, SIAM J.
  Imaging Sci. 9; Houdard, Leclaire, Papadakis & Rabin (2023), GOTEX, J. Math. Imaging Vis. 65;
  Heitz, Vanhoey, Chambon & Belcour (2021), *A sliced Wasserstein loss for
  neural texture synthesis*, CVPR; Kolkin, Salavon & Shakhnarovich (2019),
  STROTSS, CVPR. This changes the statistics; it is orthogonal to the
  user's idea (it could be combined later, e.g. sliced-W on the S1 per-scale
  histograms to fix tails).
- Warping in texture work: deformation-based statistical shape models
  (Cootes, Twining & Taylor 2004, BMVC; Hansen, Hansen & Larsen 2007, ICCV)
  synthesise *warps* from a model, not a texture whose statistics are
  matched through a warp.

### 2.5 What is new

Optimising a smooth, multi-scale displacement field (with an amplitude field
and a small residual) so that the *warped field* matches scattering
covariances, using the exact pixel gradient of a block-wise pyramid engine as
the cotangent, appears to be new. The closest relatives are field-level
inference through LPT (different loss), demons registration (a target image
instead of statistics) and metamorphosis (a metric, not a synthesis). The
transport-metric preconditioner of section 3.4 also seems new for scattering
syntheses.

## 3. Design

### 3.1 Parameterisation

    x(p) = A(p) · J(p)^β · x0(q(p)) + r(p),     q(p) = p + u(p)        (pull-back)

or, mass-conserving,

    x(p) = A(p) · Σ_q x0(q) W(p − q − u(q)) + r(p)                     (push-forward)

- `x0`: the anchor field. The default is **a random permutation of the
  target's pixels** (exact histogram, section 1). Other choices: white
  noise, a coarse synthesis upsampled, or the current iterate (section 3.5).
  With several target images, the values are drawn from the pooled target
  pixels; in the ensemble mode, from each target image in turn.
- `u`: the displacement in pixel units of level 0. Parameterisations, from
  the simplest:
  1. **multi-scale grid**: `u = Σ_ℓ U_ℓ θ_ℓ`, `θ_ℓ` a 2-vector field on level
     `ℓ` of the pyramid (grid `N / 2^ℓ`), `U_ℓ` bilinear upsampling to level
     0; only levels `ℓ ≥ ℓ_min` (displacements smooth at the scale `2^ℓ_min`).
     Coarse displacement plus finer corrections, as the user proposed.
  2. **Helmholtz split**: `u = ∇ψ + ∇^⊥ω`, with `ψ` and `ω` multi-scale
     scalars as above. The **incompressible part `∇^⊥ω` keeps the
     histogram** (section 1) and is the default with a permuted `x0`. The
     compressive part `∇ψ` (Zel'dovich, lensing) packs and spreads, and
     changes the histogram unless the Jacobian or a re-projection compensates.
     Both are one scalar per point, natural on the sphere.
  3. **stationary velocity field**: `φ = exp(v)` by scaling-and-squaring,
     `φ_0 = id + v / 2^K`, `φ_{k+1} = φ_k ∘ φ_k`, K = 6; diffeomorphic. Only if
     folds turn out to be a problem.
- `A = exp(a)`, `a` multi-scale and smooth (coarse levels only): local
  contrast.
- `β ∈ {0, 1}`: `β = 1` is the pull-back version of mass conservation
  (`J = det ∇q`), `β = 0` is pure advection of values. The push-forward form
  conserves mass exactly and creates caustics; it is the closest to
  Zel'dovich/LDL.
- `r`: Eulerian residual, scaled by `s_r` (variable `ρ`, `r = s_r ρ`) so that
  L-BFGS sees both blocks with comparable curvature. `s_r = 0` is the pure
  Lagrangian model.

### 3.2 Gradient (reusing the Rust pixel gradient)

The engine returns `L(x)` and `g = ∂L/∂x` (`PlanarObjective::loss_and_grad`
in 2D, `LocalEngine`/`GpuEngine` VJP on HEALPix). With `w = x0(q)` and
`B = A J^β`:

    ∂L/∂r(p)  = g(p)
    ∂L/∂a(p)  = g(p) · (x − r)(p)                    (then U_ℓ^T for each level)
    ∂L/∂q(p)  = g(p) · B(p) · ∇x0|_{q(p)}             (∇ of the bilinear interpolant)
              + β · ∇^T[ g B w · J (∇q)^-T ](p)        (Jacobian term, adjoint of the finite differences)
    ∂L/∂θ_ℓ   = U_ℓ^T ∂L/∂q                           (∂L/∂ψ = −div ∂L/∂u for the potential flow)

`∇x0|_q` is the exact derivative of the bilinear interpolant (piecewise
linear in each coordinate), so the gradient is exact everywhere except on
pixel edges, a measure-zero set; finite-difference tests must avoid them.

Push-forward (CIC deposit): `∂L/∂x0(q) = A·g` interpolated at `q + u(q)`, and

    ∂L/∂u(q) = x0(q) · ∇(A g)^interp |_{q + u(q)},

the gradient of the bilinearly *interpolated* cotangent at the particle
position: the deposit and the interpolation are transposes of each other.

Cost: 4 taps per pixel forward, 8 for the adjoint; O(N), memory a few
copies of the map. It is negligible next to the scattering evaluation.

### 3.3 HEALPix specifics

- Displacement in the tangent plane of each pixel, in the local
  (e_θ, e_φ) basis. The potential-flow form avoids carrying a vector field
  across faces: `u = ∇ψ` with `ψ` a scalar HEALPix map, its gradient from the
  existing stencil machinery (the 3 × 3 stencil in the tangent plane of
  `stl_core::stencil`, gauge-independent for the gradient of a scalar) or
  from the SHT (`stl_core::sht`, spin-1 derivative). That is lensing
  remapping, as in lenspix/lenspyx.
- Remapping: `n' = exp_n(u)` (the geodesic step, `cos|u| n + sin|u| û`),
  then `healpix::interp_weights_nested(n', order)`, the ring bilinear
  interpolation already ported and validated (4 pixels and weights). The
  adjoint scatters `g` with the same weights; the derivative with respect to
  `n'` needs the derivative of the ring weights in θ and φ (closed form, the
  interpolation is bilinear in (θ, φ) between two rings).
- Multi-scale `ψ` on the NESTED pyramid: `ψ = Σ_ℓ U_ℓ ψ_ℓ` with `U_ℓ` the
  parent→children replication (or the transposed decimation), cheap and
  aligned with the MPI decomposition. A remapping needs a halo of `max|u|`
  pixels per rank; bound `|u|` (tanh squashing) to a few coarse pixels.
- MPI: the warp is local up to that halo, so the existing ghost-exchange
  pattern applies. This is for later; the prototype is single-process.

### 3.4 Transport-metric preconditioner (no change of variable)

Linearising one deformation step around the current map `x`, for a smoothing
kernel `K` on `u` (Gaussian, or the multi-scale `Σ_ℓ U_ℓ U_ℓ^T`):

    δx = −∇x · u,   u = −K (g ∇x)   ⇒   δx = −P g,   P = σ^2 I + ∇x · K · ∇x^T.

`P` is symmetric positive (semi)definite and depends on the field. It is the
metamorphosis metric, with `σ^2` the weight of the Eulerian part. It
discretises the Wasserstein natural gradient (Otto metric) for signed fields
in the advective form; the mass-conserving form gives
`P = σ^2 I − div(|x| K ∇ ·)`. It can be fed to the existing
`lbfgs::minimize_preconditioned` as `H0`. Because `P` depends on `x`, it must
be frozen between L-BFGS restarts, the same issue as for re-anchoring. This
is the cheapest experiment: no new variable, about 30 lines of code.

### 3.5 Optimisation schemes

- **(E)** pixel L-BFGS (the reference).
- **(L)** L-BFGS on `θ` (and `a`), `x0` fixed: pure Lagrangian. Expected to
  stall at a loss that `x0`'s histogram cannot reach; it tests how far
  deformation alone goes.
- **(H)** joint `(θ, a, ρ)` with the residual scale `s_r`: the hybrid,
  metamorphosis-like.
- **(R)** re-anchored: every K iterations, `x0 ← x`, `θ, a, ρ ← 0`, reset the
  L-BFGS history. With `K` large this is (H); with `K = 1` and a steepest
  step it is a demons/JKO-like flow. Also the alternating variant: K_L
  Lagrangian iterations, then K_E Eulerian ones.
  At each re-anchoring, optionally re-project `x0` onto the target's
  histogram (rank matching, section 1): `--reproject`.
- **(P)** pixel L-BFGS with the transport preconditioner of section 3.4,
  frozen and refreshed every K iterations.

## 4. What to measure

On each target, with the same plan, weights, `L = 4`, `f64`:

- the loss against the number of evaluations and against wall time
  (evaluations dominate the time; the warp is extra O(N));
- iterations and evaluations needed to reach the *reference loss*: the
  loss reached by (E) after 100, 300 and 1000 iterations;
- the loss per family (mean/var, S1, S2, S3, S4) along the run: does the
  deformation mostly fix S3/S4 (morphology), as expected?
- visual structure: filaments and clumps, side by side at equal evaluations;
- cosmic variance (κ only): `tools/cv10.py`, Minkowski functionals
  (`tools/minkowski.py`), C_ℓ against the 10 Takahashi realisations.

## 5. Plan (to be agreed before implementation)

Step 0, diagnostic (about 1 h on garoupe, no new code). Run (E) on the 2D
example (10 × 256², `stl-synth2d`, PR #9) for 1000 iterations, logging the
loss per family. This shows where the iterations go. If S3/S4 at large `j`
dominate the long tail, deformation is well targeted. If S1/S2 at small `j`
dominate, it is not.

Step 0b (same day, no new code): pixel L-BFGS started from a random
permutation of the target pixels instead of Gaussian noise (`--init`). This
separates the effect of the histogram-exact start from that of the
deformation in step 2.

Step 1, the cheap test: scheme (P), the transport preconditioner, in
`stl_core::precond`. About 30 lines; one comparison run on the 2D example.

Step 2, the Lagrangian parameterisation in 2D (periodic grid):
`stl_core::warp` (pull-back bilinear and push-forward CIC, their VJPs,
multi-scale `θ_ℓ`, potential/curl forms, the Jacobian term), finite-difference
tests, then `stl-synth2d --lagrangian {pull,push} --schedule {L,H,R} ...`.
Runs (E/L/H/R/P) on the 2D example, a few minutes each on garoupe.

Step 3, only if step 2 cuts the iteration count clearly (say ≥ 2× to the
reference loss): HEALPix nside 128, Takahashi κ (data copied from hpc
`/scratch/jmdeloui/stl-bench128/data` via `-J draco`). Potential flow `∇ψ` and
ring-bilinear remapping on `interp_weights_nested`, single process;
`stl-synth --lagrangian`. Same comparison plus the cosmic-variance checks.

Step 4: results and figures here, PR (not merged).

## 6. Results

### 6.1 Step 0 and 0b: where the pixel L-BFGS iterations go (2D, 3 November)

Setup: the STL-Dev 2D example (first 10 `Turb_6` images, 256², J = 6,
L = 4, `f64`), 10 synthesised images, `stl-synth2d --gpu` on garoupe T4s,
1000 L-BFGS iterations (history 50), `--trace 10`. Two losses: `stldev`
(STL-Dev's loss: batch-mean features normalised by the target S2) and
`ensemble` (ScatCov + spectrum + Minkowski, batch mean and variance against
the targets' scatter, cold start). Two starts: Nyquist-filtered Gaussian
noise (`e_*`) and a random permutation of the target pixels (`p_*`,
`tools/lagrangian/permuted_init.py`). Script `tools/lagrangian/trace_families.py`;
figures and reports in `benchmarks/lagrangian/step0/`.

Evaluations to reach a loss (running minimum):

| run | 1e2 | 1 | 1e-2 | 1e-4 | 1e-6 |
|---|---|---|---|---|---|
| stldev, noise | 15 | 46 | 216 | 453 | 844 |
| stldev, permuted target | 16 | 49 | 211 | 429 | 848 |

| run | 1e3 | 300 | 100 | 70 | final (≈1040 evals) |
|---|---|---|---|---|---|
| ensemble, noise | 80 | 268 | 675 | 877 | 54.7 |
| ensemble, permuted target | 96 | 307 | 796 | – | 72.5 |

Findings:

1. **With STL-Dev's loss there is no slow phase to remove.** The loss falls
   by 10^12 in 850 evaluations. The maps have the target's filaments and
   voids by evaluation 100 (`e_stldev_maps.png`), and every family is
   matched at the same pace. At 256² in 2D, gathering structures in pixel
   space is not the bottleneck.
2. **The slow case is the ensemble loss from a cold start**, and its
   signature is a **transfer of variance across scales**, not in space.
   After 1000 iterations the large scales are still 70 % short in power
   (spectrum bin 0) and 40–80 % short in S1/S2 at j = 5. The small scales
   are 0.8–1.2 times too strong in S2 at j = 0–2. Early on (evaluation 30)
   the small scales were 27× too strong and the large ones 97 % short. The
   maps show the same thing: small-scale texture, no large voids or bright
   ridges (`e_ens_maps.png`). In units of the target scatter, S4 at scale
   gaps 0–1 and the spectrum dominate the residual
   (`e_ens_families.png`). This matches the known cold-start stall of the
   ensemble mode (`docs/planar_2d.md`: 145 at 500 iterations, solved by
   `--warm 100` on the stldev loss) and the large-scale C_ℓ deficits seen in
   the HEALPix ensembles.
3. **A histogram-exact start alone does nothing for pixel L-BFGS.**
   `p_stldev` follows `e_stldev` within a few percent; `p_ens` is slightly
   worse. The pixel updates destroy the permutation's histogram within the
   first steps, so the start's histogram is lost. Its benefit can only
   appear if the parameterisation keeps it, as the warp does.

Consequences for the design:

- The target of the Lagrangian scheme is the ensemble/cold regime: moving
  variance from small to large scales at a fixed one-point histogram. A
  rearrangement does exactly this. Grouping existing values into coherent
  large regions creates large-scale power without creating or destroying
  values. Eulerian L2 must instead shrink the small-scale modes and grow the
  large ones separately, and in the ensemble loss the large-scale
  coefficients weigh little (their target scatter is large). Step 2
  therefore runs on the **ensemble loss from a cold start**, with the
  permuted-target `x0` and an incompressible multi-scale displacement.
  The reference is pixel L-BFGS (loss 100 at 675 evaluations, 55 at about
  1040) and the `--warm 100` recipe.
- The transport preconditioner (step 1) acts on spatial moves at fixed
  scale. The diagnostic does not point at that, so it moves after step 2.
