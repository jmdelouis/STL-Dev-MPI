# `stl-ensemble`: ensembles with P. Campeti's method

`stl-ensemble` synthesises `B` maps (members, 10 by default) whose statistics have the **mean
and the variance of a population of reference maps**, not those of one target. It is the Rust
port of P. Campeti's method (STL_ensemble v2, variance ratio):

- the variables are Gaussian maps `g`; the physical maps are `κ = f(g) − mean`, with `f` a
  monotone lookup table pooled over the references;
- the loss is the mean of five blocks, each with a term on the mean over the members (against a
  bootstrap draw of the references) and a term on their variance (against the references'):

| Block | Statistics |
|---|---|
| `gaussian_sc` | ScatCov of `g` (full plan, `L = 4`, kernel 5) |
| `physical_sc` | ScatCov of `κ` |
| `power` | 16 log C_ℓ bands of `κ` |
| `pdf` | pixel PDF (cloud in cell) of `κ` |
| `mf` | harmonic Minkowski functionals of `κ` |

It runs in two steps: `calibrate` builds the lookup table, the targets of the blocks and the
initial C_ℓ from the references; `run` synthesises.

```bash
stl-ensemble calibrate --references r000.npy ... r009.npy --nside 128 --out cal.json --gpu
stl-ensemble run --calibration cal.json --niter 2500 --max-eval 50000 --gpu --out ens
```

---

## Key parameters

| Option | Default | Description |
|---|---|---|
| `calibrate --references MAP...` | required | the reference maps (NESTED, any nside ≥ `--nside`, block-averaged) |
| `calibrate --nside N` | required | nside of the synthesis |
| `--ensemble B` | 10 | members |
| `run --niter N`, `--max-eval N` | , `5N/4` | L-BFGS iterations and evaluations; production: 2500 and 50000 |
| `run --ps-variance-upper X` | 383 at nside 128 | upper band of the C_ℓ variance term; required at other nsides (production rule `3 nside − 1`) |
| `run --init MAP`, `--init-noise conditional` | Gaussian random fields | members of a coarser run (`{m}` in the path), put on the children with details drawn conditionally on the references |
| `run --checkpoint N` | off | the best maps every `N` evaluations, for restarts |
| `--groups G` | 1 | MPI groups of ranks, `B / G` members each |
| `--gpu`, `--harmonic gpu` | off | ScatCov on the device; C_ℓ, PDF and Minkowski terms on the device (needs cuFFT) |

Every option: {doc}`../api/cli/stl-ensemble`.

## Outputs

| File | Content |
|---|---|
| `cal.json` | calibration: LUT, targets of the five blocks, initial C_ℓ |
| `ens.json` | `loss_history`, `blocks_history`, `blocks_final` (loss, mean and variance terms per block), `timers_seconds`, device and host peaks |
| `ens_mMMM_gaussian_nested.npy` | the variable `g` of member `MMM` |
| `ens_mMMM_physical_nested.npy` | `κ = f(g) − mean` of member `MMM` |
| `ens_checkpoint_*` | with `--checkpoint` |

---

## Minimal example

Ten references at nside 64 (any ten full-sky NESTED maps of the same field):

```bash
stl-ensemble calibrate --references refs/r00?_nside256_nested.npy --nside 64 --out cal.json --gpu
stl-ensemble run --calibration cal.json --niter 300 --ps-variance-upper 191 --gpu --out ens
```

On one T4: 19 s of calibration, then 1.7 s per evaluation for the ten members together. The
complete example with figures: {doc}`../tutorials/healpix_ensemble`.

---

## Pitfalls and costs

- **Convergence is slow**: the ensemble loss needs thousands of iterations (2869 evaluations at
  nside 128 to go from 61.8 to 0.63). Start from a coarser run with `--init ... --init-noise
  conditional` ({doc}`../lagrangian_synthesis`, section 6.7) to save part of them.
- `--ps-variance-upper` is required at every nside but 128.
- **Cost per evaluation**, ten members: nside 128, 3.1 s on three T4s (`--groups 5`); nside 1024,
  on one A100, see {doc}`../campeti_production_1024`. The Minkowski functionals are about half of
  an evaluation at nside 128 ({doc}`../harmonic_speed`).
- The extremes of `κ` are wider than the references' (the LUT extrapolates linearly beyond its
  extreme knots); the bulk of the histogram is within the references' scatter.
- One ensemble mode matches a population; the other, `stl-synth --ensemble B`, matches the mean
  statistics of the members to one target map ({doc}`stl-synth`).

Background, validation against P. Campeti's Python code, and timings: {doc}`../campeti_merge`.
