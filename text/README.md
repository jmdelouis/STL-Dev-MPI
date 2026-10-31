# STL-Dev-MPI paper (first draft)

Draft of the paper on STL-Dev-MPI: scattering-covariance synthesis of
HEALPix maps on distributed-memory machines (Rust, MPI, CUDA), with the
weak-lensing convergence experiment (ensembles of 10 syntheses of the
Takahashi et al. 2017 map r000, judged against the realizations r001-r009).
Target journal: A&A.

| Path | Content |
|---|---|
| `paper.tex` | the text; red `[TODO: ...]` marks in the PDF, `% NUM2048` comments on lines whose numbers come from the nside 512/1024 ensembles |
| `refs.bib` | bibliography; every entry checked against its DOI record, arXiv or the publisher (2026-10-27) |
| `figures/` | figures and generated tables (`*_n512.*`, `*_n1024.*`, then `*_n2048.*`), with the `.txt` summaries the text quotes |
| `scripts/` | the scripts of every data figure and table (below) |
| `TODO_2048.md` | what to run and edit when the nside 2048 ensemble is available |
| `paper.pdf` | the compiled draft, committed so that it can be shared: rebuild and commit it with every change of the text |

## Building

```bash
latexmk -pdf paper.tex
```

The A&A class is not on CTAN: `aa.cls`, `aa.bst` and `linenoaa.sty` come
from the A&A LaTeX macro package 9.4 (EDP Sciences, March 2026, from
aanda.org, author instructions) and are kept in this directory; `lineno`
comes from TeX Live. The draft builds to 25 pages, with Appendix B (the
learned prior for the finest levels), with no warning (abstract under the
300-word limit the class checks). A&A 9.4 conventions: `\corrauth`
for the corresponding author, `\email{...}` after each co-author (read by the
editorial system, not printed), no ORCID, `[longauth]` only for long author
lists. The float parameters and the `\FloatBarrier` before Sect. 7 keep the
experiment's figures ahead of the discussion; the A&A production redoes the
layout. Build products are ignored by git, except `paper.pdf`.

## Overleaf

This directory is also the Overleaf project
https://www.overleaf.com/project/6ae0c881be7354f25bfc0005, through the
Overleaf git bridge (a premium feature of the project owner's account) and
`git subtree`. GitHub stays the reference; the Overleaf project holds the
history of `text/` only. In the repository root, with the remote
`overleaf` = `https://git.overleaf.com/6ae0c881be7354f25bfc0005` (user name
`git`, password: a git token from Overleaf's Account Settings):

```bash
# Overleaf -> GitHub (first, before any push)
git subtree pull --prefix text overleaf main -m "Paper: changes from Overleaf"
git push origin main
# GitHub -> Overleaf
git subtree push --prefix text overleaf main
```

Pull before pushing and avoid editing on both sides at once. Pushes from git
can drop or move Overleaf's comments and tracked changes; Overleaf has no
branches or tags. Overleaf compiles its own PDF (main document: `paper.tex`,
pdfLaTeX); `paper.pdf` here is rebuilt with `latexmk` and committed.

## Figures

All data figures are computed on the hpc cluster, next to the maps (the
maps never leave the clusters); only PDFs, PNGs and text come back.

| Script | Figure / table | Data |
|---|---|---|
| `measure.py` | caches every statistic of each map in `CACHE/n{N}/{set}/{map}.npz` | realizations `rXXX_nside{N}_nested.npy`, members `synthesis_mMMM_nested.npy` (kappa, histogram-matched) |
| `fig_cl.py` | `cl_n{N}.pdf`, `.txt` | caches |
| `fig_levels.py` | `levels_n{N}.pdf` | caches |
| `fig_peaks_pdf.py` | `peaks_pdf_n{N}.pdf` | caches |
| `fig_minkowski.py` | `minkowski_n{N}_s0.pdf` (4 pixels), `_s1.pdf` (30 arcmin), `.txt` | caches |
| `table_zscores.py` | `table_zscores_n{N}.tex` (input by `paper.tex`), `.txt` | caches |
| `mf_levels.py` (`mf_levels.slurm`) | `mf_levels_n{N}.pdf`, `.tex`, `.txt`: Minkowski functionals of the maps degraded to each level, members and r001-r009 against r000 | maps (own cache `*_mfl.npz`) |
| `fig_maps.py` (`maps.slurm`) | `maps_mollweide_n{N}.png`, `maps_gnomonic_n{N}.png`: r000, r001, a Gaussian field with r000's spectrum and members, logarithmic colour scale as in Mousset et al. (2024) | maps |
| `fig_scaling.py` | `scaling.pdf`, `.txt` | `benchmarks/` reports of the repository (runs locally) |
| `make_all.sh` | measure + every figure for one nside | |
| `paper_figures.slurm` | the hpc job: Minkowski test, nside 512, 1024, 2048 realizations (`STEPS`) | |

`common.py` holds the definitions (pyramid details, peaks, bands,
thresholds, smoothing scales). The Minkowski functionals come from
`tools/minkowski.py`, tested by `tools/test_minkowski.py` against Gaussian
fields (Appendix A of the paper).

On hpc (layout of the first run, job 2094597, 2026-10-27):

```bash
# from the repository root, in WSL: copy the scripts to the cluster
tar cz text/scripts tools/minkowski.py tools/test_minkowski.py | ssh hpc 'tar xzf - -C /scratch/jmdeloui/paper/src'
ssh hpc 'cd /scratch/jmdeloui/paper/logs && sbatch --export=ALL,P=/scratch/jmdeloui/paper /scratch/jmdeloui/paper/src/text/scripts/paper_figures.slurm'
# fetch the figures (small files only)
ssh hpc 'cd /scratch/jmdeloui/paper/figures && tar cf - *.pdf *.png *.txt *.tex' | tar xf - -C text/figures
```

`/scratch/jmdeloui/paper/cache` keeps the measurements: a re-run measures
only the maps that are new or changed (at nside 2048, about 4.5 minutes per
map on 6 threads, 5 maps at a time).

The three PNGs `kappa_*_n256.png` are the figures of
`docs/kappa_power_spectrum.md` (nside 256 study of the kappa terms), kept
until they are redrawn in the paper's style.

## Sources of the numbers

- Method, validation, CPU scaling: `CLAUDE.md`, `README.md`, `docs/USAGE.md`
  section 7, `benchmarks/` (Datarmor, hpc), commit messages of the
  milestones.
- GPU: `docs/gpu.md`; H200 numbers from the run reports on hpc
  (`~/stl-mpi/ens/runs/*/syn.json`) and from
  `notebooks/takahashi_nside2048_results.ipynb` (nside 4096 on 2 H200).
  Shared engine against one engine per objective (Table `tab:gpu`,
  footnotes c and d, the paragraph "One engine per rank" and the
  conclusions): `docs/gpu.md` section 5.1 at commit fa2e742 of branch
  `claude/musing-elgamal-9897d9` (engine badc63b, on main; memory formula
  in section 2.7): hpc, one rank, A100-SXM4-80GB, gaussianised Takahashi
  kappa target, L = 4, `--weights ergodic --scatter-weight 1 --kappa-table
  knots --kappa-weight 0.1`, level-power and marginal weights 10 and 10000
  at nside 1024, 5 and 5000 at 2048, 3 evaluations, 6 host threads (16 on
  the H200). nside 1024 x 10: c15e8ec out of memory uncapped, 86.6 s,
  setup 156 s, peak 70.2 GB with `STL_GPU_CACHE_MB=6000`; shared engine
  uncapped 64.4 s, setup 86 s, peak 62.8 GB (cache 28.4 GB; job 2285199).
  nside 2048 x 5, shared engine: 394.7 s, setup 446 s, peak 77.7 GB (cache
  8.6 GB). H200 c15e8ec: 1024 x 10 capped at 16000 MB (44.5 s in the
  3-evaluation test, peak 98.0 GB; the table keeps the 54.3 s of the
  production ensemble), 2048 x 5 capped at 6000 MB (156.7 s, 139.2 GB,
  test 1977934).
  Diagonal (DIA) form of the convolution (paragraph after "Performance" in
  the GPU section, Table `tab:gpu` footnote e): PR jmdelouis/STL-Dev-MPI#3,
  merged into main as 8120aec (30 Oct): DIA is the default of
  `stl-synth --gpu`, `--gpu-conv stencils` / `STL_GPU_CONV=stencils` gives
  the bound stencils back; DIA against stencils on the fixtures: 5.3e-17
  (statistics) and 5.3e-15 (gradients) relative. Method and per-row bytes
  in `docs/dia_convolution.md`; A100 stencils against DIA (measured with
  the engine at 74c0b0e, before the merge):
  hpc job 2353875, one A100-SXM4-80GB, nside 1024 and 2048, one map,
  L = 4, 3 L-BFGS iterations (4 evaluations) from white noise on a
  lognormal target, cache `auto` (transposition), reports
  `benchmarks/dia/a100_synth/synth_n{1024,2048}_L4_{stencils,dia}.json`
  (one run per case); T4 figures (one T4, garoupe, against a bound-stencil
  run of the same series) and the A100 ones in `docs/gpu.md` section 5.0,
  kernel ratios in `docs/dia_convolution.md` section 5.2. Every other GPU
  figure of the paper (Table `tab:gpu`, Fig. `scaling` (c), the T4 times
  of `tab:mink512` and `tab:runs`) was measured with the bound stencils;
  the text says so and gives, for each, the DIA effect measured or
  expected (expected: per-row sizes of `docs/dia_convolution.md` and the
  A100 runs). nside 4096 on two H200s: hpc job 1700256 (test, 3
  evaluations, 60.2 s per evaluation, peak 124.5 GB, `gpu_geometry` 185 s
  on rank 0 against 181 s for the evaluations; the timers include the
  target statistics of the setup), report `~/stl-gpu/runs/1700256/test.json`
  on hpc.
  Recomputed wavelet fields (paragraph "Recomputed wavelet fields", Table
  `tab:gpu` footnotes g and h, the 2048 x 5 DIA sentence of "One engine per
  rank", Sect. 2 memory, discussion, conclusions; preliminary): branch
  `claude/musing-elgamal-9897d9`, engine 13ae057 / 3177d9c
  (`--gpu-fields keep|recompute|auto`, reverse chain per block), not merged
  into main on 30 Oct; `docs/gpu.md` sections 2.2, 2.7 and 5.2 at 47c515e,
  reports in `benchmarks/fields/` of that branch. Field formulas: keep
  `8BCH0(8/3 + L(4 + 8/9 + 5/16))` = 188 B, recompute
  `8BCH0(8/3 + L(8/9 + 5/16))` = 60 B per pixel and member at L = 4 (220 B
  before the per-block chain). garoupe T4, nside 1024, DIA, lognormal
  target, 4 evaluations: keep / recompute 22.56 / 23.14 s (1 member),
  30.56 / 30.42 s (2), 48.91 / 46.18 s and peak 14.54 / 11.93 GB (4).
  hpc job 2404157, A100-SXM4-80GB, Takahashi kappa ensemble of section 5.1
  at nside 2048, DIA, 3 evaluations, 6 threads: 5 members keep 163.2 s,
  peak 75.1 GB (fields 47.3, cache 13.1, rows 11.9); recompute 162.6 s,
  63.8 GB (fields 15.0, cache 30.9); 10 members keep does not fit (fields
  94.5 GB), recompute 332.8 s, setup 341 s, peak 67.5 GB = fields 30.1 +
  cache 22.3 + rows 11.9 + 3.2, device busy 11 %; losses of the two modes
  equal to 2.7e-15, 2048 x 5 keep against the bound-stencil run 2.3e-15.
  Objective's terms on the device (paragraph "The objective on the
  device", Table `tab:gpu` footnote i, conclusions; preliminary): same
  branch, commits 96c6bf1, 2c51bd9, cceca63 (`--gpu-terms device`, the
  default with one rank per group; L-BFGS vector operations threaded),
  `docs/gpu.md` sections 2.6b and 5.3 at 27134fe, reports
  `benchmarks/terms/t4/n1024_B10_{host_f65ce8e,device_96c6bf1,device_cceca63}.json`.
  garoupe, one T4, nside 1024 x 10, production terms (kappa table
  knots1024, kurtosis 32-1024, level power 10, marginal 10000, scatter 1,
  kappa weight 0.1, L = 4), fields recomputed, 4 evaluations, 12 host
  threads: host terms 205.1 s, peak 13.47 GB; device terms 107.5 s, peak
  13.61 GB (fields 7.52 GB, rows 2.98 GB, cache 2.17 GB); losses equal to
  1e-15; marginal 35.1 -> 1.2 s, kurtosis + kappa transform + level power
  24.5 -> 0.3 s, copies and transfers 19.0 -> 1.7 s, L-BFGS 4.8 -> 1.2 s,
  device busy 42 -> 65 % (setup included). The comparison with Irene
  (128.4 s on 640 cores, job 6087067) is per evaluation of the same terms.
  Several ranks per group and occupancy (feb832c, `docs/gpu.md` sections
  2.6b and 5.3 at 4d10f1f, reports `benchmarks/terms/t4/*_feb832c.json`,
  occupancy by `nvidia-smi` every 2 s, setup excluded): 1024 x 10 on one
  T4, 107.6 s, busy 95.4 % (106.1 s and 95.7 % without the per-phase
  synchronisation); 2048 x 5 on the three T4 (3 ranks, one T4 each, 4 host
  threads per rank, `--kurtosis-nside 32-2048`, `knots2048.npy`, 4
  evaluations): host terms 184.7 s, device terms 134.6 s, peak per T4
  13.30-13.51 GB in the reports (fields 5.19 GB, rows 3.98 GB, cache 3.0 GB
  per rank), busy 49.5 -> 68.7 %, marginal 30.1 -> 16.9 s (merge of the
  ranks' orders on the host), halo exchanges 24 s per evaluation (11.6 +
  8.2 + 4.6), losses identical; device objective against host terms on 1
  and 3 ranks: same loss, gradient within 1.2e-16 of its maximum.
  A100 with the terms on the device (`docs/gpu.md` section 5.3 at 22cefc6,
  reports `benchmarks/terms/a100/n2048_B10_{host,device}.json`): hpc job
  2423067 (build 4d10f1f), one A100-SXM4-80GB, nside 2048 x 10, production
  options (level power 10, marginal 10000, kappa table weight 0.1,
  kurtosis 1.0 on nside 32-2048, scatter 1), members from the nside-1024
  syntheses, fields recomputed, 3 evaluations, 6 host threads: host terms
  333.7 s, device terms 61.7 s (5.4 x), busy 16.2 -> 90.6 % (nvidia-smi
  every 2 s, setup excluded), peak 67.5 -> 67.9 GB (cache 22.3 GB in
  both), host work about 270 -> 3 s (copies 84.2, marginal 62.1, kappa
  transform 23.3, kurtosis 11.3, level power 9.3), losses equal to 1e-15,
  setup 330 s. The "17 times" against Datarmor uses the approximate
  1070 s per evaluation of job 7775549 (Table `tab:runs`).
- Objective with several reference maps (P. Campeti's five blocks;
  paragraphs "Several reference maps" and "Harmonic terms on the device",
  preliminary): `docs/campeti_merge.md` on branch `claude/ensemble-gpu-terms`
  (cfccb47, not merged on 31 Oct; based on the merge PR head 5391d45 of
  `claude/campeti-merge-19032c`). Section 2.3: the blocks. Section 8.3:
  Rust against his Python (`tools/dump_ensemble_reference.py`,
  `fixtures/ensemble`, nside 16): SHT 7e-14, blocks 1.3e-13, gradient
  9e-14. Section 8.5: full run at nside 128 on garoupe
  (`/tmp/jmdeloui-campeti/runs/full128`, revision 5c925f4, 3 T4, 5 groups
  of 2, harmonic terms on the host): 2 h 34 min, 2869 evaluations at
  3.10 s, loss 61.81 -> 0.625; references r000-r009 block-averaged to 128
  (`cal128.json`). Section 9: host profile (`host_profile`, 4 threads:
  0.65 / 20.1 / 133 s at 128 / 512 / 1024, SHT 57 / 79 / 87 %), exactness
  (`GpuSht` 2.9e-15, blocks 1.5e-13, gradient 9.0e-14), device profile
  (`device_profile`, one T4: 0.092 / 1.86 / 11.2 s), full evaluations at
  128 x 10 (6 evaluations: 1 T4 6.04 -> 3.17 s, host user time 183 ->
  39 s; 3 T4 3.10 -> 1.24 s), and the unmeasured extrapolation to 2048
  (section 9.4).
- Kappa terms: `docs/kappa_power_spectrum.md` (section 7: kurtosis term;
  section 8: Minkowski term and its nside-256 tests). The nside-512 test of
  the Minkowski term (Table `tab:mink512`): `mf_levels.py` judge on hpc,
  `/scratch/jmdeloui/mfterm` (jobs 2144055 and 2150122);
  the runs and the other statistics: `docs/kappa_power_spectrum.md` sections 8.4
  (term added to the converged ensemble) and 8.5 (term from nside 256 on), and
  the recommendation in section 6 (branch `claude/kappa-minkowski`, commit 9b323f8).
  The tail term (`--marginal-tail-weight`, `--marginal-tail`, commit e0496a1)
  with the Minkowski term from nside 256: judged on garoupe,
  `/tmp/jmdeloui-mf/cv10_n512ft.txt` and `/tmp/jmdeloui-mf/mfl512t/mf_levels_n512.txt`
  (28 Oct; not used in production).
- Irene nside-1024 ensemble with the kurtosis term (job 6087067, analysed on
  garoupe): `figures/cv10_n1024_irene.txt`, `figures/mf_levels_n1024_irene.txt`
  (300 iterations). Continued by 300 iterations on Datarmor with the same
  setting (commit 39d2f60, Minkowski and tail terms off, INIT_NOISE=0):
  job 7775548, 400 ranks in 10 groups of 40 on 16 nodes (448 cores), Intel
  MPI, 311 evaluations at 258 s, 22.4 h, loss 2.46 to 0.802 (finished
  29 Oct); members histogram-matched to kappa (`tools/gaussianize.py
  match`), judged against r000-r009 (files from `runs/` of the main checkout):
  `figures/cv10_n1024_datarmor.txt`, `figures/mf_levels_n1024_datarmor.txt`
  (Irene 300 it and Datarmor 600 it side by side; preliminary until the
  2048 analysis). Then the 2048 ensemble from these members, in steps
  (intermediate in the paper): step 1, Datarmor job 7775549 (100
  iterations, 114 evaluations at about 1065 s on the same 448 cores,
  33.5 h, loss 2.76e5 to 212), analysed against r000-r009:
  `figures/cv10_n2048_datarmor_7775549.txt`,
  `figures/mf_levels_n2048_datarmor_7775549.txt` (copies of `runs/` of the
  main checkout); step 2, hpc A100 job 2520589 (one A100-SXM4-80GB, binary
  4d10f1f: DIA, fields recomputed, objective terms on the device, 6 host
  threads; 100 iterations, 108 evaluations at 74.5 s, 2 h 24 with setup
  and histogram matching, peak 67.8 GB, loss 212 to 19.2; `syn.json`
  timers `host_lbfgs` 1630 s, `host_marginal` 56 s); step 3, A100 job
  2535705 (+150 iterations, running). Datarmor job 7775550 runs the same
  step 2 at about 1070 s per evaluation (the "14 times"). The Irene jobs
  6090002 and 6090003 were cancelled on 29 Oct (they duplicated the
  Datarmor chain).
- Overlap mode: `stl-synth --overlap N` (branch `m8-overlap`, commit f6bbf01, on main as a1ff836; CPU engine only).
- Cosmic variance: `figures/table_zscores_n*.txt` (same definitions as
  `cv10.py` on hpc, `/scratch/jmdeloui/takahashi`).
