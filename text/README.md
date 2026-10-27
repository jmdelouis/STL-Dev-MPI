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
| `TODO_2048.md` | what to run and edit when the nside 2048 ensembles are available |
| `paper.pdf` | the compiled draft, committed so that it can be shared: rebuild and commit it with every change of the text |

## Building

```bash
latexmk -pdf paper.tex
```

The A&A class is not on CTAN: `aa.cls`, `aa.bst` and `linenoaa.sty` come
from the A&A LaTeX macro package 9.4 (EDP Sciences, March 2026, from
aanda.org, author instructions) and are kept in this directory; `lineno`
comes from TeX Live. The draft builds to 15 pages with no warning (abstract
under the 300-word limit the class checks). A&A 9.4 conventions: `\corrauth`
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
| `fig_maps.py` | `maps_mollweide_n{N}.png`, `maps_gnomonic_n{N}.png` | maps |
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
- Kappa terms: `docs/kappa_power_spectrum.md` (branch
  `claude/kappa-kurtosis` for section 7 and the kurtosis term).
- Overlap mode: `stl-synth --overlap N` (branch `m8-overlap`, commit f6bbf01, on main as a1ff836; CPU engine only).
- Cosmic variance: `figures/table_zscores_n*.txt` (same definitions as
  `cv10.py` on hpc, `/scratch/jmdeloui/takahashi`).
