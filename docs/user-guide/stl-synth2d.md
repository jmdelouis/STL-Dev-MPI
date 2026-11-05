# `stl-synth2d`: synthesis of 2D images

`stl-synth2d` synthesises periodic planar images from the statistics of a set of target images.
It runs the block-wise pyramid on a periodic grid (square tiles instead of NESTED blocks) with
the 2D kernels of STL-Dev (`STL_2D_Kernel_Torch`: `L = 4` complex 5 × 5 wavelets, stride-2
raised-cosine decimation), and matches STL-Dev's statistics and gradients to 1e-10.

Two losses:

- `--mode stldev`: the loss of STL-Dev's `synthesize_from_maps` (ScatCov and power spectrum of
  every image against the targets' mean), for comparisons with STL-Dev;
- `--mode ensemble` (default): ScatCov, power spectrum and Minkowski functionals, with the
  **mean and the variance over the batch** against those of the targets.

```bash
stl-synth2d --target Turb_6.npy --ntarget 10 --nbatch 10 --mode ensemble --warm 100 --niter 300 --gpu --out run
```

---

## Key parameters

| Option | Default | Description |
|---|---|---|
| `--target PATH` | required | `.npy` `[Bt, H, W]` or `[Bt, C, H, W]` |
| `--ntarget N`, `--nbatch B` | all, 10 | target images used, images synthesised |
| `--mode M` | `ensemble` | `ensemble` or `stldev` |
| `--J N` | `log2(min(H, W)) − 2` | scales |
| `--tile T` | 64 | tile side of the block-wise engine: the memory knob |
| `--niter N`, `--history N` | 100, 50 | L-BFGS |
| `--warm N` | 0 | first `N` iterations on the `stldev` loss (a well-conditioned start for `ensemble`) |
| `--c2f LEVELS`, `--c2f-iter`, `--c2f-detail` | off | coarse-to-fine start: pixel L-BFGS at each coarse side, then upsampling with details drawn conditionally on the coarse value and gradient |
| `--w-sc`, `--w-ps`, `--w-mf` | 1 | weights of the three blocks of the ensemble loss |
| `--gpu` | off | the batch on one CUDA device |
| `--out PREFIX` | | `PREFIX_maps.npy`, `PREFIX_target.npy`, `PREFIX_features_*.npy`, `PREFIX_mf_*.npy`, `PREFIX.json` |

Research options, documented in {doc}`../lagrangian_synthesis`: `--lagrangian` (optimise a
displacement of the images), `--transport` (transport steps), `--homotopy` (band-wise homotopy),
`--trace`. Every option: {doc}`../api/cli/stl-synth2d`.

---

## Minimal example

STL-Dev's 2D example: 10 images of 256² from `data/test/Turb_6.npy` of the STL-Dev repository.

```bash
stl-synth2d --target Turb_6.npy --ntarget 10 --nbatch 10 --mode stldev --niter 100 --out stldev
stl-synth2d --target Turb_6.npy --ntarget 10 --nbatch 10 --c2f 64,128 --niter 300 --gpu --out c2f
python tools/plot_planar.py --run c2f --fig fig
```

Walk-throughs: {doc}`../tutorials/planar_synthesis` (scripts) and
{doc}`../tutorials/turbulence_2d` (notebook).

---

## Pitfalls and costs

- **Cold starts of the ensemble loss stall** (loss around 145 from white noise at 256²): start
  with `--warm 100` or `--c2f`. Coarse-to-fine with conditional details reaches the loss of pixel
  L-BFGS 6.6 times sooner ({doc}`../lagrangian_synthesis`, section 6.4).
- **Tiles**: 32 to 64 are fastest and smallest on the CPU; on an A100, tiles of 256 or whole
  levels (tiles of 64 are latency-bound there).
- **Cost per evaluation** (STL-Dev's loss, idle machines): 10 × 256², 0.46 s on 12 cores and
  0.3 GB, 0.18 s on a T4, 0.032 s on an A100 (STL-Dev: 5.0 s and 4.5 GB on the same cores);
  one 4096² image, 29 s and 2.9 GB on the CPU, 0.6 s on an A100, where STL-Dev runs out of 80 GB.
  The ensemble loss adds the Minkowski functionals and the spectra of every image (about ×1.8).
- Threads: `STL_THREADS` (default all cores).

Design, exactness and benchmarks: {doc}`../planar_2d`.
