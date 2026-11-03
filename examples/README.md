# Examples

Worked examples, each a `run.sh` script and a short README with the figures and timings of an
actual run (garoupe: 2 × Xeon Bronze 3204, 12 cores, 3 × Tesla T4, shared with other jobs at the
time).

| | Example | Tool | Run time (garoupe) |
|---|---|---|---|
| 01 | [HEALPix synthesis of one map, CPU and GPU](01_healpix_synthesis) | `stl-synth` | 7 min CPU, 24 s GPU |
| 02 | [Ensemble synthesis, 10 members, P. Campeti's method](02_healpix_ensemble) | `stl-ensemble` | 12 min GPU |
| 03 | [Statistics of maps](03_statistics) | `stl-stats` | 25 s |
| 04 | [2D synthesis, STL-Dev's example (10 × 256²), CPU and GPU](04_planar_synthesis) | `stl-synth2d` | 2 min CPU, 1.5 + 7 min GPU |
| 05 | [Multi-rank runs under MPI](05_mpi) | `mpirun stl-synth`, `stl-mpi-check` | 7 min |

```bash
cargo build --release -p stl-cli --features gpu          # 01-04 (no feature and GPU=0 without a device)
bash examples/01_healpix_synthesis/run.sh
bash examples/03_statistics/run.sh                         # uses the synthesis of 01
REFS="$(ls /data/r00?_nside256_nested.npy)" bash examples/02_healpix_ensemble/run.sh
TURB=/path/to/STL-Dev/data/test/Turb_6.npy bash examples/04_planar_synthesis/run.sh
MPIRUN="mpirun --bind-to none" GPUS=3 bash examples/05_mpi/run.sh   # binaries built with mpi (see 05)
```

Environment of every script (`common.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `STL_BIN` | `$CARGO_TARGET_DIR/release` or `target/release` | directory of the binaries |
| `PY` | `python3` | Python with numpy, matplotlib and healpix-geo (figures only) |
| `OUT` | `examples/<name>/out` | outputs (not versioned) |
| `GPU` | 1 | 0 skips the GPU runs |
| `NITER`, `RANKS`, ... | per example | see the head of each `run.sh` |

`plotting.py` draws the figures: Mollweide views and base faces of NESTED maps, loss curves,
`stl-stats` statistics, and one-line summaries of the run reports.

Data: examples 01, 03 and 05 use the WebSky κ map at nside 64 shipped in `fixtures/`. Example 02
needs ten reference maps (the Takahashi et al. 2017 full-sky convergence realisations were used).
Example 04 needs `data/test/Turb_6.npy` from the STL-Dev repository.
