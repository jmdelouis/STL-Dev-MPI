//! Terms of the synthesis objective on the members' maps, on the device
//! (`stl-synth`): the kappa transform, the 4-children pyramid of the level
//! power and of the kurtosis per level, the pixel marginal (stable radix
//! sort), and gradient sums.  The kernels (`kernels.cu`) evaluate the
//! element-wise expressions of the host code of `stl-cli` in its order; the
//! sums over pixels go by chunks of `CHUNK` cells, each reduced in a fixed
//! tree, the chunks added in order on the host: deterministic, and equal to
//! the host's sequential sums to rounding.  The host keeps the reductions
//! of these sums (moments, losses, communicators).

use crate::device::{Arg, Gpu, NULL};
use cudarc::driver::CudaSlice;

/// Cells per partial sum.
const CHUNK: usize = 1 << 16;

/// Keys per tile of the radix sort (`RS_TILE` of `kernels.cu`).
const TILE: usize = 2048;

/// Knots, values and derivatives of a monotone cubic on the device.
pub struct DeviceTable {
    x: CudaSlice<f64>,
    y: CudaSlice<f64>,
    d: CudaSlice<f64>,
    n: usize,
}

impl DeviceTable {
    pub fn new(gpu: &Gpu, x: &[f64], y: &[f64], d: &[f64]) -> Self {
        assert!(x.len() >= 2 && y.len() == x.len() && d.len() == x.len());
        DeviceTable { x: gpu.upload(x), y: gpu.upload(y), d: gpu.upload(d), n: x.len() }
    }

    /// Standardised `kappa = (f(x gs + gm) - km) / ks` of the `n` values `x`
    /// and its derivative with respect to `x`.
    pub fn apply(&self, gpu: &Gpu, x: &CudaSlice<f64>, (gm, gs): (f64, f64), (km, ks): (f64, f64)) -> (CudaSlice<f64>, CudaSlice<f64>) {
        let n = x.len();
        let (k, dk) = (gpu.zeros::<f64>(n), gpu.zeros::<f64>(n));
        gpu.launch1(
            "kappa_pchip",
            n,
            &[
                gpu.a(x),
                n.into(),
                gpu.a(&self.x),
                gpu.a(&self.y),
                gpu.a(&self.d),
                (self.n as i32).into(),
                gm.into(),
                gs.into(),
                km.into(),
                ks.into(),
                gpu.a(&k),
                gpu.a(&dk),
            ],
        );
        (k, dk)
    }
}

/// Partial sums `[rows, n_chunks, width]` from the device, added in chunk
/// order: `[rows, width]`.
fn chunk_sums(gpu: &Gpu, buf: &CudaSlice<f64>, rows: usize, n_chunks: usize, width: usize) -> Vec<f64> {
    let h = gpu.download(buf);
    let mut out = vec![0.0; rows * width];
    for r in 0..rows {
        for c in 0..n_chunks {
            for k in 0..width {
                out[r * width + k] += h[(r * n_chunks + c) * width + k];
            }
        }
    }
    out
}

/// The 4-children pyramid of `rows` maps of `len` NESTED cells: the means
/// of the levels `1..=depth` (`[rows, len >> 2n]`); level 0 is the maps
/// themselves, passed to every method.
pub struct Pyramid {
    levels: Vec<CudaSlice<f64>>,
    rows: usize,
    len: usize,
}

impl Pyramid {
    pub fn new(gpu: &Gpu, x: &CudaSlice<f64>, rows: usize, len: usize, depth: usize) -> Self {
        assert_eq!(x.len(), rows * len);
        assert_eq!(len % (1 << (2 * depth)), 0, "rows not aligned on the pyramid");
        let mut levels: Vec<CudaSlice<f64>> = Vec::with_capacity(depth);
        for n in 1..=depth {
            let m = len >> (2 * n);
            let out = gpu.zeros::<f64>(rows * m);
            let (src, stride) = if n == 1 { (gpu.a(x), len) } else { (gpu.a(&levels[n - 2]), m * 4) };
            gpu.launch1("quad_mean", rows * m, &[src, stride.into(), m.into(), (rows as i32).into(), gpu.a(&out)]);
            levels.push(out);
        }
        Pyramid { levels, rows, len }
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    fn level(&self, gpu: &Gpu, x: &CudaSlice<f64>, n: usize) -> (Arg, usize) {
        if n == 0 {
            (gpu.a(x), self.len)
        } else {
            (gpu.a(&self.levels[n - 1]), self.len >> (2 * n))
        }
    }

    /// `(sum d^2, sum d^4)` of the details of every row and level `n <
    /// depth`, `[row * depth + n]`.
    pub fn moments(&self, gpu: &Gpu, x: &CudaSlice<f64>) -> Vec<[f64; 2]> {
        let depth = self.depth();
        let mut out = vec![[0.0; 2]; self.rows * depth];
        for n in 0..depth {
            let (a, sa) = self.level(gpu, x, n);
            let (p, sp) = self.level(gpu, x, n + 1);
            let m = self.len >> (2 * n);
            let n_chunks = m.div_ceil(CHUNK);
            let buf = gpu.zeros::<f64>(self.rows * n_chunks * 2);
            gpu.launch("detail_moments", (n_chunks as u32, self.rows as u32), 256, &[a, sa.into(), p, sp.into(), m.into(), CHUNK.into(), gpu.a(&buf)]);
            for (r, s) in chunk_sums(gpu, &buf, self.rows, n_chunks, 2).chunks(2).enumerate() {
                out[r * depth + n] = [s[0], s[1]];
            }
        }
        out
    }

    /// Addresses and row strides of the levels `0..=depth`.
    fn table(&self, gpu: &Gpu, x: &CudaSlice<f64>) -> (CudaSlice<u64>, CudaSlice<i64>) {
        let (mut ptrs, mut strides) = (Vec::new(), Vec::new());
        for n in 0..=self.depth() {
            let (a, s) = self.level(gpu, x, n);
            let Arg::Ptr(p) = a else { unreachable!() };
            ptrs.push(p);
            strides.push(s as i64);
        }
        (gpu.upload(&ptrs), gpu.upload(&strides))
    }

    /// `g[r, i] += sum_{n < L} coef[r L + n] d_n[r, i >> 2n]` (the level-power
    /// gradient), `L = coef.len() / rows <= depth`.
    pub fn level_power_grad(&self, gpu: &Gpu, x: &CudaSlice<f64>, coef: &[f64], g: &mut CudaSlice<f64>) {
        let l = coef.len() / self.rows;
        assert!(l <= self.depth() && coef.len() == l * self.rows);
        let (lv, ls) = self.table(gpu, x);
        let c = gpu.upload(coef);
        gpu.launch1(
            "level_power_grad",
            self.rows * self.len,
            &[gpu.a(&lv), gpu.a(&ls), (l as i32).into(), gpu.a(&c), self.len.into(), (self.rows as i32).into(), gpu.a(g)],
        );
    }

    /// `g[r, i] += dk[r, i] sum_k cf_0 (d^3 - c - cf_1 cf_2 d)` over the levels
    /// `lev[k]` (the kurtosis gradient), `cf` `[rows, lev.len(), 3]`.
    pub fn kurtosis_grad(&self, gpu: &Gpu, x: &CudaSlice<f64>, lev: &[usize], cf: &[f64], dk: Option<&CudaSlice<f64>>, g: &mut CudaSlice<f64>) {
        assert_eq!(cf.len(), self.rows * lev.len() * 3);
        assert!(lev.iter().all(|&n| n < self.depth()));
        let (lv, ls) = self.table(gpu, x);
        let levs: Vec<i32> = lev.iter().map(|&n| n as i32).collect();
        let (dl, c) = (gpu.upload(&levs), gpu.upload(cf));
        gpu.launch1(
            "kurtosis_grad",
            self.rows * self.len,
            &[
                gpu.a(&lv),
                gpu.a(&ls),
                gpu.a(&dl),
                (lev.len() as i32).into(),
                gpu.a(&c),
                self.len.into(),
                (self.rows as i32).into(),
                dk.map_or(NULL, |d| gpu.a(d)),
                gpu.a(g),
            ],
        );
    }
}

/// Stable radix sort of rows of `n` values (8 passes of 8-bit digits).
pub struct RadixSort {
    keys: [CudaSlice<u64>; 2],
    vals: [CudaSlice<u32>; 2],
    hist: CudaSlice<u32>,
    /// Block totals of the recursive scan of `hist`.
    sums: Vec<CudaSlice<u32>>,
    n: usize,
}

impl RadixSort {
    pub fn new(gpu: &Gpu, n: usize) -> Self {
        assert!(n < u32::MAX as usize);
        let m = 256 * n.div_ceil(TILE);
        let mut sums = Vec::new();
        let mut len = m;
        while len > 1024 {
            len = len.div_ceil(1024);
            sums.push(gpu.zeros::<u32>(len));
        }
        RadixSort {
            keys: [gpu.zeros(n), gpu.zeros(n)],
            vals: [gpu.zeros(n), gpu.zeros(n)],
            hist: gpu.zeros(m),
            sums,
            n,
        }
    }

    /// Bytes on the device for rows of `n` values.
    pub fn bytes(n: usize) -> usize {
        n * 24 + 256 * n.div_ceil(TILE) * 4 * 2
    }

    fn scan(&self, gpu: &Gpu, a: &CudaSlice<u32>, n: usize, depth: usize) {
        let blocks = n.div_ceil(1024);
        if blocks == 1 {
            gpu.launch("scan_block", (1, 1), 256, &[gpu.a(a), n.into(), NULL]);
            return;
        }
        let sums = &self.sums[depth];
        gpu.launch("scan_block", (blocks as u32, 1), 256, &[gpu.a(a), n.into(), gpu.a(sums)]);
        self.scan(gpu, sums, blocks, depth + 1);
        gpu.launch1("scan_add", n, &[gpu.a(a), n.into(), gpu.a(sums)]);
    }

    /// Positions of the `n` values at `x` in increasing order
    /// (`f64::total_cmp`, ties in position order).
    pub fn sort(&mut self, gpu: &Gpu, x: Arg) -> &CudaSlice<u32> {
        let n = self.n;
        let tiles = n.div_ceil(TILE);
        gpu.launch1("rs_init", n, &[x, n.into(), gpu.a(&self.keys[0]), gpu.a(&self.vals[0])]);
        for pass in 0..8 {
            let (src, dst) = (pass % 2, 1 - pass % 2);
            let shift = (8 * pass) as i32;
            gpu.launch("rs_hist", (tiles as u32, 1), 256, &[gpu.a(&self.keys[src]), n.into(), shift.into(), gpu.a(&self.hist)]);
            self.scan(gpu, &self.hist, self.hist.len(), 0);
            gpu.launch(
                "rs_scatter",
                (tiles as u32, 1),
                256,
                &[
                    gpu.a(&self.keys[src]),
                    gpu.a(&self.vals[src]),
                    n.into(),
                    shift.into(),
                    gpu.a(&self.hist),
                    gpu.a(&self.keys[dst]),
                    gpu.a(&self.vals[dst]),
                ],
            );
        }
        &self.vals[0]
    }
}

/// The marginal term of one row of `n` values at `x`, positions `perm` in
/// increasing order, sorted target `t`: `g[perm[q]] += coef (x[perm[q]] -
/// t[q])`; returns `sum (x[perm[q]] - t[q])^2` (chunks of positions in order).
pub fn marginal_row(gpu: &Gpu, x: Arg, perm: &CudaSlice<u32>, t: &CudaSlice<f64>, coef: f64, g: Arg) -> f64 {
    let n = t.len();
    let n_chunks = n.div_ceil(CHUNK);
    let buf = gpu.zeros::<f64>(n_chunks);
    gpu.launch("marginal_row", (n_chunks as u32, 1), 256, &[x, gpu.a(perm), gpu.a(t), n.into(), coef.into(), CHUNK.into(), g, gpu.a(&buf)]);
    chunk_sums(gpu, &buf, 1, n_chunks, 1)[0]
}

/// `g += a b` element-wise (`b` none: `g += a`).
pub fn add_product(gpu: &Gpu, g: &mut CudaSlice<f64>, a: &CudaSlice<f64>, b: Option<&CudaSlice<f64>>) {
    assert_eq!(g.len(), a.len());
    gpu.launch1("add_product", g.len(), &[gpu.a(g), gpu.a(a), b.map_or(NULL, |b| gpu.a(b)), g.len().into()]);
}
