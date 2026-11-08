//! The spherical U-Net of `stl_core::nn` on the device: same graph,
//! parameter buffer and arithmetic order where it matters (`nn.cu`), `f32`
//! activations, `f64` reductions.
//!
//! A sphere convolution is, tap by tap, a gather of the 4 interpolated
//! pixels (`nn_gather`, `[n, cin]`) and one SGEMM into the output (bias
//! first); its reverse recomputes the taps for the weight cotangent (SGEMM)
//! and sends the tap cotangent (SGEMM) back through the transposed stencil
//! of the tap (`nn_scatter`, no atomics).  Reductions over pixels (GroupNorm
//! statistics and cotangents, biases, head) are per-channel sums over
//! chunks of [`CHUNK`] pixels, then over the chunks in order: the results
//! are the same at every run.

use crate::cublas::{cublas, CuBlas, Handle, OP_N, OP_T};
use crate::device::{Arg, Gpu, NULL};
use cudarc::driver::CudaSlice;
use std::sync::Arc;
use stl_core::nn::{BlockParams, UNet};

/// Pixels per partial sum of the per-channel reductions.
pub const CHUNK: usize = 4096;

/// The stencil of one level on the device, with the transposed lists of
/// every tap.
pub struct DeviceStencil {
    n: usize,
    p: usize,
    idx: CudaSlice<u32>,
    w: CudaSlice<f32>,
    /// Per tap: offsets `[n + 1]`, rows (pixels), weights.
    taps: Vec<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<f32>)>,
}

impl DeviceStencil {
    fn new(gpu: &Gpu, st: &stl_core::nn::SphereStencil) -> Self {
        let (n, p) = (st.n, st.p);
        let mut taps = Vec::with_capacity(p);
        for t in 0..p {
            let mut ptr = vec![0u32; n + 1];
            let (mut row, mut w) = (Vec::new(), Vec::new());
            for j in 0..n {
                for e in st.t_ptr[j] as usize..st.t_ptr[j + 1] as usize {
                    let r = st.t_row[e] as usize;
                    if r % p == t {
                        row.push((r / p) as u32);
                        w.push(st.t_w[e]);
                    }
                }
                ptr[j + 1] = row.len() as u32;
            }
            taps.push((gpu.upload(&ptr), gpu.upload(&row), gpu.upload(&w)));
        }
        DeviceStencil { n, p, idx: gpu.upload(&st.idx), w: gpu.upload(&st.w), taps }
    }
}

/// Device values of one block kept for its reverse pass.
pub struct BlockTape {
    x: CudaSlice<f32>,
    y1: CudaSlice<f32>,
    s1: CudaSlice<f64>,
    h1: CudaSlice<f32>,
    y2: CudaSlice<f32>,
    s2: CudaSlice<f64>,
    /// Output of the block.
    pub out: CudaSlice<f32>,
}

/// Tape of one map through the network.
pub struct Tape {
    enc: Vec<BlockTape>,
    dec: Vec<BlockTape>,
}

/// The network on a device: stencils, parameters, their cotangents and the
/// Adam state.
pub struct GpuUNet {
    /// The device.
    pub gpu: Arc<Gpu>,
    /// The host description (layout, configuration).
    pub net: UNet,
    stencils: Vec<DeviceStencil>,
    blas: &'static CuBlas,
    handle: Handle,
    /// Parameters (the flat buffer of `UNet`).
    pub w: CudaSlice<f32>,
    /// Parameter cotangents, accumulated by [`backward`](Self::backward).
    pub g: CudaSlice<f32>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// Adam steps taken.
    pub t: u64,
}

impl Drop for GpuUNet {
    fn drop(&mut self) {
        self.blas.destroy(self.handle);
    }
}

impl GpuUNet {
    /// The network `net` with parameters `w` on `gpu`.
    pub fn new(gpu: Arc<Gpu>, net: UNet, w: &[f32]) -> Result<Self, String> {
        assert_eq!(w.len(), net.size, "parameter buffer");
        let blas = cublas()?;
        gpu.ctx.bind_to_thread().map_err(|e| format!("CUDA: {e:?}"))?;
        let handle = blas.handle(gpu.stream.cu_stream() as *mut std::ffi::c_void)?;
        let stencils = net.stencils.iter().map(|s| DeviceStencil::new(&gpu, s)).collect();
        let size = net.size;
        Ok(GpuUNet { w: gpu.upload(w), g: gpu.zeros(size), m: gpu.zeros(size), v: gpu.zeros(size), t: 0, stencils, blas, handle, net, gpu })
    }

    /// Host copy of the parameters.
    pub fn params(&self) -> Vec<f32> {
        self.gpu.download(&self.w)
    }

    /// Replaces the parameters.
    pub fn set_params(&mut self, w: &[f32]) {
        self.gpu.copy_in(w, &mut self.w);
    }

    /// Host copy of the parameter cotangents.
    pub fn grads(&self) -> Vec<f32> {
        self.gpu.download(&self.g)
    }

    /// Zeroes the parameter cotangents.
    pub fn zero_grad(&mut self) {
        self.gpu.clear(&mut self.g);
    }

    /// Resets the Adam state.
    pub fn reset_adam(&mut self) {
        self.gpu.clear(&mut self.m);
        self.gpu.clear(&mut self.v);
        self.t = 0;
    }

    /// One Adam step of learning rate `lr` from the accumulated cotangents
    /// (torch's `Adam`, `betas = (0.9, 0.999)`, `eps = 1e-8`).
    pub fn adam_step(&mut self, lr: f64) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32 as f64, 0.999f32 as f64, 1e-8f32 as f64);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        let step = (lr / bc1) as f32 as f64;
        let sbc2 = bc2.sqrt() as f32 as f64;
        let g = &self.gpu;
        g.launch1("nn_adam", self.net.size, &[g.a(&self.w), g.a(&self.g), g.a(&self.m), g.a(&self.v), self.net.size.into(), b1.into(), b2.into(), eps.into(), step.into(), sbc2.into()]);
    }

    fn wp(&self, off: usize) -> Arg {
        self.gpu.at(&self.w, off)
    }

    fn gp(&self, off: usize) -> Arg {
        self.gpu.at(&self.g, off)
    }

    fn addr(a: Arg) -> u64 {
        match a {
            Arg::Ptr(p) => p,
            _ => unreachable!(),
        }
    }

    /// Per-channel sums (mode of `nn_chan_sums`) of `[n, c]`: `[2 c]` on the device.
    #[allow(clippy::too_many_arguments)]
    fn chan_sums(&self, mode: i32, a: Arg, b: Arg, stats: Arg, gamma: Arg, beta: Arg, n: usize, c: usize, groups: usize) -> CudaSlice<f64> {
        let g = &self.gpu;
        let chunks = n.div_ceil(CHUNK);
        let partial = g.zeros::<f64>(chunks * 2 * c);
        let slope = self.net.cfg.slope as f64;
        g.launch(
            "nn_chan_sums",
            (chunks as u32, 1),
            128,
            &[mode.into(), a, b, stats, gamma, beta, slope.into(), n.into(), c.into(), groups.into(), CHUNK.into(), g.a(&partial)],
        );
        let out = g.zeros::<f64>(2 * c);
        g.launch1("nn_chan_finish", 2 * c, &[g.a(&partial), chunks.into(), (2 * c).into(), g.a(&out)]);
        out
    }

    /// Sphere convolution of `x` `[n, cin]` with the weights at `wo`, bias at `bo`.
    fn conv(&self, st: &DeviceStencil, x: &CudaSlice<f32>, cin: usize, cout: usize, wo: usize, bo: usize) -> CudaSlice<f32> {
        let g = &self.gpu;
        let (n, p) = (st.n, st.p);
        let pc = p * cin;
        let y = g.zeros::<f32>(n * cout);
        g.launch1("nn_bias", n * cout, &[g.a(&y), self.wp(bo), n.into(), cout.into()]);
        let taps = g.zeros::<f32>(n * cin);
        g.ctx.bind_to_thread().unwrap();
        for t in 0..p {
            g.launch1("nn_gather", n * cin, &[g.a(x), g.a(&st.idx), g.a(&st.w), n.into(), p.into(), t.into(), cin.into(), g.a(&taps)]);
            let wt = Self::addr(self.wp(wo + t * cin));
            unsafe { self.blas.sgemm(self.handle, OP_T, OP_N, cout, n, cin, 1.0, wt, pc, g.ptr(&taps), cin, 1.0, g.ptr(&y), cout) }.unwrap_or_else(|e| panic!("{e}"));
        }
        y
    }

    /// Reverse of [`conv`](Self::conv): adds the weight and bias cotangents,
    /// returns the input cotangent.
    #[allow(clippy::too_many_arguments)]
    fn conv_back(&self, st: &DeviceStencil, x: &CudaSlice<f32>, cin: usize, cout: usize, wo: usize, bo: usize, dy: &CudaSlice<f32>) -> CudaSlice<f32> {
        let g = &self.gpu;
        let (n, p) = (st.n, st.p);
        let pc = p * cin;
        let db = self.chan_sums(2, g.a(dy), NULL, NULL, NULL, NULL, n, cout, 1);
        g.launch1("nn_add_f64", cout, &[g.a(&db), cout.into(), self.gp(bo)]);
        let taps = g.zeros::<f32>(n * cin);
        let dtaps = g.zeros::<f32>(n * cin);
        let dx = g.zeros::<f32>(n * cin);
        g.ctx.bind_to_thread().unwrap();
        for t in 0..p {
            g.launch1("nn_gather", n * cin, &[g.a(x), g.a(&st.idx), g.a(&st.w), n.into(), p.into(), t.into(), cin.into(), g.a(&taps)]);
            let gw = Self::addr(self.gp(wo + t * cin));
            let wt = Self::addr(self.wp(wo + t * cin));
            unsafe {
                self.blas.sgemm(self.handle, OP_N, OP_T, cin, cout, n, 1.0, g.ptr(&taps), cin, g.ptr(dy), cout, 1.0, gw, pc).unwrap_or_else(|e| panic!("{e}"));
                self.blas.sgemm(self.handle, OP_N, OP_N, cin, n, cout, 1.0, wt, pc, g.ptr(dy), cout, 0.0, g.ptr(&dtaps), cin).unwrap_or_else(|e| panic!("{e}"));
            }
            let (ptr, row, tw) = &st.taps[t];
            g.launch1("nn_scatter", n * cin, &[g.a(&dtaps), g.a(ptr), g.a(row), g.a(tw), n.into(), cin.into(), g.a(&dx)]);
        }
        dx
    }

    /// GroupNorm + leaky ReLU of `y` `[n, c]` with the affine parameters at
    /// `ga`, `be`: the output and the statistics.
    fn norm(&self, y: &CudaSlice<f32>, n: usize, c: usize, groups: usize, ga: usize, be: usize) -> (CudaSlice<f32>, CudaSlice<f64>) {
        let g = &self.gpu;
        let sums = self.chan_sums(0, g.a(y), NULL, NULL, NULL, NULL, n, c, groups);
        let stats = g.zeros::<f64>(2 * groups);
        g.launch1("nn_gn_stats", groups, &[g.a(&sums), c.into(), groups.into(), n.into(), 1e-5f64.into(), g.a(&stats)]);
        let h = g.zeros::<f32>(n * c);
        let slope = self.net.cfg.slope as f64;
        g.launch1("nn_gn_apply", n * c, &[g.a(y), g.a(&stats), self.wp(ga), self.wp(be), slope.into(), n.into(), c.into(), groups.into(), g.a(&h)]);
        (h, stats)
    }

    /// Reverse of [`norm`](Self::norm): the cotangent of `y` from that of the output.
    #[allow(clippy::too_many_arguments)]
    fn norm_back(&self, y: &CudaSlice<f32>, stats: &CudaSlice<f64>, dh: &CudaSlice<f32>, n: usize, c: usize, groups: usize, ga: usize, be: usize) -> CudaSlice<f32> {
        let g = &self.gpu;
        let slope = self.net.cfg.slope as f64;
        let sums = self.chan_sums(1, g.a(y), g.a(dh), g.a(stats), self.wp(ga), self.wp(be), n, c, groups);
        let s = g.zeros::<f64>(2 * groups);
        g.launch("nn_gn_params", (1, 1), 1, &[g.a(&sums), self.wp(ga), c.into(), groups.into(), self.gp(ga), self.gp(be), g.a(&s)]);
        let dy = g.zeros::<f32>(n * c);
        g.launch1("nn_gn_back", n * c, &[g.a(y), g.a(dh), g.a(stats), self.wp(ga), self.wp(be), g.a(&s), slope.into(), n.into(), c.into(), groups.into(), g.a(&dy)]);
        dy
    }

    fn block(&self, b: &BlockParams, x: CudaSlice<f32>) -> BlockTape {
        let st = &self.stencils[b.level];
        let n = st.n;
        let y1 = self.conv(st, &x, b.cin, b.cout, b.at[0], b.at[1]);
        let (h1, s1) = self.norm(&y1, n, b.cout, b.groups, b.at[4], b.at[5]);
        let y2 = self.conv(st, &h1, b.cout, b.cout, b.at[2], b.at[3]);
        let (out, s2) = self.norm(&y2, n, b.cout, b.groups, b.at[6], b.at[7]);
        BlockTape { x, y1, s1, h1, y2, s2, out }
    }

    fn block_back(&self, b: &BlockParams, t: &BlockTape, dout: &CudaSlice<f32>) -> CudaSlice<f32> {
        let st = &self.stencils[b.level];
        let n = st.n;
        let dy2 = self.norm_back(&t.y2, &t.s2, dout, n, b.cout, b.groups, b.at[6], b.at[7]);
        let dh1 = self.conv_back(st, &t.h1, b.cout, b.cout, b.at[2], b.at[3], &dy2);
        let dy1 = self.norm_back(&t.y1, &t.s1, &dh1, n, b.cout, b.groups, b.at[4], b.at[5]);
        self.conv_back(st, &t.x, b.cin, b.cout, b.at[0], b.at[1], &dy1)
    }

    /// Column copy `nn_cols`.
    fn cols(&self, mode: i32, src: &CudaSlice<f32>, dst: &CudaSlice<f32>, n: usize, cs: usize, cd: usize, off: usize) {
        let g = &self.gpu;
        g.launch1("nn_cols", n * cs, &[mode.into(), g.a(src), g.a(dst), n.into(), cs.into(), cd.into(), off.into()]);
    }

    /// The conditioning maps `[n_0, cond]` (pixel-major) pooled to every level.
    pub fn cond_levels(&self, cond: &CudaSlice<f32>) -> Vec<CudaSlice<f32>> {
        let nc = self.net.cfg.cond;
        if nc == 0 {
            return Vec::new();
        }
        let g = &self.gpu;
        let mut out = vec![g.copy(cond)];
        for k in 1..self.stencils.len() {
            let n = self.stencils[k].n;
            let y = g.zeros::<f32>(n * nc);
            g.launch1("nn_pool", n * nc, &[g.a(out.last().unwrap()), n.into(), nc.into(), g.a(&y)]);
            out.push(y);
        }
        out
    }

    /// `(s_0, s_1)` of the per-map scales.
    fn scale_pair(&self, scales: &[f64]) -> (f64, f64) {
        assert_eq!(scales.len(), self.net.cfg.maps, "one scale per map");
        (scales[0], *scales.last().unwrap())
    }

    /// `u = d - scale r(d / scale)` of one input `d` `[maps, n_0]` on the
    /// device (`scales`: one per map), with the conditioning maps of
    /// [`cond_levels`](Self::cond_levels), and the tape.
    pub fn forward(&self, d: &CudaSlice<f64>, scales: &[f64], cond: &[CudaSlice<f32>]) -> (CudaSlice<f64>, Tape) {
        let g = &self.gpu;
        let (nc, m) = (self.net.cfg.cond, self.net.cfg.maps);
        let (s0, s1) = self.scale_pair(scales);
        let n0 = self.stencils[0].n;
        assert_eq!(d.len(), m * n0, "one input of maps");
        let mut t0 = stl_core::prof::start(|| g.sync());
        let x = g.zeros::<f32>(n0 * (m + nc));
        g.launch1("nn_input", n0, &[g.a(d), s0.into(), s1.into(), if nc > 0 { g.a(&cond[0]) } else { NULL }, n0.into(), m.into(), nc.into(), g.a(&x)]);
        let mut h = x;
        let mut enc: Vec<BlockTape> = Vec::new();
        for (k, b) in self.net.enc.iter().enumerate() {
            if k > 0 {
                let (n, c) = (self.stencils[k].n, self.net.enc[k - 1].cout);
                let y = g.zeros::<f32>(n * c);
                g.launch1("nn_pool", n * c, &[g.a(&h), n.into(), c.into(), g.a(&y)]);
                h = y;
            }
            let t = self.block(b, h);
            h = g.copy(&t.out);
            enc.push(t);
            stl_core::prof::mark(&mut t0, &format!("unet.fwd.enc.L{k}"), || g.sync());
        }
        let ch = &self.net.cfg.channels;
        let mut dec = Vec::new();
        for b in &self.net.dec {
            let k = b.level;
            let n = self.stencils[k].n;
            let (cprev, ck) = (ch[k + 1], ch[k]);
            let width = cprev + ck + nc;
            let x = g.zeros::<f32>(n * width);
            let up = g.zeros::<f32>(n * cprev);
            g.launch1("nn_unpool", n * cprev, &[g.a(&h), (n / 4).into(), cprev.into(), g.a(&up)]);
            self.cols(0, &up, &x, n, cprev, width, 0);
            self.cols(0, &enc[k].out, &x, n, ck, width, cprev);
            if nc > 0 {
                self.cols(0, &cond[k], &x, n, nc, width, cprev + ck);
            }
            let t = self.block(b, x);
            h = g.copy(&t.out);
            dec.push(t);
            stl_core::prof::mark(&mut t0, &format!("unet.fwd.dec.L{k}"), || g.sync());
        }
        let c0 = ch[0];
        let r = g.zeros::<f32>(m * n0);
        g.launch1("nn_head", m * n0, &[g.a(&h), self.wp(self.net.head[0]), self.wp(self.net.head[1]), n0.into(), c0.into(), m.into(), g.a(&r)]);
        let u = g.zeros::<f64>(m * n0);
        g.launch1("nn_output", m * n0, &[g.a(d), g.a(&r), s0.into(), s1.into(), n0.into(), m.into(), g.a(&u)]);
        stl_core::prof::mark(&mut t0, "unet.fwd.head", || g.sync());
        (u, Tape { enc, dec })
    }

    /// Adds to the parameter cotangents those of `du` `[maps, n_0]` (the
    /// cotangent of `u` of [`forward`](Self::forward), at fixed `d` and
    /// `scales`).
    pub fn backward(&self, tape: &Tape, scales: &[f64], du: &CudaSlice<f64>) {
        let g = &self.gpu;
        let (nc, m) = (self.net.cfg.cond, self.net.cfg.maps);
        let (s0, s1) = self.scale_pair(scales);
        let ch = &self.net.cfg.channels;
        let n0 = self.stencils[0].n;
        let c0 = ch[0];
        let mut t0 = stl_core::prof::start(|| g.sync());
        let dr = g.zeros::<f32>(m * n0);
        g.launch1("nn_output_back", m * n0, &[g.a(du), s0.into(), s1.into(), n0.into(), m.into(), g.a(&dr)]);
        let last = &tape.dec.last().map_or(&tape.enc[0].out, |t| &t.out);
        // Head, per output map: weight (sum dr h), bias (sum dr); input (sum over the maps of dr hw).
        for j in 0..m {
            let drj = g.at(&dr, j * n0);
            let hs = self.chan_sums(3, g.a(last), drj, NULL, NULL, NULL, n0, c0, 1);
            g.launch1("nn_add_f64", c0, &[g.a(&hs), c0.into(), self.gp(self.net.head[0] + j * c0)]);
            let hb = self.chan_sums(2, drj, NULL, NULL, NULL, NULL, n0, 1, 1);
            g.launch1("nn_add_f64", 1, &[g.a(&hb), 1usize.into(), self.gp(self.net.head[1] + j)]);
        }
        let mut dh = g.zeros::<f32>(n0 * c0);
        g.launch1("nn_head_back", n0 * c0, &[g.a(&dr), self.wp(self.net.head[0]), n0.into(), c0.into(), m.into(), g.a(&dh)]);
        stl_core::prof::mark(&mut t0, "unet.bwd.head", || g.sync());
        let levels = self.stencils.len();
        let mut dskip: Vec<Option<CudaSlice<f32>>> = (0..levels).map(|_| None).collect();
        for i in (0..self.net.dec.len()).rev() {
            let b = &self.net.dec[i];
            let k = b.level;
            let n = self.stencils[k].n;
            let din = self.block_back(b, &tape.dec[i], &dh);
            let (cprev, ck) = (ch[k + 1], ch[k]);
            let width = cprev + ck + nc;
            let dup = g.zeros::<f32>(n * cprev);
            self.cols(1, &din, &dup, n, cprev, width, 0);
            let ds = g.zeros::<f32>(n * ck);
            self.cols(1, &din, &ds, n, ck, width, cprev);
            dskip[k] = Some(ds);
            dh = g.zeros::<f32>(n / 4 * cprev);
            g.launch1("nn_unpool_back", n / 4 * cprev, &[g.a(&dup), (n / 4).into(), cprev.into(), g.a(&dh)]);
            stl_core::prof::mark(&mut t0, &format!("unet.bwd.dec.L{k}"), || g.sync());
        }
        for k in (0..levels).rev() {
            let b = &self.net.enc[k];
            if let Some(s) = dskip[k].take() {
                let n = self.stencils[k].n;
                self.cols(2, &s, &dh, n, b.cout, b.cout, 0);
            }
            let din = self.block_back(b, &tape.enc[k], &dh);
            if k > 0 {
                let (n, c) = (self.stencils[k].n, self.net.enc[k - 1].cout);
                dh = g.zeros::<f32>(4 * n * c);
                g.launch1("nn_pool_back", 4 * n * c, &[g.a(&din), n.into(), c.into(), g.a(&dh)]);
            }
            stl_core::prof::mark(&mut t0, &format!("unet.bwd.enc.L{k}"), || g.sync());
        }
    }
}
