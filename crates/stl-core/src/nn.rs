//! Spherical U-Net on the HEALPix NESTED grid, `d -> u = d - scale r(d /
//! scale)`, in `f32`, with its exact reverse pass and Adam: the native port
//! of `stlia.unet.UNet` (`conv="sphere"`, `norm="group"`) of Healpix-STL-IA.
//!
//! Convolution (`SphereConv`): for every pixel the `ksz x ksz` taps of the
//! STL stencil ([`crate::stencil::local_kernel_grid`]) rotated to the pixel
//! and its gauge ([`crate::stencil::rotation_matrices`], orientation `g` of
//! `G`), each the bilinear interpolation of 4 NESTED pixels
//! ([`crate::healpix::interp_weights_nested`]); then `y = W taps + b`, `W`
//! `[cout, ksz^2 cin]` tap-major (`p cin + c`).  The reference uses one
//! fixed gauge (`G = 1`); the stencil of any orientation is available for a
//! gauge-equivariant variant.
//!
//! Network (levels `k = 0..L`, level `k` at `nside >> k`): encoder `pool`
//! (mean of the 4 NESTED children) then `SphereBlock`; decoder
//! `cat[unpool(x), skip_k, cond_k]` then `SphereBlock`; `SphereBlock` = (conv,
//! GroupNorm(min(8, c)), LeakyReLU) twice; head `1 x 1` conv to `maps`
//! channels.  `maps` = 2 (`UNet(maps=2)`): a pair of maps, both in the
//! input and in the output, each standardised by its own scale.
//! Activations are pixel-major `[n, C]` per map; the network's inputs and
//! outputs are map-major `[maps, n]`.  Parameters are one flat
//! buffer in the order and layouts of the torch `state_dict`
//! ([`UNet::params`]).
//!
//! Host threads: `STL_THREADS` (else every CPU); every sum has a fixed
//! order, the long ones (convolutions, cotangents) in double precision.

use crate::healpix::{interp_weights_nested, nest2xyf, order_of, xyf2ang};
use crate::stencil::{apply, local_kernel_grid, rotation_matrices, Gauge};

/// Stencil of one level: for every pixel and tap, 4 source pixels and
/// weights, and the transposed lists (per source pixel: `(row, weight)`,
/// `row = pixel * taps + tap`) for the reverse pass.
pub struct SphereStencil {
    /// Pixels of the level.
    pub n: usize,
    /// Taps per pixel, `ksz^2`.
    pub p: usize,
    /// Source pixels `[n, p, 4]`.
    pub idx: Vec<u32>,
    /// Weights `[n, p, 4]` (`f32` as the torch sparse matrix).
    pub w: Vec<f32>,
    /// Offsets of the transposed lists `[n + 1]`.
    pub t_ptr: Vec<u32>,
    /// Rows of the transposed lists.
    pub t_row: Vec<u32>,
    /// Weights of the transposed lists.
    pub t_w: Vec<f32>,
}

impl SphereStencil {
    /// The stencil at `nside` for orientation `g` of `g_count` in `gauge`.
    pub fn new(nside: usize, ksz: usize, gauge: Gauge, g: usize, g_count: usize) -> Self {
        let order = order_of(nside as i64);
        let n = 12 * nside * nside;
        let p = ksz * ksz;
        let grid = local_kernel_grid(ksz, nside as i64);
        let rows = par_map_chunks(n, threads(), |range| {
            let mut idx = Vec::with_capacity(range.len() * p * 4);
            let mut w = Vec::with_capacity(range.len() * p * 4);
            for i in range {
                let (ix, iy, face) = nest2xyf(i as i64, order);
                let (theta, phi) = xyf2ang(ix, iy, face, nside as i64);
                let rot = rotation_matrices(theta, phi, g_count, gauge)[g];
                for v in &grid {
                    let (nb, wt) = interp_weights_nested(apply(&rot, v), order);
                    idx.extend(nb.iter().map(|&x| x as u32));
                    w.extend(wt.iter().map(|&x| x as f32));
                }
            }
            (idx, w)
        });
        let (mut idx, mut w) = (Vec::with_capacity(n * p * 4), Vec::with_capacity(n * p * 4));
        for (a, b) in rows {
            idx.extend(a);
            w.extend(b);
        }
        let mut count = vec![0u32; n + 1];
        for &s in &idx {
            count[s as usize + 1] += 1;
        }
        for j in 0..n {
            count[j + 1] += count[j];
        }
        let t_ptr = count.clone();
        let mut fill = count;
        let (mut t_row, mut t_w) = (vec![0u32; idx.len()], vec![0f32; idx.len()]);
        for (e, (&s, &wt)) in idx.iter().zip(&w).enumerate() {
            let at = fill[s as usize] as usize;
            t_row[at] = (e / 4) as u32;
            t_w[at] = wt;
            fill[s as usize] += 1;
        }
        SphereStencil { n, p, idx, w, t_ptr, t_row, t_w }
    }
}

/// Host threads: `STL_THREADS`, else the CPUs available.
pub fn threads() -> usize {
    std::env::var("STL_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1)
}

/// `f` on contiguous chunks of `0..n` (at most `threads`), results in order.
fn par_map_chunks<T: Send>(n: usize, threads: usize, f: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
    let t = threads.clamp(1, n.max(1));
    let chunk = n.div_ceil(t).max(1);
    let ranges: Vec<_> = (0..n).step_by(chunk).map(|a| a..(a + chunk).min(n)).collect();
    if ranges.len() <= 1 {
        return ranges.into_iter().map(&f).collect();
    }
    std::thread::scope(|s| {
        let hs: Vec<_> = ranges.into_iter().map(|r| s.spawn(|| f(r))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// `f(row, out_row)` for every row of `out` (`rows` rows of `width`), in parallel.
fn par_rows(out: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = if width == 0 { 0 } else { out.len() / width };
    let t = threads().clamp(1, rows.max(1));
    let per = rows.div_ceil(t).max(1);
    std::thread::scope(|s| {
        for (c, part) in out.chunks_mut(per * width).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (r, row) in part.chunks_mut(width).enumerate() {
                    f(c * per + r, row);
                }
            });
        }
    });
}

/// Taps of pixel `i`: `g[t cin + c] = sum_k w x[idx, c]`.
#[inline]
fn taps(st: &SphereStencil, x: &[f32], cin: usize, i: usize, g: &mut [f32]) {
    g.fill(0.0);
    for t in 0..st.p {
        let e = (i * st.p + t) * 4;
        let gt = &mut g[t * cin..(t + 1) * cin];
        for k in 0..4 {
            let (s, w) = (st.idx[e + k] as usize, st.w[e + k]);
            for (a, &v) in gt.iter_mut().zip(&x[s * cin..(s + 1) * cin]) {
                *a += w * v;
            }
        }
    }
}

/// Sphere convolution `[n, cin] -> [n, cout]`.
pub fn conv_forward(st: &SphereStencil, x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let pc = st.p * cin;
    let mut y = vec![0f32; st.n * cout];
    par_rows(&mut y, cout, |i, yi| {
        let mut g = vec![0f32; pc];
        taps(st, x, cin, i, &mut g);
        for (o, v) in yi.iter_mut().enumerate() {
            let wo = &w[o * pc..(o + 1) * pc];
            *v = (b[o] as f64 + wo.iter().zip(&g).map(|(&a, &c)| a as f64 * c as f64).sum::<f64>()) as f32;
        }
    });
    y
}

/// Reverse of [`conv_forward`]: adds the weight and bias cotangents to `dw`,
/// `db`, returns the input cotangent.
#[allow(clippy::too_many_arguments)]
pub fn conv_backward(st: &SphereStencil, x: &[f32], cin: usize, cout: usize, w: &[f32], dy: &[f32], dw: &mut [f32], db: &mut [f32]) -> Vec<f32> {
    let pc = st.p * cin;
    let n = st.n;
    // Tap cotangents [n, pc] = dy W (double accumulation).
    let mut dg = vec![0f32; n * pc];
    par_rows(&mut dg, pc, |i, gi| {
        let mut acc = vec![0f64; pc];
        for o in 0..cout {
            let d = dy[i * cout + o] as f64;
            if d != 0.0 {
                for (a, &v) in acc.iter_mut().zip(&w[o * pc..(o + 1) * pc]) {
                    *a += d * v as f64;
                }
            }
        }
        gi.iter_mut().zip(&acc).for_each(|(a, &v)| *a = v as f32);
    });
    // Weight and bias cotangents: double partial sums per chunk of pixels,
    // added in chunk order (long sums with cancellations: single precision
    // would lose about three digits at nside 64).
    let parts = par_map_chunks(n, threads(), |range| {
        let (mut pw, mut pb) = (vec![0f64; cout * pc], vec![0f64; cout]);
        let mut g = vec![0f32; pc];
        for i in range {
            taps(st, x, cin, i, &mut g);
            for o in 0..cout {
                let d = dy[i * cout + o] as f64;
                pb[o] += d;
                for (a, &v) in pw[o * pc..(o + 1) * pc].iter_mut().zip(&g) {
                    *a += d * v as f64;
                }
            }
        }
        (pw, pb)
    });
    let (mut sw, mut sb) = (vec![0f64; cout * pc], vec![0f64; cout]);
    for (pw, pb) in parts {
        sw.iter_mut().zip(&pw).for_each(|(a, b)| *a += b);
        sb.iter_mut().zip(&pb).for_each(|(a, b)| *a += b);
    }
    dw.iter_mut().zip(&sw).for_each(|(a, &b)| *a += b as f32);
    db.iter_mut().zip(&sb).for_each(|(a, &b)| *a += b as f32);
    // Input cotangent through the transposed stencil.
    let mut dx = vec![0f32; n * cin];
    par_rows(&mut dx, cin, |j, xj| {
        let mut acc = vec![0f64; cin];
        for e in st.t_ptr[j] as usize..st.t_ptr[j + 1] as usize {
            let row = st.t_row[e] as usize;
            let (i, t) = (row / st.p, row % st.p);
            let wt = st.t_w[e] as f64;
            for (a, &v) in acc.iter_mut().zip(&dg[i * pc + t * cin..i * pc + (t + 1) * cin]) {
                *a += wt * v as f64;
            }
        }
        xj.iter_mut().zip(&acc).for_each(|(a, &v)| *a = v as f32);
    });
    dx
}

/// GroupNorm statistics `(mean, rstd)` per group of `[n, c]`.
fn group_stats(x: &[f32], c: usize, groups: usize, eps: f64) -> Vec<(f64, f64)> {
    let cg = c / groups;
    let n = x.len() / c;
    (0..groups)
        .map(|g| {
            let (mut s, mut s2) = (0f64, 0f64);
            for i in 0..n {
                for &v in &x[i * c + g * cg..i * c + (g + 1) * cg] {
                    s += v as f64;
                    s2 += (v as f64) * (v as f64);
                }
            }
            let cnt = (n * cg) as f64;
            let mean = s / cnt;
            let var = (s2 / cnt - mean * mean).max(0.0);
            (mean, 1.0 / (var + eps).sqrt())
        })
        .collect()
}

/// GroupNorm (affine) of `[n, c]`; returns the output and the statistics.
pub fn group_norm_forward(x: &[f32], c: usize, groups: usize, gamma: &[f32], beta: &[f32], eps: f64) -> (Vec<f32>, Vec<(f64, f64)>) {
    let stats = group_stats(x, c, groups, eps);
    let cg = c / groups;
    let mut y = vec![0f32; x.len()];
    par_rows(&mut y, c, |i, yi| {
        for (ch, v) in yi.iter_mut().enumerate() {
            let (m, r) = stats[ch / cg];
            *v = (((x[i * c + ch] as f64 - m) * r) as f32) * gamma[ch] + beta[ch];
        }
    });
    (y, stats)
}

/// Reverse of [`group_norm_forward`] (adds to `dgamma`, `dbeta`).
#[allow(clippy::too_many_arguments)]
pub fn group_norm_backward(x: &[f32], c: usize, groups: usize, gamma: &[f32], stats: &[(f64, f64)], dy: &[f32], dgamma: &mut [f32], dbeta: &mut [f32]) -> Vec<f32> {
    let cg = c / groups;
    let n = x.len() / c;
    let xhat = |i: usize, ch: usize| {
        let (m, r) = stats[ch / cg];
        (x[i * c + ch] as f64 - m) * r
    };
    // Per group: sum dxhat, sum dxhat xhat; per channel: dgamma, dbeta.
    let mut s1 = vec![0f64; groups];
    let mut s2 = vec![0f64; groups];
    let (mut dg, mut db) = (vec![0f64; c], vec![0f64; c]);
    for i in 0..n {
        for ch in 0..c {
            let d = dy[i * c + ch] as f64;
            let xh = xhat(i, ch);
            dg[ch] += d * xh;
            db[ch] += d;
            let dxh = d * gamma[ch] as f64;
            s1[ch / cg] += dxh;
            s2[ch / cg] += dxh * xh;
        }
    }
    for ch in 0..c {
        dgamma[ch] += dg[ch] as f32;
        dbeta[ch] += db[ch] as f32;
    }
    let cnt = (n * cg) as f64;
    let mut dx = vec![0f32; x.len()];
    par_rows(&mut dx, c, |i, di| {
        for (ch, v) in di.iter_mut().enumerate() {
            let g = ch / cg;
            let dxh = dy[i * c + ch] as f64 * gamma[ch] as f64;
            *v = (stats[g].1 * (dxh - s1[g] / cnt - xhat(i, ch) * s2[g] / cnt)) as f32;
        }
    });
    dx
}

/// Leaky ReLU.
pub fn leaky(x: &[f32], slope: f32) -> Vec<f32> {
    x.iter().map(|&v| if v > 0.0 { v } else { v * slope }).collect()
}

/// Reverse of [`leaky`] at its input `x`.
pub fn leaky_backward(x: &[f32], dy: &[f32], slope: f32) -> Vec<f32> {
    x.iter().zip(dy).map(|(&v, &d)| if v > 0.0 { d } else { d * slope }).collect()
}

/// Mean of the 4 NESTED children, `[n, c] -> [n / 4, c]`.
pub fn pool(x: &[f32], c: usize) -> Vec<f32> {
    let n = x.len() / c / 4;
    let mut y = vec![0f32; n * c];
    for i in 0..n {
        for ch in 0..c {
            let s = x[(4 * i) * c + ch] + x[(4 * i + 1) * c + ch] + x[(4 * i + 2) * c + ch] + x[(4 * i + 3) * c + ch];
            y[i * c + ch] = s / 4.0;
        }
    }
    y
}

/// Reverse of [`pool`].
pub fn pool_backward(dy: &[f32], c: usize) -> Vec<f32> {
    let mut dx = vec![0f32; dy.len() * 4];
    for (i, row) in dy.chunks(c).enumerate() {
        for k in 0..4 {
            dx[(4 * i + k) * c..(4 * i + k + 1) * c].iter_mut().zip(row).for_each(|(a, &b)| *a = b / 4.0);
        }
    }
    dx
}

/// Parent copied into its 4 children, `[n, c] -> [4 n, c]`.
pub fn unpool(x: &[f32], c: usize) -> Vec<f32> {
    let mut y = Vec::with_capacity(x.len() * 4);
    for row in x.chunks(c) {
        for _ in 0..4 {
            y.extend_from_slice(row);
        }
    }
    y
}

/// Reverse of [`unpool`].
pub fn unpool_backward(dy: &[f32], c: usize) -> Vec<f32> {
    let n = dy.len() / c / 4;
    let mut dx = vec![0f32; n * c];
    for i in 0..n {
        for ch in 0..c {
            dx[i * c + ch] = dy[(4 * i) * c + ch] + dy[(4 * i + 1) * c + ch] + dy[(4 * i + 2) * c + ch] + dy[(4 * i + 3) * c + ch];
        }
    }
    dx
}

/// Channel concatenation of pixel-major parts `[(x, c)]`.
pub fn concat(parts: &[(&[f32], usize)]) -> Vec<f32> {
    let n = parts[0].0.len() / parts[0].1;
    let ct: usize = parts.iter().map(|p| p.1).sum();
    let mut y = Vec::with_capacity(n * ct);
    for i in 0..n {
        for (x, c) in parts {
            y.extend_from_slice(&x[i * c..(i + 1) * c]);
        }
    }
    y
}

/// Reverse of [`concat()`]: the cotangent of every part.
pub fn concat_backward(dy: &[f32], widths: &[usize]) -> Vec<Vec<f32>> {
    let ct: usize = widths.iter().sum();
    let n = dy.len() / ct;
    let mut out: Vec<Vec<f32>> = widths.iter().map(|&c| Vec::with_capacity(n * c)).collect();
    for row in dy.chunks(ct) {
        let mut at = 0;
        for (o, &c) in out.iter_mut().zip(widths) {
            o.extend_from_slice(&row[at..at + c]);
            at += c;
        }
    }
    out
}

/// One named parameter tensor of the flat buffer.
#[derive(Clone, Debug)]
pub struct ParamSpec {
    /// Name in the torch `state_dict`.
    pub name: String,
    /// Shape in the torch `state_dict`.
    pub shape: Vec<usize>,
    /// Offset in the flat buffer.
    pub offset: usize,
    /// Number of values.
    pub len: usize,
}

/// Configuration of the network (the constructor arguments of `stlia.unet.UNet`).
#[derive(Clone, Debug)]
pub struct UNetConfig {
    /// Resolution of the input maps.
    pub nside: usize,
    /// Channels per level (`levels` of them).
    pub channels: Vec<usize>,
    /// Kernel size (`ksz x ksz` taps).
    pub ksz: usize,
    /// Kernel gauge.
    pub gauge: Gauge,
    /// Negative slope of the leaky ReLU.
    pub slope: f32,
    /// Number of conditioning maps.
    pub cond: usize,
    /// Maps denoised together (1 or 2).
    pub maps: usize,
    /// Fixed input scale; `None`: the standard deviation of the input batch.
    pub scale: Option<f64>,
}

/// Offsets of the parameters of one `SphereBlock`.
#[derive(Clone, Debug)]
pub struct BlockParams {
    /// Input and output channels.
    pub cin: usize,
    /// Output channels.
    pub cout: usize,
    /// Level of the block.
    pub level: usize,
    /// Groups of its GroupNorms.
    pub groups: usize,
    /// `c1.weight, c1.bias, c2.weight, c2.bias, n1.weight, n1.bias, n2.weight, n2.bias`.
    pub at: [usize; 8],
}

/// The network: stencils per level, parameter layout.
pub struct UNet {
    /// Configuration.
    pub cfg: UNetConfig,
    /// Stencils of the levels.
    pub stencils: Vec<SphereStencil>,
    /// Encoder blocks, level 0 first.
    pub enc: Vec<BlockParams>,
    /// Decoder blocks, level `L - 2` first.
    pub dec: Vec<BlockParams>,
    /// Offsets of `head.weight`, `head.bias`.
    pub head: [usize; 2],
    /// Named parameters in `state_dict` order.
    pub params: Vec<ParamSpec>,
    /// Number of parameter values.
    pub size: usize,
}

const EPS: f64 = 1e-5;

impl UNet {
    /// The network of `cfg`, its stencils built for orientation 0 of 1 in the gauge.
    pub fn new(cfg: UNetConfig) -> Self {
        let levels = cfg.channels.len();
        assert!(levels >= 1 && cfg.nside >> (levels - 1) >= 1, "too many levels for nside");
        assert!(cfg.ksz % 2 == 1, "odd kernel size");
        assert!(cfg.maps == 1 || cfg.maps == 2, "maps: 1 or 2");
        let stencils = (0..levels).map(|k| SphereStencil::new(cfg.nside >> k, cfg.ksz, cfg.gauge, 0, 1)).collect();
        let mut params = Vec::new();
        let mut size = 0;
        let mut add = |name: String, shape: Vec<usize>| {
            let len = shape.iter().product();
            params.push(ParamSpec { name, shape, offset: size, len });
            size += len;
            size - len
        };
        let p = cfg.ksz * cfg.ksz;
        let ch = &cfg.channels;
        let mut block = |prefix: String, cin: usize, cout: usize, level: usize| {
            let groups = cout.min(8);
            assert!(cout % groups == 0, "GroupNorm: {cout} channels in {groups} groups");
            let c1w = add(format!("{prefix}.c1.weight"), vec![cout, p * cin]);
            let c1b = add(format!("{prefix}.c1.bias"), vec![cout]);
            let c2w = add(format!("{prefix}.c2.weight"), vec![cout, p * cout]);
            let c2b = add(format!("{prefix}.c2.bias"), vec![cout]);
            let n1w = add(format!("{prefix}.n1.weight"), vec![cout]);
            let n1b = add(format!("{prefix}.n1.bias"), vec![cout]);
            let n2w = add(format!("{prefix}.n2.weight"), vec![cout]);
            let n2b = add(format!("{prefix}.n2.bias"), vec![cout]);
            BlockParams { cin, cout, level, groups, at: [c1w, c1b, c2w, c2b, n1w, n1b, n2w, n2b] }
        };
        let enc: Vec<_> = (0..levels).map(|k| block(format!("enc.{k}"), if k == 0 { cfg.maps + cfg.cond } else { ch[k - 1] }, ch[k], k)).collect();
        let dec: Vec<_> = (0..levels.saturating_sub(1))
            .map(|i| {
                let k = levels - 2 - i;
                block(format!("dec.{i}"), ch[k + 1] + ch[k] + cfg.cond, ch[k], k)
            })
            .collect();
        let hw = add("head.weight".into(), vec![cfg.maps, ch[0], 1]);
        let hb = add("head.bias".into(), vec![cfg.maps]);
        UNet { cfg, stencils, enc, dec, head: [hw, hb], params, size }
    }

    /// Torch's initialisation: conv weights `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`
    /// (kaiming_uniform with `a = sqrt 5`), conv biases 0, GroupNorm 1 and 0,
    /// head 0 (an untrained network is the identity).
    pub fn init(&self, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut uniform = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut out = vec![0f32; self.size];
        for spec in &self.params {
            let v = &mut out[spec.offset..spec.offset + spec.len];
            if spec.name.starts_with("head") {
                continue;
            }
            if spec.name.ends_with(".weight") && spec.shape.len() == 2 {
                let bound = 1.0 / (spec.shape[1] as f64).sqrt();
                v.iter_mut().for_each(|x| *x = (bound * uniform()) as f32);
            } else if spec.name.contains(".n") && spec.name.ends_with(".weight") {
                v.fill(1.0);
            }
        }
        out
    }

    /// Pixels of level `k`.
    pub fn pixels(&self, k: usize) -> usize {
        self.stencils[k].n
    }

    /// The conditioning maps `[cond][n_0]` pooled to every level: `[level][n_k, cond]`.
    pub fn cond_levels(&self, cond: &[Vec<f32>]) -> Vec<Vec<f32>> {
        assert_eq!(cond.len(), self.cfg.cond, "conditioning maps");
        if self.cfg.cond == 0 {
            return vec![Vec::new(); self.stencils.len()];
        }
        let parts: Vec<(&[f32], usize)> = cond.iter().map(|c| (&c[..], 1)).collect();
        let mut out = vec![concat(&parts)];
        for _ in 1..self.stencils.len() {
            let next = pool(out.last().unwrap(), self.cfg.cond);
            out.push(next);
        }
        out
    }

    fn block_forward(&self, b: &BlockParams, w: &[f32], x: Vec<f32>) -> BlockTape {
        let st = &self.stencils[b.level];
        let at = |i: usize, len: usize| &w[b.at[i]..b.at[i] + len];
        let p = st.p;
        let y1 = conv_forward(st, &x, b.cin, b.cout, at(0, b.cout * p * b.cin), at(1, b.cout));
        let (a1, s1) = group_norm_forward(&y1, b.cout, b.groups, at(4, b.cout), at(5, b.cout), EPS);
        let h1 = leaky(&a1, self.cfg.slope);
        let y2 = conv_forward(st, &h1, b.cout, b.cout, at(2, b.cout * p * b.cout), at(3, b.cout));
        let (a2, s2) = group_norm_forward(&y2, b.cout, b.groups, at(6, b.cout), at(7, b.cout), EPS);
        let out = leaky(&a2, self.cfg.slope);
        BlockTape { x, y1, s1, a1, h1, y2, s2, a2, out }
    }

    fn block_backward(&self, b: &BlockParams, w: &[f32], t: &BlockTape, dout: &[f32], g: &mut [f32]) -> Vec<f32> {
        let st = &self.stencils[b.level];
        let p = st.p;
        let (c, cin) = (b.cout, b.cin);
        let slope = self.cfg.slope;
        let lens = [c * p * cin, c, c * p * c, c, c, c, c, c];
        // The cotangents of the 8 tensors of the block, added to `g` at the end.
        let mut d: Vec<Vec<f32>> = lens.iter().map(|&l| vec![0f32; l]).collect();
        let at = |i: usize| &w[b.at[i]..b.at[i] + lens[i]];
        let da2 = leaky_backward(&t.a2, dout, slope);
        let (d6, d7) = two(&mut d, 6, 7);
        let dy2 = group_norm_backward(&t.y2, c, b.groups, at(6), &t.s2, &da2, d6, d7);
        let (d2, d3) = two(&mut d, 2, 3);
        let dh1 = conv_backward(st, &t.h1, c, c, at(2), &dy2, d2, d3);
        let da1 = leaky_backward(&t.a1, &dh1, slope);
        let (d4, d5) = two(&mut d, 4, 5);
        let dy1 = group_norm_backward(&t.y1, c, b.groups, at(4), &t.s1, &da1, d4, d5);
        let (d0, d1) = two(&mut d, 0, 1);
        let dx = conv_backward(st, &t.x, cin, c, at(0), &dy1, d0, d1);
        for (i, di) in d.iter().enumerate() {
            g[b.at[i]..b.at[i] + lens[i]].iter_mut().zip(di).for_each(|(a, v)| *a += v);
        }
        dx
    }

    /// The correction `r` `[maps, n_0]` of one standardised input `x`
    /// `[maps, n_0]`, with the conditioning maps of
    /// [`cond_levels`](Self::cond_levels), and the tape of its reverse pass.
    pub fn forward(&self, w: &[f32], x: &[f32], cond: &[Vec<f32>]) -> (Vec<f32>, Tape) {
        assert_eq!(w.len(), self.size, "parameter buffer");
        let levels = self.stencils.len();
        let (nc, m) = (self.cfg.cond, self.cfg.maps);
        let n0 = self.pixels(0);
        assert_eq!(x.len(), m * n0, "input maps");
        let xp = transpose(x, m, n0);
        let mut h = if nc > 0 { concat(&[(&xp, m), (&cond[0], nc)]) } else { xp };
        let mut enc = Vec::with_capacity(levels);
        for (k, b) in self.enc.iter().enumerate() {
            if k > 0 {
                h = pool(&h, self.enc[k - 1].cout);
            }
            let t = self.block_forward(b, w, h);
            h = t.out.clone();
            enc.push(t);
        }
        let mut dec = Vec::with_capacity(levels - 1);
        for b in &self.dec {
            let k = b.level;
            let cprev = self.cfg.channels[k + 1];
            let up = unpool(&h, cprev);
            let mut parts: Vec<(&[f32], usize)> = vec![(&up, cprev), (&enc[k].out, self.cfg.channels[k])];
            if nc > 0 {
                parts.push((&cond[k], nc));
            }
            let t = self.block_forward(b, w, concat(&parts));
            h = t.out.clone();
            dec.push(t);
        }
        let c0 = self.cfg.channels[0];
        let mut r = Vec::with_capacity(m * n0);
        for j in 0..m {
            let hw = &w[self.head[0] + j * c0..self.head[0] + (j + 1) * c0];
            let hb = w[self.head[1] + j];
            r.extend(h.chunks(c0).map(|row| hb + row.iter().zip(hw).map(|(a, b)| a * b).sum::<f32>()));
        }
        (r, Tape { enc, dec, last: h })
    }

    /// Adds to `g` the parameter cotangent of `dr` `[maps, n_0]` (the
    /// cotangent of `r` of [`forward`](Self::forward)); returns the cotangent
    /// of `x` `[maps, n_0]`.
    pub fn backward(&self, w: &[f32], tape: &Tape, dr: &[f32], g: &mut [f32]) -> Vec<f32> {
        let c0 = self.cfg.channels[0];
        let (nc, m) = (self.cfg.cond, self.cfg.maps);
        let n0 = self.pixels(0);
        assert_eq!(dr.len(), m * n0, "output cotangent");
        let mut dh = vec![0f32; tape.last.len()];
        for j in 0..m {
            let hw = &w[self.head[0] + j * c0..self.head[0] + (j + 1) * c0];
            for (i, (&d, row)) in dr[j * n0..(j + 1) * n0].iter().zip(tape.last.chunks(c0)).enumerate() {
                g[self.head[1] + j] += d;
                for ch in 0..c0 {
                    g[self.head[0] + j * c0 + ch] += d * row[ch];
                    dh[i * c0 + ch] += d * hw[ch];
                }
            }
        }
        let levels = self.stencils.len();
        // Skip cotangents, added to the encoder outputs.
        let mut dskip: Vec<Option<Vec<f32>>> = vec![None; levels];
        // Decoder in reverse order (level 0 first).
        for i in (0..self.dec.len()).rev() {
            let (b, t) = (&self.dec[i], &tape.dec[i]);
            let k = b.level;
            let din = self.block_backward(b, w, t, &dh, g);
            let cprev = self.cfg.channels[k + 1];
            let mut widths = vec![cprev, self.cfg.channels[k]];
            if nc > 0 {
                widths.push(nc);
            }
            let parts = concat_backward(&din, &widths);
            dskip[k] = Some(parts[1].clone());
            dh = unpool_backward(&parts[0], cprev);
        }
        // Encoder, coarsest first.
        for k in (0..levels).rev() {
            let b = &self.enc[k];
            if let Some(s) = dskip[k].take() {
                dh.iter_mut().zip(&s).for_each(|(a, b)| *a += b);
            }
            let din = self.block_backward(b, w, &tape.enc[k], &dh, g);
            dh = if k > 0 { pool_backward(&din, self.enc[k - 1].cout) } else { din };
        }
        let dx = if nc > 0 { concat_backward(&dh, &[m, nc]).swap_remove(0) } else { dh };
        transpose(&dx, n0, m)
    }

    /// `u = d - scale r(d / scale)` of a batch `d` `[B, n_0]` (`scale`: the
    /// configured one, else the standard deviation of the batch, unbiased),
    /// with the tapes and the scale (one map per input: `maps` = 1).
    pub fn apply(&self, w: &[f32], d: &[f64], cond: &[Vec<f32>]) -> (Vec<f64>, Vec<Tape>, f64) {
        assert_eq!(self.cfg.maps, 1, "apply: one map per input");
        let n = self.pixels(0);
        let scale = self.cfg.scale.unwrap_or_else(|| std_unbiased(d));
        let levels = self.cond_levels(cond);
        let mut u = Vec::with_capacity(d.len());
        let mut tapes = Vec::new();
        for di in d.chunks(n) {
            let x: Vec<f32> = di.iter().map(|&v| (v / scale) as f32).collect();
            let (r, t) = self.forward(w, &x, &levels);
            u.extend(di.iter().zip(&r).map(|(&a, &b)| a - scale * b as f64));
            tapes.push(t);
        }
        (u, tapes, scale)
    }

    /// Parameter cotangent (added to `g`) of the cotangent `du` `[B, n_0]` of
    /// `u` of [`apply`](Self::apply) at fixed `d` and scale.
    pub fn apply_backward(&self, w: &[f32], tapes: &[Tape], scale: f64, du: &[f64], g: &mut [f32]) {
        let n = self.pixels(0);
        for (t, dui) in tapes.iter().zip(du.chunks(n)) {
            let dr: Vec<f32> = dui.iter().map(|&v| (-scale * v) as f32).collect();
            self.backward(w, t, &dr, g);
        }
    }
}

/// `[a, b]` -> `[b, a]` (row-major); a copy when either is 1.
fn transpose(x: &[f32], a: usize, b: usize) -> Vec<f32> {
    if a == 1 || b == 1 {
        return x.to_vec();
    }
    let mut y = vec![0f32; x.len()];
    for i in 0..a {
        for j in 0..b {
            y[j * a + i] = x[i * b + j];
        }
    }
    y
}

/// Mutable references to entries `i < j` of `v`.
fn two(v: &mut [Vec<f32>], i: usize, j: usize) -> (&mut [f32], &mut [f32]) {
    let (lo, hi) = v.split_at_mut(j);
    (&mut lo[i], &mut hi[0])
}

/// Standard deviation with Bessel's correction (torch's `Tensor.std`).
pub fn std_unbiased(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Forward values of one block kept for its reverse pass.
pub struct BlockTape {
    x: Vec<f32>,
    y1: Vec<f32>,
    s1: Vec<(f64, f64)>,
    a1: Vec<f32>,
    h1: Vec<f32>,
    y2: Vec<f32>,
    s2: Vec<(f64, f64)>,
    a2: Vec<f32>,
    /// Output of the block.
    pub out: Vec<f32>,
}

/// Tape of one map through the network.
pub struct Tape {
    enc: Vec<BlockTape>,
    dec: Vec<BlockTape>,
    last: Vec<f32>,
}

/// Adam (torch's `Adam`: `betas = (0.9, 0.999)`, `eps = 1e-8`, no weight
/// decay), state on the host.
pub struct Adam {
    /// First moments.
    pub m: Vec<f32>,
    /// Second moments.
    pub v: Vec<f32>,
    /// Steps taken.
    pub t: u64,
    /// `beta1`, `beta2`, `eps`.
    pub betas: (f32, f32, f32),
}

impl Adam {
    /// Zero state for `n` parameters.
    pub fn new(n: usize) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0, betas: (0.9, 0.999, 1e-8) }
    }

    /// One step of learning rate `lr` (torch's update order and bias corrections).
    pub fn step(&mut self, w: &mut [f32], g: &[f32], lr: f64) {
        self.t += 1;
        let (b1, b2, eps) = self.betas;
        let bc1 = 1.0 - (b1 as f64).powi(self.t as i32);
        let bc2 = 1.0 - (b2 as f64).powi(self.t as i32);
        let step = (lr / bc1) as f32;
        let sbc2 = bc2.sqrt() as f32;
        for i in 0..w.len() {
            self.m[i] = self.m[i] * b1 + g[i] * (1.0 - b1);
            self.v[i] = self.v[i] * b2 + g[i] * g[i] * (1.0 - b2);
            let denom = self.v[i].sqrt() / sbc2 + eps;
            w[i] -= step * self.m[i] / denom;
        }
    }
}

/// Learning rate of step `step` (1-based) of a cosine schedule over `steps`
/// (torch's `CosineAnnealingLR`, `eta_min = 0`, stepped after every update).
pub fn cosine_lr(lr0: f64, step: usize, steps: usize) -> f64 {
    lr0 * 0.5 * (1.0 + (std::f64::consts::PI * (step - 1) as f64 / steps as f64).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
            })
            .collect()
    }

    /// The stencil of every pixel sums to 1 per tap; the transposed lists
    /// hold every entry once.
    #[test]
    fn stencil_partition_of_unity() {
        let st = SphereStencil::new(4, 3, Gauge::Phi, 0, 1);
        for row in st.w.chunks(4) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        assert_eq!(st.t_ptr[st.n] as usize, st.idx.len());
    }

    /// `<conv(x), y> = <x, conv^T y>` and the weight cotangent as a dot
    /// product, at nside 2.
    #[test]
    fn conv_adjoint() {
        let st = SphereStencil::new(2, 3, Gauge::Phi, 0, 1);
        let (cin, cout) = (3, 2);
        let w = noise(cout * 9 * cin, 1);
        let b = vec![0f32; cout];
        let x = noise(st.n * cin, 2);
        let dy = noise(st.n * cout, 3);
        let y = conv_forward(&st, &x, cin, cout, &w, &b);
        let (mut dw, mut db) = (vec![0f32; w.len()], vec![0f32; cout]);
        let dx = conv_backward(&st, &x, cin, cout, &w, &dy, &mut dw, &mut db);
        let lhs: f64 = y.iter().zip(&dy).map(|(a, b)| (a * b) as f64).sum();
        let rhs: f64 = x.iter().zip(&dx).map(|(a, b)| (a * b) as f64).sum();
        assert!((lhs - rhs).abs() < 1e-4 * lhs.abs().max(1.0), "{lhs} {rhs}");
        let rw: f64 = w.iter().zip(&dw).map(|(a, b)| (a * b) as f64).sum();
        assert!((lhs - rw).abs() < 1e-4 * lhs.abs().max(1.0), "{lhs} {rw}");
    }
}
