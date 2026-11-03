// ---------------------------------------------------------------------------
// Planar (2D periodic) kernels of stl_gpu::planar, the device counterpart of
// stl_core::planar::grid.  A plane of a level is H x W row-major; a window
// starts at (r0, c0) (any integers, taken modulo the level) and holds h x w
// pixels.  Every kernel is a gather (one thread per output value, a fixed
// order of the terms), so the results do not depend on the launch.
// ---------------------------------------------------------------------------

__device__ __forceinline__ i64 pmod(i64 v, i64 m) {
    i64 r = v % m;
    return r < 0 ? r + m : r;
}

// Valid cross-correlation of P real planes with the L complex 5 x 5 taps:
// out[(p L + l), i, j] = sum_ab w[l, a, b] src[p, r0 + i + a - 2, c0 + j + b - 2]
// (indices modulo sh, sw); re at out, im at out + im_off.
extern "C" __global__ void pl_conv(
    const double* src, i64 sh, i64 sw, i64 r0, i64 c0, i64 P,
    const double* wre, const double* wim, int L, i64 oh, i64 ow, double* out, i64 im_off)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 on = oh * ow;
    if (t >= P * on) return;
    i64 p = t / on, q = t - p * on, i = q / ow, j = q - i * ow;
    const double* s = src + p * sh * sw;
    double x[25];
    for (int a = 0; a < 5; a++) {
        const double* row = s + pmod(r0 + i + a - 2, sh) * sw;
        for (int b = 0; b < 5; b++) x[a * 5 + b] = row[pmod(c0 + j + b - 2, sw)];
    }
    for (int l = 0; l < L; l++) {
        double re = 0.0, im = 0.0;
        for (int k = 0; k < 25; k++) {
            re += wre[l * 25 + k] * x[k];
            im += wim[l * 25 + k] * x[k];
        }
        out[(p * L + l) * on + q] = re;
        out[im_off + (p * L + l) * on + q] = im;
    }
}

// Adjoint of pl_conv on a local source (r0 = c0 = 2, no wrap): for P real
// planes of (oh + 4) x (ow + 4), dst (+)= sum_l sum_ab Re(g[p L + l, s - a, t - b] conj w[l, a, b]).
extern "C" __global__ void pl_conv_adj(
    const double* g, i64 g_im_off, i64 P, int L, i64 oh, i64 ow,
    const double* wre, const double* wim, double* dst, int accumulate)
{
    i64 dh = oh + 4, dw = ow + 4, dn = dh * dw, on = oh * ow;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * dn) return;
    i64 p = t / dn, q = t - p * dn, s = q / dw, u = q - s * dw;
    double v = accumulate ? dst[t] : 0.0;
    for (int l = 0; l < L; l++) {
        const double* gr = g + (p * L + l) * on;
        const double* gi = gr + g_im_off;
        for (int a = 4; a >= 0; a--) {
            i64 i = s - a;
            if (i < 0 || i >= oh) continue;
            for (int b = 0; b < 5; b++) {
                i64 j = u - b;
                if (j < 0 || j >= ow) continue;
                v += gr[i * ow + j] * wre[l * 25 + a * 5 + b] + gi[i * ow + j] * wim[l * 25 + a * 5 + b];
            }
        }
    }
    dst[t] = v;
}

// Window (r0, c0, h, w) of P level planes H x W: dst[p, i, j].
extern "C" __global__ void pl_read(const double* src, i64 H, i64 W, i64 P, i64 r0, i64 c0, i64 h, i64 w, double* dst) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 n = h * w;
    if (t >= P * n) return;
    i64 p = t / n, q = t - p * n, i = q / w, j = q - i * w;
    dst[t] = src[(p * H + pmod(r0 + i, H)) * W + pmod(c0 + j, W)];
}

// Adjoint of pl_read: one thread per level pixel the window covers, adding
// every window position that falls on it (positions in increasing order).
extern "C" __global__ void pl_accum(double* dst, i64 H, i64 W, i64 P, i64 r0, i64 c0, i64 h, i64 w, const double* src) {
    i64 ch = h < H ? h : H, cw = w < W ? w : W, n = ch * cw;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 p = t / n, q = t - p * n, i = q / cw, j = q - i * cw;
    double v = 0.0;
    for (i64 a = i; a < h; a += H)
        for (i64 b = j; b < w; b += W) v += src[(p * h + a) * w + b];
    double* d = dst + (p * H + pmod(r0 + i, H)) * W + pmod(c0 + j, W);
    *d += v;
}

// P planes sh x sw shrunk by `off` on every side.
extern "C" __global__ void pl_crop(const double* src, i64 P, i64 sh, i64 sw, i64 off, double* dst) {
    i64 h = sh - 2 * off, w = sw - 2 * off, n = h * w;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 p = t / n, q = t - p * n, i = q / w, j = q - i * w;
    dst[t] = src[(p * sh + i + off) * sw + j + off];
}

// dst (P planes (h + 2 off) x (w + 2 off)) += src (P planes h x w) at the offset.
extern "C" __global__ void pl_embed_add(const double* src, i64 P, i64 h, i64 w, i64 off, double* dst) {
    i64 n = h * w;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 p = t / n, q = t - p * n, i = q / w, j = q - i * w;
    dst[(p * (h + 2 * off) + i + off) * (w + 2 * off) + j + off] += src[t];
}

// out = |z| (num_complex norm: hypot).
extern "C" __global__ void pl_abs(const double* z, i64 im_off, i64 n, double* out) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = hypot(z[t], z[im_off + t]);
}

// gz += gmod z / |z| (|z| = 0: divided by 1).
extern "C" __global__ void pl_modvjp(const double* z, i64 z_im, const double* gmod, i64 n, double* gz, i64 g_im) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    double re = z[t], im = z[z_im + t];
    double mag = hypot(re, im);
    double d = mag == 0.0 ? 1.0 : mag;
    double gm = gmod[t];
    gz[t] += (gm * re) / d;
    gz[g_im + t] += (gm * im) / d;
}

// dst += src.
extern "C" __global__ void pl_add(double* dst, const double* src, i64 n) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    dst[t] += src[t];
}

// Decimation of P planes of a fine window (fr0, fh, Hf) x (fc0, fw, Wf),
// zero outside, onto the coarse window (cr0, ch) x (cc0, cw):
// y[p, i, j] = sum_ab s[a, b] x[p, pos(2 (cr0 + i) + a - 2), pos(2 (cc0 + j) + b - 2)].
extern "C" __global__ void pl_dec(
    const double* x, i64 P, i64 fr0, i64 fh, i64 Hf, i64 fc0, i64 fw, i64 Wf,
    i64 cr0, i64 ch, i64 cc0, i64 cw, const double* s, double* y)
{
    i64 n = ch * cw;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 p = t / n, q = t - p * n, i = q / cw, j = q - i * cw;
    const double* xp = x + p * fh * fw;
    i64 R = cr0 + i, C = cc0 + j;
    double v = 0.0;
    for (int a = 0; a < 6; a++) {
        i64 pr = pmod(2 * R + a - 2 - fr0, Hf);
        if (pr >= fh) continue;
        for (int b = 0; b < 6; b++) {
            i64 pc = pmod(2 * C + b - 2 - fc0, Wf);
            if (pc >= fw) continue;
            v += s[a * 6 + b] * xp[pr * fw + pc];
        }
    }
    y[t] = v;
}

// Adjoint of pl_dec: the coarse cotangent g pulled back to the fine window.
extern "C" __global__ void pl_dec_adj(
    const double* g, i64 P, i64 fr0, i64 fh, i64 Hf, i64 fc0, i64 fw, i64 Wf,
    i64 cr0, i64 ch, i64 cc0, i64 cw, const double* s, double* x)
{
    i64 n = fh * fw, Hc = Hf / 2, Wc = Wf / 2;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 p = t / n, q = t - p * n, pr = q / fw, pc = q - pr * fw;
    const double* gp = g + p * ch * cw;
    i64 r = fr0 + pr, c = fc0 + pc;
    double v = 0.0;
    for (int a = 0; a < 6; a++) {
        if (pmod(r - a, 2) != 0) continue;
        i64 i = pmod((r - a + 2) / 2 - cr0, Hc);
        if (i >= ch) continue;
        for (int b = 0; b < 6; b++) {
            if (pmod(c - b, 2) != 0) continue;
            i64 j = pmod((c - b + 2) / 2 - cc0, Wc);
            if (j >= cw) continue;
            v += s[a * 6 + b] * gp[i * cw + j];
        }
    }
    x[t] = v;
}

// x *= s.
extern "C" __global__ void pl_scale(double* x, double s, i64 n) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    x[t] *= s;
}

// ---------------------------------------------------------------------------
// Fourier terms of the planar objective (stl_core::planar::spectrum): complex
// planes are interleaved (re, im) pairs, P planes of H x W.
// ---------------------------------------------------------------------------

// out = (x, 0).
extern "C" __global__ void pl_to_complex(const double* x, i64 n, double* out) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[2 * t] = x[t];
    out[2 * t + 1] = 0.0;
}

// out[p, j, i] = in[p, i, j] (complex), in P planes of H x W.
extern "C" __global__ void pl_transpose(const double* in, i64 H, i64 W, i64 P, double* out) {
    i64 n = H * W;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 p = t / n, q = t - p * n, i = q / W, j = q - i * W;
    i64 o = p * n + j * H + i;
    out[2 * o] = in[2 * t];
    out[2 * o + 1] = in[2 * t + 1];
}

// x *= s (n complex values, real s).
extern "C" __global__ void pl_cscale(double* x, double s, i64 n) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    x[2 * t] *= s;
    x[2 * t + 1] *= s;
}

// out (+)= s Re(x), n values.
extern "C" __global__ void pl_real_part(const double* x, double s, i64 n, double* out, int accumulate) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    double v = x[2 * t] * s;
    out[t] = accumulate ? out[t] + v : v;
}

__device__ __forceinline__ i64 int_freq(i64 k, i64 n) {
    return k < (n + 1) / 2 ? k : k - n;
}

// Multiplier `which` of Fft2::derivatives at the frequency (kr, kc) of an
// H x W grid: 0, 1: i 2 pi k / n along rows / columns (0 at Nyquist),
// 2, 3: -(2 pi k / n)^2, 4: the product of 0 and 1.
__device__ void pl_multiplier(int which, i64 kr, i64 kc, i64 H, i64 W, double* re, double* im) {
    const double tau = 6.283185307179586;
    double ar = 0.0, ai = 0.0, br = 0.0, bi = 0.0;
    if (which == 0 || which == 4) {
        if (2 * (kr < 0 ? -kr : kr) != H) ai = tau * (double)kr / (double)H;
    }
    if (which == 1 || which == 4) {
        if (2 * (kc < 0 ? -kc : kc) != W) bi = tau * (double)kc / (double)W;
    }
    if (which == 0) { *re = ar; *im = ai; return; }
    if (which == 1) { *re = br; *im = bi; return; }
    if (which == 2) { double t = tau * (double)kr / (double)H; *re = -t * t; *im = 0.0; return; }
    if (which == 3) { double t = tau * (double)kc / (double)W; *re = -t * t; *im = 0.0; return; }
    *re = ar * br - ai * bi;
    *im = ar * bi + ai * br;
}

// out[p, w] = H_(first + w) in[p] (conjugate multipliers if conj) for
// w < nw, in P planes; out [P, nw] planes.
extern "C" __global__ void pl_spec_mult(const double* in, i64 H, i64 W, i64 P, int first, int nw, int conj, double* out) {
    i64 n = H * W;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 q = t % n, kr = int_freq(q / W, H), kc = int_freq(q % W, W);
    double xr = in[2 * t], xi = in[2 * t + 1];
    for (int w = 0; w < nw; w++) {
        double hr, hi;
        pl_multiplier(first + w, kr, kc, H, W, &hr, &hi);
        if (conj) hi = -hi;
        i64 o = ((t / n) * nw + w) * n + q;
        out[2 * o] = xr * hr - xi * hi;
        out[2 * o + 1] = xr * hi + xi * hr;
    }
}

// out[p] = sum_w conj(H_w) in[p, w] over the 5 multipliers: in [P, 5] planes.
extern "C" __global__ void pl_spec_mult_sum(const double* in, i64 H, i64 W, i64 P, double* out) {
    i64 n = H * W;
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 q = t % n, kr = int_freq(q / W, H), kc = int_freq(q % W, W);
    double sr = 0.0, si = 0.0;
    for (int w = 0; w < 5; w++) {
        double hr, hi;
        pl_multiplier(w, kr, kc, H, W, &hr, &hi);
        hi = -hi;
        i64 o = ((t / n) * 5 + w) * n + q;
        double xr = in[2 * o], xi = in[2 * o + 1];
        sr += xr * hr - xi * hi;
        si += xr * hi + xi * hr;
    }
    out[2 * t] = sr;
    out[2 * t + 1] = si;
}

// Weight of the Gaussian ring at the log-frequency lr (GaussianRings).
__device__ __forceinline__ double ring_weight(double lr, double lc, double ls, double cut) {
    double d = lr - lc;
    if (!(fabs(d) <= cut * ls)) return 0.0;
    return exp((-0.5 * (d * d)) / (ls * ls));
}

// Partial sums of w_k(p) X_a(p) conj X_b(p) over chunks, for the pairs
// (a, b) of planes: out[((q nb + k) n_chunks + c) 2 + {re, im}].
// Grid (n_chunks, nb, Q), 256 threads.
extern "C" __global__ void pl_rings(
    const double* X, const int* pairs, const double* lr, const double* lc, const double* ls,
    double cut, i64 n, i64 chunk, double* out)
{
    __shared__ double sr[256], si[256];
    i64 c = blockIdx.x;
    int k = blockIdx.y, q = blockIdx.z, nb = gridDim.y;
    i64 n_chunks = gridDim.x;
    const double* xa = X + 2 * (i64)pairs[2 * q] * n;
    const double* xb = X + 2 * (i64)pairs[2 * q + 1] * n;
    double lck = lc[k], lsk = ls[k];
    i64 lo = c * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double ar = 0.0, ai = 0.0;
    for (i64 p = lo + threadIdx.x; p < hi; p += blockDim.x) {
        double l = lr[p];
        if (isnan(l)) continue;
        double w = ring_weight(l, lck, lsk, cut);
        if (w == 0.0) continue;
        double vr, vi;
        mul_conj(xa[2 * p], xa[2 * p + 1], xb[2 * p], xb[2 * p + 1], &vr, &vi);
        ar += w * vr;
        ai += w * vi;
    }
    sr[threadIdx.x] = ar;
    si[threadIdx.x] = ai;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            sr[threadIdx.x] += sr[threadIdx.x + o];
            si[threadIdx.x] += si[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double* w = out + (((i64)q * nb + k) * n_chunks + c) * 2;
        w[0] = sr[0];
        w[1] = si[0];
    }
}

// Cotangent of the planes X from that of the pair spectra g[q, k] (already
// divided by the bin norms): for every plane and pixel, the pairs in order,
// G = sum_k g w_k, then G X_b (plane a) and conj(G) X_a (plane b).
extern "C" __global__ void pl_rings_vjp(
    const double* X, const int* pairs, int Q, const double* g, int nb, const double* lr,
    const double* lc, const double* ls, double cut, i64 n, i64 P, double* gx)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * n) return;
    i64 plane = t / n, p = t - plane * n;
    double l = lr[p];
    double outr = 0.0, outi = 0.0;
    if (!isnan(l)) {
        for (int q = 0; q < Q; q++) {
            int a = pairs[2 * q], b = pairs[2 * q + 1];
            if (a != plane && b != plane) continue;
            double Gr = 0.0, Gi = 0.0;
            for (int k = 0; k < nb; k++) {
                double w = ring_weight(l, lc[k], ls[k], cut);
                if (w == 0.0) continue;
                Gr += g[2 * (q * nb + k)] * w;
                Gi += g[2 * (q * nb + k) + 1] * w;
            }
            if (a == plane) {
                double xr = X[2 * ((i64)b * n + p)], xi = X[2 * ((i64)b * n + p) + 1];
                outr += Gr * xr - Gi * xi;
                outi += Gr * xi + Gi * xr;
            }
            if (b == plane) {
                double xr = X[2 * ((i64)a * n + p)], xi = X[2 * ((i64)a * n + p) + 1];
                outr += Gr * xr + Gi * xi;
                outi += Gr * xi - Gi * xr;
            }
        }
    }
    gx[2 * t] = outr;
    gx[2 * t + 1] = outi;
}
