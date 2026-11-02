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
