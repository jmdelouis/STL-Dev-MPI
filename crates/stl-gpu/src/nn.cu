// Kernels of the spherical U-Net on the device (`nn.rs`; host reference
// `stl_core::nn`).  Activations are pixel-major `[n, c]` float; reductions
// over pixels run in double, chunk by chunk in a fixed order.

// Taps of tap `t`: g[i, c] = sum_k w[i, t, k] x[idx[i, t, k], c].
extern "C" __global__ void nn_gather(const float* x, const unsigned* idx, const float* w, long long n, long long p,
                                     long long t, long long cin, float* g) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * cin) return;
    long long i = e / cin, c = e % cin;
    long long b = (i * p + t) * 4;
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += w[b + k] * x[(long long)idx[b + k] * cin + c];
    g[e] = s;
}

// dx[j, c] += sum over the transposed list of tap t at j of w dg[i, c].
extern "C" __global__ void nn_scatter(const float* dg, const unsigned* ptr, const unsigned* row, const float* tw,
                                      long long n, long long cin, float* dx) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * cin) return;
    long long j = e / cin, c = e % cin;
    float s = 0.0f;
    for (unsigned q = ptr[j]; q < ptr[j + 1]; q++) s += tw[q] * dg[(long long)row[q] * cin + c];
    dx[e] += s;
}

// y[i, o] = b[o].
extern "C" __global__ void nn_bias(float* y, const float* b, long long n, long long c) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * c) return;
    y[e] = b[e % c];
}

__device__ static float nn_leaky(float v, float slope) { return v > 0.0f ? v : v * slope; }

// Partial sums per channel over chunks of `chunk` pixels (block = chunk,
// threads over channels), `partial[chunk][2 c]`:
//   mode 0: sum a, sum a^2                               (GroupNorm statistics)
//   mode 1: sum da xhat, sum da, a = y, b = dh            (GroupNorm reverse)
//   mode 2: sum a                                         (bias cotangent)
//   mode 3: sum a[i, c] b[i]                              (head weight cotangent)
extern "C" __global__ void nn_chan_sums(int mode, const float* a, const float* b, const double* stats,
                                        const float* gamma, const float* beta, double slope, long long n, long long c,
                                        long long groups, long long chunk, double* partial) {
    long long i0 = blockIdx.x * chunk;
    long long i1 = i0 + chunk < n ? i0 + chunk : n;
    long long cg = c / groups;
    for (long long ch = threadIdx.x; ch < c; ch += blockDim.x) {
        double s0 = 0.0, s1 = 0.0;
        double m = 0.0, r = 0.0;
        if (mode == 1) {
            m = stats[2 * (ch / cg)];
            r = stats[2 * (ch / cg) + 1];
        }
        for (long long i = i0; i < i1; i++) {
            double v = (double)a[i * c + ch];
            if (mode == 0) {
                s0 += v;
                s1 += v * v;
            } else if (mode == 1) {
                double xh = (v - m) * r;
                float av = (float)xh * gamma[ch] + beta[ch];
                float d = b[i * c + ch];
                double da = (double)(av > 0.0f ? d : d * (float)slope);
                s0 += da * xh;
                s1 += da;
            } else if (mode == 2) {
                s0 += v;
            } else {
                s0 += v * (double)b[i];
            }
        }
        partial[blockIdx.x * 2 * c + ch] = s0;
        partial[blockIdx.x * 2 * c + c + ch] = s1;
    }
}

// out[k] = sum over chunks (in order) of partial[chunk][k], k < width.
extern "C" __global__ void nn_chan_finish(const double* partial, long long chunks, long long width, double* out) {
    long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k >= width) return;
    double s = 0.0;
    for (long long q = 0; q < chunks; q++) s += partial[q * width + k];
    out[k] = s;
}

// GroupNorm statistics (mean, rstd) per group from the channel sums.
extern "C" __global__ void nn_gn_stats(const double* sums, long long c, long long groups, long long n, double eps,
                                       double* stats) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= groups) return;
    long long cg = c / groups;
    double s = 0.0, s2 = 0.0;
    for (long long ch = g * cg; ch < (g + 1) * cg; ch++) {
        s += sums[ch];
        s2 += sums[c + ch];
    }
    double cnt = (double)(n * cg);
    double mean = s / cnt;
    double var = s2 / cnt - mean * mean;
    if (var < 0.0) var = 0.0;
    stats[2 * g] = mean;
    stats[2 * g + 1] = 1.0 / sqrt(var + eps);
}

// h = leaky(((y - mean) rstd) gamma + beta).
extern "C" __global__ void nn_gn_apply(const float* y, const double* stats, const float* gamma, const float* beta,
                                       double slope, long long n, long long c, long long groups, float* h) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * c) return;
    long long ch = e % c, g = ch / (c / groups);
    float a = (float)(((double)y[e] - stats[2 * g]) * stats[2 * g + 1]) * gamma[ch] + beta[ch];
    h[e] = nn_leaky(a, (float)slope);
}

// From the channel sums of mode 1 (dgamma, dbeta): adds them to the
// parameter cotangents and writes s[2 g] = sum gamma dbeta, s[2 g + 1] =
// sum gamma dgamma per group.  One thread.
extern "C" __global__ void nn_gn_params(const double* sums, const float* gamma, long long c, long long groups,
                                        float* dgamma, float* dbeta, double* s) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    long long cg = c / groups;
    for (long long g = 0; g < groups; g++) {
        s[2 * g] = 0.0;
        s[2 * g + 1] = 0.0;
    }
    for (long long ch = 0; ch < c; ch++) {
        dgamma[ch] += (float)sums[ch];
        dbeta[ch] += (float)sums[c + ch];
        s[2 * (ch / cg)] += (double)gamma[ch] * sums[c + ch];
        s[2 * (ch / cg) + 1] += (double)gamma[ch] * sums[ch];
    }
}

// dy = rstd (da gamma - s1 / N - xhat s2 / N), da the cotangent through the leaky ReLU.
extern "C" __global__ void nn_gn_back(const float* y, const float* dh, const double* stats, const float* gamma,
                                      const float* beta, const double* s, double slope, long long n, long long c,
                                      long long groups, float* dy) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * c) return;
    long long ch = e % c, cg = c / groups, g = ch / cg;
    double xh = ((double)y[e] - stats[2 * g]) * stats[2 * g + 1];
    float av = (float)xh * gamma[ch] + beta[ch];
    float d = dh[e];
    float da = av > 0.0f ? d : d * (float)slope;
    double cnt = (double)(n * cg);
    double dxh = (double)da * (double)gamma[ch];
    dy[e] = (float)(stats[2 * g + 1] * (dxh - s[2 * g] / cnt - xh * s[2 * g + 1] / cnt));
}

// Mean of the 4 NESTED children: y[i, c] = (x[4i] + x[4i+1] + x[4i+2] + x[4i+3]) / 4.
extern "C" __global__ void nn_pool(const float* x, long long n, long long c, float* y) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * c) return;
    long long i = e / c, ch = e % c;
    float s = x[(4 * i) * c + ch] + x[(4 * i + 1) * c + ch] + x[(4 * i + 2) * c + ch] + x[(4 * i + 3) * c + ch];
    y[e] = s / 4.0f;
}

extern "C" __global__ void nn_pool_back(const float* dy, long long n, long long c, float* dx) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= 4 * n * c) return;
    long long j = e / c, ch = e % c;
    dx[e] = dy[(j / 4) * c + ch] / 4.0f;
}

extern "C" __global__ void nn_unpool(const float* x, long long n, long long c, float* y) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= 4 * n * c) return;
    long long j = e / c, ch = e % c;
    y[e] = x[(j / 4) * c + ch];
}

extern "C" __global__ void nn_unpool_back(const float* dy, long long n, long long c, float* dx) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * c) return;
    long long i = e / c, ch = e % c;
    dx[e] = dy[(4 * i) * c + ch] + dy[(4 * i + 1) * c + ch] + dy[(4 * i + 2) * c + ch] + dy[(4 * i + 3) * c + ch];
}

// Columns: dst[i, off + ch] (width cd) = src[i, ch] (width cs) (mode 0),
// dst[i, ch] (width cs) = src[i, off + ch] (width cd) (mode 1), or += (mode 2).
extern "C" __global__ void nn_cols(int mode, const float* src, float* dst, long long n, long long cs, long long cd,
                                   long long off) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * cs) return;
    long long i = e / cs, ch = e % cs;
    if (mode == 0) dst[i * cd + off + ch] = src[e];
    else if (mode == 1) dst[e] = src[i * cd + off + ch];
    else dst[e] += src[i * cd + off + ch];
}

// r[i] = hb + sum_c hw[c] h[i, c].
extern "C" __global__ void nn_head(const float* h, const float* hw, const float* hb, long long n, long long c, float* r) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (long long ch = 0; ch < c; ch++) s += h[i * c + ch] * hw[ch];
    r[i] = hb[0] + s;
}

// dh[i, c] = dr[i] hw[c].
extern "C" __global__ void nn_head_back(const float* dr, const float* hw, long long n, long long c, float* dh) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n * c) return;
    dh[e] = dr[e / c] * hw[e % c];
}

// x[i, 0] = d[i] / scale, x[i, 1 + k] = cond[i, k].
extern "C" __global__ void nn_input(const double* d, double scale, const float* cond, long long n, long long nc, float* x) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i * (1 + nc)] = (float)(d[i] / scale);
    for (long long k = 0; k < nc; k++) x[i * (1 + nc) + 1 + k] = cond[i * nc + k];
}

// u = d - scale r.
extern "C" __global__ void nn_output(const double* d, const float* r, double scale, long long n, double* u) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    u[i] = d[i] - scale * (double)r[i];
}

// dr = -scale du.
extern "C" __global__ void nn_output_back(const double* du, double scale, long long n, float* dr) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    dr[i] = (float)(-scale * du[i]);
}

// dst[i] = a[i] + s b[i] (b null: dst = a); in place when dst == a.
extern "C" __global__ void nn_axpy(double* dst, const double* a, const double* b, double s, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = b ? a[i] + s * b[i] : a[i];
}

// dst[k] += (float) src[k].
extern "C" __global__ void nn_add_f64(const double* src, long long n, float* dst) {
    long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k >= n) return;
    dst[k] += (float)src[k];
}

// Adam (torch's update order): step = lr / bc1, sbc2 = sqrt(bc2).
extern "C" __global__ void nn_adam(float* w, const float* g, float* m, float* v, long long n, double b1, double b2,
                                   double eps, double step, double sbc2) {
    long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k >= n) return;
    float fb1 = (float)b1, fb2 = (float)b2;
    m[k] = m[k] * fb1 + g[k] * (1.0f - fb1);
    v[k] = v[k] * fb2 + g[k] * g[k] * (1.0f - fb2);
    float denom = sqrtf(v[k]) / (float)sbc2 + (float)eps;
    w[k] -= (float)step * m[k] / denom;
}
