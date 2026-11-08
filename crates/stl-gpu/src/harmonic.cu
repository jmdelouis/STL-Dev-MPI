
// ---------------------------------------------------------------------------
// Spherical harmonic transforms of full-sky HEALPix maps (port of
// stl-core/src/sht.rs) and the harmonic terms of the ensemble objective
// (stl-cli/src/ensemble/fields.rs: C_l, CIC PDF, Minkowski functionals).
// Complex arrays are interleaved (re, im) doubles (`double2`), the layout
// of num_complex::Complex64; the packed a_lm are m-major as on the host.
//
// Orders of evaluation (the host's unless stated):
// - forward Legendre sums (sht_leg_syn): one thread per (m, ring pair),
//   the recursion and the sums over l in the host's order;
// - adjoint Legendre sums (sht_leg_adj): one block per (m, chunk of
//   SHT_T * SHT_R ring pairs); each thread adds its rings j = 0..SHT_R in
//   order, the threads are added in a fixed tree (warp shuffles, then the
//   warps in order), the chunks in order (sht_chunk_sum).  The host adds
//   the rings one after the other: the two differ by rounding;
// - ring sums over m as on the host; ring FFTs by cuFFT (deterministic for
//   a plan; rustfft on the host: they differ by rounding);
// - pixel sums: chunks of `chunk` pixels, each in a fixed tree, the chunks
//   added in order on the host (the host adds the pixels in order); the
//   CIC histogram per chunk of 2^14 pixels in pixel order, as on the host.
// The starting value of the recursion uses the device exp, the ring phases
// the device cos / sin, the functionals exp and erfc: within 1-2 ulp of
// glibc.  The Gaussians of the Minkowski thresholds are evaluated directly
// (the host uses a recurrence over the thresholds).
// ---------------------------------------------------------------------------

#define SHT_T 256
#define SHT_R 4
#define SHT_LT 16

__device__ __forceinline__ double2 c_add(double2 a, double2 b) { return make_double2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ double2 c_sub(double2 a, double2 b) { return make_double2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ double2 c_mul(double2 a, double2 b) { return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
__device__ __forceinline__ double2 c_scale(double2 a, double s) { return make_double2(a.x * s, a.y * s); }
__device__ __forceinline__ double2 c_div(double2 a, double s) { return make_double2(a.x / s, a.y / s); }
__device__ __forceinline__ double2 c_conj(double2 a) { return make_double2(a.x, -a.y); }
__device__ __forceinline__ double2 c_neg(double2 a) { return make_double2(-a.x, -a.y); }
__device__ __forceinline__ double2 c_real(double v) { return make_double2(v, 0.0); }
__device__ __forceinline__ double2 c_polar(double r, double t) { return make_double2(r * cos(t), r * sin(t)); }

// Packed index of (l, m) (Sht::index).
__device__ __forceinline__ i64 alm_index(i64 lmax, i64 l, i64 m) {
    return m * (lmax + 1) - m * (m > 0 ? m - 1 : 0) / 2 + (l - m);
}

// Recursion coefficients (a_l, b_l, c_l) of every (l, m), at the packed
// index (Sht::coefficients).  Grid (ceil((lmax + 1) / 256), lmax + 1).
extern "C" __global__ void sht_coef(i64 lmax, double* coef) {
    i64 m = blockIdx.y;
    i64 l = m + (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    double* c = coef + 3 * alm_index(lmax, l, m);
    if (l == m) {
        c[0] = 0.0;
        c[1] = 0.0;
        c[2] = 0.0;
        return;
    }
    double lf = (double)l, mf = (double)m;
    c[0] = sqrt((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf));
    double l1 = lf - 1.0;
    c[1] = sqrt((l1 * l1 - mf * mf) / (4.0 * l1 * l1 - 1.0));
    c[2] = sqrt((2.0 * lf + 1.0) * (lf * lf - mf * mf) / (2.0 * lf - 1.0));
}

// Constants of the scaled recursion (from the host: -650 ln 2, 300 ln 2,
// LN_SCALE, SCALE, LIMIT of stl-core).
struct RecConst {
    double lo, c300, ln_scale, scale, limit;
};

// Starting value lambda_mm (scaled by SCALE^k) of the recursion on a ring of
// ln(sin) `lnsin` (Sht::recurse_block).
__device__ __forceinline__ void rec_start(i64 m, double lnsin, const double* log_mm, RecConst rc, double* cur, int* k) {
    double lg = log_mm[m] + (double)m * lnsin;
    int kk = lg > rc.lo ? 0 : (int)ceil((lg - rc.c300) / rc.ln_scale);
    *cur = exp(lg - (double)kk * rc.ln_scale) * ((m & 1) ? -1.0 : 1.0);
    *k = kk;
}

// One step l - 1 -> l > m (coefficients `cf`); 1 when lambda_l is in range.
__device__ __forceinline__ int rec_step(const double* cf, double cosv, double* prev, double* cur, int* k, RecConst rc) {
    double next = cf[0] * (cosv * *cur - cf[1] * *prev);
    *prev = *cur;
    *cur = next;
    if (*k < 0) {
        if (fabs(*cur) > rc.limit) {
            *cur /= rc.scale;
            *prev /= rc.scale;
            *k += 1;
        }
        if (*k < 0) return 0;
    }
    return 1;
}

// Legendre sums [m][ring] of `alm` (Sht::legendre): A, and B, C when
// `deriv`.  Grid (ceil((pairs + 1) / blockDim), lmax + 1).
extern "C" __global__ void sht_leg_syn(
    const double2* alm, i64 lmax, const double* coef, const double* log_mm,
    const double* cs, const double* sn, const double* lnsin, i64 nr, i64 pairs, int deriv,
    double rc_lo, double rc_c300, double rc_ln, double rc_scale, double rc_limit,
    const i64* p0, double2* A, double2* B, double2* C)
{
    RecConst rc = {rc_lo, rc_c300, rc_ln, rc_scale, rc_limit};
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 m = blockIdx.y;
    // Rings before p0[m]: negligible up to lmax (the sums stay zero).
    if (p > pairs || p < p0[m]) return;
    double cosv = cs[p], sinv = sn[p];
    i64 off = alm_index(lmax, m, m);
    const double2* a = alm + off;
    const double* cf = coef + 3 * off;
    double2 z = make_double2(0.0, 0.0);
    double2 sa0 = z, sa1 = z, sb0 = z, sb1 = z, sc0 = z, sc1 = z;
    double prev = 0.0, cur;
    int k;
    rec_start(m, lnsin[p], log_mm, rc, &cur, &k);
    for (i64 l = m; l <= lmax; l++) {
        double dlam;
        if (l == m) {
            if (k != 0) continue;
            dlam = (double)m * cosv * cur;
        } else {
            const double* c3 = cf + 3 * (l - m);
            if (!rec_step(c3, cosv, &prev, &cur, &k, rc)) continue;
            dlam = (double)l * cosv * cur - c3[2] * prev;
        }
        double lam = cur;
        double2 x = a[l - m];
        if (((l + m) & 1) == 0) {
            sa0 = c_add(sa0, c_scale(x, lam));
            if (deriv) {
                sb0 = c_add(sb0, c_scale(x, dlam / sinv));
                sc0 = c_add(sc0, c_scale(x, (double)(l * (l + 1)) * lam));
            }
        } else {
            sa1 = c_add(sa1, c_scale(x, lam));
            if (deriv) {
                sb1 = c_add(sb1, c_scale(x, dlam / sinv));
                sc1 = c_add(sc1, c_scale(x, (double)(l * (l + 1)) * lam));
            }
        }
    }
    A[m * nr + p] = c_add(sa0, sa1);
    if (deriv) {
        B[m * nr + p] = c_add(sb0, sb1);
        C[m * nr + p] = c_add(sc0, sc1);
    }
    if (p < pairs) {
        i64 s = nr - 1 - p;
        A[m * nr + s] = c_sub(sa0, sa1);
        if (deriv) {
            B[m * nr + s] = c_sub(sb1, sb0);
            C[m * nr + s] = c_sub(sc0, sc1);
        }
    }
}

// Transpose of sht_leg_syn (Sht::legendre_adjoint): out[chunk][packed
// (l, m)] = sum over the chunk's ring pairs of lambda A + lambda' B +
// l (l + 1) lambda C (B, C when `deriv`).  Grid (lmax + 1, chunks), SHT_T
// threads; thread t holds the rings p0[m] + chunk * SHT_T * SHT_R + j SHT_T
// + t (the rings before p0[m] are negligible: Sht::mlim); a chunk without
// rings writes zeros.
extern "C" __global__ void sht_leg_adj(
    const double2* A, const double2* B, const double2* C, int deriv,
    i64 lmax, const double* coef, const double* log_mm,
    const double* cs, const double* sn, const double* lnsin, i64 nr, i64 pairs, const i64* p0,
    double rc_lo, double rc_c300, double rc_ln, double rc_scale, double rc_limit,
    double2* out, i64 nalm)
{
    __shared__ double red_re[SHT_T / 32][SHT_LT], red_im[SHT_T / 32][SHT_LT];
    RecConst rc = {rc_lo, rc_c300, rc_ln, rc_scale, rc_limit};
    i64 m = blockIdx.x;
    i64 chunk = blockIdx.y;
    int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    i64 off = alm_index(lmax, m, m);
    const double* cf = coef + 3 * off;
    double2 z = make_double2(0.0, 0.0);
    i64 first = p0[m] + chunk * (SHT_T * SHT_R);
    if (first > pairs) {
        for (i64 l = m + t; l <= lmax; l += SHT_T) out[chunk * nalm + off + (l - m)] = z;
        return;
    }
    double prev[SHT_R], cur[SHT_R];
    int k[SHT_R];
#pragma unroll
    for (int j = 0; j < SHT_R; j++) {
        i64 p = first + (i64)j * SHT_T + t;
        prev[j] = 0.0;
        cur[j] = 0.0;
        k[j] = 0;
        if (p <= pairs) rec_start(m, lnsin[p], log_mm, rc, &cur[j], &k[j]);
    }
    for (i64 l0 = m; l0 <= lmax; l0 += SHT_LT) {
        double ar[SHT_LT], ai[SHT_LT];
#pragma unroll
        for (int i = 0; i < SHT_LT; i++) {
            ar[i] = 0.0;
            ai[i] = 0.0;
        }
#pragma unroll
        for (int j = 0; j < SHT_R; j++) {
            i64 p = first + (i64)j * SHT_T + t;
            if (p > pairs) continue;
            double cosv = cs[p], sinv = sn[p];
            i64 s = nr - 1 - p;
            int south = p < pairs;
            double2 n = A[m * nr + p], v = south ? A[m * nr + s] : z;
            double2 ta0 = c_add(n, v), ta1 = c_sub(n, v);
            double2 tb0 = z, tb1 = z, tc0 = z, tc1 = z;
            if (deriv) {
                n = B[m * nr + p];
                v = south ? B[m * nr + s] : z;
                tb0 = c_sub(n, v);
                tb1 = c_add(n, v);
                n = C[m * nr + p];
                v = south ? C[m * nr + s] : z;
                tc0 = c_add(n, v);
                tc1 = c_sub(n, v);
            }
            double pv = prev[j], cu = cur[j];
            int kk = k[j];
#pragma unroll
            for (int i = 0; i < SHT_LT; i++) {
                i64 l = l0 + i;
                if (l > lmax) continue;
                double dlam;
                int ok;
                if (l == m) {
                    ok = kk == 0;
                    dlam = (double)m * cosv * cu;
                } else {
                    const double* c3 = cf + 3 * (l - m);
                    ok = rec_step(c3, cosv, &pv, &cu, &kk, rc);
                    dlam = (double)l * cosv * cu - c3[2] * pv;
                }
                if (!ok) continue;
                double lam = cu;
                int par = (int)((l + m) & 1);
                double2 w = c_scale(par ? ta1 : ta0, lam);
                if (deriv) {
                    double2 e = c_add(c_scale(par ? tb1 : tb0, dlam / sinv), c_scale(par ? tc1 : tc0, (double)(l * (l + 1)) * lam));
                    w = c_add(w, e);
                }
                ar[i] += w.x;
                ai[i] += w.y;
            }
            prev[j] = pv;
            cur[j] = cu;
            k[j] = kk;
        }
#pragma unroll
        for (int i = 0; i < SHT_LT; i++) {
            double r = ar[i], q = ai[i];
            for (int o = 16; o > 0; o >>= 1) {
                r += __shfl_down_sync(0xffffffffu, r, o);
                q += __shfl_down_sync(0xffffffffu, q, o);
            }
            if (lane == 0) {
                red_re[warp][i] = r;
                red_im[warp][i] = q;
            }
        }
        __syncthreads();
        if (t < SHT_LT && l0 + t <= lmax) {
            double r = red_re[0][t], q = red_im[0][t];
            for (int w = 1; w < SHT_T / 32; w++) {
                r += red_re[w][t];
                q += red_im[w][t];
            }
            out[chunk * nalm + off + (l0 + t - m)] = make_double2(r, q);
        }
        __syncthreads();
    }
}

// out[i] = sum over the chunks c in order of part[c][i].
extern "C" __global__ void sht_chunk_sum(const double2* part, i64 nalm, int chunks, double2* out) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nalm) return;
    double2 s = part[i];
    for (int c = 1; c < chunks; c++) s = c_add(s, part[(i64)c * nalm + i]);
    out[i] = s;
}

// The derivative maps as R[ka A + kb B + kc C] (derivative_coefficients):
// the factors of output `o` on a ring of (x, s) = (cos, sin) at m.
__device__ __forceinline__ void deriv_coef(int o, double mf, double x, double s, double2* ka, double2* kb, double2* kc) {
    double2 im = make_double2(0.0, mf);
    switch (o) {
        case 0: *ka = c_real(0.0); *kb = c_real(1.0); *kc = c_real(0.0); break;
        case 1: *ka = c_div(im, s); *kb = c_real(0.0); *kc = c_real(0.0); break;
        case 2: *ka = c_real(mf * mf / (s * s)); *kb = c_real(-x / s); *kc = c_real(-1.0); break;
        case 3: *ka = c_real(-mf * mf / (s * s)); *kb = c_real(x / s); *kc = c_real(0.0); break;
        default: *ka = c_div(c_scale(c_neg(im), x), s * s); *kb = c_div(im, s); *kc = c_real(0.0); break;
    }
}

// Ring values before the inverse FFT (Sht::synthesise): h[o, start_r + j]
// = sum over m = j, j + n, ... <= lmax of H_o(r, m) w_m e^{i m phi0_r}, H =
// A (mode 0) or the five derivative maps (mode 1).  Grid (ceil(max nphi /
// blockDim), nr).
extern "C" __global__ void sht_ring_syn(
    const double2* A, const double2* B, const double2* C, int mode, i64 lmax, i64 nr,
    const i64* start, const int* nphi, const double* phi0, const double* cs, const double* sn,
    double2* h, i64 npix)
{
    i64 r = blockIdx.y;
    i64 j = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 n = nphi[r];
    if (j >= n) return;
    double x = cs[r], s = sn[r], ph = phi0[r];
    int K = mode ? 5 : 1;
    double2 acc[5];
    for (int o = 0; o < 5; o++) acc[o] = make_double2(0.0, 0.0);
    for (i64 m = j; m <= lmax; m += n) {
        double2 shift = c_polar(m == 0 ? 1.0 : 2.0, (double)m * ph);
        i64 i = m * nr + r;
        if (mode == 0) {
            acc[0] = c_add(acc[0], c_mul(A[i], shift));
        } else {
            double2 a = A[i], b = B[i], c = C[i];
            for (int o = 0; o < 5; o++) {
                double2 ka, kb, kc;
                deriv_coef(o, (double)m, x, s, &ka, &kb, &kc);
                double2 v = c_add(c_add(c_mul(ka, a), c_mul(kb, b)), c_mul(kc, c));
                acc[o] = c_add(acc[o], c_mul(v, shift));
            }
        }
    }
    for (int o = 0; o < K; o++) h[o * npix + start[r] + j] = acc[o];
}

// out[o, nest[q]] = Re h[o, q] (RING -> NESTED), q < npix, o < K.
extern "C" __global__ void sht_ring_store(const double2* h, i64 K, i64 npix, const u32* nest, double* out) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= K * npix) return;
    i64 o = t / npix, q = t % npix;
    out[o * npix + nest[q]] = h[t].x;
}

// h[o, q] = x[o, nest[q]] + 0 i (NESTED -> RING).
extern "C" __global__ void sht_ring_load(const double* x, i64 K, i64 npix, const u32* nest, double2* h) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= K * npix) return;
    i64 o = t / npix, q = t % npix;
    h[t] = make_double2(x[o * npix + nest[q]], 0.0);
}

// Ring spectra -> Legendre sums of the transposed synthesis (Sht::analyse):
// f_o = F_o(r, m % n) w e^{-i m phi0} (w = 2 for m > 0 with `with_mult`),
// then mode 0: A = f_0; mode 1: (A, B, C) = sum_o conj(k_o) f_o.  Grid
// (ceil((lmax + 1) / blockDim), nr).
extern "C" __global__ void sht_ring_ana(
    const double2* h, int mode, int with_mult, i64 lmax, i64 nr,
    const i64* start, const int* nphi, const double* phi0, const double* cs, const double* sn, i64 npix,
    double2* A, double2* B, double2* C)
{
    i64 r = blockIdx.y;
    i64 m = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (m > lmax) return;
    i64 n = nphi[r];
    double w = with_mult && m > 0 ? 2.0 : 1.0;
    double2 shift = c_polar(w, -((double)m) * phi0[r]);
    i64 q = start[r] + m % n;
    i64 i = m * nr + r;
    if (mode == 0) {
        A[i] = c_mul(h[q], shift);
        return;
    }
    double x = cs[r], s = sn[r];
    double2 sa = make_double2(0.0, 0.0), sb = sa, sc = sa;
    for (int o = 0; o < 5; o++) {
        double2 f = c_mul(h[o * npix + q], shift);
        double2 ka, kb, kc;
        deriv_coef(o, (double)m, x, s, &ka, &kb, &kc);
        sa = c_add(sa, c_mul(c_conj(ka), f));
        sb = c_add(sb, c_mul(c_conj(kb), f));
        sc = c_add(sc, c_mul(c_conj(kc), f));
    }
    A[i] = sa;
    B[i] = sb;
    C[i] = sc;
}

// alm[(l, m)] *= (m == 0 ? f0 : f1).  Grid (ceil((lmax + 1) / blockDim), lmax + 1).
extern "C" __global__ void sht_scale_m(double2* alm, i64 lmax, double f0, double f1) {
    i64 m = blockIdx.y;
    i64 l = m + (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    i64 i = alm_index(lmax, l, m);
    alm[i] = c_scale(alm[i], m == 0 ? f0 : f1);
}

// c[(l, m)] += a[(l, m)] (m == 0 ? f0 : f1).  Grid as sht_scale_m.
extern "C" __global__ void sht_add_scaled_m(double2* c, const double2* a, i64 lmax, double f0, double f1) {
    i64 m = blockIdx.y;
    i64 l = m + (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    i64 i = alm_index(lmax, l, m);
    c[i] = c_add(c[i], c_scale(a[i], m == 0 ? f0 : f1));
}

// out = p a + q b over n doubles.
extern "C" __global__ void sht_combine(const double* a, double p, const double* b, double q, i64 n, double* out) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = p * a[i] + q * b[i];
}

// C_l = sum_m mult |a_lm|^2 / (2 l + 1), m in increasing order (Sht::power).
extern "C" __global__ void sht_power(const double2* alm, i64 lmax, double* cl) {
    i64 l = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    double s = 0.0;
    for (i64 m = 0; m <= l; m++) {
        double2 a = alm[alm_index(lmax, l, m)];
        s += (m == 0 ? 1.0 : 2.0) * (a.x * a.x + a.y * a.y);
    }
    cl[l] = s / (double)(2 * l + 1);
}

// out = (a * (2 mult dcl[l] / (2 l + 1))) * (m == 0 ? f0 : f1) (Sht::power_vjp,
// then the scaling of Sht::analysis_vjp).  Grid as sht_scale_m.
extern "C" __global__ void sht_power_vjp(const double2* alm, i64 lmax, const double* dcl, double f0, double f1, double2* out) {
    i64 m = blockIdx.y;
    i64 l = m + (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    i64 i = alm_index(lmax, l, m);
    double2 c = c_scale(alm[i], 2.0 * (m == 0 ? 1.0 : 2.0) * dcl[l] / (double)(2 * l + 1));
    out[i] = c_scale(c, m == 0 ? f0 : f1);
}

// Partial sums over chunks of `chunk` pixels, in a fixed tree: mode 0 sum
// a, 1 sum (a - shift)^2, 2 sum a^2 + b^2, 3 sum a b.  Grid n_chunks, 256 threads.
extern "C" __global__ void px_sums(const double* a, const double* b, i64 n, i64 chunk, double shift, int mode, double* out) {
    __shared__ double sh[256];
    i64 c = blockIdx.x;
    i64 lo = c * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double s = 0.0;
    for (i64 p = lo + threadIdx.x; p < hi; p += blockDim.x) {
        double v = a[p];
        if (mode == 0) s += v;
        else if (mode == 1) s += (v - shift) * (v - shift);
        else if (mode == 2) s += v * v + b[p] * b[p];
        else s += v * b[p];
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[c] = sh[0];
}

// u = (x - mean) / sigma.
extern "C" __global__ void mk_standard(const double* x, i64 n, double mean, double sigma, double* u) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    u[p] = (x[p] - mean) / sigma;
}

// Gradient norm and curvature of every pixel from the derivative maps d
// [5, n]: mag = sqrt(q + 1e-30), curv = (2 d1 d2 d12 - d1^2 d22 - d2^2 d11)
// / (q + floor), q = d1^2 + d2^2.
extern "C" __global__ void mk_geom(const double* d, i64 n, double floor, double* mag, double* curv) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    double d1 = d[p], d2 = d[n + p], d11 = d[2 * n + p], d22 = d[3 * n + p], d12 = d[4 * n + p];
    double q = d1 * d1 + d2 * d2;
    mag[p] = sqrt(q + 1e-30);
    curv[p] = (2.0 * d1 * d2 * d12 - d1 * d1 * d22 - d2 * d2 * d11) / (q + floor);
}

// Normal CDF as the host's ndtr.
__device__ __forceinline__ double mk_ndtr(double z) {
    if (z >= 8.5) return 1.0;
    if (z <= -38.5) return 0.0;
    return 0.5 * erfc(-z / 1.4142135623730951);
}

// Minkowski functionals, partial sums per chunk of pixels and threshold t:
// out[(c K + t) 3 + {0, 1, 2}] = sums of ndtr(z), 0.25 delta mag, delta curv
// / tau, z = (u - nu_t) / eps, delta = exp(-z^2 / 2) cn.  Grid (n_chunks, K), 256 threads.
extern "C" __global__ void mk_forward(
    const double* u, const double* mag, const double* curv, i64 n, i64 chunk,
    const double* nu, double eps, double cn, double tau, double* out)
{
    __shared__ double s0[256], s1[256], s2[256];
    i64 c = blockIdx.x;
    int t = blockIdx.y, K = gridDim.y;
    i64 lo = c * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double nt = nu[t];
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (i64 p = lo + threadIdx.x; p < hi; p += blockDim.x) {
        double z = (u[p] - nt) / eps;
        double delta = exp(-0.5 * z * z) * cn;
        a0 += mk_ndtr(z);
        a1 += 0.25 * delta * mag[p];
        a2 += delta * curv[p] / tau;
    }
    s0[threadIdx.x] = a0;
    s1[threadIdx.x] = a1;
    s2[threadIdx.x] = a2;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            s0[threadIdx.x] += s0[threadIdx.x + o];
            s1[threadIdx.x] += s1[threadIdx.x + o];
            s2[threadIdx.x] += s2[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double* w = out + (c * K + t) * 3;
        w[0] = s0[0];
        w[1] = s1[0];
        w[2] = s2[0];
    }
}

// Cotangent of the Minkowski functionals per pixel (Minkowski::vjp, the
// floor held fixed): gu = d/du (divided by n) and gd [5, n], the cotangents
// of the derivative maps; the thresholds in the host's order (nearest
// first, then upwards, then downwards).
extern "C" __global__ void mk_vjp(
    const double* u, const double* d, i64 n, double floor, const double* nu, int K,
    double eps, double cn, double tau, const double* cot, double* gu, double* gd)
{
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    double d1 = d[p], d2 = d[n + p], d11 = d[2 * n + p], d22 = d[3 * n + p], d12 = d[4 * n + p];
    double q = d1 * d1 + d2 * d2;
    double mag = sqrt(q + 1e-30);
    double den = q + floor;
    double num = 2.0 * d1 * d2 * d12 - d1 * d1 * d22 - d2 * d2 * d11;
    double curv = num / den;
    double up = u[p];
    double h = nu[1] - nu[0];
    double cf = round((up - nu[0]) / h);
    cf = cf < 0.0 ? 0.0 : (cf > (double)(K - 1) ? (double)(K - 1) : cf);
    int c = (int)cf;
    double aa = 0.0, bb = 0.0, g = 0.0;
    for (int i = 0; i < K; i++) {
        int t = i < K - c ? c + i : c - 1 - (i - (K - c));
        double z = (up - nu[t]) / eps;
        double delta = exp(-0.5 * z * z) * cn;
        double dprime = -z * delta / eps;
        g += cot[t] * delta + cot[K + t] * dprime * mag / 4.0 + cot[2 * K + t] * dprime * curv / tau;
        aa += cot[K + t] * delta / 4.0;
        bb += cot[2 * K + t] * delta / tau;
    }
    double nn = (double)n;
    aa = aa / nn;
    bb = bb / nn;
    gu[p] = g / nn;
    gd[p] = aa * d1 / mag + bb * ((2.0 * d2 * d12 - 2.0 * d1 * d22) / den - 2.0 * d1 * num / (den * den));
    gd[n + p] = aa * d2 / mag + bb * ((2.0 * d1 * d12 - 2.0 * d2 * d11) / den - 2.0 * d2 * num / (den * den));
    gd[2 * n + p] = -bb * d2 * d2 / den;
    gd[3 * n + p] = -bb * d1 * d1 / den;
    gd[4 * n + p] = bb * 2.0 * d1 * d2 / den;
}

// Map cotangent of u = (x - mean) / sqrt(mean((x - mean)^2) + 1e-12):
// out = (g - mg - u mu) / sigma.
extern "C" __global__ void mk_final(const double* g, const double* u, i64 n, double mg, double mu, double sigma, double* out) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    out[p] = (g[p] - mg - u[p] * mu) / sigma;
}

// Cloud-in-cell histogram per chunk of `chunk` pixels in pixel order (Cic::forward):
// out[c bins + b]; one thread per chunk, bins <= 256.
extern "C" __global__ void cic_partial(const double* x, i64 n, i64 chunk, double lo, double dx, int bins, double* out) {
    i64 c = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (c * chunk >= n) return;
    double h[256];
    for (int b = 0; b < bins; b++) h[b] = 0.0;
    i64 hi = (c + 1) * chunk < n ? (c + 1) * chunk : n;
    for (i64 p = c * chunk; p < hi; p++) {
        double pos = (x[p] - lo) / dx;
        double f = floor(pos);
        double frac = pos - f;
        i64 i0 = (i64)f;
        if (i0 >= 0 && i0 < bins) h[i0] += 1.0 - frac;
        if (i0 + 1 >= 0 && i0 + 1 < bins) h[i0 + 1] += frac;
    }
    for (int b = 0; b < bins; b++) out[c * bins + b] = h[b];
}

// Map cotangent of the CIC density (Cic::vjp): g = (cot[i + 1] - cot[i]) / norm.
extern "C" __global__ void cic_vjp(const double* x, i64 n, double lo, double dx, int bins, const double* cot, double norm, double* g) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    i64 i = (i64)floor((x[p] - lo) / dx);
    double v = 0.0;
    if (i >= 0 && i < bins) v -= cot[i];
    if (i + 1 >= 0 && i + 1 < bins) v += cot[i + 1];
    g[p] = v / norm;
}

// ---------------------------------------------------------------------------
// Batched Legendre sums as GEMMs (GpuSht with cuBLAS).  For a block of m
// [m0, m0 + gridDim.y) and each parity q of l - m, the table
// T_q(m)[p, j] = lambda_{l m}(ring pair p), l = m + q + 2 j, is built once
// (leg_table: the scaled recursion of sht_leg_syn, 0 where it is still
// below range) and shared by the G members and the sums of a stage through
// one DGEMM per (m, q).  cum[2 (m - m0) + q] counts the table rows before
// (m, q) in the block: the table of (m, q) starts at P cum (P ring pairs,
// the equator last), its coefficients X at N cum (N columns, [n_q, N]
// column-major), its ring values O at (2 (m - m0) + q) P N ([P, N]).
// Column ((kind G) + g) 2 + ri holds the real (ri = 0) or imaginary part
// of member g.  Synthesis kinds: a_lm, l (l + 1) a_lm, l a_lm and
// c_{l+1} a_{l+1, m}; the derivative sum is B = (x S_2 - S_3) / s, since
// dlambda_l = l x lambda_l - c_l lambda_{l-1}.  Adjoint kinds: A, C,
// x B / s and B / s, then a_lm = Y_0 + l (l + 1) Y_1 + l Y_2 - c_l Y_3(l - 1).
// The sums differ from sht_leg_syn / sht_leg_adj by rounding (order of the
// GEMM; B and the ring combinations formed before the division).
// ---------------------------------------------------------------------------

__device__ __forceinline__ i64 leg_rows(i64 lmax, i64 m, int q) {
    return q ? (lmax - m + 1) / 2 : (lmax - m) / 2 + 1;
}

// Grid (ceil(P / blockDim), m-block).
extern "C" __global__ void leg_table(
    i64 m0, i64 lmax, const double* coef, const double* log_mm, const double* cs, const double* lnsin, i64 P,
    double rc_lo, double rc_c300, double rc_ln, double rc_scale, double rc_limit, const i64* cum, const i64* p0, double* T)
{
    RecConst rc = {rc_lo, rc_c300, rc_ln, rc_scale, rc_limit};
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 m = m0 + blockIdx.y;
    // Rows before p0[m] (negligible rings) are neither written nor read.
    if (p >= P || p < p0[m]) return;
    double cosv = cs[p];
    const double* cf = coef + 3 * alm_index(lmax, m, m);
    double* t0 = T + P * cum[2 * (m - m0)];
    double* t1 = T + P * cum[2 * (m - m0) + 1];
    double prev = 0.0, cur;
    int k;
    rec_start(m, lnsin[p], log_mm, rc, &cur, &k);
    for (i64 l = m; l <= lmax; l++) {
        int ok = l == m ? k == 0 : rec_step(cf + 3 * (l - m), cosv, &prev, &cur, &k, rc);
        double v = ok ? cur : 0.0;
        i64 j = (l - m) >> 1;
        if ((l - m) & 1) t1[j * P + p] = v;
        else t0[j * P + p] = v;
    }
}

__device__ __forceinline__ void leg_put(double* x, i64 n, i64 j, int kind, int G, int g, double2 v) {
    i64 col = ((i64)kind * G + g) * 2;
    x[col * n + j] = v.x;
    x[(col + 1) * n + j] = v.y;
}

__device__ __forceinline__ double2 leg_get(const double* o, i64 n, i64 j, int kind, int G, int g) {
    i64 col = ((i64)kind * G + g) * 2;
    return make_double2(o[col * n + j], o[(col + 1) * n + j]);
}

// Synthesis coefficients X of G members (rows of nalm complex a_lm).
// Grid (ceil((lmax + 1) / blockDim), m-block).
extern "C" __global__ void leg_pack_syn(const double2* alm, i64 nalm, int G, int kinds, i64 m0, i64 lmax, const double* coef, const i64* cum, double* X) {
    i64 m = m0 + blockIdx.y;
    i64 l = m + (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    int q = (int)((l - m) & 1);
    i64 j = (l - m) >> 1, n = leg_rows(lmax, m, q);
    i64 N = 2 * (i64)G * kinds;
    double* x = X + N * cum[2 * (m - m0) + q];
    i64 i = alm_index(lmax, l, m);
    double lf = (double)l;
    double cn = l < lmax ? coef[3 * (i + 1) + 2] : 0.0;
    for (int g = 0; g < G; g++) {
        double2 a = alm[g * nalm + i];
        leg_put(x, n, j, 0, G, g, a);
        if (kinds == 4) {
            double2 an = l < lmax ? alm[g * nalm + i + 1] : make_double2(0.0, 0.0);
            leg_put(x, n, j, 1, G, g, c_scale(a, lf * (lf + 1.0)));
            leg_put(x, n, j, 2, G, g, c_scale(a, lf));
            leg_put(x, n, j, 3, G, g, c_scale(an, cn));
        }
    }
}

// Ring sums [g][m][ring] (rows of `stride` complex) of G members from the
// GEMM values O: north ring p, south ring nr - 1 - p.  Grid (ceil(P /
// blockDim), m-block).
extern "C" __global__ void leg_unpack_syn(
    const double* O, int G, int kinds, i64 m0, i64 P, i64 nr, i64 pairs, const double* cs, const double* sn,
    double2* A, double2* B, double2* C, i64 stride)
{
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 m = m0 + blockIdx.y;
    if (p >= P) return;
    i64 N = 2 * (i64)G * kinds;
    const double* o0 = O + (2 * (m - m0)) * P * N;
    const double* o1 = o0 + P * N;
    double x = cs[p], s = sn[p];
    int south = p < pairs;
    i64 rn = m * nr + p, rs = m * nr + (nr - 1 - p);
    for (int g = 0; g < G; g++) {
        i64 row = g * stride;
        double2 e = leg_get(o0, P, p, 0, G, g), o = leg_get(o1, P, p, 0, G, g);
        A[row + rn] = c_add(e, o);
        if (south) A[row + rs] = c_sub(e, o);
        if (kinds == 4) {
            e = leg_get(o0, P, p, 1, G, g);
            o = leg_get(o1, P, p, 1, G, g);
            C[row + rn] = c_add(e, o);
            if (south) C[row + rs] = c_sub(e, o);
            double2 e2 = leg_get(o0, P, p, 2, G, g), o2 = leg_get(o1, P, p, 2, G, g);
            double2 e3 = leg_get(o0, P, p, 3, G, g), o3 = leg_get(o1, P, p, 3, G, g);
            B[row + rn] = c_div(c_sub(c_scale(c_add(e2, o2), x), c_add(e3, o3)), s);
            if (south) B[row + rs] = c_div(c_sub(c_scale(c_sub(e2, o2), -x), c_sub(e3, o3)), s);
        }
    }
}

// Adjoint columns T ([P, N] per (m, q)) of G members from their ring sums.
// Grid (ceil(P / blockDim), m-block).
extern "C" __global__ void leg_pack_adj(
    const double2* A, const double2* B, const double2* C, i64 stride, int G, int kinds, i64 m0, i64 P, i64 nr, i64 pairs,
    const double* cs, const double* sn, double* T)
{
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 m = m0 + blockIdx.y;
    if (p >= P) return;
    i64 N = 2 * (i64)G * kinds;
    double* t0 = T + (2 * (m - m0)) * P * N;
    double* t1 = t0 + P * N;
    double x = cs[p], s = sn[p];
    int south = p < pairs;
    i64 rn = m * nr + p, rs = m * nr + (nr - 1 - p);
    double2 z = make_double2(0.0, 0.0);
    for (int g = 0; g < G; g++) {
        i64 row = g * stride;
        double2 n = A[row + rn], v = south ? A[row + rs] : z;
        leg_put(t0, P, p, 0, G, g, c_add(n, v));
        leg_put(t1, P, p, 0, G, g, c_sub(n, v));
        if (kinds == 4) {
            n = C[row + rn];
            v = south ? C[row + rs] : z;
            leg_put(t0, P, p, 1, G, g, c_add(n, v));
            leg_put(t1, P, p, 1, G, g, c_sub(n, v));
            n = B[row + rn];
            v = south ? B[row + rs] : z;
            leg_put(t0, P, p, 2, G, g, c_div(c_scale(c_sub(n, v), x), s));
            leg_put(t1, P, p, 2, G, g, c_div(c_scale(c_add(n, v), x), s));
            leg_put(t0, P, p, 3, G, g, c_div(c_add(n, v), s));
            leg_put(t1, P, p, 3, G, g, c_div(c_sub(n, v), s));
        }
    }
}

// a_lm of G members (rows of nalm complex) from the adjoint GEMM values Y.
// Grid (ceil((lmax + 1) / blockDim), m-block).
extern "C" __global__ void leg_unpack_adj(const double* Y, int G, int kinds, i64 m0, i64 lmax, const double* coef, const i64* cum, double2* alm, i64 nalm) {
    i64 m = m0 + blockIdx.y;
    i64 l = m + (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    int q = (int)((l - m) & 1);
    i64 j = (l - m) >> 1, n = leg_rows(lmax, m, q);
    i64 N = 2 * (i64)G * kinds;
    const double* y = Y + N * cum[2 * (m - m0) + q];
    i64 i = alm_index(lmax, l, m);
    double lf = (double)l, cl = coef[3 * i + 2];
    // The other parity, at the row of l - 1.
    const double* y1 = Y + N * cum[2 * (m - m0) + (1 - q)];
    i64 n1 = leg_rows(lmax, m, 1 - q), j1 = (l - 1 - m) >> 1;
    for (int g = 0; g < G; g++) {
        double2 v = leg_get(y, n, j, 0, G, g);
        if (kinds == 4) {
            v = c_add(v, c_scale(leg_get(y, n, j, 1, G, g), lf * (lf + 1.0)));
            v = c_add(v, c_scale(leg_get(y, n, j, 2, G, g), lf));
            if (l > m) v = c_sub(v, c_scale(leg_get(y1, n1, j1, 3, G, g), cl));
        }
        alm[g * nalm + i] = v;
    }
}

// Spin-2 transforms through the covariant Hessians of scalar fields (GpuEb):
// healpy's alm2map_spin(spin 2) of (a_E, a_B) is
//   Q = -(H_tt - H_pp)[psi_E] + 2 H_tp[psi_B],  U = -2 H_tp[psi_E] - (H_tt - H_pp)[psi_B],
// psi_X = Y(N_l a_X), N_l = sqrt((l - 2)! / (l + 2)!) (0 for l < 2), the
// Hessians the derivative maps 2, 3, 4 of Sht::derivatives.

// alm[(l, m)] *= N_l (m == 0 ? f0 : f1).  Grid as sht_scale_m.
extern "C" __global__ void sht_scale_spin2(double2* alm, i64 lmax, double f0, double f1) {
    i64 m = blockIdx.y;
    i64 l = m + (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (l > lmax) return;
    i64 i = alm_index(lmax, l, m);
    double nl = l < 2 ? 0.0 : 1.0 / sqrt((double)(l - 1) * (double)l * (double)(l + 1) * (double)(l + 2));
    alm[i] = c_scale(alm[i], nl * (m == 0 ? f0 : f1));
}

// (Q, U) of the derivative maps de, db [5, n] of psi_E and psi_B.
extern "C" __global__ void eb_from_hessians(const double* de, const double* db, i64 n, double* q, double* u) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    q[p] = -(de[2 * n + p] - de[3 * n + p]) + 2.0 * db[4 * n + p];
    u[p] = -2.0 * de[4 * n + p] - (db[2 * n + p] - db[3 * n + p]);
}

// Transpose of eb_from_hessians: the cotangents ce, cb [5, n] of the
// derivative maps from those of (Q, U).
extern "C" __global__ void eb_to_hessians(const double* gq, const double* gu, i64 n, double* ce, double* cb) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    double a = gq[p], b = gu[p];
    ce[p] = 0.0; ce[n + p] = 0.0; ce[2 * n + p] = -a; ce[3 * n + p] = a; ce[4 * n + p] = -2.0 * b;
    cb[p] = 0.0; cb[n + p] = 0.0; cb[2 * n + p] = -b; cb[3 * n + p] = b; cb[4 * n + p] = 2.0 * a;
}

// out = a w over n doubles (a map times a mask).
extern "C" __global__ void px_mul(const double* a, const double* w, i64 n, double* out) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * w[i];
}

// dst += a w over n doubles.
extern "C" __global__ void px_mul_add(double* dst, const double* a, const double* w, i64 n) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += a[i] * w[i];
}
