// Device kernels of stl-gpu, compiled at run time by NVRTC with
// `--fmad=false` (no contraction of a * b + c into an FMA), IEEE division
// and square root: every arithmetic expression below is evaluated in the
// order of its Rust counterpart in stl-core, so the only differences with
// the CPU engine come from the transcendental functions (CUDA libm, within
// 1-2 ulp of glibc) and from the documented summation orders.
//
// Layouts follow stl-core: row-major, pixel axis last.  A complex field is
// stored as two planes per row ("planar"): `[rows, 2, n]`, re then im,
// which is the output layout of the convolution (`[R, G * C_out, K]` with
// `C_out = 2`).

typedef long long i64;
typedef unsigned long long u64;
typedef unsigned int u32;

#define PI_D 3.141592653589793
#define RAD2DEG 57.295779513082320876798154814105170332405472466564
#define DEG2RAD 0.017453292519943295769236907684886127134428718885417

__constant__ i64 JRLL[12] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
__constant__ i64 JPLL[12] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

// ---------------------------------------------------------------------------
// HEALPix (port of stl-core/src/healpix.rs)
// ---------------------------------------------------------------------------

__device__ __forceinline__ double rad2deg(double x) { return x * RAD2DEG; }
__device__ __forceinline__ double deg2rad(double x) { return x * DEG2RAD; }

// torch.remainder / Python % for floats.
__device__ __forceinline__ double py_remainder(double a, double b) {
    double m = fmod(a, b);
    if (m != 0.0 && ((b < 0.0) != (m < 0.0))) m += b;
    return m;
}

__device__ __forceinline__ i64 floor_div(i64 a, i64 b) {
    i64 q = a / b;
    i64 r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}

__device__ __forceinline__ i64 compress_bits(i64 v) {
    v &= 0x5555555555555555LL;
    v = (v | (v >> 1)) & 0x3333333333333333LL;
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0FLL;
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FFLL;
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFFLL;
    return (v | (v >> 16)) & 0x00000000FFFFFFFFLL;
}

__device__ __forceinline__ i64 spread_bits(i64 v) {
    v &= 0x00000000FFFFFFFFLL;
    v = (v | (v << 16)) & 0x0000FFFF0000FFFFLL;
    v = (v | (v << 8)) & 0x00FF00FF00FF00FFLL;
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0FLL;
    v = (v | (v << 2)) & 0x3333333333333333LL;
    return (v | (v << 1)) & 0x5555555555555555LL;
}

__device__ __forceinline__ i64 xyf2nest(i64 ix, i64 iy, i64 face, int order) {
    return (face << (2 * order)) + spread_bits(ix) + (spread_bits(iy) << 1);
}

// Colatitude and longitude of a NESTED pixel centre.
__device__ void pix2ang_nest(i64 pix, i64 nside, int order, double* theta_out, double* phi_out) {
    i64 face = pix >> (2 * order);
    i64 local = pix & ((1LL << (2 * order)) - 1);
    i64 ix = compress_bits(local), iy = compress_bits(local >> 1);
    i64 jr = JRLL[face] * nside - ix - iy - 1;
    double ns = (double)nside;
    double fact2 = 4.0 / (12.0 * ns * ns);
    double fact1 = 2.0 / (3.0 * ns);
    bool north = jr < nside;
    bool south = jr > 3 * nside;
    i64 nr = north ? jr : (south ? 4 * nside - jr : nside);
    i64 kshift = (north || south) ? 0 : ((jr - nside) & 1);
    double nrf = (double)nr;
    double theta;
    if (north || south) {
        double tmp = nrf * nrf * fact2;
        double z_cap = 1.0 - tmp;
        double sth_cap = sqrt(tmp * (2.0 - tmp));
        double theta_cap = atan2(sth_cap, z_cap);
        theta = north ? theta_cap : PI_D - theta_cap;
    } else {
        theta = acos((2.0 * ns - (double)jr) * fact1);
    }
    i64 jp = floor_div(JPLL[face] * nr + ix - iy + 1 + kshift, 2);
    if (jp > 4 * nr) jp -= 4 * nr;
    if (jp < 1) jp += 4 * nr;
    double phi = ((double)jp - ((double)kshift + 1.0) * 0.5) * (0.5 * PI_D / nrf);
    *theta_out = theta;
    *phi_out = phi;
}

__device__ i64 isqrt_i(i64 v) {
    i64 r = (i64)sqrt((double)v);
    if (r * r > v) r -= 1;
    if ((r + 1) * (r + 1) <= v) r += 1;
    return r;
}

__device__ i64 ring2nest(i64 pix, int order) {
    i64 nside = 1LL << order;
    i64 npix = 12 * nside * nside;
    i64 ncap = 2 * nside * (nside - 1);
    i64 nl2 = 2 * nside;
    i64 iring, iphi, kshift, nr, face;
    if (pix < ncap) {
        i64 ir = (1 + isqrt_i(1 + 2 * pix)) >> 1;
        i64 ip = pix + 1 - 2 * ir * (ir - 1);
        iring = ir; iphi = ip; kshift = 0; nr = ir;
        face = floor_div(ip - 1, ir > 1 ? ir : 1);
    } else if (pix < npix - ncap) {
        i64 ip = pix - ncap;
        i64 tmp = ip >> (order + 2);
        i64 ir = tmp + nside;
        i64 ip2 = ip - tmp * 4 * nside + 1;
        i64 ire = ir - nside + 1;
        i64 irm = nl2 + 2 - ire;
        i64 ifm = (ip2 - floor_div(ire, 2) + nside - 1) >> order;
        i64 ifp = (ip2 - floor_div(irm, 2) + nside - 1) >> order;
        iring = ir; iphi = ip2; kshift = (ir + nside) & 1; nr = nside;
        face = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : ifm + 8);
    } else {
        i64 ip = npix - pix;
        i64 ir = (1 + isqrt_i(2 * ip - 1)) >> 1;
        i64 ip2 = 4 * ir + 1 - (ip - 2 * ir * (ir - 1));
        iring = 2 * nl2 - ir; iphi = ip2; kshift = 0; nr = ir;
        face = 8 + floor_div(ip2 - 1, ir > 1 ? ir : 1);
    }
    i64 irt = iring - JRLL[face] * nside + 1;
    i64 ipt = 2 * iphi - JPLL[face] * nr - kshift - 1;
    if (ipt >= nl2) ipt -= 8 * nside;
    i64 ix = (ipt - irt) >> 1;
    i64 iy = (-ipt - irt) >> 1;
    return xyf2nest(ix, iy, face, order);
}

__device__ i64 ring_above(double z, i64 nside) {
    double az = fabs(z);
    double ns = (double)nside;
    if (az <= 2.0 / 3.0) return (i64)trunc(ns * (2.0 - 1.5 * z));
    i64 iring = (i64)trunc(ns * sqrt(3.0 * (1.0 - az)));
    return z > 0.0 ? iring : 4 * nside - iring - 1;
}

__device__ void ring_info(i64 ring, i64 nside, i64* startpix_o, i64* ringpix_o, double* theta_o, bool* shifted_o) {
    i64 ncap = 2 * nside * (nside - 1);
    i64 npix = 12 * nside * nside;
    double ns = (double)nside;
    double fact1 = 2.0 / (3.0 * ns);
    double fact2 = 1.0 / (3.0 * ns * ns);
    i64 northring = ring > 2 * nside ? 4 * nside - ring : ring;
    bool cap = northring < nside;
    double theta;
    i64 ringpix, startpix;
    bool shifted;
    if (cap) {
        double nrf = (double)northring;
        double tmp = nrf * nrf * fact2;
        theta = atan2(sqrt(tmp * (2.0 - tmp)), 1.0 - tmp);
        ringpix = 4 * northring;
        shifted = true;
        startpix = 2 * northring * (northring - 1);
    } else {
        theta = acos((double)(2 * nside - northring) * fact1);
        ringpix = 4 * nside;
        shifted = ((northring - nside) & 1) == 0;
        startpix = ncap + (northring - nside) * 4 * nside;
    }
    if (northring != ring) {
        theta = PI_D - theta;
        startpix = npix - startpix - ringpix;
    }
    *startpix_o = startpix; *ringpix_o = ringpix; *theta_o = theta; *shifted_o = shifted;
}

__device__ void ring_pair(double phi, i64 ring, i64 nside, i64* pa, i64* pb, double* wa, double* wb, double* theta_ring) {
    i64 sp, nr;
    bool shifted;
    ring_info(ring, nside, &sp, &nr, theta_ring, &shifted);
    double dphi = (2.0 * PI_D) / (double)nr;
    double half = shifted ? 0.5 : 0.0;
    double tmp = phi / dphi - half;
    i64 i1 = (i64)trunc(tmp);
    if (tmp < 0.0) i1 -= 1;
    double w1 = (phi - ((double)i1 + half) * dphi) / dphi;
    i64 i2 = i1 + 1;
    if (i1 < 0) i1 += nr;
    if (i2 >= nr) i2 -= nr;
    *pa = sp + i1; *pb = sp + i2; *wa = 1.0 - w1; *wb = w1;
}

__device__ void get_interpol_ring(double theta, double phi, i64 nside, i64 pix[4], double wgt[4]) {
    i64 npix = 12 * nside * nside;
    double z = cos(theta);
    i64 ir1 = ring_above(z, nside);
    i64 ir2 = ir1 + 1;
    i64 p[4] = {0, 0, 0, 0};
    double w[4] = {0.0, 0.0, 0.0, 0.0};
    i64 p0, p1, p2, p3;
    double w0, w1, w2, w3, theta1, theta2;
    ring_pair(phi, ir1 > 1 ? ir1 : 1, nside, &p0, &p1, &w0, &w1, &theta1);
    if (ir1 > 0) { p[0] = p0; p[1] = p1; w[0] = w0; w[1] = w1; }
    ring_pair(phi, ir2 < 4 * nside - 1 ? ir2 : 4 * nside - 1, nside, &p2, &p3, &w2, &w3, &theta2);
    if (ir2 < 4 * nside) { p[2] = p2; p[3] = p3; w[2] = w2; w[3] = w3; }
    bool north = ir1 == 0;
    bool south = ir2 == 4 * nside && !north;
    if (north) {
        double wn = theta / theta2;
        double facn = (1.0 - wn) * 0.25;
        pix[0] = (p[2] + 2) & 3; pix[1] = (p[3] + 2) & 3; pix[2] = p[2]; pix[3] = p[3];
        wgt[0] = facn; wgt[1] = facn; wgt[2] = w[2] * wn + facn; wgt[3] = w[3] * wn + facn;
    } else if (south) {
        double ws = (theta - theta1) / (PI_D - theta1);
        double facs = ws * 0.25;
        pix[0] = p[0]; pix[1] = p[1];
        pix[2] = ((p[0] + 2) & 3) + npix - 4; pix[3] = ((p[1] + 2) & 3) + npix - 4;
        wgt[0] = w[0] * (1.0 - ws) + facs; wgt[1] = w[1] * (1.0 - ws) + facs; wgt[2] = facs; wgt[3] = facs;
    } else {
        double wm = (theta - theta1) / (theta2 - theta1);
        for (int i = 0; i < 4; i++) pix[i] = p[i];
        wgt[0] = w[0] * (1.0 - wm); wgt[1] = w[1] * (1.0 - wm); wgt[2] = w[2] * wm; wgt[3] = w[3] * wm;
    }
}

__device__ __forceinline__ double norm3(double a, double b, double c) { return sqrt(a * a + b * b + c * c); }

__device__ void interp_weights_nested(double v0, double v1, double v2, int order, i64 pix[4], double w[4]) {
    double n = fmax(norm3(v0, v1, v2), 1e-12);
    double x = v0 / n, y = v1 / n, z = v2 / n;
    double theta = acos(fmin(fmax(z, -1.0), 1.0));
    double phi = py_remainder(atan2(y, x), 2.0 * PI_D);
    double lon_deg = rad2deg(phi);
    double lat_deg = 90.0 - rad2deg(theta);
    double theta2 = deg2rad(90.0 - lat_deg);
    double phi2 = py_remainder(deg2rad(lon_deg), 2.0 * PI_D);
    i64 rp[4];
    get_interpol_ring(theta2, phi2, 1LL << order, rp, w);
    for (int i = 0; i < 4; i++) pix[i] = ring2nest(rp[i], order);
}

// ---------------------------------------------------------------------------
// Convolution geometry (port of ConvGeometry::rows and rotation_matrices)
// ---------------------------------------------------------------------------

// Position of `v` in the sorted grid, or -1.  Cells of the contiguous range
// [own_start, own_end) sit at own_pos0 + (v - own_start).
__device__ __forceinline__ i64 grid_find(const i64* grid, i64 n_grid, i64 own_start, i64 own_end, i64 own_pos0, i64 v) {
    if (v >= own_start && v < own_end) return own_pos0 + (v - own_start);
    i64 lo = 0, hi = n_grid;
    while (lo < hi) {
        i64 mid = (lo + hi) >> 1;
        if (grid[mid] < v) lo = mid + 1; else hi = mid;
    }
    return (lo < n_grid && grid[lo] == v) ? lo : -1;
}

// Bound stencils of the output rows `rows[K]` reading `grid`, one thread
// per (row, gauge).  Entry ((g * 4 + n) * K + k) * P + p.  `complete[k * G + g]`.
extern "C" __global__ void geom_build(
    const i64* grid, i64 n_grid, i64 own_start, i64 own_end, i64 own_pos0,
    const i64* rows, i64 K, i64 nside, int order, int G, int P, int gauge,
    const double* stencil, int p_ref,
    u32* pos, double* wout, unsigned char* complete)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= K * G) return;
    i64 k = t / G;
    int g = (int)(t - k * G);
    double th, ph;
    pix2ang_nest(rows[k], nside, order, &th, &ph);
    double theta = deg2rad(90.0 - rad2deg(PI_D / 2.0 - th));
    double phi = deg2rad(rad2deg(ph));
    // rotation_matrices(theta, phi, G, gauge)[g]
    double ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin(phi);
    double base[3][3] = {{cp * ct, -sp, cp * st}, {sp * ct, cp, sp * st}, {-st, 0.0, ct}};
    double nn = fmax(norm3(base[0][2], base[1][2], base[2][2]), 1e-12);
    double nv[3] = {base[0][2] / nn, base[1][2] / nn, base[2][2] / nn};
    double alpha_base, sign_g;
    if (gauge == 0) {
        if (theta > PI_D / 2.0) { alpha_base = phi; sign_g = -1.0; } else { alpha_base = -phi; sign_g = 1.0; }
    } else {
        alpha_base = 0.0; sign_g = 1.0;
    }
    double skew[3][3] = {{0.0, -nv[2], nv[1]}, {nv[2], 0.0, -nv[0]}, {-nv[1], nv[0], 0.0}};
    double step = PI_D / (double)G;
    double alpha = alpha_base + sign_g * ((double)g * step);
    double ca = cos(alpha), sa = sin(alpha);
    double rg[3][3], r[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double eye = i == j ? 1.0 : 0.0;
            rg[i][j] = eye * ca + skew[i][j] * sa + nv[i] * nv[j] * (1.0 - ca);
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r[i][j] = rg[i][0] * base[0][j] + rg[i][1] * base[1][j] + rg[i][2] * base[2][j];

    i64 ref_idx[4];
    double ref_w[4];
    {
        const double* v = stencil + 3 * p_ref;
        interp_weights_nested(r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
                              r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
                              r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2], order, ref_idx, ref_w);
    }
    bool comp = true;
    for (int p = 0; p < P; p++) {
        const double* v = stencil + 3 * p;
        i64 ix[4];
        double wx[4];
        interp_weights_nested(r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
                              r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
                              r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2], order, ix, wx);
        i64 found[4];
        bool any = false;
        for (int n = 0; n < 4; n++) {
            found[n] = grid_find(grid, n_grid, own_start, own_end, own_pos0, ix[n]);
            any = any || found[n] >= 0;
        }
        if (!any) {
            for (int n = 0; n < 4; n++) {
                ix[n] = ref_idx[n];
                wx[n] = ref_w[n];
                found[n] = grid_find(grid, n_grid, own_start, own_end, own_pos0, ix[n]);
            }
        }
        double w[4];
        for (int n = 0; n < 4; n++) w[n] = wx[n] * (found[n] >= 0 ? 1.0 : 0.0);
        double colsum = w[0] + w[1] + w[2] + w[3];
        if (colsum == 0.0) {
            w[0] = found[0] >= 0 ? 1.0 : 0.0;
            colsum = w[0] + w[1] + w[2] + w[3];
        }
        for (int n = 0; n < 4; n++) {
            i64 a = (((i64)g * 4 + n) * K + k) * P + p;
            wout[a] = w[n] / fmax(colsum, 1e-12);
            pos[a] = found[n] >= 0 ? (u32)found[n] : 0u;
            comp = comp && found[n] >= 0;
        }
    }
    if (complete) complete[k * G + g] = comp ? 1 : 0;
}

// Transposition of the bound stencils: entries grouped by grid position,
// in increasing entry order within a group (deterministic adjoint).
extern "C" __global__ void geom_count(const u32* pos, u64 n_entries, u32* count) {
    for (u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x; e < n_entries; e += (u64)gridDim.x * blockDim.x)
        atomicAdd(&count[pos[e]], 1u);
}

extern "C" __global__ void geom_fill(const u32* pos, u64 n_entries, u32* cursor, u32* idx) {
    for (u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x; e < n_entries; e += (u64)gridDim.x * blockDim.x) {
        u32 slot = atomicAdd(&cursor[pos[e]], 1u);
        idx[slot] = (u32)e;
    }
}

// One block per segment: bitonic sort in shared memory (segments up to
// SEG_CAP entries), serial insertion sort above.
#define SEG_CAP 2048
extern "C" __global__ void seg_sort(const u32* offsets, i64 n_seg, u32* idx) {
    __shared__ u32 buf[SEG_CAP];
    for (i64 s = blockIdx.x; s < n_seg; s += gridDim.x) {
        u32 a = offsets[s], b = offsets[s + 1];
        u32 len = b - a;
        if (len <= 1) continue;
        if (len > SEG_CAP) {
            if (threadIdx.x == 0) {
                for (u32 i = a + 1; i < b; i++) {
                    u32 v = idx[i];
                    u32 j = i;
                    while (j > a && idx[j - 1] > v) { idx[j] = idx[j - 1]; j--; }
                    idx[j] = v;
                }
            }
            __syncthreads();
            continue;
        }
        u32 m = 1;
        while (m < len) m <<= 1;
        for (u32 i = threadIdx.x; i < m; i += blockDim.x) buf[i] = i < len ? idx[a + i] : 0xFFFFFFFFu;
        __syncthreads();
        for (u32 size = 2; size <= m; size <<= 1) {
            for (u32 stride = size >> 1; stride > 0; stride >>= 1) {
                for (u32 i = threadIdx.x; i < m; i += blockDim.x) {
                    u32 j = i ^ stride;
                    if (j > i) {
                        bool up = (i & size) == 0;
                        u32 x = buf[i], y = buf[j];
                        if ((x > y) == up) { buf[i] = y; buf[j] = x; }
                    }
                }
                __syncthreads();
            }
        }
        for (u32 i = threadIdx.x; i < len; i += blockDim.x) idx[a + i] = buf[i];
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Convolution forward / adjoint (AdjointOps, C_in = 1, C_out = 2)
// ---------------------------------------------------------------------------

// out[((r * G + g) * 2 + o) * out_stride + opos(i)] = sum_p (sum_n x[r, pos] w) ker[g, o, p]
// for the geometry rows k = row0 + i (i < n_sel; opos null: i).
extern "C" __global__ void conv_fwd(
    const u32* pos, const double* w, int G, i64 K, int P, const double* ker,
    const double* x, i64 x_stride, int R,
    i64 row0, i64 n_sel, const u32* opos, i64 out_stride, double* out)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 total = n_sel * G * (i64)R;
    if (t >= total) return;
    i64 i = t % n_sel;
    i64 rest = t / n_sel;
    int g = (int)(rest % G);
    int r = (int)(rest / G);
    i64 k = row0 + i;
    const double* xs = x + (i64)r * x_stride;
    const double* k0 = ker + ((i64)g * 2 + 0) * P;
    const double* k1 = ker + ((i64)g * 2 + 1) * P;
    double s0 = 0.0, s1 = 0.0;
    for (int p = 0; p < P; p++) {
        double acc = 0.0;
        for (int n = 0; n < 4; n++) {
            i64 a = (((i64)g * 4 + n) * K + k) * P + p;
            acc += xs[pos[a]] * w[a];
        }
        s0 += acc * k0[p];
        s1 += acc * k1[p];
    }
    i64 o = opos ? (i64)opos[i] : i;
    out[(((i64)r * G + g) * 2 + 0) * out_stride + o] = s0;
    out[(((i64)r * G + g) * 2 + 1) * out_stride + o] = s1;
}

// Adjoint, gathered per destination: segment s holds the entries reading
// grid position dest[s], in increasing order, i.e. gauge, bilinear
// neighbour, row, tap (the scatter order of
// AdjointOps.convolution_adjoint).  Only the rows k in [row0, row0 +
// n_rows) take part; gin `[R, G, 2, gin_stride]` at index k - row0.  The
// result goes to out[r * out_stride + dmap(dest[s])] (dmap null:
// dest[s]; -1: dropped), added when `accumulate`.
extern "C" __global__ void conv_adj(
    const u32* dest, const u32* offsets, i64 n_dest, const u32* idx, const double* w,
    int G, i64 K, int P, const double* ker,
    const double* gin, i64 gin_stride, int R, i64 row0, i64 n_rows,
    const int* dmap, double* out, i64 out_stride, int accumulate)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_dest * (i64)R) return;
    i64 s = t % n_dest;
    int r = (int)(t / n_dest);
    u32 a = offsets[s], b = offsets[s + 1];
    i64 dst = dmap ? (i64)dmap[dest[s]] : (i64)dest[s];
    if (dst < 0) return;
    double acc = 0.0;
    bool any = false;
    for (u32 q = a; q < b; q++) {
        u32 e = idx[q];
        int p = (int)(e % (u32)P);
        u64 kk = (u64)(e / (u32)P);
        i64 k = (i64)(kk % (u64)K);
        int gn = (int)(kk / (u64)K);
        int g = gn >> 2;
        i64 ri = k - row0;
        if (ri < 0 || ri >= n_rows) continue;
        double taps = 0.0;
        taps += gin[(((i64)r * G + g) * 2 + 0) * gin_stride + ri] * ker[((i64)g * 2 + 0) * P + p];
        taps += gin[(((i64)r * G + g) * 2 + 1) * gin_stride + ri] * ker[((i64)g * 2 + 1) * P + p];
        acc += taps * w[e];
        any = true;
    }
    if (!any) return;
    double* o = out + (i64)r * out_stride + dst;
    if (accumulate) *o += acc; else *o = acc;
}

// ---------------------------------------------------------------------------
// Sparse rows (decimation and its transpose)
// ---------------------------------------------------------------------------

// y[r * y_stride + y0 + i] (+)= sum_q vals[q] x[r * x_stride + cols[q]] for
// the rows i < n_rows, each row summed in column order from 0.
extern "C" __global__ void csr_apply(
    const u32* indptr, const u32* cols, const double* vals, i64 n_rows,
    const double* x, i64 x_stride, int R, double* y, i64 y_stride, i64 y0, int accumulate)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_rows * (i64)R) return;
    i64 i = t % n_rows;
    int r = (int)(t / n_rows);
    const double* xs = x + (i64)r * x_stride;
    double s = 0.0;
    for (u32 q = indptr[i]; q < indptr[i + 1]; q++) s = s + vals[q] * xs[cols[q]];
    double* o = y + (i64)r * y_stride + y0 + i;
    if (accumulate) *o += s; else *o = s;
}

// ---------------------------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------------------------

// |z| of a planar complex field at the positions pos[i] (rows R):
// out[r * out_stride + pos[i]] = hypot(z[(2r) * z_stride + pos[i]], z[(2r+1) * z_stride + pos[i]]).
extern "C" __global__ void modulus_at(const double* z, i64 z_stride, const u32* pos, i64 n, int R, double* out, i64 out_stride) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 i = t % n;
    int r = (int)(t / n);
    i64 p = pos[i];
    double re = z[(2 * (i64)r) * z_stride + p], im = z[(2 * (i64)r + 1) * z_stride + p];
    out[(i64)r * out_stride + p] = hypot(re, im);
}

// gz += modulus_vjp(z, g) (planar complex gz, z; real g), element i of
// rows R: z at z_off + zpos(i), g at g_off + gpos(i), gz at gz_off + i
// (positions: index arrays or contiguous when null).
extern "C" __global__ void modvjp_add(
    const double* z, i64 z_stride, i64 z_off, const u32* zpos,
    const double* g, i64 g_stride, i64 g_off, const u32* gpos,
    double* gz, i64 gz_stride, i64 gz_off, i64 n, int R)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 i = t % n;
    int r = (int)(t / n);
    i64 zp = z_off + (zpos ? (i64)zpos[i] : i);
    i64 gp = g_off + (gpos ? (i64)gpos[i] : i);
    double re = z[(2 * (i64)r) * z_stride + zp], im = z[(2 * (i64)r + 1) * z_stride + zp];
    double gi = g[(i64)r * g_stride + gp];
    double mag = hypot(re, im);
    double d = mag == 0.0 ? 1.0 : mag;
    gz[(2 * (i64)r) * gz_stride + gz_off + i] += (gi * re) / d;
    gz[(2 * (i64)r + 1) * gz_stride + gz_off + i] += (gi * im) / d;
}

// dst[r * d_stride + d_off + dpos(i)] += src[r * s_stride + s_off + spos(i)].
extern "C" __global__ void gather_add(
    const double* src, i64 s_stride, i64 s_off, const u32* spos,
    double* dst, i64 d_stride, i64 d_off, const u32* dpos, i64 n, int R)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 i = t % n;
    int r = (int)(t / n);
    double v = src[(i64)r * s_stride + s_off + (spos ? (i64)spos[i] : i)];
    dst[(i64)r * d_stride + d_off + (dpos ? (i64)dpos[i] : i)] += v;
}

// dst[r * d_stride + d_off + dpos(i)] = src[r * s_stride + s_off + spos(i)].
extern "C" __global__ void gather_copy(
    const double* src, i64 s_stride, i64 s_off, const u32* spos,
    double* dst, i64 d_stride, i64 d_off, const u32* dpos, i64 n, int R)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 i = t % n;
    int r = (int)(t / n);
    double v = src[(i64)r * s_stride + s_off + (spos ? (i64)spos[i] : i)];
    dst[(i64)r * d_stride + d_off + (dpos ? (i64)dpos[i] : i)] = v;
}

// dmap[j] = -1 for all j, then dmap[pos[i]] = i.
extern "C" __global__ void fill_i32(int* a, i64 n, int v) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) a[t] = v;
}

extern "C" __global__ void dmap_set(int* dmap, const u32* pos, i64 n) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) dmap[pos[t]] = (int)t;
}

// out[i] = in[pos[i]] for each of R rows (pack for the halo exchange) or
// the reverse (unpack: out[pos[i]] = in[i], add: out[pos[i]] += in[i]).
extern "C" __global__ void pack_rows(const double* in, i64 in_stride, const u32* pos, i64 n, int R, double* out) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 i = t % n;
    int r = (int)(t / n);
    out[t] = in[(i64)r * in_stride + pos[i]];
}

extern "C" __global__ void unpack_rows(const double* in, const u32* pos, i64 n, int R, double* out, i64 out_stride, int mode) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 i = t % n;
    int r = (int)(t / n);
    double* o = out + (i64)r * out_stride + pos[i];
    if (mode == 0) *o = in[t];
    else if (mode == 1) *o += in[t];
    else *o = 0.0;
}

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

// An operand: value (b-th item) at pixel i is (re[i], im[i]) with
// re = ptr, im = ptr + im_off (im_off < 0: real operand, im = 0).
struct Operand {
    u64 ptr;
    i64 im_off;
};

__device__ __forceinline__ void load(const Operand& o, i64 i, double* re, double* im) {
    const double* p = (const double*)o.ptr;
    *re = p[i];
    *im = o.im_off >= 0 ? p[o.im_off + i] : 0.0;
}

// z / sqrt(|z| + 1e-8)
__device__ __forceinline__ void root_normalise(double re, double im, double* ore, double* oim) {
    double s = sqrt(hypot(re, im) + 1e-8);
    *ore = re / s;
    *oim = im / s;
}

// x * conj(y), as num_complex computes x * y.conj().
__device__ __forceinline__ void mul_conj(double xr, double xi, double yr, double yi, double* r, double* i) {
    double cr = yr, ci = -yi;
    *r = xr * cr - xi * ci;
    *i = xr * ci + xi * cr;
}

#define F_MEAN 0
#define F_SQ 1      // var, S2 with c1 == c2: |a|^2 (abs then square)
#define F_ABS 2     // S1 with c1 == c2
#define F_S1X 3     // S1 cross: rn(a) conj(rn(b))
#define F_PROD 4    // a conj(b)

struct ForwardItem {
    int kind;
    int pad;
    Operand a;
    Operand b;
};

// Partial sums of the items over n pixels, by chunks of `chunk` pixels:
// out[(item * n_chunks + c) * 3 + {0, 1, 2}] = (sum re, sum im, sum |f|^2)
// over chunk c, reduced in a fixed tree.
extern "C" __global__ void coef_partial(const ForwardItem* items, i64 n, i64 chunk, int moments, double* out) {
    __shared__ double sre[256], sim[256], ssq[256];
    int item = blockIdx.y;
    i64 c = blockIdx.x;
    i64 n_chunks = gridDim.x;
    ForwardItem it = items[item];
    i64 lo = c * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double are, aim, bre, bim;
    double tre = 0.0, tim = 0.0, tsq = 0.0;
    for (i64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        load(it.a, i, &are, &aim);
        double vre = 0.0, vim = 0.0, sq = 0.0;
        switch (it.kind) {
            case F_MEAN:
                vre = are;
                sq = are * are + aim * aim;
                break;
            case F_SQ: {
                double r = hypot(are, aim);
                vre = r * r;
                sq = vre * vre;
                break;
            }
            case F_ABS:
                vre = hypot(are, aim);
                sq = are * are + aim * aim;
                break;
            case F_S1X: {
                load(it.b, i, &bre, &bim);
                double xr, xi, yr, yi;
                root_normalise(are, aim, &xr, &xi);
                root_normalise(bre, bim, &yr, &yi);
                mul_conj(xr, xi, yr, yi, &vre, &vim);
                sq = vre * vre + vim * vim;
                break;
            }
            default:
                load(it.b, i, &bre, &bim);
                mul_conj(are, aim, bre, bim, &vre, &vim);
                sq = vre * vre + vim * vim;
        }
        tre += vre;
        tim += vim;
        if (moments) tsq += sq;
    }
    sre[threadIdx.x] = tre;
    sim[threadIdx.x] = tim;
    ssq[threadIdx.x] = tsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sre[threadIdx.x] += sre[threadIdx.x + s];
            sim[threadIdx.x] += sim[threadIdx.x + s];
            ssq[threadIdx.x] += ssq[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double* o = out + ((i64)item * n_chunks + c) * 3;
        o[0] = sre[0];
        o[1] = sim[0];
        o[2] = ssq[0];
    }
}

#define R_MEAN 0     // g.re / N                      (real)
#define R_VAR 1      // 2 g.re a.re / N               (real)
#define R_S1SAME 2   // modulus_vjp(a, g.re / N)
#define R_S2SAME 3   // 2 g.re a / N
#define R_S1X_A 4    // root_modulus_vjp(a, g rn(b) / N)
#define R_S1X_B 5    // root_modulus_vjp(b, conj(g) rn(a) / N)
#define R_GA 6       // g b / N
#define R_GB 7       // conj(g) a / N

struct Contrib {
    int kind;
    int pad;
    Operand a;
    Operand b;
    double g_re, g_im, n_total;
};

struct Target {
    u64 ptr;      // re at pixel 0
    i64 im_off;   // < 0: real target (receives the real part)
    u32 start, end;
};

__device__ void root_modulus_vjp(double vr, double vi, double gr, double gi, double* outr, double* outi) {
    double r = hypot(vr, vi);
    // (gi.conj() * v).re
    double cr = gr, ci = -gi;
    double radial = cr * vr - ci * vi;
    double mag = r == 0.0 ? 1.0 : r;
    double s = sqrt(r + 1e-8);
    double c = 0.5 * radial / pow(r + 1e-8, 1.5);
    *outr = gr / s - (c * vr) / mag;
    *outi = gi / s - (c * vi) / mag;
}

// For each (target, pixel): dst (+)= the contributions of the target in
// order (the coefficient order of the plan).
extern "C" __global__ void coef_reverse(const Target* targets, const Contrib* contribs, i64 n) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Target t = targets[blockIdx.y];
    double* d = (double*)t.ptr;
    double dre = d[i];
    double dim = t.im_off >= 0 ? d[t.im_off + i] : 0.0;
    for (u32 q = t.start; q < t.end; q++) {
        Contrib c = contribs[q];
        double are, aim, bre, bim, vr = 0.0, vi = 0.0;
        load(c.a, i, &are, &aim);
        switch (c.kind) {
            case R_MEAN:
                vr = c.g_re / c.n_total;
                break;
            case R_VAR:
                vr = 2.0 * c.g_re * are / c.n_total;
                break;
            case R_S1SAME: {
                double scale = c.g_re / c.n_total;
                double mag = hypot(are, aim);
                double dd = mag == 0.0 ? 1.0 : mag;
                vr = (scale * are) / dd;
                vi = (scale * aim) / dd;
                break;
            }
            case R_S2SAME: {
                double s = 2.0 * c.g_re;
                vr = (s * are) / c.n_total;
                vi = (s * aim) / c.n_total;
                break;
            }
            case R_S1X_A: {
                load(c.b, i, &bre, &bim);
                double yr, yi, pr, pi;
                root_normalise(bre, bim, &yr, &yi);
                // g * rn(b)
                pr = c.g_re * yr - c.g_im * yi;
                pi = c.g_re * yi + c.g_im * yr;
                root_modulus_vjp(are, aim, pr / c.n_total, pi / c.n_total, &vr, &vi);
                break;
            }
            case R_S1X_B: {
                load(c.b, i, &bre, &bim);
                double yr, yi, pr, pi;
                root_normalise(are, aim, &yr, &yi);
                // conj(g) * rn(a)
                double gr = c.g_re, gi = -c.g_im;
                pr = gr * yr - gi * yi;
                pi = gr * yi + gi * yr;
                root_modulus_vjp(bre, bim, pr / c.n_total, pi / c.n_total, &vr, &vi);
                break;
            }
            case R_GA: {
                load(c.b, i, &bre, &bim);
                double pr = c.g_re * bre - c.g_im * bim;
                double pi = c.g_re * bim + c.g_im * bre;
                vr = pr / c.n_total;
                vi = pi / c.n_total;
                break;
            }
            default: {
                double gr = c.g_re, gi = -c.g_im;
                double pr = gr * are - gi * aim;
                double pi = gr * aim + gi * are;
                vr = pr / c.n_total;
                vi = pi / c.n_total;
            }
        }
        dre += vr;
        dim += vi;
    }
    d[i] = dre;
    if (t.im_off >= 0) d[t.im_off + i] = dim;
}
