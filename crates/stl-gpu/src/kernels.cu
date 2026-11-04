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

// Colatitudes of the rings 0..4 nside - 1 (ring_info's theta), so that the
// interpolation reads them instead of evaluating acos / atan2 per tap.
extern "C" __global__ void ring_thetas(i64 nside, double* out) {
    i64 r = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < 1 || r >= 4 * nside) return;
    i64 sp, nr;
    bool shifted;
    ring_info(r, nside, &sp, &nr, &out[r], &shifted);
}

// ring_info with the colatitude read from the table.
__device__ __forceinline__ void ring_info_t(i64 ring, i64 nside, const double* thetas, i64* startpix_o, i64* ringpix_o, double* theta_o, bool* shifted_o) {
    i64 ncap = 2 * nside * (nside - 1);
    i64 npix = 12 * nside * nside;
    i64 northring = ring > 2 * nside ? 4 * nside - ring : ring;
    i64 ringpix, startpix;
    bool shifted;
    if (northring < nside) {
        ringpix = 4 * northring;
        shifted = true;
        startpix = 2 * northring * (northring - 1);
    } else {
        ringpix = 4 * nside;
        shifted = ((northring - nside) & 1) == 0;
        startpix = ncap + (northring - nside) * 4 * nside;
    }
    if (northring != ring) startpix = npix - startpix - ringpix;
    *startpix_o = startpix; *ringpix_o = ringpix; *theta_o = thetas[ring]; *shifted_o = shifted;
}

__device__ void ring_pair(double phi, i64 ring, i64 nside, const double* thetas, i64* pa, i64* pb, double* wa, double* wb, double* theta_ring) {
    i64 sp, nr;
    bool shifted;
    ring_info_t(ring, nside, thetas, &sp, &nr, theta_ring, &shifted);
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

__device__ void get_interpol_ring(double theta, double phi, i64 nside, const double* thetas, i64 pix[4], double wgt[4]) {
    i64 npix = 12 * nside * nside;
    double z = cos(theta);
    i64 ir1 = ring_above(z, nside);
    i64 ir2 = ir1 + 1;
    i64 p[4] = {0, 0, 0, 0};
    double w[4] = {0.0, 0.0, 0.0, 0.0};
    i64 p0, p1, p2, p3;
    double w0, w1, w2, w3, theta1, theta2;
    ring_pair(phi, ir1 > 1 ? ir1 : 1, nside, thetas, &p0, &p1, &w0, &w1, &theta1);
    if (ir1 > 0) { p[0] = p0; p[1] = p1; w[0] = w0; w[1] = w1; }
    ring_pair(phi, ir2 < 4 * nside - 1 ? ir2 : 4 * nside - 1, nside, thetas, &p2, &p3, &w2, &w3, &theta2);
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

__device__ void interp_weights_nested(double v0, double v1, double v2, int order, const double* thetas, i64 pix[4], double w[4]) {
    double n = fmax(norm3(v0, v1, v2), 1e-12);
    double x = v0 / n, y = v1 / n, z = v2 / n;
    double theta = acos(fmin(fmax(z, -1.0), 1.0));
    double phi = py_remainder(atan2(y, x), 2.0 * PI_D);
    double lon_deg = rad2deg(phi);
    double lat_deg = 90.0 - rad2deg(theta);
    double theta2 = deg2rad(90.0 - lat_deg);
    double phi2 = py_remainder(deg2rad(lon_deg), 2.0 * PI_D);
    i64 rp[4];
    get_interpol_ring(theta2, phi2, 1LL << order, thetas, rp, w);
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

// rotation_matrices(theta, phi, G, gauge)[g] at the centre of `pix`,
// rounded through degrees as in ConvGeometry::rows.
__device__ void stencil_rotation(i64 pix, i64 nside, int order, int G, int g, int gauge, double r[3][3]) {
    double th, ph;
    pix2ang_nest(pix, nside, order, &th, &ph);
    double theta = deg2rad(90.0 - rad2deg(PI_D / 2.0 - th));
    double phi = deg2rad(rad2deg(ph));
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
    double rg[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double eye = i == j ? 1.0 : 0.0;
            rg[i][j] = eye * ca + skew[i][j] * sa + nv[i] * nv[j] * (1.0 - ca);
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r[i][j] = rg[i][0] * base[0][j] + rg[i][1] * base[1][j] + rg[i][2] * base[2][j];
}

__device__ __forceinline__ void rotate_interp(double r[3][3], const double* v, int order, const double* thetas, i64 ix[4], double wx[4]) {
    interp_weights_nested(r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
                          r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
                          r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2], order, thetas, ix, wx);
}

// Tap `v` of the rotated stencil `r` interpolated and bound on the grid
// (ConvGeometry::rows): its cells `ix`, their grid positions `found` (-1:
// absent) and weights `w` renormalised over the cells found; a tap with no
// cell found takes the central tap (`ref_idx`, `ref_w`).
__device__ void bind_tap(double r[3][3], const double* v, int order, const double* thetas,
    const i64* ref_idx, const double* ref_w,
    const i64* grid, i64 n_grid, i64 own_start, i64 own_end, i64 own_pos0, double eps,
    i64 ix[4], i64 found[4], double w[4])
{
    double wx[4];
    rotate_interp(r, v, order, thetas, ix, wx);
    bool any = false;
    for (int n = 0; n < 4; n++) {
        found[n] = grid_find(grid, n_grid, own_start, own_end, own_pos0, ix[n]);
        if (eps > 0.0 && wx[n] <= eps) found[n] = -1;
        any = any || found[n] >= 0;
    }
    if (!any) {
        for (int n = 0; n < 4; n++) {
            ix[n] = ref_idx[n];
            wx[n] = ref_w[n];
            found[n] = grid_find(grid, n_grid, own_start, own_end, own_pos0, ix[n]);
            if (eps > 0.0 && wx[n] <= eps) found[n] = -1;
        }
    }
    for (int n = 0; n < 4; n++) w[n] = wx[n] * (found[n] >= 0 ? 1.0 : 0.0);
    double colsum = w[0] + w[1] + w[2] + w[3];
    if (colsum == 0.0) {
        w[0] = found[0] >= 0 ? 1.0 : 0.0;
        colsum = w[0] + w[1] + w[2] + w[3];
    }
    for (int n = 0; n < 4; n++) w[n] = w[n] / fmax(colsum, 1e-12);
}

// Bound stencils of the output rows `rows[K]` reading `grid`, one thread
// per (row, gauge); with eps > 0 the cells of weight <= eps are not read
// (ConvGeometry::rows_eps).  Entry ((g * 4 + n) * P + p) * K + k (rows fastest, so
// that consecutive threads read consecutive entries).  `complete[k * G + g]`.
extern "C" __global__ void geom_build(
    const i64* grid, i64 n_grid, i64 own_start, i64 own_end, i64 own_pos0,
    const i64* rows, i64 K, i64 nside, int order, int G, int P, int gauge,
    const double* stencil, int p_ref, const double* thetas, double eps,
    u32* pos, double* wout, unsigned char* complete)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= K * G) return;
    i64 k = t / G;
    int g = (int)(t - k * G);
    double r[3][3];
    stencil_rotation(rows[k], nside, order, G, g, gauge, r);
    i64 ref_idx[4];
    double ref_w[4];
    rotate_interp(r, stencil + 3 * p_ref, order, thetas, ref_idx, ref_w);
    bool comp = true;
    for (int p = 0; p < P; p++) {
        i64 ix[4], found[4];
        double w[4];
        bind_tap(r, stencil + 3 * p, order, thetas, ref_idx, ref_w, grid, n_grid, own_start, own_end, own_pos0, eps, ix, found, w);
        for (int n = 0; n < 4; n++) {
            i64 a = (((i64)g * 4 + n) * P + p) * K + k;
            wout[a] = w[n];
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

// Order of an entry in the scatter of AdjointOps.convolution_adjoint:
// gauge, neighbour, row, tap, i.e. ((g * 4 + n) * K + k) * P + p, and back.
__device__ __forceinline__ u32 entry_key(u32 e, u64 K, u64 P) {
    u64 gn = e / (P * K), rem = e - gn * P * K;
    u64 p = rem / K, k = rem - p * K;
    return (u32)((gn * K + k) * P + p);
}

__device__ __forceinline__ u32 key_entry(u32 key, u64 K, u64 P) {
    u64 gn = key / (K * P), rem = key - gn * K * P;
    u64 k = rem / P, p = rem - k * P;
    return (u32)((gn * P + p) * K + k);
}

// One block per segment: entries sorted by their scatter order, bitonic
// sort in shared memory (segments up to SEG_CAP entries), in global memory
// above (cells read by many rows, e.g. on the coarse levels of a disc).
#define SEG_CAP 2048
extern "C" __global__ void seg_sort(const u32* offsets, i64 n_seg, u32* idx, i64 K, int P) {
    __shared__ u32 buf[SEG_CAP];
    for (i64 s = blockIdx.x; s < n_seg; s += gridDim.x) {
        u32 a = offsets[s], b = offsets[s + 1];
        u32 len = b - a;
        if (len <= 1) continue;
        if (len > SEG_CAP) {
            // Bitonic sort in global memory, in the form whose compares are
            // all ascending (mirror pairs i ^ (size - 1), then i ^ stride):
            // the virtual padding up to m (+inf) never moves.  The keys are
            // unique, so the order is that of any sort.
            u32* h = idx + a;
            for (u32 i = threadIdx.x; i < len; i += blockDim.x) h[i] = entry_key(h[i], K, P);
            __syncthreads();
            u32 m = 1;
            while (m < len) m <<= 1;
            for (u32 size = 2; size <= m; size <<= 1) {
                for (u32 stride = size >> 1; stride > 0; stride >>= 1) {
                    for (u32 i = threadIdx.x; i < m; i += blockDim.x) {
                        u32 j = stride == (size >> 1) ? i ^ (size - 1) : i ^ stride;
                        if (j > i && j < len) {
                            u32 x = h[i], y = h[j];
                            if (x > y) { h[i] = y; h[j] = x; }
                        }
                    }
                    __syncthreads();
                }
            }
            for (u32 i = threadIdx.x; i < len; i += blockDim.x) h[i] = key_entry(h[i], K, P);
            __syncthreads();
            continue;
        }
        u32 m = 1;
        while (m < len) m <<= 1;
        for (u32 i = threadIdx.x; i < m; i += blockDim.x) buf[i] = i < len ? entry_key(idx[a + i], K, P) : 0xFFFFFFFFu;
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
        for (u32 i = threadIdx.x; i < len; i += blockDim.x) idx[a + i] = key_entry(buf[i], K, P);
        __syncthreads();
    }
}

// After the sort: each transposed entry as (row k << 11 | (g * 4 + n) << 6
// | tap) and its weight, contiguous in the scatter order.
extern "C" __global__ void geom_pack(const u32* idx, u64 n_entries, const double* w, i64 K, int P, u32* kgp, double* wt) {
    for (u64 q = (u64)blockIdx.x * blockDim.x + threadIdx.x; q < n_entries; q += (u64)gridDim.x * blockDim.x) {
        u32 e = idx[q];
        u32 kp = e / (u32)K;
        u32 k = e - kp * (u32)K;
        u32 p = kp % (u32)P;
        u32 gn = kp / (u32)P;
        kgp[q] = (k << 11) | (gn << 6) | p;
        wt[q] = w[e];
    }
}

// The weights by cell from the weights by row and the packed entries.
extern "C" __global__ void geom_wt(const u32* kgp, u64 n_entries, const double* w, i64 K, int P, double* wt) {
    for (u64 q = (u64)blockIdx.x * blockDim.x + threadIdx.x; q < n_entries; q += (u64)gridDim.x * blockDim.x) {
        u32 c = kgp[q];
        u64 k = c >> 11, gn = (c >> 6) & 31u, p = c & 63u;
        wt[q] = w[(gn * (u64)P + p) * (u64)K + k];
    }
}

// ---------------------------------------------------------------------------
// Convolution forward / adjoint (AdjointOps, C_in = 1, C_out = 2)
// ---------------------------------------------------------------------------

// Input rows handled by one thread (the geometry is read once for them).
#define RC 4

// out[((r * G + g) * 2 + o) * out_stride + opos(i)] = sum_p (sum_n x[r, pos] w) ker[g, o, p]
// for the geometry rows k = row0 + i (i < n_sel; opos null: i).  The grid
// cell pos is read at xmap[pos] when xmap is given (-1: a zero input).  One
// thread per (i, g, group of RC input rows r).
extern "C" __global__ void conv_fwd(
    const u32* pos, const double* w, int G, i64 K, int P, const double* ker,
    const double* x, const int* xmap, i64 x_stride, int R,
    i64 row0, i64 n_sel, const u32* opos, i64 out_stride, double* out)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    int n_rc = (R + RC - 1) / RC;
    if (t >= n_sel * G * (i64)n_rc) return;
    i64 i = t % n_sel;
    i64 rest = t / n_sel;
    int g = (int)(rest % G);
    int r0 = (int)(rest / G) * RC;
    int nr = R - r0 < RC ? R - r0 : RC;
    i64 k = row0 + i;
    const double* k0 = ker + ((i64)g * 2 + 0) * P;
    const double* k1 = ker + ((i64)g * 2 + 1) * P;
    double s0[RC], s1[RC];
#pragma unroll
    for (int j = 0; j < RC; j++) { s0[j] = 0.0; s1[j] = 0.0; }
    for (int p = 0; p < P; p++) {
        double acc[RC];
#pragma unroll
        for (int j = 0; j < RC; j++) acc[j] = 0.0;
        for (int n = 0; n < 4; n++) {
            i64 a = (((i64)g * 4 + n) * P + p) * K + k;
            i64 ps = xmap ? (i64)xmap[pos[a]] : (i64)pos[a];
            double wv = w[a];
#pragma unroll
            for (int j = 0; j < RC; j++)
                if (j < nr) acc[j] += (ps >= 0 ? x[(i64)(r0 + j) * x_stride + ps] : 0.0) * wv;
        }
        double kp0 = k0[p], kp1 = k1[p];
#pragma unroll
        for (int j = 0; j < RC; j++) {
            s0[j] += acc[j] * kp0;
            s1[j] += acc[j] * kp1;
        }
    }
    i64 o = opos ? (i64)opos[i] : i;
#pragma unroll
    for (int j = 0; j < RC; j++) {
        if (j < nr) {
            out[(((i64)(r0 + j) * G + g) * 2 + 0) * out_stride + o] = s0[j];
            out[(((i64)(r0 + j) * G + g) * 2 + 1) * out_stride + o] = s1[j];
        }
    }
}

// Adjoint, gathered per destination: segment s holds the entries reading
// grid position dest[s], in increasing order, i.e. gauge, bilinear
// neighbour, row, tap (the scatter order of
// AdjointOps.convolution_adjoint).  Only the rows k in [row0, row0 +
// n_rows) take part; gin `[R, G, 2, gin_stride]` at index k - row0.  The
// result goes to out[r * out_stride + dmap(dest[s])] (dmap null:
// dest[s]; -1: dropped), added when `accumulate`.
//
// One warp per (segment, group of RC input rows): lane l adds the terms l,
// l + 32, l + 64, ... of the segment (coalesced reads), then the 32 lane
// sums are combined by a fixed shuffle tree (deterministic; the CPU adds
// the terms one after the other).
extern "C" __global__ void conv_adj(
    const u32* dest, const u32* offsets, i64 n_dest, const u32* kgp, const double* wt,
    int G, i64 K, int P, const double* ker,
    const double* gin, i64 gin_stride, int R, i64 row0, i64 n_rows,
    const int* dmap, double* out, i64 out_stride, int accumulate)
{
    i64 warp = ((i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int n_rc = (R + RC - 1) / RC;
    if (warp >= n_dest * (i64)n_rc) return;
    i64 s = warp % n_dest;
    int r0 = (int)(warp / n_dest) * RC;
    int nr = R - r0 < RC ? R - r0 : RC;
    u32 a = offsets[s], b = offsets[s + 1];
    i64 dst = dmap ? (i64)dmap[dest[s]] : (i64)dest[s];
    if (dst < 0) return;
    double acc[RC];
#pragma unroll
    for (int j = 0; j < RC; j++) acc[j] = 0.0;
    bool any = false;
    for (u32 q = a + lane; q < b; q += 32) {
        u32 c = kgp[q];
        i64 k = (i64)(c >> 11);
        int g = (int)((c >> 8) & 7u);
        int p = (int)(c & 63u);
        i64 ri = k - row0;
        if (ri < 0 || ri >= n_rows) continue;
        double wv = wt[q];
        double kg0 = ker[((i64)g * 2 + 0) * P + p], kg1 = ker[((i64)g * 2 + 1) * P + p];
#pragma unroll
        for (int j = 0; j < RC; j++) {
            if (j < nr) {
                const double* gr = gin + (((i64)(r0 + j) * G + g) * 2) * gin_stride + ri;
                double taps = 0.0;
                taps += gr[0] * kg0;
                taps += gr[gin_stride] * kg1;
                acc[j] += taps * wv;
            }
        }
        any = true;
    }
    any = __any_sync(0xffffffffu, any);
#pragma unroll
    for (int j = 0; j < RC; j++)
        for (int off = 16; off > 0; off >>= 1) acc[j] += __shfl_down_sync(0xffffffffu, acc[j], off);
    if (!any || lane != 0) return;
#pragma unroll
    for (int j = 0; j < RC; j++) {
        if (j < nr) {
            double* o = out + (i64)(r0 + j) * out_stride + dst;
            if (accumulate) *o += acc[j]; else *o = acc[j];
        }
    }
}

// ---------------------------------------------------------------------------
// DIA convolution (port of stl_core::dia): the cells of a row are found
// from its face coordinates and a fixed set of D offsets (diagonals); the
// kernel and the bound bilinear weights are folded into per-row
// coefficients, dense[((g * 2 + o) * D + d) * K + k] (dia_build), then
// kept for the diagonals a row uses only: mask[3 k + w] bit b for diagonal
// 32 w + b, coef[((s * K + k) * G + g) * 2 + o] for its s-th used
// diagonal (dia_slots, dia_pack); the adjoint gathers the rows at the
// opposite offsets, so there is no transposition.
// ---------------------------------------------------------------------------

// HEALPix C++ nb_facearray (face beyond side / corner 4 + xdir + 3 ydir,
// -1 at the vertices of three faces) and nb_swaparray (by face / 4).
__constant__ signed char NB_FACE[9][12] = {
    {8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9},
    {5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8},
    {-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1},
    {4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    {1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4},
    {-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1},
    {3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7},
    {2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3}};
__constant__ unsigned char NB_SWAP[9][3] = {{0, 0, 3}, {0, 0, 6}, {0, 0, 0}, {0, 0, 5}, {0, 0, 0}, {5, 0, 0}, {0, 0, 0}, {6, 0, 0}, {3, 0, 0}};

__device__ __forceinline__ void nest2xyf(i64 pix, int order, i64* x, i64* y, i64* f) {
    *f = pix >> (2 * order);
    i64 local = pix & ((1LL << (2 * order)) - 1);
    *x = compress_bits(local);
    *y = compress_bits(local >> 1);
}

// Pixel at the extended coordinates (ex, ey) of `face` (-nside <= ex, ey <
// 2 nside), -1 in the quadrant missing at a vertex of three faces.
__device__ i64 ext_pix(i64 face, i64 ex, i64 ey, i64 nside, int order) {
    int nb = 4;
    if (ex < 0) { ex += nside; nb -= 1; } else if (ex >= nside) { ex -= nside; nb += 1; }
    if (ey < 0) { ey += nside; nb -= 3; } else if (ey >= nside) { ey -= nside; nb += 3; }
    if (nb == 4) return xyf2nest(ex, ey, face, order);
    int f = NB_FACE[nb][face];
    if (f < 0) return -1;
    int bits = NB_SWAP[nb][face >> 2];
    if (bits & 1) ex = nside - ex - 1;
    if (bits & 2) ey = nside - ey - 1;
    if (bits & 4) { i64 t = ex; ex = ey; ey = t; }
    return xyf2nest(ex, ey, (i64)f, order);
}

// Diagonal of `cell` seen from (x, y) on face f (extended frame), -1 if the
// cell is off the diagonals.  `dindex[(dy + W) (2 W + 1) + dx + W]`.
__device__ int diag_of(i64 x, i64 y, i64 f, i64 cell, i64 nside, int order, const int* dindex, int W) {
    i64 xq, yq, fq;
    nest2xyf(cell, order, &xq, &yq, &fq);
    i64 dx, dy;
    if (fq == f) {
        dx = xq - x;
        dy = yq - y;
    } else {
        int nb = 0;
        while (nb < 9 && (nb == 4 || NB_FACE[nb][f] != fq)) nb++;
        if (nb == 9) return -1;
        int bits = NB_SWAP[nb][f >> 2];
        i64 a = xq, b = yq;
        if (bits & 4) { i64 t = a; a = b; b = t; }
        if (bits & 2) b = nside - b - 1;
        if (bits & 1) a = nside - a - 1;
        dx = a + (nb % 3 - 1) * nside - x;
        dy = b + (nb / 3 - 1) * nside - y;
    }
    if (dx < -W || dx > W || dy < -W || dy > W) return -1;
    return dindex[(dy + W) * (2 * W + 1) + dx + W];
}

// flag[j] = 1 for the grid cells j on a diagonal of a row (one thread per
// (row, diagonal)): the cells the adjoint produces.
extern "C" __global__ void dia_reach(
    const i64* grid, i64 n_grid, i64 own_start, i64 own_end, i64 own_pos0,
    const i64* rows, i64 K, i64 nside, int order, const int* doff, int D, unsigned char* flag)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= K * D) return;
    int d = (int)(t / K);
    i64 k = t - (i64)d * K;
    i64 x, y, f;
    nest2xyf(rows[k], order, &x, &y, &f);
    i64 q = ext_pix(f, x + doff[2 * d], y + doff[2 * d + 1], nside, order);
    if (q < 0) return;
    i64 j = grid_find(grid, n_grid, own_start, own_end, own_pos0, q);
    if (j >= 0) flag[j] = 1;
}

// Compaction of the flagged grid cells, in increasing order, by chunks of
// 1024 positions (blocks of 256 threads, 4 positions each): flag_count
// counts a chunk, flag_fill writes it from offsets[chunk] (the exclusive
// prefix of the counts).
extern "C" __global__ void flag_count(const unsigned char* flag, i64 n, u32* counts) {
    __shared__ u32 sh[256];
    i64 base = (i64)blockIdx.x * 1024 + threadIdx.x * 4;
    u32 c = 0;
    for (int i = 0; i < 4; i++)
        if (base + i < n && flag[base + i]) c++;
    sh[threadIdx.x] = c;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[blockIdx.x] = sh[0];
}

extern "C" __global__ void flag_fill(const unsigned char* flag, i64 n, const u32* offsets, u32* dest) {
    __shared__ u32 sh[256];
    int t = threadIdx.x;
    i64 base = (i64)blockIdx.x * 1024 + t * 4;
    u32 c = 0;
    for (int i = 0; i < 4; i++)
        if (base + i < n && flag[base + i]) c++;
    sh[t] = c;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        u32 v = t >= off ? sh[t - off] : 0u;
        __syncthreads();
        sh[t] += v;
        __syncthreads();
    }
    u32 o = offsets[blockIdx.x] + sh[t] - c;
    for (int i = 0; i < 4; i++)
        if (base + i < n && flag[base + i]) dest[o++] = (u32)(base + i);
}

// Dense coefficients of the output rows `rows[K]` reading `grid` (coef zeroed),
// one thread per (gauge, row): the bound stencil of geom_build, each
// nonzero weight w of a tap p folded as ker[g, o, p] * w into the diagonal
// of its cell, taps then bilinear neighbours in order (DiaGeometry::
// from_conv).  Without `pmask`, into the dense coef[((g 2 + o) D + d) K +
// k], the diagonals that receive a term or-ed into the row's mask
// (mask[3 k + w], zeroed); with the masks of a previous build `pmask`,
// directly into the packed coef[((s K + k) G + g) 2 + o] (the same values:
// each coefficient adds its terms in the same order).  errors[0] counts the
// nonzero reads off the diagonals (or off the row's mask).
extern "C" __global__ void dia_build(
    const i64* grid, i64 n_grid, i64 own_start, i64 own_end, i64 own_pos0,
    const i64* rows, i64 K, i64 nside, int order, int G, int P, int gauge,
    const double* stencil, int p_ref, const double* thetas,
    const double* ker, const int* dindex, int W, int D, double* coef, u32* mask, const u32* pmask, u32* errors)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= K * G) return;
    int g = (int)(t / K);
    i64 k = t - (i64)g * K;
    i64 row = rows[k];
    double r[3][3];
    stencil_rotation(row, nside, order, G, g, gauge, r);
    i64 ref_idx[4];
    double ref_w[4];
    rotate_interp(r, stencil + 3 * p_ref, order, thetas, ref_idx, ref_w);
    i64 x, y, f;
    nest2xyf(row, order, &x, &y, &f);
    const double* k0 = ker + ((i64)g * 2 + 0) * P;
    const double* k1 = ker + ((i64)g * 2 + 1) * P;
    double* c0 = coef + ((i64)g * 2 + 0) * D * K + k;
    double* c1 = coef + ((i64)g * 2 + 1) * D * K + k;
    u32 used[3] = {0u, 0u, 0u};
    u32 pm[3] = {0u, 0u, 0u};
    if (pmask)
        for (int w = 0; w < 3; w++) pm[w] = pmask[3 * k + w];
    for (int p = 0; p < P; p++) {
        i64 ix[4], found[4];
        double w[4];
        bind_tap(r, stencil + 3 * p, order, thetas, ref_idx, ref_w, grid, n_grid, own_start, own_end, own_pos0, 0.0, ix, found, w);
        for (int n = 0; n < 4; n++) {
            if (w[n] == 0.0) continue;
            int d = diag_of(x, y, f, ix[n], nside, order, dindex, W);
            if (d < 0) {
                atomicAdd(&errors[0], 1u);
                continue;
            }
            if (pmask) {
                u32 mw = pm[d >> 5];
                if (!((mw >> (d & 31)) & 1u)) {
                    atomicAdd(&errors[0], 1u);
                    continue;
                }
                i64 slot = __popc(mw & ((1u << (d & 31)) - 1u));
                if (d >= 32) slot += __popc(pm[0]);
                if (d >= 64) slot += __popc(pm[1]);
                double* cp = coef + ((slot * K + k) * G + g) * 2;
                cp[0] += k0[p] * w[n];
                cp[1] += k1[p] * w[n];
            } else {
                c0[(i64)d * K] += k0[p] * w[n];
                c1[(i64)d * K] += k1[p] * w[n];
                used[d >> 5] |= 1u << (d & 31);
            }
        }
    }
    if (!pmask)
        for (int w = 0; w < 3; w++)
            if (used[w]) atomicOr(&mask[3 * k + w], used[w]);
}

// *slots = the most diagonals a row uses (popcount of its mask).
extern "C" __global__ void dia_slots(const u32* mask, i64 K, u32* slots) {
    i64 k = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    atomicMax(slots, (u32)(__popc(mask[3 * k]) + __popc(mask[3 * k + 1]) + __popc(mask[3 * k + 2])));
}

// coef[((s K + k) G + g) 2 + o] = dense[((g 2 + o) D + d) K + k] for the
// s-th diagonal d used by row k (increasing d; the threads of a warp walk
// the diagonals together, so that the dense reads are coalesced).
extern "C" __global__ void dia_pack(const double* dense, i64 K, int G, int D, const u32* mask, double* coef) {
    i64 k = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    u32 m[3] = {mask[3 * k], mask[3 * k + 1], mask[3 * k + 2]};
    i64 s = 0;
    for (int d = 0; d < D; d++) {
        if (!((m[d >> 5] >> (d & 31)) & 1u)) continue;
        double* o = coef + (s * K + k) * G * 2;
        for (int go = 0; go < G * 2; go++) o[go] = dense[((i64)go * D + d) * K + k];
        s++;
    }
}

// Gauges per thread of dia_fwd.
#define GB 4

// out[((r * G + g) * 2 + o) * out_stride + opos(i)] = sum_d coef[g, o, d, k]
// x[r, cell(k, d)] for the rows k = row0 + i (i < n_sel), over the
// diagonals the row uses in increasing order (the others are zero for
// every gauge).  The grid cell is read at xmap[cell] when xmap is given
// (-1: skipped, as a cell off the grid).  One thread per (i, GB gauges, RC
// input rows): the cell of a diagonal and its RC inputs serve every gauge.
extern "C" __global__ void dia_fwd(
    const double* coef, const u32* mask, int G, i64 K, const int* doff,
    const i64* rows, i64 nside, int order,
    const i64* grid, i64 n_grid, i64 own_start, i64 own_end, i64 own_pos0,
    const double* x, const int* xmap, i64 x_stride, int R,
    i64 row0, i64 n_sel, const u32* opos, i64 out_stride, double* out)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    int n_rc = (R + RC - 1) / RC;
    int n_gb = (G + GB - 1) / GB;
    if (t >= n_sel * n_gb * (i64)n_rc) return;
    i64 i = t % n_sel;
    i64 rest = t / n_sel;
    int g0 = (int)(rest % n_gb) * GB;
    int r0 = (int)(rest / n_gb) * RC;
    int ng = G - g0 < GB ? G - g0 : GB;
    int nr = R - r0 < RC ? R - r0 : RC;
    i64 k = row0 + i;
    i64 xr, yr, fr;
    nest2xyf(rows[k], order, &xr, &yr, &fr);
    double s[GB][2][RC];
#pragma unroll
    for (int gb = 0; gb < GB; gb++)
#pragma unroll
        for (int j = 0; j < RC; j++) { s[gb][0][j] = 0.0; s[gb][1][j] = 0.0; }
    i64 slot = 0;
    for (int w = 0; w < 3; w++) {
        u32 m = mask[3 * k + w];
        while (m) {
            int d = 32 * w + __ffs(m) - 1;
            m &= m - 1;
            const double* cg = coef + ((slot * K + k) * G + g0) * 2;
            slot++;
            i64 q = ext_pix(fr, xr + doff[2 * d], yr + doff[2 * d + 1], nside, order);
            if (q < 0) continue;
            i64 ps = grid_find(grid, n_grid, own_start, own_end, own_pos0, q);
            if (ps >= 0 && xmap) ps = xmap[ps];
            if (ps < 0) continue;
            double xv[RC];
#pragma unroll
            for (int j = 0; j < RC; j++) xv[j] = j < nr ? x[(i64)(r0 + j) * x_stride + ps] : 0.0;
#pragma unroll
            for (int gb = 0; gb < GB; gb++) {
                if (gb < ng) {
                    double c0 = cg[2 * gb], c1 = cg[2 * gb + 1];
#pragma unroll
                    for (int j = 0; j < RC; j++) {
                        s[gb][0][j] += c0 * xv[j];
                        s[gb][1][j] += c1 * xv[j];
                    }
                }
            }
        }
    }
    i64 o = opos ? (i64)opos[i] : i;
#pragma unroll
    for (int gb = 0; gb < GB; gb++)
#pragma unroll
        for (int j = 0; j < RC; j++)
            if (gb < ng && j < nr) {
                out[(((i64)(r0 + j) * G + g0 + gb) * 2 + 0) * out_stride + o] = s[gb][0][j];
                out[(((i64)(r0 + j) * G + g0 + gb) * 2 + 1) * out_stride + o] = s[gb][1][j];
            }
}

// Row of the NESTED cell `id` among the sorted rows: the block's own rows
// [blk_start, blk_start + blk_len) sit at blk_f0 + (id - blk_start).
__device__ __forceinline__ i64 row_find(const i64* rows, i64 K, i64 blk_start, i64 blk_len, i64 blk_f0, i64 id) {
    if (id >= blk_start && id < blk_start + blk_len) return blk_f0 + (id - blk_start);
    i64 lo = 0, hi = K;
    while (lo < hi) {
        i64 mid = (lo + hi) >> 1;
        if (rows[mid] < id) lo = mid + 1; else hi = mid;
    }
    return (lo < K && rows[lo] == id) ? lo : -1;
}

// gt[i * (R G 2) + (r G + g) 2 + o] = gin[((r G + g) 2 + o) gin_stride + i]
// for the rows i < n_rows: the cotangents of a row together (dia_adj).
extern "C" __global__ void dia_gin_t(const double* gin, i64 gin_stride, int RG2, i64 n_rows, double* gt) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_rows * (i64)RG2) return;
    i64 i = t / RG2;
    int c = (int)(t - i * RG2);
    gt[t] = gin[(i64)c * gin_stride + i];
}

// Adjoint of dia_fwd for the rows k in [row0, row0 + n_rows) (gt from
// dia_gin_t: `[n_rows, R, G, 2]` at k - row0), gathered per grid cell
// dest[s] over the rows at the offsets v of the window [-W, W]^2 in the
// cell's extended frame, each through the diagonal on which it reads the
// cell (diagonals unused by the row skipped).  The result goes to out[r *
// out_stride + dmap(dest[s])] (dmap null: dest[s]; -1: dropped), added
// when `accumulate`; cells without a row in range are left alone.
//
// `lpc` lanes (a power of two <= 32) per (cell, RC input rows): lane l
// takes the offsets l, l + lpc, ... of the window (vy then vx increasing),
// each row summed over gauges and channels then added to the lane's sum;
// the lane sums are combined by a fixed shuffle tree (deterministic; with
// lpc = 1 the order of DiaGeometry::adjoint on the CPU).
extern "C" __global__ void dia_adj(
    const u32* dest, i64 n_dest, const i64* grid,
    const double* coef, const u32* mask, int G, i64 K, const int* dindex, int W,
    const i64* rows, i64 blk_start, i64 blk_len, i64 blk_f0, i64 nside, int order,
    const double* gt, int R, i64 row0, i64 n_rows,
    const int* dmap, double* out, i64 out_stride, int accumulate, int lpc)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int sub = lane & (lpc - 1);
    i64 unit = t / lpc;
    int n_rc = (R + RC - 1) / RC;
    bool valid = unit < n_dest * (i64)n_rc;
    i64 s = valid ? unit % n_dest : 0;
    int r0 = valid ? (int)(unit / n_dest) * RC : 0;
    int nr = R - r0 < RC ? R - r0 : RC;
    u32 j = valid ? dest[s] : 0;
    i64 dst = !valid ? -1 : (dmap ? (i64)dmap[j] : (i64)j);
    valid = valid && dst >= 0;
    int side = 2 * W + 1;
    int RG2 = R * G * 2;
    double acc[RC];
#pragma unroll
    for (int jj = 0; jj < RC; jj++) acc[jj] = 0.0;
    bool any = false;
    if (valid) {
        i64 q = grid[j];
        i64 xq, yq, fq;
        nest2xyf(q, order, &xq, &yq, &fq);
        for (int vi = sub; vi < side * side; vi += lpc) {
            int vy = vi / side - W, vx = vi - (vi / side) * side - W;
            i64 ex = xq + vx, ey = yq + vy;
            bool inside = ex >= 0 && ex < nside && ey >= 0 && ey < nside;
            int d = -1;
            if (inside) {
                // Same face: the row reads the cell at the opposite offset.
                d = dindex[(-vy + W) * side - vx + W];
                if (d < 0) continue;
            }
            i64 rid = ext_pix(fq, ex, ey, nside, order);
            if (rid < 0) continue;
            i64 k = row_find(rows, K, blk_start, blk_len, blk_f0, rid);
            i64 ri = k - row0;
            if (k < 0 || ri < 0 || ri >= n_rows) continue;
            if (!inside) {
                i64 xr, yr, fr;
                nest2xyf(rid, order, &xr, &yr, &fr);
                d = diag_of(xr, yr, fr, q, nside, order, dindex, W);
                if (d < 0) continue;
            }
            // Diagonals the row does not use are zero for every gauge.
            const u32* mk = mask + 3 * k;
            u32 mw = mk[d >> 5];
            if (!((mw >> (d & 31)) & 1u)) continue;
            i64 slot = __popc(mw & ((1u << (d & 31)) - 1u));
            if (d >= 32) slot += __popc(mk[0]);
            if (d >= 64) slot += __popc(mk[1]);
            any = true;
            // The row's coefficients and cotangents are contiguous; per
            // input row the gauges and channels in order.
            const double* cr = coef + (slot * K + k) * G * 2;
            const double* gr = gt + ri * RG2 + (i64)r0 * G * 2;
            double sj[RC];
#pragma unroll
            for (int jj = 0; jj < RC; jj++) sj[jj] = 0.0;
            for (int go = 0; go < G * 2; go++) {
                double c = cr[go];
#pragma unroll
                for (int jj = 0; jj < RC; jj++)
                    if (jj < nr) sj[jj] += c * gr[jj * G * 2 + go];
            }
#pragma unroll
            for (int jj = 0; jj < RC; jj++) acc[jj] += sj[jj];
        }
    }
    unsigned ballot = __ballot_sync(0xffffffffu, any);
    unsigned group = lpc == 32 ? 0xffffffffu : ((1u << lpc) - 1u) << (lane & ~(lpc - 1));
    any = (ballot & group) != 0u;
#pragma unroll
    for (int jj = 0; jj < RC; jj++)
        for (int off = lpc >> 1; off > 0; off >>= 1) acc[jj] += __shfl_down_sync(0xffffffffu, acc[jj], off, lpc);
    if (!valid || !any || sub != 0) return;
#pragma unroll
    for (int jj = 0; jj < RC; jj++) {
        if (jj < nr) {
            double* o = out + (i64)(r0 + jj) * out_stride + dst;
            if (accumulate) *o += acc[jj]; else *o = acc[jj];
        }
    }
}

// Upper bound only, not the operator (bench_kernels): a constant
// kernel_sz^2 complex kernel per gauge on face images (no rotation, no
// interpolation, zero outside the face), out[((r G + g) 2 + o) n + i] for
// the first n pixels i of the faces (raster order, x fastest).
#define FACE_CONV(NAME, T)                                                                         \
    extern "C" __global__ void NAME(const T* x, i64 npix, int R, i64 nside, int G, int ksz,        \
                                    const T* ker, i64 n, T* out) {                                 \
        i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;                                        \
        if (t >= n * R) return;                                                                    \
        i64 i = t % n;                                                                             \
        int r = (int)(t / n);                                                                      \
        i64 f = i / (nside * nside), loc = i - f * nside * nside;                                  \
        i64 y = loc / nside, xx = loc - y * nside;                                                 \
        int h = ksz / 2, P = ksz * ksz;                                                            \
        const T* xs = x + (i64)r * npix + f * nside * nside;                                       \
        for (int g = 0; g < G; g++) {                                                              \
            T s0 = 0, s1 = 0;                                                                      \
            for (int ty = 0; ty < ksz; ty++) {                                                     \
                i64 yy = y + ty - h;                                                               \
                if (yy < 0 || yy >= nside) continue;                                               \
                for (int tx = 0; tx < ksz; tx++) {                                                 \
                    i64 x2 = xx + tx - h;                                                          \
                    if (x2 < 0 || x2 >= nside) continue;                                           \
                    T v = xs[yy * nside + x2];                                                     \
                    s0 += ker[(g * 2) * P + ty * ksz + tx] * v;                                    \
                    s1 += ker[(g * 2 + 1) * P + ty * ksz + tx] * v;                                \
                }                                                                                  \
            }                                                                                      \
            out[(((i64)r * G + g) * 2) * n + i] = s0;                                              \
            out[(((i64)r * G + g) * 2 + 1) * n + i] = s1;                                          \
        }                                                                                          \
    }
FACE_CONV(face_conv_f32, float)
FACE_CONV(face_conv_f64, double)

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

// |z| of a planar complex field, element i of rows R:
// out[r * out_stride + opos(i)] = hypot(z[(2r) * z_stride + zpos(i)], z[(2r+1) * z_stride + zpos(i)])
// (positions: index arrays or contiguous when null).
extern "C" __global__ void modulus_at(
    const double* z, i64 z_stride, const u32* zpos, i64 n, int R, double* out, i64 out_stride, const u32* opos)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 i = t % n;
    int r = (int)(t / n);
    i64 p = zpos ? (i64)zpos[i] : i;
    double re = z[(2 * (i64)r) * z_stride + p], im = z[(2 * (i64)r + 1) * z_stride + p];
    out[(i64)r * out_stride + (opos ? (i64)opos[i] : i)] = hypot(re, im);
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

// ---------------------------------------------------------------------------
// Terms of the synthesis objective on the members' maps (stl-cli): kappa
// transform, 4-children pyramid (level power, kurtosis per level), pixel
// marginal (stable radix sort), gradient sums.  Rows are members, the pixel
// axis last; the element-wise expressions are those of stl-cli, the sums by
// chunks of a fixed size reduced in a fixed tree, chunks added in order on
// the host.
// ---------------------------------------------------------------------------

// Standardised kappa = (f(x gs + gm) - km) / ks and its derivative dk = f'
// gs / ks, f the monotone cubic of the knots (kx, ky, kd), linear beyond the
// ends (kappa::Monotone::eval, kappa::Transform::apply).
extern "C" __global__ void kappa_pchip(
    const double* x, i64 n, const double* kx, const double* ky, const double* kd, int nk,
    double gm, double gs, double km, double ks, double* k, double* dk)
{
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double t = x[i] * gs + gm;
    double v, dv;
    if (t <= kx[0]) {
        v = ky[0] + kd[0] * (t - kx[0]);
        dv = kd[0];
    } else if (t >= kx[nk - 1]) {
        v = ky[nk - 1] + kd[nk - 1] * (t - kx[nk - 1]);
        dv = kd[nk - 1];
    } else {
        int lo = 0, hi = nk;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (kx[mid] <= t) lo = mid + 1; else hi = mid;
        }
        int j = lo - 1;
        double h = kx[j + 1] - kx[j];
        double u = (t - kx[j]) / h;
        double y0 = ky[j], y1 = ky[j + 1], d0 = kd[j] * h, d1 = kd[j + 1] * h;
        double u2 = u * u, u3 = u * u * u;
        v = (2.0 * u3 - 3.0 * u2 + 1.0) * y0 + (u3 - 2.0 * u2 + u) * d0 + (3.0 * u2 - 2.0 * u3) * y1 + (u3 - u2) * d1;
        double w = (6.0 * u2 - 6.0 * u) * (y0 - y1) + (3.0 * u2 - 4.0 * u + 1.0) * d0 + (3.0 * u2 - 2.0 * u) * d1;
        dv = w / h;
    }
    k[i] = (v - km) / ks;
    dk[i] = dv * gs / ks;
}

// Parents of a pyramid level: out[r * n + j] = (c0 + c1 + c2 + c3) / 4 over
// the children x[r * x_stride + 4 j + .] (level_power::pyramid).
extern "C" __global__ void quad_mean(const double* x, i64 x_stride, i64 n, int R, double* out) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (i64)R) return;
    i64 j = t % n;
    int r = (int)(t / n);
    const double* c = x + (i64)r * x_stride + 4 * j;
    out[t] = (c[0] + c[1] + c[2] + c[3]) / 4.0;
}

// Partial sums of the details d_j = xn[r, j] - xp[r, j / 4] of one level
// over chunks of `chunk` cells: out[(r * n_chunks + c) * 2 + {0, 1}] =
// (sum d^2, sum d^4) over chunk c of row r, in a fixed tree.  Grid (n_chunks, R).
extern "C" __global__ void detail_moments(
    const double* xn, i64 xn_stride, const double* xp, i64 xp_stride, i64 n, i64 chunk, double* out)
{
    __shared__ double s2[256], s4[256];
    int r = blockIdx.y;
    i64 c = blockIdx.x, n_chunks = gridDim.x;
    i64 lo = c * chunk, hi = lo + chunk < n ? lo + chunk : n;
    const double* a = xn + (i64)r * xn_stride;
    const double* p = xp + (i64)r * xp_stride;
    double t2 = 0.0, t4 = 0.0;
    for (i64 j = lo + threadIdx.x; j < hi; j += blockDim.x) {
        double d = a[j] - p[j >> 2];
        t2 += d * d;
        t4 += d * d * d * d;
    }
    s2[threadIdx.x] = t2;
    s4[threadIdx.x] = t4;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s2[threadIdx.x] += s2[threadIdx.x + s];
            s4[threadIdx.x] += s4[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[((i64)r * n_chunks + c) * 2] = s2[0];
        out[((i64)r * n_chunks + c) * 2 + 1] = s4[0];
    }
}

// Level-power gradient: g[r, i] += sum_{n < L} coef[r L + n] (lv[n][r, i >> 2n]
// - lv[n + 1][r, i >> 2(n + 1)]), summed from 0 in level order (the gradient
// of level_power::loss_and_grad, then added to g).  lv: L + 1 level
// addresses (lv[0] the map), ls their row strides.
extern "C" __global__ void level_power_grad(const u64* lv, const i64* ls, int L, const double* coef, i64 len, int R, double* g) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len * (i64)R) return;
    i64 i = t % len;
    int r = (int)(t / len);
    double acc = 0.0;
    for (int n = 0; n < L; n++) {
        const double* a = (const double*)lv[n] + (i64)r * ls[n];
        const double* p = (const double*)lv[n + 1] + (i64)r * ls[n + 1];
        double d = a[i >> (2 * n)] - p[i >> (2 * (n + 1))];
        acc += coef[(i64)r * L + n] * d;
    }
    g[t] += acc;
}

// Kurtosis gradient: g4[r, i] = sum over the term's levels k (level lev[k]
// = n) of coef (d^3 - c - K V d) at the level-n detail d of pixel i, c the
// mean of d^3 over its 4 siblings, cf[(r T + k) * 3 + {0, 1, 2}] = (coef,
// K, V) (kurtosis::loss_and_grad); then g[r, i] += g4 dk[r, i] (dk null:
// g4).  lv, ls as level_power_grad, down to the deepest level of the term + 1.
extern "C" __global__ void kurtosis_grad(
    const u64* lv, const i64* ls, const int* lev, int T, const double* cf, i64 len, int R, const double* dk, double* g)
{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len * (i64)R) return;
    i64 i = t % len;
    int r = (int)(t / len);
    double acc = 0.0;
    for (int k = 0; k < T; k++) {
        int n = lev[k];
        const double* a = (const double*)lv[n] + (i64)r * ls[n];
        const double* p = (const double*)lv[n + 1] + (i64)r * ls[n + 1];
        i64 j = i >> (2 * n);
        i64 j0 = j & ~(i64)3;
        double pv = p[j >> 2];
        double e0 = a[j0] - pv, e1 = a[j0 + 1] - pv, e2 = a[j0 + 2] - pv, e3 = a[j0 + 3] - pv;
        double c = (e0 * e0 * e0 + e1 * e1 * e1 + e2 * e2 * e2 + e3 * e3 * e3) / 4.0;
        double d = a[j] - pv;
        const double* q = cf + ((i64)r * T + k) * 3;
        acc += q[0] * (d * d * d - c - q[1] * q[2] * d);
    }
    g[t] += dk ? acc * dk[t] : acc;
}

// g[i] += a[i] b[i] (b null: a[i]).
extern "C" __global__ void add_product(double* g, const double* a, const double* b, i64 n) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g[i] += b ? a[i] * b[i] : a[i];
}

// Pixel marginal of one row: q-th smallest value at pixel perm[q], d =
// x[perm[q]] - t[q], g[perm[q]] += coef d (marginal::loss_and_grad, coef = 2
// w / N); out[c] = sum of d^2 over chunk c of positions, in a fixed tree.
extern "C" __global__ void marginal_row(const double* x, const u32* perm, const double* tsorted, i64 n, double coef, i64 chunk, double* g, double* out) {
    __shared__ double s[256];
    i64 c = blockIdx.x;
    i64 lo = c * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double acc = 0.0;
    for (i64 q = lo + threadIdx.x; q < hi; q += blockDim.x) {
        u32 p = perm[q];
        double d = x[p] - tsorted[q];
        acc += d * d;
        g[p] += coef * d;
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s[threadIdx.x] += s[threadIdx.x + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[c] = s[0];
}

// Stable LSD radix sort of f64 values (8-bit digits, 8 passes): tiles of
// RS_TILE keys, per-tile digit counts, an exclusive scan of the counts
// (digit-major), and a stable scatter.  The keys order as f64::total_cmp;
// the payload is the index, so ties keep the index order.
#define RS_ITEMS 8
#define RS_TILE (256 * RS_ITEMS)

extern "C" __global__ void rs_init(const double* x, i64 n, u64* keys, u32* vals) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 b = (u64)__double_as_longlong(x[i]);
    keys[i] = (b >> 63) ? ~b : (b | 0x8000000000000000ULL);
    vals[i] = (u32)i;
}

// hist[d * n_tiles + tile] = number of keys of the tile with digit d.  One block of 256 per tile.
extern "C" __global__ void rs_hist(const u64* keys, i64 n, int shift, u32* hist) {
    __shared__ u32 counts[256];
    i64 tile = blockIdx.x, n_tiles = gridDim.x;
    counts[threadIdx.x] = 0;
    __syncthreads();
    for (int k = 0; k < RS_ITEMS; k++) {
        i64 i = tile * RS_TILE + (i64)k * 256 + threadIdx.x;
        if (i < n) atomicAdd(&counts[(keys[i] >> shift) & 255], 1u);
    }
    __syncthreads();
    hist[(i64)threadIdx.x * n_tiles + tile] = counts[threadIdx.x];
}

// Exclusive scan in place by blocks of 1024 (256 threads x 4), the block
// totals to sums[block] (sums null: dropped).
extern "C" __global__ void scan_block(u32* a, i64 n, u32* sums) {
    __shared__ u32 s[256];
    i64 base = (i64)blockIdx.x * 1024 + threadIdx.x * 4;
    u32 v[4], tot = 0;
    for (int k = 0; k < 4; k++) {
        v[k] = base + k < n ? a[base + k] : 0;
        tot += v[k];
    }
    s[threadIdx.x] = tot;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        u32 add = threadIdx.x >= off ? s[threadIdx.x - off] : 0;
        __syncthreads();
        s[threadIdx.x] += add;
        __syncthreads();
    }
    u32 run = s[threadIdx.x] - tot;
    for (int k = 0; k < 4; k++) {
        if (base + k < n) a[base + k] = run;
        run += v[k];
    }
    if (threadIdx.x == 255 && sums) sums[blockIdx.x] = s[255];
}

// a[i] += offs[i / 1024].
extern "C" __global__ void scan_add(u32* a, i64 n, const u32* offs) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += offs[i >> 10];
}

// Stable scatter of a tile: rounds of 256 keys (one per thread, in index
// order); the rank of a key among the tile's keys of its digit = the
// earlier rounds + the lower warps of its round + the lower lanes of its
// warp (match_any); its position adds the scanned count of the tile.
extern "C" __global__ void rs_scatter(const u64* kin, const u32* vin, i64 n, int shift, const u32* offs, u64* kout, u32* vout) {
    __shared__ u32 base[256];
    __shared__ u32 wc[8][257];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    i64 tile = blockIdx.x, n_tiles = gridDim.x;
    base[tid] = offs[(i64)tid * n_tiles + tile];
    for (int w = 0; w < 8; w++) wc[w][tid] = 0;
    __syncthreads();
    for (int k = 0; k < RS_ITEMS; k++) {
        i64 i = tile * RS_TILE + (i64)k * 256 + tid;
        bool valid = i < n;
        u64 key = valid ? kin[i] : 0;
        int d = valid ? (int)((key >> shift) & 255) : 256;
        unsigned peers = __match_any_sync(0xffffffffu, d);
        int below = __popc(peers & ((1u << lane) - 1u));
        if (valid && below == 0) wc[warp][d] = __popc(peers);
        __syncthreads();
        if (valid) {
            u32 pos = base[d] + below;
            for (int w = 0; w < warp; w++) pos += wc[w][d];
            kout[pos] = key;
            vout[pos] = vin[i];
        }
        __syncthreads();
        u32 sum = 0;
        for (int w = 0; w < 8; w++) {
            sum += wc[w][tid];
            wc[w][tid] = 0;
        }
        base[tid] += sum;
        __syncthreads();
    }
}
