
// ---------------------------------------------------------------------------
// Disc patches (disc.rs): the term sums and their cotangents for a batch of
// patches grouped by disc.  The operands of level n are fields restricted to
// the averaging (core) pixels of the level, one buffer per (group, slot):
// slot s of member m holds rows[s] rows of nc[group] values, a real row r at
// (m rows[s] + r) nc, a complex row at 2 (m rows[s] + r) nc (re) and nc
// further (im), the layout of the convolution outputs.
// ---------------------------------------------------------------------------

struct DiscOp {
    int slot;   // < 0: none
    int row;
    int cplx;
    int pad;
};

// A coefficient of the level: term kind (F_* forward, R_* reverse) and plan index.
struct DiscTerm {
    int kind;
    int index;
    DiscOp a;
    DiscOp b;
};

struct DiscPatch {
    int group;
    int member;
    int out;     // patch index of the outputs and cotangents
    int pad;
};

struct DiscTarget {
    DiscOp t;
    u32 start, end;
};

__device__ __forceinline__ i64 disc_at(const DiscOp& o, const int* rows, int member, i64 nc) {
    i64 r = (i64)member * rows[o.slot] + o.row;
    return o.cplx ? 2 * r * nc : r * nc;
}

__device__ __forceinline__ void disc_load(const DiscOp& o, const u64* base, const int* rows, int n_slots, const DiscPatch& pt, i64 nc, i64 i, double* re, double* im) {
    const double* p = (const double*)base[(i64)pt.group * n_slots + o.slot] + disc_at(o, rows, pt.member, nc);
    *re = p[i];
    *im = o.cplx ? p[nc + i] : 0.0;
}

// Sums over the core pixels of every (term, patch): out[(out * n_plan +
// index) * 3 + {0, 1, 2}] = (sum re, sum im, sum |f|^2), reduced in a fixed
// tree.  Grid (terms, patches), DISC_T threads.
#define DISC_T 128
extern "C" __global__ void disc_terms(
    const DiscTerm* terms, const DiscPatch* patches, const u64* base, const int* rows, int n_slots,
    const i64* ncs, i64 n_plan, int moments, double* out)
{
    __shared__ double sre[DISC_T], sim[DISC_T], ssq[DISC_T];
    DiscTerm it = terms[blockIdx.x];
    DiscPatch pt = patches[blockIdx.y];
    i64 nc = ncs[pt.group];
    double are, aim, bre, bim;
    double tre = 0.0, tim = 0.0, tsq = 0.0;
    for (i64 i = threadIdx.x; i < nc; i += blockDim.x) {
        disc_load(it.a, base, rows, n_slots, pt, nc, i, &are, &aim);
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
                disc_load(it.b, base, rows, n_slots, pt, nc, i, &bre, &bim);
                double xr, xi, yr, yi;
                root_normalise(are, aim, &xr, &xi);
                root_normalise(bre, bim, &yr, &yi);
                mul_conj(xr, xi, yr, yi, &vre, &vim);
                sq = vre * vre + vim * vim;
                break;
            }
            default:
                disc_load(it.b, base, rows, n_slots, pt, nc, i, &bre, &bim);
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
        double* o = out + ((i64)pt.out * n_plan + it.index) * 3;
        o[0] = sre[0];
        o[1] = sim[0];
        o[2] = ssq[0];
    }
}

// Cotangents of the operands: for every (target, patch, core pixel), the
// contributions of the target in plan order (R_* kinds of coef_reverse,
// n_total = nc) added to the gradient slot (gbase) of the target.  Grid
// (pixel blocks, targets, patches).
extern "C" __global__ void disc_terms_rev(
    const DiscTarget* targets, const DiscTerm* contribs, const DiscPatch* patches,
    const u64* base, const u64* gbase, const int* rows, int n_slots, const i64* ncs,
    const double* cot, i64 n_plan)
{
    DiscPatch pt = patches[blockIdx.z];
    i64 nc = ncs[pt.group];
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    DiscTarget t = targets[blockIdx.y];
    double n_total = (double)nc;
    double dre = 0.0, dim = 0.0;
    for (u32 q = t.start; q < t.end; q++) {
        DiscTerm c = contribs[q];
        const double* gp = cot + ((i64)pt.out * n_plan + c.index) * 2;
        double g_re = gp[0], g_im = gp[1];
        if (g_re == 0.0 && g_im == 0.0) continue;
        double are, aim, bre, bim, vr = 0.0, vi = 0.0;
        disc_load(c.a, base, rows, n_slots, pt, nc, i, &are, &aim);
        switch (c.kind) {
            case R_MEAN:
                vr = g_re / n_total;
                break;
            case R_VAR:
                vr = 2.0 * g_re * are / n_total;
                break;
            case R_S1SAME: {
                double scale = g_re / n_total;
                double mag = hypot(are, aim);
                double dd = mag == 0.0 ? 1.0 : mag;
                vr = (scale * are) / dd;
                vi = (scale * aim) / dd;
                break;
            }
            case R_S2SAME: {
                double s = 2.0 * g_re;
                vr = (s * are) / n_total;
                vi = (s * aim) / n_total;
                break;
            }
            case R_S1X_A: {
                disc_load(c.b, base, rows, n_slots, pt, nc, i, &bre, &bim);
                double yr, yi;
                root_normalise(bre, bim, &yr, &yi);
                double pr = g_re * yr - g_im * yi;
                double pi = g_re * yi + g_im * yr;
                root_modulus_vjp(are, aim, pr / n_total, pi / n_total, &vr, &vi);
                break;
            }
            case R_S1X_B: {
                disc_load(c.b, base, rows, n_slots, pt, nc, i, &bre, &bim);
                double yr, yi;
                root_normalise(are, aim, &yr, &yi);
                double gr = g_re, gi = -g_im;
                double pr = gr * yr - gi * yi;
                double pi = gr * yi + gi * yr;
                root_modulus_vjp(bre, bim, pr / n_total, pi / n_total, &vr, &vi);
                break;
            }
            case R_GA: {
                disc_load(c.b, base, rows, n_slots, pt, nc, i, &bre, &bim);
                double pr = g_re * bre - g_im * bim;
                double pi = g_re * bim + g_im * bre;
                vr = pr / n_total;
                vi = pi / n_total;
                break;
            }
            default: {
                double gr = g_re, gi = -g_im;
                double pr = gr * are - gi * aim;
                double pi = gr * aim + gi * are;
                vr = pr / n_total;
                vi = pi / n_total;
            }
        }
        dre += vr;
        dim += vi;
    }
    double* d = (double*)gbase[(i64)pt.group * n_slots + t.t.slot] + disc_at(t.t, rows, pt.member, nc);
    d[i] += dre;
    if (t.t.cplx) d[nc + i] += dim;
}
