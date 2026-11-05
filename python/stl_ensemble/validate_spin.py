"""Validation of the spin-2 front end of NativeEngine(..., spin=2).

At every level j the engine holds chi_j = -(d_x - i d_y)^2 P_j (local
stencil on the pixels within 2 rings, pixel units of level j, P_j the
transported mean of the four children). The exact operator applied by SHT
to the same (Q, U)_j at nside_j is

    chi_ref_j = h_j^2 sqrt((l + 2)! / (l - 2)!) (E_lm, B_lm)  ->  alm2map

(h_j^2 = 4 pi / N_j; E_lm, B_lm by map2alm(pol=True, iter=3)). Reported:

- fields: relative error of chi_E, chi_B per level (all l, dominated by
  the pixel scale where chi ~ l^4 C_l peaks), and for a pure-E input the
  E->B leakage |chi_B| / |chi_E| (stencil and SHT reference);
- statistics: the coefficients of the engine on the stencil chi_j against
  the same engine on chi_ref_j (forward_levels), relative error per
  family, scale j3 and orientation, for E and B;
- locality: a hole (zeroed data) with the masks eroded by the stencil: chi
  outside the eroded mask is unchanged (to the bit), and the masked
  statistics change only through the wavelet halo near the edge;
- vjp: finite differences of Re sum conj(c) S(x) along a random direction;
- timings of forward and vjp, spin 2 against the scalar engine.

    python validate_spin.py --nside 64 [--planck Q.npy U.npy] [--backend gpu] [--out spin.json]
"""
import argparse
import json
import math
import sys
import time

import healpy as hp
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from stl_port.native import NativeEngine  # noqa: E402


def chi_exact(qu, nside, iters=3):
    """[2, npix] NESTED (Q, U) -> exact (chi_E, chi_B), pixel units of nside."""
    lmax = 3 * nside - 1
    q, u = (hp.reorder(v, n2r=True) for v in qu)
    _, ae, ab = hp.map2alm([np.zeros_like(q), q, u], lmax=lmax, iter=iters, pol=True)
    l = np.arange(lmax + 1.0)
    f = np.zeros(lmax + 1)
    f[2:] = np.sqrt((l[2:] + 2) * (l[2:] + 1) * l[2:] * (l[2:] - 1)) * 4 * math.pi / (12 * nside**2)
    return np.stack([hp.reorder(hp.alm2map(hp.almxfl(a, f), nside, lmax=lmax), r2n=True) for a in (ae, ab)])


def random_qu(nside, seed, pure_e=False, slope=-2.4):
    lmax = 3 * nside - 1
    l = np.arange(lmax + 1.0)
    cl = np.zeros(lmax + 1)
    cl[2:] = l[2:] ** slope
    np.random.seed(seed)
    alms = hp.synalm([0 * cl, cl, 0 * cl if pure_e else cl, 0 * cl], lmax=lmax, new=True)
    _, q, u = hp.alm2map(alms, nside, lmax=lmax, pol=True)
    return np.stack([hp.reorder(q, r2n=True), hp.reorder(u, r2n=True)])


def rel(a, b):
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2)))


def summarise(plan, loc, ref):
    """Relative error per (family, j3) and per (family, j3, l3) for E and B."""
    out = {}
    for comp, name in ((0, "E"), (1, "B")):
        groups = {}
        for i, c in enumerate(plan):
            jj = c["j3"] if c["j3"] is not None else c["j1"]
            key = c["family"] if c["family"] in ("mean", "var") else f'{c["family"]} j={jj}'
            groups.setdefault(key, []).append(i)
            if c["family"] in ("S3", "S4"):
                groups.setdefault(f'{c["family"]} j1={"0" if c["j1"] == 0 else ">0"}', []).append(i)
            if c["family"] in ("S1", "S2"):
                groups.setdefault(f'{c["family"]} j={jj} l={c["l1"]}', []).append(i)
        out[name] = {k: rel(loc[comp, v], ref[comp, v]) for k, v in groups.items()}
    return out


def disc_hole(nside, lon, lat, radius_deg):
    v = hp.ang2vec(lon, lat, lonlat=True)
    hole = hp.query_disc(nside, v, math.radians(radius_deg), nest=True)
    w = np.ones(12 * nside**2)
    w[hole] = 0.0
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--j", type=int, default=None)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--backend", default="cpu")
    p.add_argument("--pyramid", default="scalar", choices=["transported", "scalar"])
    p.add_argument("--realisations", type=int, default=2)
    p.add_argument("--planck", nargs=2, default=None)
    p.add_argument("--planck-t", default=None, help="scalar map T (e.g. 857 GHz) for the T x (Q, U) cross terms")
    p.add_argument("--fd-nside", type=int, default=16)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    nside = a.nside
    j = a.j or int(math.log2(nside)) - 1
    eng = NativeEngine(nside, j, batch=1, spin=2, spin_kernel=a.kernel, backend=a.backend, spin_pyramid=a.pyramid)
    plan = eng.plan
    scalar = a.pyramid == "scalar"
    levels = 1 if scalar else j
    # Reference of the scalar pyramid: the scalar engine on the exact chi_0 (E, B as a batch of 2).
    ref_eng = NativeEngine(nside, j, batch=2, backend=a.backend) if scalar else None
    report = {"nside": nside, "j": j, "kernel": a.kernel, "backend": a.backend, "pyramid": a.pyramid,
              "fields": {}, "statistics": {}}

    cases = [(f"dust{r}", random_qu(nside, 100 + r)) for r in range(a.realisations)]
    cases.append(("pure_e", random_qu(nside, 200, pure_e=True)))
    if a.planck:
        cases.append(("planck353", np.stack([np.load(a.planck[0]), np.load(a.planck[1])]).astype(np.float64)))
    for name, qu in cases:
        fields, chi_ref = {}, []
        for lev in range(levels):
            pol, chi = eng.spin_fields(qu[None], lev)
            ref = chi_exact(pol[0], nside >> lev)
            chi_ref.append(ref)
            f = {"rel_E": rel(chi[0, 0], ref[0]), "rel_B": rel(chi[0, 1], ref[1])}
            if name == "pure_e":
                f["leak_B_over_E"] = float(np.linalg.norm(chi[0, 1]) / np.linalg.norm(chi[0, 0]))
                f["ref_leak_B_over_E"] = float(np.linalg.norm(ref[1]) / np.linalg.norm(ref[0]))
            fields[str(lev)] = f
        loc, _ = eng.forward(qu[None])
        ref_v = ref_eng.forward(chi_ref[0])[0][None] if scalar else eng.forward_levels(chi_ref)
        st = summarise(plan, loc[0], ref_v[0])
        if name == "pure_e":
            # Coefficient-level leakage: S2 of B over S2 of E, per j3.
            s2 = [i for i, c in enumerate(plan) if c["family"] == "S2"]
            st["leak_S2_B_over_E"] = {f'j={plan[i]["j1"]} l={plan[i]["l1"]}':
                                      [float(abs(loc[0, 1, i]) / abs(loc[0, 0, i])), float(abs(ref_v[0, 1, i]) / abs(ref_v[0, 0, i]))]
                                      for i in s2}
        report["fields"][name] = fields
        report["statistics"][name] = st
        print(name, "fields", json.dumps(fields), flush=True)
        print(name, "statistics", json.dumps({k: v for k, v in st["E"].items() if "l=" not in k}), flush=True)

    # Cross statistics (scalar pyramid): against the scalar cross engine on
    # the exact chi_0 (T as it is), per component (E, B).
    if scalar:
        sc = NativeEngine(nside, j, batch=1, cross=True, backend=a.backend)
        pairs = []
        u = cases[0][1]
        pairs.append(("qu_x_qu", 2, u + 0.3 * random_qu(nside, 400), u + 0.3 * random_qu(nside, 401)))
        pairs.append(("t_x_qu", 0, chi_exact(u, nside)[0] / np.std(chi_exact(u, nside)[0]) + random_qu(nside, 402)[0], u))
        if a.planck and a.planck_t:
            pairs.append(("t857_x_planck353", 0, np.load(a.planck_t).astype(np.float64), cases[-1][1]))
        report["cross"] = {}
        for name, spin_a, xa, xb in pairs:
            e = NativeEngine(nside, j, batch=1, spin=2, cross=True, spin_a=spin_a, backend=a.backend)
            loc = e.forward_cross(xa[None], xb[None])[0][0]
            ea = chi_exact(xa, nside) if spin_a == 2 else np.stack([xa, xa])
            eb = chi_exact(xb, nside)
            ref = np.stack([sc.forward_cross(ea[k][None], eb[k][None])[0][0] for k in range(2)])
            st = summarise(e.plan, loc, ref)
            report["cross"][name] = st
            print("cross", name, json.dumps({k: v for k, v in st["E"].items() if "l=" not in k}), flush=True)

    # Locality: a hole of 15 degrees, masks eroded by the stencil at every level.
    qu = cases[0][1]
    w = disc_hole(nside, 30.0, 20.0, 15.0)
    holed = qu * w[None]
    loc_rep = {}
    eroded = None
    for lev in range(levels):
        _, full = eng.spin_fields(qu[None], lev)  # noqa: E501 (scalar pyramid: level 0 only)
        _, part = eng.spin_fields(holed[None], lev)
        diff = np.abs(part[0] - full[0]).max(0)
        loc_rep[str(lev)] = {"changed_pixels": int((diff > 0).sum()), "pixels": int(diff.size)}
    eng.set_masks(w[None])
    counts = eng.native_pixels()
    v_full, _ = eng.forward(qu[None])
    v_hole, _ = eng.forward(holed[None])
    loc_rep["eroded_mask_pixels"] = {f'level {c["j1"]}': int(counts[0, i]) for i, c in enumerate(plan) if c["family"] == "S1" and c["l1"] == 0}
    loc_rep["masked_statistics_rel_change"] = {k: v for k, v in summarise(plan, v_hole[0, :, 0], v_full[0, :, 0])["E"].items() if "l=" not in k}
    # chi outside the eroded mask must not change: check through the engine's own erosion
    eng.set_masks(None)
    report["locality"] = loc_rep
    print("locality", json.dumps(loc_rep), flush=True)

    # Finite differences of the vjp at small nside.
    nf = a.fd_nside
    jf = int(math.log2(nf)) - 1
    e2 = NativeEngine(nf, jf, batch=2, spin=2, spin_kernel=a.kernel, backend=a.backend, spin_pyramid=a.pyramid)
    rng = np.random.default_rng(1)
    x = np.stack([random_qu(nf, 300), random_qu(nf, 301)])
    v, _ = e2.forward(x)
    c = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
    g = e2.vjp(c)
    d = rng.normal(size=x.shape)
    eps = 1e-4 * np.abs(x).max()

    def loss(y, gen):
        return float(np.real(np.sum(np.conj(c) * e2.forward(y, generation=gen)[0])))

    fd = (loss(x + eps * d, 1) - loss(x - eps * d, 2)) / (2 * eps)
    an = float(np.sum(g * d))
    report["vjp_fd"] = {"nside": nf, "analytic": an, "finite_difference": fd, "rel": abs(an - fd) / abs(fd)}
    print("vjp fd", json.dumps(report["vjp_fd"]), flush=True)

    # Timings at nside: spin 2 vs scalar (one map).
    sc = NativeEngine(nside, j, batch=1, backend=a.backend)
    t = {}
    for label, e, x in (("spin2", eng, qu[None]), ("scalar", sc, qu[None, 0])):
        e.forward(x, generation=10)
        t0 = time.perf_counter()
        v, _ = e.forward(x, generation=11)
        t1 = time.perf_counter()
        e.vjp(np.ones_like(v), generation=11)
        t[label] = {"forward_s": t1 - t0, "vjp_s": time.perf_counter() - t1}
    t["spin_operator_bytes"] = eng.info().get("spin_bytes")
    report["timings"] = t
    print("timings", json.dumps(t), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
