#!/usr/bin/env python3
"""The options of train_native_unet.py that complete the best configuration
of Healpix-STL-IA (--loss data,l1,l2,l3,cl --stat-masks --cl-masks --cond
log857+sigma --bias frozen --bias-warmup) against train_denoise.py written
with torch autograd over NativeEngine statistics: the same network weights,
the same noise draws, the loss terms and the parameter gradients of a
frozen-bias step and of a live-bias (warmup) step, both under the
statistics masks and with the cl term (stlia.spectra, SpectrumEngine and
torch backends); the masks and the conditioning maps themselves.

Needs torch, healpy and the stlia package (STLIA_PATH); STL_TEST_BACKEND=gpu
runs the native side and the engines on the GPU. Synthetic maps at nside 16
under pytest; with --data DIR --nside 64 (as a script) the PR3 maps and the
FFP10 noise of train_denoise.py, --out FILE writes the report."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeEngine, NativeUNet
from stl_port.denoise import BinnedSpectra, SpectrumLoss, conditioning, stat_masks

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("STLIA_PATH", str(ROOT.parent / "Healpix-STL-IA")))
try:
    import torch
    from stlia.unet import UNet
    from stlia.stl_torch import DenoiseLoss, StatsFunction, as_real
    from stlia.train_denoise import CrossStats, unbiased
    from stlia import spectra as sp
except ImportError as e:  # pragma: no cover
    torch = None
    print(f"torch or stlia missing ({e}): skipped")

BACKEND = os.environ.get("STL_TEST_BACKEND", "cpu")
TERMS = ("data", "l1", "l2", "l3", "cl")
STAT_CUTS, CL_CUTS = [0.0, 20.0, 30.0, 45.0], [0.0, 30.0, 45.0, 60.0]


def _rel(a, b, floor=0.0):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), floor))


def _synthetic(nside, seed=0):
    rng = np.random.default_rng(seed)
    npix = 12 * nside**2
    sky = rng.normal(size=npix) * 2.0
    sims = 6
    pools = {f: [rng.normal(size=(sims, npix)) * 0.5 * (1 + 0.3 * np.arange(npix) / npix) for _ in range(3)]
             for f in ("Q", "U")}
    p = pools["Q"]
    return dict(d=sky + p[0][0] * 1.1, d1=sky + p[1][0] * 1.1, d2=sky + p[2][0] * 1.1,
                t857=np.exp(sky * 0.3 + rng.normal(size=npix)), pools=pools)


def _pr3(data, nside, sims=range(10, 50)):
    """The training simulations present for the three noise pools, as the driver."""
    sims = [k for k in sims if all(os.path.exists(os.path.join(data, f"planck353_{kd}_{f}_n{nside}_mc{k}.npy"))
                                   for kd in ("noise", "noisehm1", "noisehm2") for f in ("Q", "U"))]
    load = lambda name: np.load(os.path.join(data, f"{name}_n{nside}.npy")).astype(np.float64)
    pools = {f: [np.stack([np.load(os.path.join(data, f"planck353_{kd}_{f}_n{nside}_mc{k}.npy")) for k in sims])
                 .astype(np.float64) for kd in ("noise", "noisehm1", "noisehm2")] for f in ("Q", "U")}
    return dict(d=load("planck353_Q"), d1=load("planck353hm1_Q"), d2=load("planck353hm2_Q"),
                t857=load("planck857_I"), pools=pools)


def _stlia_cond(o):
    """The block "conds" of train_denoise.py (log857+sigma), on torch tensors."""
    i857 = o["t857"]
    conds = [np.log(np.clip(i857, np.percentile(i857[i857 > 0], 0.1), None))]
    var = torch.stack([torch.as_tensor(o["pools"][f][0]).square().mean(0) for f in ("Q", "U")]).mean(0)
    conds.append(0.5 * np.log(var.cpu().numpy()))
    return np.stack([(x - x.mean()) / x.std() for x in conds]).astype(np.float32)


class Reference:
    """train_denoise.py (--real, one field, --stat-masks, cl): frozen_losses,
    estimate_bias, losses + delouis with explicit draws."""

    def __init__(self, nside, j, o, cl_backend):
        self.nside, self.j, self.o = nside, j, o
        self.mk = {"masks": [np.flatnonzero(sp.galactic_mask(nside, b, apod=0) > 0.5) for b in STAT_CUTS]}
        K = len(STAT_CUTS)
        kw = dict(backend=BACKEND, **self.mk)
        self.one = NativeEngine(nside, j, batch=1, **kw)
        self.loss_fn = DenoiseLoss(self.one.plan, nside, masks=K, npix=self.one.native_pixels())
        self.xplan = NativeEngine(nside, j, batch=1, cross=True, **kw).plan
        self.xloss = DenoiseLoss(self.xplan, nside, masks=K, npix=NativeEngine(nside, j, batch=1, cross=True, **kw).native_pixels())
        self.exact = {t: torch.as_tensor(np.tile(unbiased(self.one.plan if t == "data" else self.xplan, t), K))
                      for t in TERMS[:4]}
        dev = "cuda:0" if BACKEND == "gpu" and cl_backend == "stl" else "cpu"
        self.spectra = sp.BinnedSpectra(nside, masks=CL_CUTS, device=dev, backend=cl_backend)
        self.dev = dev
        d1, d2 = (torch.as_tensor(o[k], device=dev) for k in ("d1", "d2"))
        self.cl = sp.SpectrumLoss(self.spectra, d1, d2)
        self.T, self.d, self.d1, self.d2 = (torch.as_tensor(o[k]) for k in ("t857", "d", "d1", "d2"))

    def auto(self, xs):
        e = NativeEngine(self.nside, self.j, batch=len(xs), backend=BACKEND, **self.mk)
        v, _ = e.forward(np.stack(xs))
        return torch.as_tensor(as_real(v)).reshape(len(xs), -1, 2)

    def cross(self, a, b, grad=False):
        e = NativeEngine(self.nside, self.j, batch=len(a), backend=BACKEND, cross=True, **self.mk)
        if grad:
            run = {"engine": e, "generation": 0}
            return CrossStats.apply(torch.stack(a), torch.stack(b), run), run["var"]
        v, _ = e.forward_cross(np.stack(a), np.stack(b))
        return torch.as_tensor(as_real(v)).reshape(len(a), -1, 2)

    def pick(self, pool, draws):
        return [g * self.o["pools"]["Q"][pool][s] for s, g in zip(*draws)]

    def bias(self, ud, draws):
        n0, n1, n2 = (self.pick(h, draws) for h in range(3))
        m, T = len(n0), self.o["t857"]
        with torch.no_grad():
            b = {"data": self.auto([ud + n for n in n0]).mean(0) - self.auto([ud])[0]}
            uu, tu = self.cross([ud, T], [ud, ud])
            b["l1"] = self.cross([ud + x for x in n1], [ud + y for y in n2]).mean(0) - uu
            b["l2"] = self.cross([ud + x for x in n0], [ud] * m).mean(0) - uu
            b["l3"] = self.cross([T] * m, [ud + x for x in n0]).mean(0) - tu
            for t in b:
                b[t][self.exact[t]] = 0
        return b

    def cl_term(self, u):
        uc = u.to(self.dev)
        return self.cl(self.spectra(uc, uc)).cpu()

    def frozen(self, u, bias):
        vt, tv = self.one.forward(self.o["d"][None], moments=True)
        target, tvar = torch.as_tensor(as_real(vt[0])).reshape(-1, 2), torch.as_tensor(tv[0]).reshape(-1)
        terms = {"data": self.loss_fn((StatsFunction.apply(u[None], self.one, 1)[0] + bias["data"])[None], target, tvar)}
        T, d = self.T, self.d
        st, var = self.cross([self.d1, d, T, u, T], [self.d2, u, d, u, u], grad=True)
        model = {"l1": st[3], "l2": st[3], "l3": st[4]}
        for t, name in enumerate(("l1", "l2", "l3")):
            target = st[t] if name == "l2" else st[t].detach()
            terms[name] = self.xloss((model[name] + bias[name])[None], target, var[t])
        terms["cl"] = self.cl_term(u)
        return terms

    def live(self, u, ddraws, hdraws):
        k, kh = len(ddraws[0]), len(hdraws[0])
        many = NativeEngine(self.nside, self.j, batch=k, backend=BACKEND, **self.mk)
        vt, tv = self.one.forward(self.o["d"][None], moments=True)
        st = StatsFunction.apply(u + torch.as_tensor(np.stack(self.pick(0, ddraws))), many, 1)
        terms = {"data": self.loss_fn(st, torch.as_tensor(as_real(vt[0])).reshape(-1, 2), torch.as_tensor(tv[0]).reshape(-1))}
        n, n1, n2 = (torch.as_tensor(np.stack(self.pick(h, hdraws))) for h in range(3))
        T, d = self.T, self.d
        A = list(torch.cat([self.d1[None], d[None], T[None], u + n1, u + n, T.expand(kh, -1)]))
        Bm = list(torch.cat([self.d2[None], u[None], d[None], u + n2, u.expand(kh, -1), u + n]))
        xs, var = self.cross(A, Bm, grad=True)
        for t, name in enumerate(("l1", "l2", "l3")):
            target = xs[t] if name == "l2" else xs[t].detach()
            terms[name] = self.xloss(xs[3 + t * kh:3 + (t + 1) * kh], target, var[t])
        terms["cl"] = self.cl_term(u)
        return terms


def _compare(nside, j, channels, o, seed=3, cl_backend="stl", report=None):
    torch.manual_seed(seed)
    cond = conditioning(["log857", "sigma"], o["t857"], [np.mean(np.square(o["pools"][f][0]), 0) for f in ("Q", "U")])
    ec = _rel(cond, _stlia_cond(o))
    ids, ref_ids = stat_masks(nside, STAT_CUTS), [np.flatnonzero(sp.galactic_mask(nside, b, apod=0) > 0.5) for b in STAT_CUTS]
    assert all(np.array_equal(x, y) for x, y in zip(ids, ref_ids))
    net = UNet(nside, len(channels), channels, cond=2)
    with torch.no_grad():
        for name, q in net.named_parameters():
            if name.startswith("head"):
                q.copy_(0.05 * torch.randn_like(q))
    ref = Reference(nside, j, o, cl_backend)
    w = dict(data=1.0, l1=1.0, l2=1.0, l3=1.0, cl=1.0)
    nat = NativeUNet(nside, channels, cond=2, j=j, loss=w, backend=BACKEND)
    nat.load_state_dict(net.state_dict())
    nat.set_masks(ids)
    nat.set_cond(cond)
    nat.set_common(o["t857"])
    nat.set_field("Q", o["d"], o["d1"], o["d2"], *o["pools"]["Q"])
    spectra = BinnedSpectra(nside, masks=CL_CUTS, backend=BACKEND)
    cl = SpectrumLoss(spectra, o["d1"], o["d2"])
    tc = torch.as_tensor(cond)[None]
    d = torch.as_tensor(o["d"])
    out = {"nside": nside, "j": j, "channels": channels, "cl_backend": cl_backend, "cond_max_rel": ec,
           "stat_mask_pixels": [len(x) for x in ids], "cl_bins": int(len(spectra.edges) - 1)}

    def native_terms(terms):
        u = nat.apply(o["d"])
        v, g = cl(u, grad=True)
        nat.apply(o["d"], cot=g)
        terms["cl"] = v
        return terms

    def check(name, ref_terms):
        total = sum(ref_terms.values())
        net.zero_grad()
        total.backward()
        rg = {x: q.grad.numpy().copy() for x, q in net.named_parameters()}
        g = nat.grads_dict()
        floor = 1e-2 * max(np.abs(v).max() for v in rg.values())
        eg = max(_rel(g[x], rg[x], floor) for x in rg)
        en = float(np.sqrt(sum(np.sum((g[x] - rg[x])**2) for x in rg) / sum(np.sum(rg[x]**2) for x in rg)))
        et = {t: abs(got[t] - float(ref_terms[t])) / abs(float(ref_terms[t])) for t in TERMS}
        out[name] = {"terms_native": {t: got[t] for t in TERMS}, "terms_reference": {t: float(ref_terms[t]) for t in TERMS},
                     "terms_max_rel": max(et.values()), "terms_rel": et, "gradients_max_rel": eg,
                     "gradients_norm_rel": en}
        print(f"{name}: terms {max(et.values()):.2e} ({', '.join(f'{t} {v:.1e}' for t, v in et.items())}), "
              f"gradients {eg:.2e} (norm {en:.2e})", flush=True)
        # float32 network: the GroupNorm reverse limits the gradients to ~1.5e-3 at nside 64
        assert max(et.values()) < 1e-4 and eg < (1e-3 if nside < 64 else 5e-3) and en < 1e-3, (name, et, eg, en)

    # frozen-bias step, biases from the same draws
    draws = (np.array([1, 3, 2, 4, 0, 5]), np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]))
    u = net(d[None], d.std(), cond=tc)[0]
    bias = ref.bias(u.detach().numpy(), draws)
    nat.estimate_bias("Q", *draws)
    nat.zero_grad()
    got = native_terms(nat.loss("Q", backward=True))
    check("frozen", ref.frozen(u, bias))
    # live-bias (warmup) step
    ddraws = (np.array([2, 0, 4, 5]), np.array([-1.0, 1.0, 1.0, -1.0]))
    hdraws = (np.array([1, 3, 5, 0]), np.array([1.0, -1.0, 1.0, 1.0]))
    u = net(d[None], d.std(), cond=tc)[0]
    nat.zero_grad()
    got = native_terms(nat.live("Q", ddraws, hdraws, backward=True))
    check("live", ref.live(u, ddraws, hdraws))
    print(f"conditioning maps: {ec:.2e}; stat masks {out['stat_mask_pixels']}", flush=True)
    assert ec < 1e-5
    if report:
        with open(report, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def test_best_configuration_matches_torch():
    if torch is None:
        return
    _compare(16, 3, [8, 16, 24], _synthetic(16), cl_backend="torch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--cl-backend", default="torch", choices=["torch", "stl"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if torch is None:
        sys.exit(0)
    if a.data:
        n = a.nside
        levels = int(np.log2(n // 8)) + 1
        _compare(n, int(np.log2(n)) - 1, [32, 64, 96, 128, 128, 128, 128][:levels], _pr3(a.data, n),
                 cl_backend=a.cl_backend, report=a.out)
    else:
        test_best_configuration_matches_torch()
    print("native denoise tests passed")
