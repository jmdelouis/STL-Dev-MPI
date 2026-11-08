#!/usr/bin/env python3
"""The joint network of NativeUNet (maps=2) against Healpix-STL-IA:

- the network: stlia.unet.UNet(maps=2) with one input scale per map, its
  output and parameter gradients, the state_dict both ways;
- the front end "eth": NativeEngine(spin=2)'s chi_0 of the same (Q, U), and
  its transpose (dot-product test);
- --pol eb (the network on the E and B maps, statistics per field, cl the
  scalar spectra of E and B): the frozen-bias and live-bias terms and
  gradients of a joint step against train_denoise.py's per-field terms
  (frozen_losses, losses) summed over the two fields of a UNet(maps=2);
- --pol spin (the network on (Q, U), statistics of chi_E and chi_B by the
  scalar engine, cl the EE/BB spectra): against stlia train_denoise.ChiTerms
  (--pol-stats chi, NativeEngine(spin=2)) and BinnedSpectra(spin=2), the same
  draws (GPU only, as ChiTerms).

Needs torch, healpy and the stlia package (STLIA_PATH); STL_TEST_BACKEND=gpu
runs the native side on the GPU. Synthetic maps at nside 16 (pytest); with
--nside 64 --data DIR (script) the PR3 maps and FFP10 noise; --out FILE
writes the report."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from stl_port.native import NativeEngine, NativeUNet
from stl_port.denoise import BinnedSpectra, SpectrumLoss, conditioning, stat_masks

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("STLIA_PATH", str(ROOT.parent / "Healpix-STL-IA")))
try:
    import torch
    from stlia.unet import UNet
    from stlia import spectra as sp
    from test_native_denoise import Reference, STAT_CUTS, CL_CUTS, _rel
except ImportError as e:  # pragma: no cover
    torch = None
    print(f"torch or stlia missing ({e}): skipped")

BACKEND = os.environ.get("STL_TEST_BACKEND", "cpu")
TERMS = ("data", "l1", "l2", "l3", "cl")


def _synthetic(nside, seed=0, sims=6):
    """Two fields (Q, U) with noise pools of the same simulations."""
    rng = np.random.default_rng(seed)
    npix = 12 * nside**2
    out = {"t857": None, "fields": {}}
    sky = {f: rng.normal(size=npix) * (2.0 if f == "Q" else 1.5) for f in ("Q", "U")}
    for f in ("Q", "U"):
        pools = [rng.normal(size=(sims, npix)) * 0.5 * (1 + 0.3 * np.arange(npix) / npix) for _ in range(3)]
        out["fields"][f] = dict(d=sky[f] + pools[0][0] * 1.1, d1=sky[f] + pools[1][0] * 1.1,
                                d2=sky[f] + pools[2][0] * 1.1, pools=pools)
    out["t857"] = np.exp(sky["Q"] * 0.3 + rng.normal(size=npix))
    return out


def _pr3(data, nside, sims=range(10, 50)):
    sims = [k for k in sims if all(os.path.exists(os.path.join(data, f"planck353_{kd}_{f}_n{nside}_mc{k}.npy"))
                                   for kd in ("noise", "noisehm1", "noisehm2") for f in ("Q", "U"))]
    load = lambda name: np.load(os.path.join(data, f"{name}_n{nside}.npy")).astype(np.float64)
    out = {"t857": load("planck857_I"), "fields": {}}
    for f in ("Q", "U"):
        pools = [np.stack([np.load(os.path.join(data, f"planck353_{kd}_{f}_n{nside}_mc{k}.npy")) for k in sims])
                 .astype(np.float64) for kd in ("noise", "noisehm1", "noisehm2")]
        out["fields"][f] = dict(d=load(f"planck353_{f}"), d1=load(f"planck353hm1_{f}"), d2=load(f"planck353hm2_{f}"),
                                pools=pools)
    return out


def _pair(o, key):
    return np.stack([o["fields"][f][key] for f in ("Q", "U")])


def _cond(o):
    return conditioning(["log857", "sigma"], o["t857"], [np.mean(np.square(o["fields"][f]["pools"][0]), 0)
                                                        for f in ("Q", "U")])


def _torch_net(nside, channels, seed):
    torch.manual_seed(seed)
    net = UNet(nside, len(channels), channels, cond=2, maps=2)
    with torch.no_grad():
        for name, q in net.named_parameters():
            if name.startswith("head"):
                q.copy_(0.05 * torch.randn_like(q))
    return net


def _grad_errors(net, nat):
    rg = {x: q.grad.numpy().copy() for x, q in net.named_parameters()}
    g = nat.grads_dict()
    floor = 1e-2 * max(np.abs(v).max() for v in rg.values())
    eg = max(_rel(g[x], rg[x], floor) for x in rg)
    en = float(np.sqrt(sum(np.sum((g[x] - rg[x])**2) for x in rg) / sum(np.sum(rg[x]**2) for x in rg)))
    return eg, en


def check_network(nside=16, channels=(8, 16, 24)):
    """UNet(maps=2) and NativeUNet(maps=2): outputs and parameter gradients, per-map scales."""
    channels = list(channels)
    o = _synthetic(nside)
    net = _torch_net(nside, channels, 5)
    nat = NativeUNet(nside, channels, cond=2, maps=2, backend=BACKEND)
    nat.load_state_dict(net.state_dict())
    back = {k: torch.as_tensor(v) for k, v in nat.state_dict().items()}
    assert all(torch.equal(back[k], v) for k, v in net.state_dict().items()), "state_dict round trip"
    cond = _cond(o)
    nat.set_cond(cond)
    d2 = _pair(o, "d")
    scales = d2.std(1, ddof=1)
    rng = np.random.default_rng(1)
    cot = rng.normal(size=d2.shape)
    d = torch.as_tensor(d2)[None]
    u = net(d, torch.as_tensor(scales)[None, :, None], cond=torch.as_tensor(cond)[None])
    net.zero_grad()
    (u * torch.as_tensor(cot)[None]).sum().backward()
    nat.zero_grad()
    un = nat.apply(d2, cot=cot)
    eu = _rel(d2 - un, (d - u)[0].detach().numpy())
    eg, en = _grad_errors(net, nat)
    print(f"network maps=2: output {eu:.2e}, gradients {eg:.2e} (norm {en:.2e})", flush=True)
    assert eu < 1e-4 and eg < 1e-3 and en < 1e-4, (eu, eg, en)
    return {"output_max_rel": eu, "gradients_max_rel": eg, "gradients_norm_rel": en}


def check_front(nside=16):
    """The "eth" front end: chi_0 of NativeEngine(spin=2), and its transpose."""
    o = _synthetic(nside)
    nat = NativeUNet(nside, [8, 16], maps=2, backend="cpu")
    qu = _pair(o, "d")
    nat.set_joint(qu, ("E", "B"), front="eth")
    chi = nat.front(qu)
    eng = NativeEngine(nside, 3, batch=1, spin=2, backend="cpu")
    ref = np.asarray(eng.spin_fields(qu[None], level=0)[1]).reshape(2, -1)
    e = _rel(chi, ref)
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(2, 2, qu.shape[-1]))
    dot = abs(np.sum(nat.front(x) * y) - np.sum(x * nat.front(y, adjoint=True))) / abs(np.sum(nat.front(x) * y))
    print(f"front eth: chi vs NativeEngine(spin=2) {e:.2e}, adjoint {dot:.2e}", flush=True)
    assert e < 1e-12 and dot < 1e-12, (e, dot)
    return {"chi_max_rel": e, "adjoint_rel": dot}


def _check_terms(name, out, got, ref_terms, net, nat, nside):
    total = sum(ref_terms.values())
    net.zero_grad()
    total.backward()
    eg, en = _grad_errors(net, nat)
    et = {t: abs(got[t] - float(ref_terms[t])) / abs(float(ref_terms[t])) for t in TERMS}
    out[name] = {"terms_native": {t: got[t] for t in TERMS}, "terms_reference": {t: float(ref_terms[t]) for t in TERMS},
                 "terms_max_rel": max(et.values()), "terms_rel": et, "gradients_max_rel": eg, "gradients_norm_rel": en}
    print(f"{name}: terms {max(et.values()):.2e} ({', '.join(f'{t} {v:.1e}' for t, v in et.items())}), "
          f"gradients {eg:.2e} (norm {en:.2e})", flush=True)
    # float32 network: the GroupNorm reverse limits the gradients to ~1.5e-3 at nside 64
    assert max(et.values()) < 1e-4 and eg < (1e-3 if nside < 64 else 5e-3) and en < 1e-3, (name, et, eg, en)


def check_eb(nside, j, channels, o, report=None):
    """--pol eb: the joint step on E and B fields against the per-field terms of train_denoise.py, summed."""
    channels = list(channels)
    fields = ("E", "B")
    # the E and B maps stand for any pair of scalar fields here (synthetic: the Q, U pools as they are)
    fo = {f: o["fields"][g] for f, g in zip(fields, ("Q", "U"))}
    net = _torch_net(nside, channels, 3)
    w = dict(data=1.0, l1=1.0, l2=1.0, l3=1.0, cl=1.0)
    nat = NativeUNet(nside, channels, cond=2, maps=2, j=j, loss=w, backend=BACKEND)
    nat.load_state_dict(net.state_dict())
    ids = stat_masks(nside, STAT_CUTS)
    nat.set_masks(ids)
    cond = _cond(o)
    nat.set_cond(cond)
    nat.set_common(o["t857"])
    refs, cls = {}, {}
    spectra = BinnedSpectra(nside, masks=CL_CUTS, backend=BACKEND)
    for f in fields:
        x = fo[f]
        nat.set_field(f, x["d"], x["d1"], x["d2"], *x["pools"])
        refs[f] = Reference(nside, j, dict(d=x["d"], d1=x["d1"], d2=x["d2"], t857=o["t857"], pools={"Q": x["pools"]}),
                            "torch")
        cls[f] = SpectrumLoss(spectra, x["d1"], x["d2"])
    d2 = np.stack([fo[f]["d"] for f in fields])
    scales = d2.std(1, ddof=1)
    nat.set_joint(d2, fields, scales=scales)
    td, ts, tc = torch.as_tensor(d2)[None], torch.as_tensor(scales)[None, :, None], torch.as_tensor(cond)[None]
    out = {"pol": "eb", "nside": nside, "j": j, "channels": channels}

    def native_terms(terms):
        u = nat.apply(d2, scale=scales)
        parts = [cls[f](u[i], grad=True) for i, f in enumerate(fields)]
        nat.apply(d2, scale=scales, cot=np.stack([g for _, g in parts]))
        terms["cl"] = sum(v for v, _ in parts)
        return terms

    def summed(per):
        return {t: sum(p[t] for p in per) for t in TERMS}

    draws = (np.array([1, 3, 2, 4, 0, 5]), np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]))
    u = net(td, ts, cond=tc)[0]
    bias = {f: refs[f].bias(u[i].detach().numpy(), draws) for i, f in enumerate(fields)}
    nat.estimate_bias("joint", *draws)
    nat.zero_grad()
    got = native_terms(nat.loss("joint", backward=True))
    _check_terms("eb frozen", out, got, summed([refs[f].frozen(u[i], bias[f]) for i, f in enumerate(fields)]),
                 net, nat, nside)
    ddraws = (np.array([2, 0, 4, 5]), np.array([-1.0, 1.0, 1.0, -1.0]))
    hdraws = (np.array([1, 3, 5, 0]), np.array([1.0, -1.0, 1.0, 1.0]))
    u = net(td, ts, cond=tc)[0]
    nat.zero_grad()
    got = native_terms(nat.live("joint", ddraws, hdraws, backward=True))
    _check_terms("eb live", out, got, summed([refs[f].live(u[i], ddraws, hdraws) for i, f in enumerate(fields)]),
                 net, nat, nside)
    if report:
        with open(report, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def check_ebqu(nside, j, channels, o, report=None):
    """--pol ebqu: the front end "eb" against healpy (stl_port.denoise.eb_maps) and its
    transpose (dot-product test); the joint step on (Q, U) whose terms are those of the
    E and B maps of the output, against train_denoise.py's per-field terms of eb_maps(u)
    summed, the gradient chained through the front end's transpose (frozen and live
    biases); a central finite difference of the whole loss in the output u (float64)."""
    from stl_port.denoise import eb_maps
    channels = list(channels)
    qu, fields = ("Q", "U"), ("E", "B")
    net = _torch_net(nside, channels, 7)
    w = dict(data=1.0, l1=1.0, l2=1.0, l3=1.0, cl=1.0)
    nat = NativeUNet(nside, channels, cond=2, maps=2, j=j, loss=w, backend=BACKEND)
    nat.load_state_dict(net.state_dict())
    nat.set_masks(stat_masks(nside, STAT_CUTS))
    cond = _cond(o)
    nat.set_cond(cond)
    nat.set_common(o["t857"])
    d2 = _pair(o, "d")
    scales = d2.std(1, ddof=1)
    nat.set_joint(d2, fields, front="eb", scales=scales)
    out = {"pol": "ebqu", "nside": nside, "j": j, "channels": channels}
    # the front end: healpy's E/B maps, and the transpose
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(2, 3, 2, d2.shape[-1]))
    fx = nat.front(x)
    ef = _rel(fx, eb_maps(x))
    dot = abs(np.sum(fx * y) - np.sum(x * nat.front(y, adjoint=True))) / abs(np.sum(fx * y))
    out["front"] = {"eb_maps_max_rel": ef, "adjoint_rel": dot}
    print(f"front eb: vs healpy eb_maps {ef:.2e}, adjoint {dot:.2e}", flush=True)
    assert ef < 1e-10 and dot < 1e-12, (ef, dot)
    # the fields of the terms: the E and B maps of the observations and pools, once
    obs = {k: nat.front(_pair(o, k)) for k in ("d", "d1", "d2")}
    pools = [nat.front(np.stack([o["fields"][f]["pools"][h] for f in qu], 1)) for h in range(3)]  # [S, 2, npix]
    refs, cls = {}, {}
    spectra = BinnedSpectra(nside, masks=CL_CUTS, backend=BACKEND)
    for i, f in enumerate(fields):
        fp = [np.ascontiguousarray(p[:, i]) for p in pools]
        nat.set_field(f, obs["d"][i], obs["d1"][i], obs["d2"][i], *fp)
        refs[f] = Reference(nside, j, dict(d=obs["d"][i], d1=obs["d1"][i], d2=obs["d2"][i], t857=o["t857"],
                                           pools={"Q": fp}), "torch")
        cls[f] = SpectrumLoss(spectra, obs["d1"][i], obs["d2"][i])
    td, ts, tc = torch.as_tensor(d2)[None], torch.as_tensor(scales)[None, :, None], torch.as_tensor(cond)[None]

    def native_terms(terms, backward=True):
        u = nat.apply(d2, scale=scales)
        parts = [cls[f](v, grad=True) for f, v in zip(fields, nat.front(u))]
        if backward:
            nat.apply(d2, scale=scales, cot=nat.front(np.stack([g for _, g in parts]), adjoint=True))
        terms["cl"] = sum(v for v, _ in parts)
        return terms

    def reference(name, terms_of, got):
        """terms_of(i, y_i) the reference terms of field i at the E/B map y_i (torch)."""
        u = net(td, ts, cond=tc)[0]
        yv = torch.tensor(eb_maps(u.detach().numpy()), requires_grad=True)
        per = [terms_of(i, yv[i]) for i in range(2)]
        ref = {t: sum(p[t] for p in per) for t in TERMS}
        sum(ref.values()).backward()
        gu = torch.as_tensor(nat.front(yv.grad.numpy(), adjoint=True))
        et = {t: abs(got[t] - float(ref[t])) / abs(float(ref[t])) for t in TERMS}
        net.zero_grad()
        (u * gu).sum().backward()
        eg, en = _grad_errors(net, nat)
        out[name] = {"terms_native": {t: got[t] for t in TERMS}, "terms_reference": {t: float(ref[t]) for t in TERMS},
                     "terms_max_rel": max(et.values()), "terms_rel": et, "gradients_max_rel": eg,
                     "gradients_norm_rel": en}
        print(f"{name}: terms {max(et.values()):.2e} ({', '.join(f'{t} {v:.1e}' for t, v in et.items())}), "
              f"gradients {eg:.2e} (norm {en:.2e})", flush=True)
        assert max(et.values()) < 1e-4 and eg < (1e-3 if nside < 64 else 5e-3) and en < 1e-3, (name, et, eg, en)

    draws = (np.array([1, 3, 2, 4, 0, 5]), np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]))
    u0 = net(td, ts, cond=tc)[0].detach().numpy()
    y0 = eb_maps(u0)
    bias = {f: refs[f].bias(y0[i], draws) for i, f in enumerate(fields)}
    nat.estimate_bias("joint", *draws)
    nat.zero_grad()
    got = native_terms(nat.loss("joint", backward=True))
    reference("ebqu frozen", lambda i, y: refs[fields[i]].frozen(y, bias[fields[i]]), got)
    # the loss (frozen biases) as a function of the output u in float64, L(EB(u)), along a
    # direction v of u: central difference against <F^T grad_y L, v>. (The network parameters
    # are float32: a difference over them is dominated by rounding.) Without l2, whose chi2
    # weights, the variances of Phi(d, u), depend on u but are held fixed in the gradient
    # (as in train_denoise.py); the weights of data, l1, l3 and cl do not depend on u.
    def loss_u(uu, grad=False):
        yv = torch.tensor(eb_maps(uu), requires_grad=grad)
        per = [refs[f].frozen(yv[i], bias[f]) for i, f in enumerate(fields)]
        tot = sum(sum(p[t] for p in per) for t in TERMS if t != "l2")
        if grad:
            tot.backward()
            return float(tot), nat.front(yv.grad.numpy(), adjoint=True)
        return float(tot)

    _, gu = loss_u(u0, grad=True)
    v = rng.normal(size=u0.shape) * u0.std()
    dirn = float(np.sum(gu * v))
    fds = [(loss_u(u0 + h * v) - loss_u(u0 - h * v)) / (2 * h) for h in (1e-3, 1e-4)]
    efd = abs(fds[-1] - dirn) / abs(dirn)
    out["finite_difference"] = {"steps": [1e-3, 1e-4], "fd": fds, "gradient": dirn, "rel": efd}
    print(f"ebqu finite difference in u: {', '.join(f'{x:.8e}' for x in fds)} vs gradient {dirn:.8e}, rel {efd:.2e}",
          flush=True)
    assert efd < 1e-4, efd
    ddraws = (np.array([2, 0, 4, 5]), np.array([-1.0, 1.0, 1.0, -1.0]))
    hdraws = (np.array([1, 3, 5, 0]), np.array([1.0, -1.0, 1.0, 1.0]))
    nat.zero_grad()
    got = native_terms(nat.live("joint", ddraws, hdraws, backward=True))
    reference("ebqu live", lambda i, y: refs[fields[i]].live(y, ddraws, hdraws), got)
    if report:
        with open(report, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def check_spin(nside, j, channels, o, report=None):
    """--pol spin: the joint step on (Q, U) with the eth front end against ChiTerms (--pol-stats chi) and
    the EE/BB spectrum loss of train_denoise.py (--cl-spin 2), the same draws."""
    from stlia.train_denoise import ChiTerms
    channels = list(channels)
    dev = "cuda:0"
    qu = ("Q", "U")
    net = _torch_net(nside, channels, 4)
    w = dict(data=1.0, l1=1.0, l2=1.0, l3=1.0, cl=1.0)
    nat = NativeUNet(nside, channels, cond=2, maps=2, j=j, loss=w, backend=BACKEND)
    nat.load_state_dict(net.state_dict())
    ids = stat_masks(nside, STAT_CUTS)
    cond = _cond(o)
    nat.set_cond(cond)
    nat.set_common(o["t857"])
    d2 = _pair(o, "d")
    scales = d2.std(1, ddof=1)
    nat.set_joint(d2, ("E", "B"), front="eth", scales=scales, weight=0.5)
    nat.set_masks(ids)
    chi = {k: nat.front(_pair(o, k)) for k in ("d", "d1", "d2")}
    pools = [nat.front(np.stack([o["fields"][f]["pools"][h] for f in qu], 1)) for h in range(3)]  # [S, 2, npix]
    for i, f in enumerate(("E", "B")):
        nat.set_field(f, chi["d"][i], chi["d1"][i], chi["d2"][i], *[p[:, i] for p in pools])
    spectra = BinnedSpectra(nside, masks=CL_CUTS, backend=BACKEND, spin=2)
    cl = SpectrumLoss(spectra, _pair(o, "d1"), _pair(o, "d2"))
    # the reference: ChiTerms and BinnedSpectra(spin=2) of train_denoise.py
    T = torch.as_tensor(o["t857"], device=dev)
    tt = lambda x: torch.as_tensor(x, device=dev)
    fixed = {f: (None, tt(o["fields"][f]["d"]), (tt(o["fields"][f]["d1"]), tt(o["fields"][f]["d2"]))) for f in qu}
    noise = {f: types.SimpleNamespace(train=[tt(p) for p in o["fields"][f]["pools"]]) for f in qu}
    M, B = 6, 4
    a = types.SimpleNamespace(j=j, nside=nside, bias_realizations=M, bias_all=False, device=0)
    ref = ChiTerms(a, w, list(qu), fixed, noise, 6, B, {"masks": ids}, T, dev)
    rspec = sp.BinnedSpectra(nside, masks=CL_CUTS, device=dev, backend="stl", spin=2)
    rcl = sp.SpectrumLoss(rspec, tt(_pair(o, "d1")), tt(_pair(o, "d2")))
    queue = []  # [(sims, signs), position]: the draws, taken in turn (every term takes all of them)

    def draw(k, generator=None):
        sims, signs = queue[0]
        idx = (queue[1] + np.arange(k)) % len(sims)
        queue[1] += k
        s = torch.as_tensor(sims[idx], device=dev)
        g = torch.as_tensor(signs[idx], device=dev)[:, None, None]
        return tuple(torch.stack([p[h][s] for p in ref.pools], 1) * g for h in range(3))
    ref.draw = draw
    td, ts, tc = tt(d2)[None], tt(scales)[None, :, None], tt(cond)[None]
    net = net.to(dev)
    out = {"pol": "spin", "nside": nside, "j": j, "channels": channels}

    def native_terms(terms):
        u = nat.apply(d2, scale=scales)
        v, g = cl(u, grad=True)
        nat.apply(d2, scale=scales, cot=g)
        terms["cl"] = v
        return terms

    def reference(u, bias):
        terms = ref.terms(u, bias)
        terms["cl"] = rcl(rspec(u, u))
        return {t: v.cpu() for t, v in terms.items()}

    draws = (np.array([1, 3, 2, 4, 0, 5]), np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]))
    u = net(td, ts, cond=tc)[0]
    queue[:] = [draws, 0]
    bias = ref.estimate_bias(u.detach())
    nat.estimate_bias("joint", *draws)
    nat.zero_grad()
    got = native_terms(nat.loss("joint", backward=True))

    def check(name, ref_terms):
        total = sum(ref_terms.values())
        net.zero_grad()
        total.backward()
        rg = {x: q.grad.cpu().numpy().copy() for x, q in net.named_parameters()}
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
        assert max(et.values()) < 1e-4 and eg < (1e-3 if nside < 64 else 5e-3) and en < 1e-3, (name, et, eg, en)

    check("spin frozen", reference(u, bias))
    ddraws = (np.array([2, 0, 4, 5]), np.array([-1.0, 1.0, 1.0, -1.0]))
    hdraws = (np.array([1, 3, 5, 0]), np.array([1.0, -1.0, 1.0, 1.0]))
    u = net(td, ts, cond=tc)[0]
    nat.zero_grad()
    got = native_terms(nat.live("joint", ddraws, hdraws, backward=True))
    # ChiTerms.terms(u, None) draws the data noise, then the (n, n1, n2) of the l terms
    seq = iter([ddraws, hdraws])

    def draw_live(k, generator=None):
        queue[:] = [next(seq), 0]
        return draw(k)
    ref.draw = draw_live
    check("spin live", reference(u, None))
    if report:
        with open(report, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def test_joint_network():
    if torch is None:
        return
    check_network()
    check_front()


def test_joint_eb_matches_torch():
    if torch is None:
        return
    check_eb(16, 3, [8, 16, 24], _synthetic(16))


def test_joint_ebqu_matches_torch():
    if torch is None:
        return
    check_ebqu(16, 3, [8, 16, 24], _synthetic(16))


def test_joint_spin_matches_chiterms():
    if torch is None or BACKEND != "gpu" or not torch.cuda.is_available():
        return
    check_spin(16, 3, [8, 16, 24], _synthetic(16))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--pol", default="both", choices=["eb", "ebqu", "spin", "both"])
    p.add_argument("--out", default=None, help="report prefix: OUT_eb.json, OUT_spin.json")
    a = p.parse_args()
    if torch is None:
        sys.exit(0)
    if a.data:
        n = a.nside
        levels = int(np.log2(n // 8)) + 1
        ch, j = [32, 64, 96, 128, 128, 128, 128][:levels], int(np.log2(n)) - 1
        o = _pr3(a.data, n)
    else:
        test_joint_network()
        n, ch, j, o = 16, [8, 16, 24], 3, _synthetic(16)
    if a.pol in ("eb", "both"):
        check_eb(n, j, ch, o, report=a.out and f"{a.out}_eb.json")
    if a.pol in ("ebqu", "both"):
        check_ebqu(n, j, ch, o, report=a.out and f"{a.out}_ebqu.json")
    if a.pol in ("spin", "both") and BACKEND == "gpu":
        check_spin(n, j, ch, o, report=a.out and f"{a.out}_spin.json")
    print("native joint tests passed")
