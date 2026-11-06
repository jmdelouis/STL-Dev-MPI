#!/usr/bin/env python3
"""NativeUNet (stl_port.native) against the torch U-Net of Healpix-STL-IA:
network output and parameter gradients, then one full training step with
frozen noise biases (data, l1, l2, l3 of stlia train_denoise.py) against
the same step written with torch autograd over NativeEngine statistics
(the reference implementation of Healpix-STL-IA), then the Adam update
and an L-BFGS round. Needs torch and the stlia package (STLIA_PATH, default
../Healpix-STL-IA next to STL-Dev-MPI); STL_TEST_BACKEND=gpu runs the
native side on the GPU. Runs under pytest or as a script."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeEngine, NativeUNet

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("STLIA_PATH", str(ROOT.parent / "Healpix-STL-IA")))
try:
    import torch
    from stlia.unet import UNet
    from stlia.stl_torch import DenoiseLoss, StatsFunction, as_real
    from stlia.train_denoise import CrossStats, unbiased
except ImportError as e:  # pragma: no cover
    torch = None
    print(f"torch or stlia missing ({e}): skipped")

NSIDE, J, CH = 16, 3, [8, 16, 24]
NPIX = 12 * NSIDE**2
BACKEND = os.environ.get("STL_TEST_BACKEND", "cpu")


def _rel(a, b, floor=0.0):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), floor))


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = UNet(NSIDE, len(CH), CH, cond=1)
    with torch.no_grad():
        for name, q in net.named_parameters():
            if name.startswith("head"):
                q.copy_(0.05 * torch.randn_like(q))
    sky = rng.normal(size=NPIX) * 2.0
    sims = 5
    n0, n1, n2 = (rng.normal(size=(sims, NPIX)) * 0.5 for _ in range(3))
    d, d1, d2 = sky + n0[0] * 1.1, sky + n1[0] * 1.1, sky + n2[0] * 1.1
    t857 = sky * 0.7 + rng.normal(size=NPIX)
    cond = rng.normal(size=(1, NPIX)).astype(np.float32)
    draws = (np.array([1, 3, 2, 4]), np.array([1.0, -1.0, -1.0, 1.0]))
    return net, dict(d=d, d1=d1, d2=d2, t857=t857, n0=n0, n1=n1, n2=n2, cond=cond, draws=draws)


def _torch_step(net, o, bias_all=False):
    """Frozen-bias loss terms and parameter gradients, as train_denoise.py
    (Healpix-STL-IA >= a2141c1: no bias on the coefficients of unbiased(),
    unless bias_all)."""
    one = NativeEngine(NSIDE, J, batch=1)
    xeng = NativeEngine(NSIDE, J, batch=5, cross=True)
    loss_fn = DenoiseLoss(one.plan, NSIDE)
    xloss = DenoiseLoss(xeng.plan, NSIDE)
    xrun = {"engine": xeng, "generation": 0}
    T = torch.as_tensor(o["t857"])
    d, d1, d2 = (torch.as_tensor(o[k]) for k in ("d", "d1", "d2"))
    cond = torch.as_tensor(o["cond"])[None]
    u = net(d[None], d.std(), cond=cond)[0]

    def auto(x):
        v, _ = one.forward(x[None])
        return torch.as_tensor(as_real(v[0]))

    def cross(a, b):
        e = NativeEngine(NSIDE, J, batch=len(a), cross=True)
        v, _ = e.forward_cross(np.stack(a), np.stack(b))
        return torch.as_tensor(as_real(v))

    with torch.no_grad():
        ud = u.detach().numpy()
        sims, signs = o["draws"]
        n0 = [s * o["n0"][k] for k, s in zip(sims, signs)]
        n1 = [s * o["n1"][k] for k, s in zip(sims, signs)]
        n2 = [s * o["n2"][k] for k, s in zip(sims, signs)]
        m = len(sims)
        bias = {"data": sum(auto(ud + n) for n in n0) / m - auto(ud)}
        uu, tu = cross([ud, o["t857"]], [ud, ud])
        bias["l1"] = cross([ud + a for a in n1], [ud + b for b in n2]).mean(0) - uu
        bias["l2"] = cross([ud + a for a in n0], [ud] * m).mean(0) - uu
        bias["l3"] = cross([o["t857"]] * m, [ud + a for a in n0]).mean(0) - tu
        if not bias_all:
            for t in bias:
                bias[t][torch.as_tensor(unbiased(one.plan if t == "data" else xeng.plan, t))] = 0
    vt, tv = one.forward(o["d"][None], moments=True)
    target, tvar = torch.as_tensor(as_real(vt[0])), torch.as_tensor(tv[0])
    model = StatsFunction.apply(u[None], one, 1)[0] + bias["data"]
    terms = {"data": loss_fn(model[None], target, tvar)}
    st = CrossStats.apply(torch.stack([d1, d, T, u, T]), torch.stack([d2, u, d, u, u]), xrun)
    var = xrun["var"]
    models = {"l1": st[3], "l2": st[3], "l3": st[4]}
    for t, name in enumerate(("l1", "l2", "l3")):
        target = st[t] if name == "l2" else st[t].detach()
        terms[name] = xloss((models[name] + bias[name])[None], target, var[t])
    total = sum(terms.values())
    net.zero_grad()
    total.backward()
    return {k: float(v) for k, v in terms.items()} | {"total": float(total)}, {n: q.grad.numpy().copy() for n, q in net.named_parameters()}


def _native(net, o, **kw):
    nat = NativeUNet(NSIDE, CH, cond=1, j=J, loss=dict(data=1, l1=1, l2=1, l3=1), backend=BACKEND, **kw)
    nat.load_state_dict(net.state_dict())
    nat.set_cond(o["cond"])
    nat.set_common(o["t857"])
    nat.set_field("Q", o["d"], o["d1"], o["d2"], o["n0"], o["n1"], o["n2"])
    nat.estimate_bias("Q", *o["draws"])
    return nat


def test_network_matches_torch():
    if torch is None:
        return
    net, o = _setup(1)
    d = np.stack([o["d"], o["d1"]])
    cot = np.random.default_rng(2).normal(size=d.shape)
    u = net(torch.as_tensor(d), cond=torch.as_tensor(o["cond"])[None].expand(2, -1, -1))
    (torch.as_tensor(cot) * u).sum().backward()
    with NativeUNet(NSIDE, CH, cond=1, backend=BACKEND) as nat:
        nat.load_state_dict(net.state_dict())
        sd = nat.state_dict()
        assert all(np.array_equal(sd[n], q.detach().numpy()) for n, q in net.named_parameters())
        nat.set_cond(o["cond"])
        un = nat.apply(d, cot=cot)
        r, rt = d - un, d - u.detach().numpy()
        assert _rel(r, rt) < 1e-5, _rel(r, rt)
        g = nat.grads_dict()
        floor = 1e-2 * max(np.abs(q.grad.numpy()).max() for q in net.parameters())
        err = max(_rel(g[n], q.grad.numpy(), floor) for n, q in net.named_parameters())
        print(f"network: correction {_rel(r, rt):.2e}, gradients {err:.2e}")
        assert err < 1e-4


def test_training_step_bias_all_matches_torch():
    """--bias-all: the bias of every coefficient estimated."""
    if torch is None:
        return
    net, o = _setup(3)
    ref_terms, ref_grads = _torch_step(net, o, bias_all=True)
    with _native(net, o, bias_all=True) as nat:
        nat.zero_grad()
        terms = nat.loss("Q", backward=True)
        g = nat.grads_dict()
        floor = 1e-2 * max(np.abs(v).max() for v in ref_grads.values())
        eg = max(_rel(g[n], ref_grads[n], floor) for n in ref_grads)
        et = max(abs(terms[k] - ref_terms[k]) / abs(ref_terms[k]) for k in ref_terms)
        print(f"training step, bias_all: terms {et:.2e}, gradients {eg:.2e}")
        assert et < 1e-4 and eg < 1e-3


def test_training_step_matches_torch():
    if torch is None:
        return
    net, o = _setup(3)
    ref_terms, ref_grads = _torch_step(net, o)
    with _native(net, o) as nat:
        nat.zero_grad()
        terms = nat.loss("Q", backward=True)
        g = nat.grads_dict()
        floor = 1e-2 * max(np.abs(v).max() for v in ref_grads.values())
        eg = max(_rel(g[n], ref_grads[n], floor) for n in ref_grads)
        et = max(abs(terms[k] - ref_terms[k]) / abs(ref_terms[k]) for k in ref_terms)
        print(f"training step: terms {et:.2e} ({terms}), gradients {eg:.2e}")
        assert et < 1e-4 and eg < 1e-3
        # Adam from the same state and the same gradients (a first step moves
        # every parameter by about lr: the rounding-level gradients of the
        # biases before a one-channel GroupNorm would flip signs).
        with torch.no_grad():
            for n, q in net.named_parameters():
                q.grad.copy_(torch.as_tensor(g[n]))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        opt.step()
        nat.adam_step(1e-3)
        sd = nat.state_dict()
        ea = max(float(np.abs(sd[n] - q.detach().numpy()).max()) for n, q in net.named_parameters())
        print(f"Adam step: {ea:.2e}")
        assert ea < 1e-6


def test_live_step_matches_torch():
    """Live biases (train_denoise.py losses() and delouis()), the same draws."""
    if torch is None:
        return
    net, o = _setup(5)
    ddraws = (np.array([2, 0, 4]), np.array([-1.0, 1.0, 1.0]))
    hdraws = (np.array([1, 3]), np.array([1.0, -1.0]))
    one = NativeEngine(NSIDE, J, batch=1)
    k, kh = len(ddraws[0]), len(hdraws[0])
    many = NativeEngine(NSIDE, J, batch=k)
    xeng = NativeEngine(NSIDE, J, batch=3 + 3 * kh, cross=True)
    loss_fn, xloss = DenoiseLoss(one.plan, NSIDE), DenoiseLoss(xeng.plan, NSIDE)
    xrun = {"engine": xeng, "generation": 0}
    T = torch.as_tensor(o["t857"])
    d, d1, d2 = (torch.as_tensor(o[x]) for x in ("d", "d1", "d2"))
    u = net(d[None], d.std(), cond=torch.as_tensor(o["cond"])[None])[0]
    pick = lambda pool, dr: torch.as_tensor(np.stack([g * o[pool][s] for s, g in zip(*dr)]))
    vt, tv = one.forward(o["d"][None], moments=True)
    st = StatsFunction.apply(u + pick("n0", ddraws), many, 1)
    terms = {"data": loss_fn(st, torch.as_tensor(as_real(vt[0])), torch.as_tensor(tv[0]))}
    n, n1, n2 = pick("n0", hdraws), pick("n1", hdraws), pick("n2", hdraws)
    A = torch.cat([d1[None], d[None], T[None], u + n1, u + n, T.expand(kh, -1)])
    Bm = torch.cat([d2[None], u[None], d[None], u + n2, u.expand(kh, -1), u + n])
    xs = CrossStats.apply(A, Bm, xrun)
    for t, name in enumerate(("l1", "l2", "l3")):
        target = xs[t] if name == "l2" else xs[t].detach()
        terms[name] = xloss(xs[3 + t * kh:3 + (t + 1) * kh], target, xrun["var"][t])
    total = sum(terms.values())
    net.zero_grad()
    total.backward()
    ref = {x: float(v) for x, v in terms.items()} | {"total": float(total)}
    with _native(net, o) as nat:
        nat.zero_grad()
        got = nat.live("Q", ddraws, hdraws, backward=True)
        g = nat.grads_dict()
        floor = 1e-2 * max(np.abs(q.grad.numpy()).max() for q in net.parameters())
        eg = max(_rel(g[x], q.grad.numpy(), floor) for x, q in net.named_parameters())
        et = max(abs(got[x] - ref[x]) / abs(ref[x]) for x in ref)
        print(f"live step: terms {et:.2e}, gradients {eg:.2e}")
        assert et < 1e-4 and eg < 1e-3


def test_lbfgs_round_decreases_the_loss():
    if torch is None:
        return
    net, o = _setup(4)
    with _native(net, o) as nat:
        start = nat.loss("Q")["total"]
        report = nat.lbfgs("Q", {"max_iter": 5, "history_size": 5})
        end = nat.loss("Q")["total"]
        print(f"L-BFGS round: {start:.4g} -> {end:.4g} in {report['evaluations']} evaluations")
        assert end < start and abs(report["loss_history"][0] - start) <= 1e-12 * abs(start)


if __name__ == "__main__":
    test_network_matches_torch()
    test_training_step_matches_torch()
    test_training_step_bias_all_matches_torch()
    test_live_step_matches_torch()
    test_lbfgs_round_decreases_the_loss()
    print("native U-Net tests passed")
