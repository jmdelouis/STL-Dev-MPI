#!/usr/bin/env python3
"""The cl term in the native trainer (NativeUNet.set_cl) against the driver's
Python path (stl_port.denoise.SpectrumLoss over BinnedSpectra, an extra
network forward and backward with the map cotangent), for every --pol mode of
train_native_unet.py:

- qu: one field, the scalar spectrum of u;
- eb: the joint network on the E and B maps, the scalar spectra of both outputs, summed;
- ebqu: the joint network on (Q, U), the scalar spectra of the harmonic E and B
  maps of the output (front "eb"), summed;
- spin: the joint network on (Q, U) (front "eth"), the EE and BB spectra of the output.

Two networks with the same parameters, the same frozen biases and the same
draws: the cl value, the total and the parameter gradients of one step
(frozen and live biases) must agree to float32 accumulation (the Python path
runs the network backward twice).

    STL_TEST_BACKEND=gpu python tests/test_native_cl.py [--nside 16]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeUNet
from stl_port.denoise import BinnedSpectra, SpectrumLoss, eb_maps, stat_masks

BACKEND = os.environ.get("STL_TEST_BACKEND", "cpu")
CL_CUTS = (0.0, 30.0)


def _synthetic(nside, seed=0, sims=6):
    rng = np.random.default_rng(seed)
    npix = 12 * nside**2
    sky = {f: rng.normal(size=npix) * (2.0 if f == "Q" else 1.5) for f in ("Q", "U")}
    out = {"t857": np.exp(sky["Q"] * 0.3 + rng.normal(size=npix))}
    for f in ("Q", "U"):
        pools = [rng.normal(size=(sims, npix)) * 0.5 * (1 + 0.3 * np.arange(npix) / npix) for _ in range(3)]
        out[f] = dict(d=sky[f] + pools[0][0] * 1.1, d1=sky[f] + pools[1][0] * 1.1, d2=sky[f] + pools[2][0] * 1.1,
                      pools=pools)
    return out


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))


def setup(pol, nside, j, channels, o, engine_cl):
    """A network and its training data as train_native_unet.py --pol `pol`; with
    engine_cl the cl term in the trainer, else the Python cl of the driver (returned)."""
    w = dict(data=1.0, l1=1.0, l2=1.0, l3=1.0, cl=1.0)
    joint = pol != "qu"
    net = NativeUNet(nside, channels, maps=2 if joint else 1, j=j, loss=w, backend=BACKEND, seed=3)
    net.set_masks(stat_masks(nside, (0.0, 30.0)))
    net.set_common(o["t857"])
    spectra = BinnedSpectra(nside, masks=CL_CUTS, lmin=4, width=6, backend=BACKEND, spin=2 if pol == "spin" else 0)
    pair = lambda k: np.stack([o[f][k] for f in ("Q", "U")])  # noqa: E731
    cl_loss, front = {}, None
    if not joint:
        f = "Q"
        net.set_field(f, o[f]["d"], o[f]["d1"], o[f]["d2"], *o[f]["pools"])
        cl_loss[f] = SpectrumLoss(spectra, o[f]["d1"], o[f]["d2"])
        x, scales, name = o[f]["d"], None, f
    else:
        d_qu = pair("d")
        hm_qu = [pair("d1"), pair("d2")]
        pools_qu = [np.stack([o[f]["pools"][h] for f in ("Q", "U")], 1) for h in range(3)]
        if pol == "eb":
            tf = eb_maps
            x = tf(d_qu)
            net.set_joint(x, ("E", "B"))
        else:
            x = d_qu
            front = "eth" if pol == "spin" else "eb"
            net.set_joint(x, ("E", "B"), front=front)
            tf = net.front
        scales, name = None, "joint"
        d_f, hm_f, pools_f = tf(d_qu), [tf(h) for h in hm_qu], [tf(p) for p in pools_qu]
        for i, f in enumerate(("E", "B")):
            net.set_field(f, d_f[i], hm_f[0][i], hm_f[1][i], *[np.ascontiguousarray(p[:, i]) for p in pools_f])
            if pol != "spin":
                cl_loss[f] = SpectrumLoss(spectra, hm_f[0][i], hm_f[1][i])
        if pol == "spin":
            cl_loss["pair"] = SpectrumLoss(spectra, *hm_qu)
    if engine_cl:
        kw = dict(d1=pair("d1"), d2=pair("d2")) if pol == "spin" else {}
        net.set_cl(spectra.m, spectra.edges, lmax=spectra.lmax, spin=2 if pol == "spin" else 0, **kw)

    def python_cl(terms):
        """train_native_unet.py add_cl (backward): the cl of the output and its cotangent."""
        u = net.apply(x, scale=scales)
        if pol == "spin":
            v, g = cl_loss["pair"](u, grad=True)
        elif not joint:
            v, g = cl_loss["Q"](u, grad=True)
        else:
            y = net.front(u) if pol == "ebqu" else u
            parts = [cl_loss[f](y[i], grad=True) for i, f in enumerate(("E", "B"))]
            v = sum(p for p, _ in parts)
            g = np.stack([q for _, q in parts])
            g = net.front(g, adjoint=True) if pol == "ebqu" else g
        net.apply(x, scale=scales, cot=g)
        terms["cl"] = v
        terms["total"] += v
        return terms

    return net, name, python_cl


def check(pol, nside=16, j=3, channels=(8, 16, 24), report=None):
    o = _synthetic(nside)
    channels = list(channels)
    a, name, python_cl = setup(pol, nside, j, channels, o, engine_cl=False)
    b, _, _ = setup(pol, nside, j, channels, o, engine_cl=True)
    rng = np.random.default_rng(11)
    draws = (rng.integers(0, 6, 8), rng.choice([-1.0, 1.0], 8))
    live = [(rng.permutation(6)[:3], rng.choice([-1.0, 1.0], 3)) for _ in range(2)]
    out = {}
    for mode in ("frozen", "live"):
        for net in (a, b):
            net.estimate_bias(name, *draws)
            net.zero_grad()
        if mode == "frozen":
            ta = python_cl(a.loss(name, backward=True))
            tb = b.loss(name, backward=True)
        else:
            ta = python_cl(a.live(name, *live, backward=True))
            tb = b.live(name, *live, backward=True)
        ga, gb = a.grads(), b.grads()
        ecl, etot = abs(tb["cl"] - ta["cl"]) / abs(ta["cl"]), abs(tb["total"] - ta["total"]) / abs(ta["total"])
        eg = _rel(gb, ga)
        en = float(np.linalg.norm(gb - ga) / np.linalg.norm(ga))
        out[mode] = dict(cl_python=ta["cl"], cl_native=tb["cl"], cl_rel=ecl, total_rel=etot, gradients_max_rel=eg,
                         gradients_norm_rel=en)
        print(f"{pol} {mode}: cl {ta['cl']:.6g} vs {tb['cl']:.6g} (rel {ecl:.1e}), total rel {etot:.1e}, "
              f"gradients max rel {eg:.1e}, norm rel {en:.1e}", flush=True)
        # float32 network: two backward passes (Python path) against one; the host network sums in another order
        assert ecl < 1e-9 and etot < 1e-9 and eg < 1e-5 and en < 1e-5, (pol, mode, ecl, etot, eg, en)
    if report is not None:
        report[pol] = out
    return out


def test_native_cl_matches_python():
    for pol in ("qu", "eb", "ebqu", "spin"):
        check(pol)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--pol", default="qu,eb,ebqu,spin")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rep = {}
    for pol in a.pol.split(","):
        check(pol, nside=a.nside, j=a.nside.bit_length() - 2, report=rep)
    if a.out:
        import json
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)
    print("native cl tests passed")
