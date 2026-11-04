#!/usr/bin/env python3
"""PatchEngine(backend="gpu") against the CPU engine: values, variances and
gradients of auto and cross statistics with several patches per disc, and
the device-tensor path (torch) against the host one.  Needs a library built
with --features gpu and a CUDA device; skipped otherwise."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from stl_port.native import PatchEngine

NSIDE, J = 32, 4


def gpu_engine(**kw):
    try:
        return PatchEngine(NSIDE, J, backend="gpu", **kw)
    except Exception as e:  # no GPU build or no device
        pytest.skip(f"GPU patch engine unavailable: {e}")


def rel(a, b):
    a, b = np.concatenate([np.ravel(x) for x in a]), np.concatenate([np.ravel(x) for x in b])
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("cross", [False, True])
def test_gpu_matches_cpu(cross):
    rng = np.random.default_rng(3)
    centres = np.repeat(rng.normal(size=(4, 3)), [3, 1, 2, 4], axis=0)
    x = rng.normal(size=(2, 12 * NSIDE**2))
    with PatchEngine(NSIDE, J, cross=cross) as cpu, gpu_engine(cross=cross) as gpu:
        for e in (cpu, gpu):
            e.select(centres, 35.0, 20.0)
        if cross:
            a, b = cpu.gather(x[0]), cpu.gather(x[1])
            vc, sc = cpu.forward_cross(a, b)
            vg, sg = gpu.forward_cross(a, b)
        else:
            a = cpu.gather(x[0])
            vc, sc = cpu.forward(a)
            vg, sg = gpu.forward(a)
        cot = rng.normal(size=vc.shape) + 1j * rng.normal(size=vc.shape)
        gc, gg = cpu.vjp(cot), gpu.vjp(cot)
        assert rel([vg], [vc]) < 1e-12 and rel([sg], [sc]) < 1e-11
        assert rel(gg, gc) < 1e-11
        assert gpu.info()["backend"] == "gpu"


def test_device_tensors_match_host():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device for torch")
    rng = np.random.default_rng(4)
    with gpu_engine(cross=True) as e:
        ids = e.select(rng.normal(size=(5, 3)), 30.0, 15.0)
        x = rng.normal(size=(2, 12 * NSIDE**2))
        pairs = [x[:, i] for i in ids]
        v, s = e.forward(pairs, generation=1)
        cot = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
        g = e.vjp(cot, generation=1)
        dev = f"cuda:{e.info()['device']['ordinal']}"
        xt = torch.cat([torch.from_numpy(p.reshape(-1)) for p in pairs]).to(dev)
        vd, sd = e.forward_device(xt, generation=2)
        gd = e.vjp_device(cot, generation=2).cpu().numpy()
        np.testing.assert_array_equal(vd, v)
        np.testing.assert_array_equal(sd, s)
        np.testing.assert_array_equal(gd, np.concatenate([p.reshape(-1) for p in g]))
