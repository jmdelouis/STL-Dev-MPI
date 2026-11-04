"""Directional derivative of the ensemble loss through `StatsFunction`
against a central finite difference, and through the generator.

    python check_grad.py [--nside 16 --batch 4 --device 0 --backend gpu]
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python", "stl_ensemble"))
from generator import Generator  # noqa: E402
from stl_torch import EnsembleLoss, Stats  # noqa: E402
from stl_port.native import NativeEngine  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--nside", type=int, default=16)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--device", type=int, default=0)
p.add_argument("--backend", default="gpu")
a = p.parse_args()
n, b = a.nside, a.batch
j = int(np.log2(n)) - 1
rng = np.random.default_rng(1)
t = rng.standard_normal(12 * n * n)
e1 = NativeEngine(n, j, 1, backend=a.backend, device=a.device)
tv, tvar = e1.forward(t[None], moments=True)
eng = NativeEngine(n, j, b, backend=a.backend, device=a.device)
stats = Stats(eng)
loss_fn = EnsembleLoss(tv[0], tvar[0], eng.plan, n, b, 1.0)


def loss_of(x):
    return loss_fn(stats(torch.as_tensor(x)))[0].item()


x0 = rng.standard_normal((b, 12 * n * n)) * 1.3
v = rng.standard_normal(x0.shape)
x = torch.tensor(x0, requires_grad=True)
loss = loss_fn(stats(x))[0]
loss.backward()
analytic = float((x.grad.numpy() * v).sum())
for h in (1e-3, 1e-4, 1e-5):
    fd = (loss_of(x0 + h * v) - loss_of(x0 - h * v)) / (2 * h)
    print(f"maps: h {h:g} fd {fd:.12g} analytic {analytic:.12g} rel {abs(fd - analytic) / abs(analytic):.2e}")

# Through the generator (float64 on the CPU for a clean finite difference).
torch.manual_seed(0)
gen = Generator(n, 4).double()
z = gen.noise(b).double()
q = gen.heads[-1].weight
loss = loss_fn(stats(gen(z)))[0]
loss.backward()
d = torch.randn_like(q)
analytic = float((q.grad * d).sum())
with torch.no_grad():
    w0 = q.clone()
    vals = []
    for s in (1, -1):
        q.copy_(w0 + s * 1e-5 * d)
        vals.append(loss_fn(stats(gen(z)))[0].item())
    q.copy_(w0)
fd = (vals[0] - vals[1]) / 2e-5
print(f"generator head: fd {fd:.12g} analytic {analytic:.12g} rel {abs(fd - analytic) / abs(analytic):.2e}")
