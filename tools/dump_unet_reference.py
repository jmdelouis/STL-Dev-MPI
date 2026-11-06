#!/usr/bin/env python3
"""Fixtures of the native U-Net: stlia.unet.UNet (Healpix-STL-IA, torch,
conv="sphere", norm="group") with random weights, its output u for a batch
of maps and the parameter gradients of sum(cot * u).

    python tools/dump_unet_reference.py --stlia PATH/Healpix-STL-IA --out fixtures/unet/n16

Writes config.json, d.npy [B, npix] (float64), cond.npy [cond, npix]
(float32), cot.npy [B, npix], u.npy [B, npix], scale.npy, and per parameter
p_<name>.npy and g_<name>.npy (state_dict names and shapes), plus the
gradient after one Adam step (adam_<name>.npy: the parameters after the
step, lr 1e-3).
"""
import argparse
import json
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--stlia", required=True)
p.add_argument("--out", required=True)
p.add_argument("--nside", type=int, default=16)
p.add_argument("--channels", default="8,16,24")
p.add_argument("--cond", type=int, default=1)
p.add_argument("--batch", type=int, default=2)
p.add_argument("--gauge", default="phi")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--double", default=None,
               help="a fixture directory: the same parameters and inputs run in float64 (the exact reference "
                    "of the float32 implementations); outputs and gradients saved as there")
a = p.parse_args()
sys.path.insert(0, a.stlia)
import torch  # noqa: E402
if a.double:
    # stlia.unet runs in float32 by construction: float64 everywhere for this reference.
    torch.Tensor.float = torch.Tensor.double
    torch.float32 = torch.float64
from stlia.unet import UNet  # noqa: E402

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
channels = [int(c) for c in a.channels.split(",")]
net = UNet(a.nside, len(channels), channels, gauge=a.gauge, cond=a.cond).double().float()
with torch.no_grad():
    for name, q in net.named_parameters():
        if ".n" in name and name.endswith("weight"):
            q.copy_(1 + 0.1 * torch.randn_like(q))
        elif name.endswith("bias") or name.startswith("head"):
            q.copy_(0.1 * torch.randn_like(q))
npix = 12 * a.nside**2
d = torch.as_tensor(rng.normal(size=(a.batch, npix)) * 3.0 + 1.0)
cond = torch.as_tensor(rng.normal(size=(a.cond, npix)), dtype=torch.float32)
cot = torch.as_tensor(rng.normal(size=(a.batch, npix)))
if a.double:
    net = net.double()
    src = a.double
    with torch.no_grad():
        for name, q in net.named_parameters():
            q.copy_(torch.as_tensor(np.load(os.path.join(src, f"p_{name}.npy")), dtype=torch.float64))
    d = torch.as_tensor(np.load(os.path.join(src, "d.npy")))
    cond = torch.as_tensor(np.load(os.path.join(src, "cond.npy")), dtype=torch.float64)
    cot = torch.as_tensor(np.load(os.path.join(src, "cot.npy")))
u = net(d, cond=cond[None].expand(a.batch, -1, -1) if a.cond else None)
(cot * u).sum().backward()
os.makedirs(a.out, exist_ok=True)
names = [n for n, _ in net.named_parameters()]
with open(os.path.join(a.out, "config.json"), "w") as f:
    json.dump({**net.config(), "batch": a.batch, "params": names,
               "shapes": {n: list(q.shape) for n, q in net.named_parameters()}}, f, indent=1)
np.save(os.path.join(a.out, "d.npy"), d.numpy())
np.save(os.path.join(a.out, "cond.npy"), cond.numpy().astype(np.float32))
np.save(os.path.join(a.out, "cot.npy"), cot.numpy())
np.save(os.path.join(a.out, "u.npy"), u.detach().numpy())
np.save(os.path.join(a.out, "scale.npy"), np.array(float(d.std())))
for n, q in net.named_parameters():
    np.save(os.path.join(a.out, f"p_{n}.npy"), q.detach().numpy().astype(np.float32))
    np.save(os.path.join(a.out, f"g_{n}.npy"), q.grad.numpy().astype(np.float32))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
opt.step()
for n, q in net.named_parameters():
    np.save(os.path.join(a.out, f"adam_{n}.npy"), q.detach().numpy().astype(np.float32))
print("wrote", a.out, "params", sum(q.numel() for q in net.parameters()))
