"""Gradient of the ensemble loss at the initial maps: STL-Dev legacy autograd
against finite differences (per family) and against the Rust gradient of
`cargo run --release -p stl-core --example ensemble_gradient`.

    python gradient_check.py --stl-dev DIR --target T_f64.npy --init init --ensemble 4 \
        --J 6 --rust grad_rust.f64 --device cuda

The init files are PREFIX_mMMM.npy (standardised units, make_inits.py).
"""

import argparse
import os
import shutil
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stl-dev", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--init", required=True, help="prefix of PREFIX_mMMM.npy")
    p.add_argument("--ensemble", type=int, default=4)
    p.add_argument("--J", type=int, required=True)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--rust", default=None, help="raw f64 [B, N] gradient of the Rust example")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    # Exact geometry: no float32 reload from a previous process (legacy_synth.py).
    scratch = tempfile.mkdtemp(prefix="hpxconv_")
    os.environ["HEALPIXCONV_CACHE"] = scratch
    sys.path.insert(0, args.stl_dev)
    import numpy as np
    import torch
    from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch

    raw = np.load(args.target).astype(np.float64)
    t = torch.as_tensor((raw - raw.mean()) / raw.std()).to(args.device)
    tgt = STL_Healpix_Kernel_Torch(t)
    op = tgt.get_ST_op(J=args.J, L=args.L, norm="vanilla", compute_PS=False, has_fewer_convolutions=False,
                       wavelet_op_kwargs=dict(kernel_size=5, ellipsoid="sphere"))
    plan = op.build_plan()
    fam = np.array([c.family for c in plan])

    def values(x):
        s = op.apply(tgt.new_like(x))
        return torch.stack([getattr(s, c.family)[c.legacy_index] for c in plan], -1)

    with torch.no_grad():
        tv = values(t.reshape(1, 1, -1))[0]
    B = args.ensemble
    u = torch.as_tensor(np.stack([np.load(f"{args.init}_m{m:03d}.npy") for m in range(B)])).reshape(B, 1, -1).to(args.device)

    def loss(x, mask=None):
        r = values(x).mean(0) - tv
        return (r if mask is None else r * mask).abs().square().sum()

    grad_all = None
    for f in (None, "mean", "var", "S1", "S2", "S3", "S4"):
        mask = None if f is None else torch.as_tensor(fam == f).to(args.device)
        x = u.clone().requires_grad_(True)
        value = loss(x, mask)
        (g,) = torch.autograd.grad(value, x)
        if f is None:
            grad_all = g.reshape(B, -1).cpu().numpy()
        # Central differences along -g, Richardson-extrapolated.
        v = -g / g.norm()
        eps = 1e-3 * float(value) / float(g.norm())
        with torch.no_grad():
            d1 = (loss(u + eps * v, mask) - loss(u - eps * v, mask)) / (2 * eps)
            d2 = (loss(u + 2 * eps * v, mask) - loss(u - 2 * eps * v, mask)) / (4 * eps)
        fd = float((4 * d1 - d2) / 3)
        ad = float((g * v).sum())
        print(f"{f or 'all'}: loss {float(value):.15e}, autograd {ad:.15e}, finite differences {fd:.15e}, rel {abs(ad - fd) / abs(ad):.1e}")
    if args.rust:
        gr = np.fromfile(args.rust).reshape(B, -1)
        d = gr - grad_all
        print(f"Rust vs legacy gradient: |d| / |g| {np.linalg.norm(d) / np.linalg.norm(grad_all):.2e}, "
              f"max |d| / max |g| {np.abs(d).max() / np.abs(grad_all).max():.2e}, "
              f"cos {float((gr * grad_all).sum() / np.linalg.norm(gr) / np.linalg.norm(grad_all)):.16f}")
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
