"""torch.autograd wrappers of the differentiable engines of stl_port.native.

    scattering(engine, x)   NativeEngine: x [B, npix] ([B, 2, npix] cross) -> complex [B, (K,) ncoeff]
    patches(engine, x)      PatchEngine after select(): full-sky x [npix] ([2, npix] cross) -> complex [B, ncoeff]
    discs(engine, x)        PatchEngine after select(): one patch per disc, x [B, n] ([B, 2, n] cross) -> complex [B, ncoeff]
    spectrum(engine, x)     SpectrumEngine: x shaped as engine.forward expects -> real C_l

The engines run on the host (backend "cpu" or "gpu"): the tensors are copied
to numpy and the results back to the device of x. Complex outputs are
complex128 tensors; torch hands their backward the cotangent dL/dRe + i dL/dIm,
which is the convention of the engines' vjp.

An engine keeps the state of its last forward only: run backward before the
next forward of the same engine (one engine per term of a loss, or one batch
for several terms). The generation counter makes a stale backward an error.
"""
import itertools

import numpy as np
import torch

_generation = itertools.count(1)


class _Engine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward, vjp):
        g = next(_generation)
        y = forward(x.detach().cpu().numpy(), g)
        ctx.vjp, ctx.generation = vjp, g
        return torch.from_numpy(np.ascontiguousarray(y)).to(x.device)

    @staticmethod
    def backward(ctx, gy):
        gx = ctx.vjp(gy.detach().cpu().numpy(), ctx.generation)
        return torch.from_numpy(np.ascontiguousarray(gx)).to(gy.device), None, None


def scattering(engine, x):
    """Statistics of NativeEngine (full sky, masks, cross) as a torch op."""
    return _Engine.apply(x, lambda a, g: engine.forward(a, generation=g)[0],
                         lambda c, g: engine.vjp(c, generation=g))


def spectrum(engine, x):
    """C_l of SpectrumEngine (spin 0 or s, masks, cross) as a torch op."""
    return _Engine.apply(x, lambda a, g: engine.forward(a, generation=g),
                         lambda c, g: engine.vjp(c, generation=g))


def patches(engine, x):
    """Statistics of the discs selected in PatchEngine, read from the
    full-sky map x [npix] ([2, npix] with cross=True); the gradient is
    scattered back to the full sky by torch's indexing."""
    ids = [torch.from_numpy(i).to(x.device) for i in engine.ids]
    flat = torch.cat([x[..., i].reshape(-1) for i in ids])
    sizes = [x[..., i].numel() for i in ids]
    shapes = [tuple(x[..., i].shape) for i in ids]

    def forward(a, g):
        parts = np.split(a, np.cumsum(sizes)[:-1])
        return engine.forward([p.reshape(s) for p, s in zip(parts, shapes)], generation=g)[0]

    def vjp(c, g):
        return np.concatenate([np.ravel(p) for p in engine.vjp(c, generation=g)])

    return _Engine.apply(flat, forward, vjp)


def discs(engine, x):
    """Statistics of a dense batch of patches, x[i] being the patch of disc i
    ([B, n] or [B, 2, n] with cross=True, every disc with n pixels; with
    discs of 180 deg, n = npix and the patches are full-sky maps)."""
    return _Engine.apply(x, lambda a, g: engine.forward(a, generation=g)[0],
                         lambda c, g: engine.vjp(c, generation=g))
