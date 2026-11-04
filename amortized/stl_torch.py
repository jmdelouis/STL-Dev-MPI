"""The Rust scattering-covariance engine as a PyTorch layer, and the
ensemble loss of `stl-synth --ensemble B --weights ergodic --scatter-weight X`.

`StatsFunction` wraps `NativeEngine.forward` / `.vjp` of `stl-bridge`
(`python/stl_ensemble/stl_port/native.py`): statistics of a batch of maps
`[B, 12 nside^2]` (float64) as a real tensor `[B, ncoeff, 2]` (real and
imaginary parts), and in the backward pass the exact map cotangent from the
pyramidal reverse sweep.  The bridge's convention for complex coefficients,
`Re(sum(conj(cotangent) dS))`, is the one of a real loss differentiated by
torch with respect to the real and imaginary parts:
`cotangent = dL/dRe + i dL/dIm`.

`EnsembleLoss` is `stl_mpi::synthesis::Ensemble` with `ergodic_weights`,
transcribed: `sigma_i^2 = V_i / N_i` (spatial variance of the terms of
coefficient `i` on the target over the pixels of its native level),
`w_i = B / (n sigma_i^2)`,

    loss = sum_i w_i |mean_m S_{m,i} - T_i|^2
         + X / n sum_i (D_i / sigma_i^2 - 1)^2,
    D_i  = sum_m |S_{m,i} - mean_i|^2 / (B - 1).
"""

import numpy as np
import torch

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


class StatsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, engine, generation):
        values, _ = engine.forward(x.detach().cpu().numpy(), generation=generation)
        ctx.engine, ctx.generation, ctx.device = engine, generation, x.device
        out = values.view(np.float64).reshape(values.shape + (2,))
        return torch.from_numpy(out.copy()).to(x.device)

    @staticmethod
    def backward(ctx, g):
        g = g.detach().cpu().numpy()
        grad = ctx.engine.vjp(g[..., 0] + 1j * g[..., 1], generation=ctx.generation)
        return torch.from_numpy(grad).to(ctx.device), None, None


class Stats:
    """Statistics of a batch with gradient; one engine, one forward state at a time."""

    def __init__(self, engine):
        self.engine = engine
        self.generation = 0

    def __call__(self, x):
        self.generation += 1
        return StatsFunction.apply(x, self.engine, self.generation)


def native_pixels(plan, nside):
    return np.array([12 * (nside >> c["native_dg"]) ** 2 for c in plan], dtype=np.float64)


class EnsembleLoss:
    def __init__(self, target, target_var, plan, nside, members, scatter_weight=1.0, device="cpu"):
        """`target` complex `[ncoeff]`, `target_var` real `[ncoeff]` (the
        moments of the target's forward)."""
        n = len(plan)
        sigma2 = np.maximum(target_var / native_pixels(plan, nside), 1e-300)
        w = np.where(target_var > 0, 1.0 / sigma2, np.inf)
        w[~np.isfinite(w)] = w[np.isfinite(w)].max()
        self.n, self.members, self.x = n, members, scatter_weight
        self.weights = torch.as_tensor(w * members / n, device=device)
        self.sigma2 = torch.as_tensor(sigma2, device=device)
        self.target = torch.as_tensor(np.stack([target.real, target.imag], -1), device=device)
        self.family = np.array([c["family"] for c in plan])

    def __call__(self, s):
        """`s`: `[B, ncoeff, 2]`.  Returns (loss, mean term, scatter term)."""
        mean = s.mean(0)
        lm = (self.weights * ((mean - self.target) ** 2).sum(-1)).sum()
        if self.x > 0 and s.shape[0] > 1:
            d = ((s - mean) ** 2).sum((0, 2)) / (s.shape[0] - 1)
            ls = self.x / self.n * ((d / self.sigma2 - 1) ** 2).sum()
        else:
            ls = torch.zeros((), dtype=s.dtype, device=s.device)
        return lm + ls, lm.detach(), ls.detach()

    def report(self, s):
        """Per family, as in the `stl-synth` report: residual of the mean in
        sigma / sqrt(M) and in sigma, per-member residual and scatter (rms, in
        sigma), and the scatter over the ergodic variance.  `s`: numpy `[M, ncoeff, 2]`."""
        s = np.asarray(s)
        t = self.target.cpu().numpy()
        sig = np.sqrt(self.sigma2.cpu().numpy())
        m = s.shape[0]
        mean = s.mean(0)
        zmean = np.sqrt(((mean - t) ** 2).sum(-1)) / sig
        zmem = np.sqrt(((s - t) ** 2).sum(-1)) / sig
        zsc = np.sqrt(((s - mean) ** 2).sum(-1)) / sig
        d = ((s - mean) ** 2).sum((0, 2)) / max(m - 1, 1) / sig**2
        out = {}
        for f in FAMILIES:
            k = self.family == f
            if k.any():
                out[f] = {
                    "count": int(k.sum()),
                    "mean_rms_sigma_over_sqrt_m": float(np.sqrt((zmean[k] ** 2).mean()) * np.sqrt(m)),
                    "mean_rms_sigma": float(np.sqrt((zmean[k] ** 2).mean())),
                    "residual_rms_sigma": float(np.sqrt((zmem[:, k] ** 2).mean())),
                    "scatter_rms_sigma": float(np.sqrt((zsc[:, k] ** 2).mean())),
                    "scatter_over_ergodic_variance": float(d[k].mean()),
                }
        return out
