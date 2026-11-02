"""Dump STL-Dev planar (2D kernel) reference outputs as fixtures for the
Rust tests of `stl_core::planar`.

Run with an STL-Dev checkout (branch HealpixUpdate) on the path, from the
repository root:

    PYTHONPATH=<STL-Dev> python tools/dump_planar_reference.py

Writes ``fixtures/planar/<case>/`` (same layout as tools/dump_reference.py):

Torch's default dtype is left as is (float32), as in STL-Dev's own use: the
Fourier wavelet bank and the radial frequencies of the power spectrum are
float32 there before being cast to float64.

- ``kernels``: the complex wavelet kernel ``[L, K, K]`` of
  ``WaveletOperator2Dkernel_torch`` (Bump-Steerable, built in float32 by
  STL-Dev, hence stored rather than recomputed) and the 6 x 6 smoothing
  kernel of the decimation;
- ``ops_*``: one wavelet convolution and one decimation of a random
  rectangular map, with their adjoints applied to random cotangents
  (autograd);
- ``stats_*``: raw statistics (``norm="vanilla"``) of random maps in the
  order of ``stl_core::plan::build_plan`` (mean, var, S1..S4 per level) plus
  the cross spectra, and the map gradient of ``Re sum conj(g) S`` for random
  complex cotangents ``g``;
- ``loss_*``: the complete loss of ``Synthesis.optimize_from_maps``
  (Nyquist prefilter, standardisation, ``store_ref``/``load_ref``
  normalisation, S3/S4 ad-hoc weights, batch mean, power spectrum), its
  gradient at the initial map, and a short L-BFGS trajectory.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from STL_main import Synthesis
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch, WaveletOperator2Dkernel_torch

ROOT = Path(__file__).resolve().parents[1] / "fixtures"
DATA = None  # Turb_6.npy path, set in main


class Suite:
    def __init__(self, name):
        self.dir = ROOT / name
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.index = []

    def case(self, kind, name, meta, **arrays):
        out = self.dir / name
        out.mkdir()
        for key, value in arrays.items():
            a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if np.iscomplexobj(a):
                a = np.stack([a.real, a.imag], axis=-1)
            if a.dtype == np.bool_:
                a = a.astype(np.uint8)
            elif np.issubdtype(a.dtype, np.integer):
                a = a.astype(np.int64)
            elif np.issubdtype(a.dtype, np.floating):
                a = a.astype(np.float64)
            np.save(out / f"{key}.npy", np.ascontiguousarray(a))
        meta = dict(kind=kind, name=name, **meta)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        self.index.append(meta)

    def close(self):
        (self.dir / "index.json").write_text(json.dumps(self.index, indent=1))
        size = sum(p.stat().st_size for p in self.dir.rglob("*") if p.is_file())
        print(f"{self.dir}: {len(self.index)} cases, {size / 1e6:.1f} MB")


def plan(J, L, C, cross=True):
    """Coefficients in the order of stl_core::plan::build_plan (identity
    j_to_dg, every family): tuples (family, c1, c2, j1, j2, j3, l1, l2, l3)."""
    out = [("mean", c, c) + (None,) * 6 for c in range(C)]
    out += [("var", c, c) + (None,) * 6 for c in range(C)]
    for j3 in range(J):
        for c1 in range(C):
            for c2 in range(c1, C):
                if c1 != c2 and not cross:
                    continue
                for fam in ("S1", "S2"):
                    out += [(fam, c1, c2, j3, None, None, l, None, None) for l in range(L)]
                pairs = [(c1, c2)] if c1 == c2 else [(c1, c2), (c2, c1)]
                for ca, cb in pairs:
                    for j2 in range(j3 + 1):
                        out += [("S3", ca, cb, j2, None, j3, l2, None, l3) for l2 in range(L) for l3 in range(L)]
                        for j1 in range(j2 + 1):
                            out += [
                                ("S4", ca, cb, j1, j2, j3, l1, l2, l3)
                                for l1 in range(L)
                                for l2 in range(L)
                                for l3 in range(L)
                            ]
    return out


def gather(st, coeffs):
    """[B, n] complex tensor of the statistics `st` in plan order."""
    cols = []
    for fam, c1, c2, j1, j2, j3, l1, l2, l3 in coeffs:
        if fam == "mean":
            v = st.mean[:, c1].to(torch.complex128)
        elif fam == "var":
            v = st.var[:, c1].to(torch.complex128)
        elif fam == "S1":
            v = st.S1[:, c1, c2, j1, l1]
        elif fam == "S2":
            v = st.S2[:, c1, c2, j1, l1]
        elif fam == "S3":
            v = st.S3[:, c1, c2, j1, j3, l1, l3]
        else:
            v = st.S4[:, c1, c2, j1, j2, j3, l1, l2, l3]
        cols.append(v)
    return torch.stack(cols, dim=1)


def gather_ps(ps, C):
    """[B, n] cross spectra, pairs c1 <= c2 then bins."""
    return torch.stack([ps[:, c1, c2, k] for c1 in range(C) for c2 in range(c1, C) for k in range(ps.shape[-1])], dim=1)


def random_maps(rng, shape):
    """Positive, skewed maps with spatial correlations (exp of smoothed noise)."""
    x = rng.standard_normal(shape)
    k = np.fft.fftfreq(shape[-2])[:, None] ** 2 + np.fft.fftfreq(shape[-1])[None, :] ** 2
    x = np.fft.ifft2(np.fft.fft2(x) * np.exp(-30 * k)).real
    x /= x.std()
    return np.exp(0.7 * x)


def dump_kernels(suite):
    op = WaveletOperator2Dkernel_torch(J=3, device="cpu", dtype=torch.float64)
    w = op._wav_kernel[0]
    s = op._get_smooth_kernel(device=torch.device("cpu"), dtype=torch.float64)
    suite.case("kernels", "kernels", dict(L=op.L, K=op.KERNELSZ), wavelet=w, smooth=s)


def dump_ops(suite, rng, shape, pbc=True):
    op = WaveletOperator2Dkernel_torch(J=3, device="cpu", dtype=torch.float64)
    mode = op._get_padding_mode(pbc)
    x = torch.tensor(rng.standard_normal(shape), requires_grad=True)
    z = op._semicomplex_conv2d_circular(x, op._wav_kernel[0], padding_mode=mode)  # [L, H, W]
    gz = torch.tensor(rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))
    (gx_conv,) = torch.autograd.grad((z * gz.conj()).real.sum(), x)
    s = op._get_smooth_kernel(device=torch.device("cpu"), dtype=torch.float64)
    y = op._downsample_tensor(x, smooth_kernel=s, dg_inc=1, padding_mode=mode)
    gy = torch.tensor(rng.standard_normal(y.shape))
    (gx_down,) = torch.autograd.grad((y * gy).sum(), x)
    name = f"ops_{shape[0]}x{shape[1]}"
    suite.case("ops", name, dict(H=shape[0], W=shape[1]), x=x, z=z, gz=gz, gx_conv=gx_conv, y=y, gy=gy, gx_down=gx_down)


def dump_stats(suite, rng, H, W, J, C, B, fewer):
    x = torch.tensor(random_maps(rng, (B, C, H, W)), requires_grad=True)
    data = STL_2D_Kernel_Torch(x, pbc=True)
    st_op = data.get_ST_op(J=J, compute_PS=True)
    st = st_op.apply(data, norm="vanilla", has_fewer_convolutions=fewer)
    coeffs = plan(J, st_op.L, C)
    s = gather(st, coeffs)
    ps = gather_ps(st.PS, C)
    g = torch.tensor(rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
    gp = torch.tensor(rng.standard_normal(ps.shape) + 1j * rng.standard_normal(ps.shape))
    (grad,) = torch.autograd.grad((s * g.conj()).real.sum(), x, retain_graph=True)
    (grad_ps,) = torch.autograd.grad((ps * gp.conj()).real.sum(), x)
    name = f"stats_{H}x{W}_J{J}_C{C}_{'fewer' if fewer else 'full'}"
    meta = dict(H=H, W=W, J=J, L=st_op.L, C=C, B=B, fewer=fewer, n_bins=st_op.n_bins, n=len(coeffs))
    suite.case("stats", name, meta, x=x, stats=s, cot=g, grad=grad, ps=ps, cot_ps=gp, grad_ps=grad_ps)


def dump_loss(suite, size, ntarget, nbatch, niter):
    """optimize_from_maps on `ntarget` Turb_6 images cropped to size^2."""
    target = np.load(DATA)[:ntarget, :size, :size].astype(np.float64)
    data_target = STL_2D_Kernel_Torch(target[:, None], pbc=True)
    trace = {}
    original = Synthesis.optimize_lbfgs

    def recording(model, loss_fn, lr, max_iter, history_size, verbose, print_iter):
        u0 = model.u.detach().clone()
        out = model()
        loss = loss_fn(out)
        (grad,) = torch.autograd.grad(loss, model.u)
        trace.update(u0=u0, loss0=loss.item(), grad0=grad)
        history = []

        def fn(o):
            v = loss_fn(o)
            history.append(v.item())
            return v

        u = original(model, fn, lr, max_iter, history_size, False, print_iter)
        trace.update(history=history, u_final=u)
        return u

    Synthesis.optimize_lbfgs = recording
    try:
        Synthesis.synthesize_from_maps(data_target, nbatch=nbatch, pbc_running=True, max_iter=niter, verbose=False)
    finally:
        Synthesis.optimize_lbfgs = original
    meta = dict(size=size, ntarget=ntarget, nbatch=nbatch, niter=niter, J=int(math.log2(size)) - 2, L=4, n_bins=2 ** (int(math.log2(size)) - 4))
    meta["loss0"] = trace["loss0"]
    meta["history"] = trace["history"]
    suite.case(
        "loss",
        f"loss_{size}_t{ntarget}_b{nbatch}",
        meta,
        target=target,
        u0=trace["u0"][:, 0],
        grad0=trace["grad0"][:, 0],
        u_final=trace["u_final"][:, 0],
    )


def main():
    global DATA
    import STL_main

    DATA = Path(STL_main.__file__).resolve().parents[1] / "data" / "test" / "Turb_6.npy"
    rng = np.random.default_rng(2026)
    suite = Suite("planar")
    dump_kernels(suite)
    dump_ops(suite, rng, (16, 24))
    dump_ops(suite, rng, (32, 8))
    for H, W, J, C, B, fewer in [
        (16, 16, 2, 1, 2, False),
        (32, 32, 3, 1, 2, False),
        (32, 32, 3, 1, 2, True),
        (32, 16, 2, 2, 2, False),
        (32, 32, 3, 2, 1, True),
        (64, 64, 4, 1, 1, False),
        (64, 64, 4, 1, 1, True),
    ]:
        dump_stats(suite, rng, H, W, J, C, B, fewer)
    dump_loss(suite, 64, 4, 3, 10)
    suite.close()


if __name__ == "__main__":
    main()
