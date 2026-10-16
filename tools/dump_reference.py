"""Dump STL-Dev reference outputs as fixtures for the Rust tests.

Run with STL-Dev's CPU venv and the synced STL-Dev tree on the path, from
the repository root:

    PYTHONPATH=~/stl-work ~/stl-deps/venv-cpu/bin/python tools/dump_reference.py geometry

Every case is a directory ``fixtures/<suite>/<case>/`` holding ``.npy``
arrays (int64, float64 or uint8 for booleans) and a ``meta.json`` with the
parameters.  ``fixtures/<suite>/index.json`` lists the cases.  The suites
follow the validation strategy of CLAUDE.md section 5; ``geometry`` covers
milestone 1 (``TorchGeometry``: centres, ring2nest, interpolation, stencil
rotation, binding, convolution, decimation weights).
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from healpix_geo.nested import kth_neighbourhood

from STL_main import TorchGeometry as tg
from STL_main.LocalGeometry import dilate

ROOT = Path(__file__).resolve().parents[1] / "fixtures"


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


def order_of(nside):
    return int(round(math.log2(nside)))


def random_unit_vectors(gen, n):
    v = torch.randn(n, 3, dtype=torch.float64, generator=gen)
    return v / v.norm(dim=1, keepdim=True)


def special_vectors():
    """Poles, ring boundaries of the polar caps and the equator, phi = 0 / 2 pi."""
    out = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1e-9, 0, 1], [1e-9, 1e-9, -1]]
    for z in (2.0 / 3.0, -2.0 / 3.0, 0.999, -0.999, 1e-12, -1e-12):
        for phi in (0.0, 1e-12, math.pi / 4, math.pi, 2 * math.pi - 1e-12):
            s = math.sqrt(1 - z * z)
            out.append([s * math.cos(phi), s * math.sin(phi), z])
    return torch.tensor(out, dtype=torch.float64)


def block_starts(nside, block, gen, n_random):
    """Blocks at the face corners (equator, poles) and at random positions."""
    npix = 12 * nside * nside
    face = npix // 12
    starts = {0, face - block, 4 * face, 5 * face - block, npix - block}
    nblocks = npix // block
    starts.update(int(s) * block for s in torch.randperm(nblocks, generator=gen)[:n_random])
    return sorted(s for s in starts if 0 <= s <= npix - block)


def candidates(coarse, nside):
    """Candidate fine pixels of every coarse row: ``LocalGeometry.torch_down``."""
    if nside // 2 >= 4 and 2 < nside // 2:
        nb = kth_neighbourhood(coarse.numpy().astype(np.uint64), order_of(nside // 2), 2)
        nb = torch.as_tensor(np.asarray(nb, dtype=np.int64))
    else:
        nb = torch.arange(12 * (nside // 2) ** 2).expand(coarse.numel(), -1)
    ch = (nb[:, :, None] * 4 + torch.arange(4)).reshape(nb.shape[0], -1)
    return torch.where(nb[:, :, None].expand(-1, -1, 4).reshape(nb.shape[0], -1) < 0, -1, ch), nb


def dump_geometry():
    s = Suite("geometry")
    gen = torch.Generator().manual_seed(20261016)

    for nside in (8, 32, 128, 1024):
        order = order_of(nside)
        npix = 12 * nside * nside
        ids = torch.randperm(npix, generator=gen)[:5000].sort().values
        theta, phi = tg.xyf2ang(*tg.nest2xyf(ids, order), nside)
        ix, iy, face = tg.nest2xyf(ids, order)
        lon, lat = tg.centres_lonlat(ids, nside)
        s.case("centres", f"centres_n{nside}", dict(nside=nside), ids=ids, ix=ix, iy=iy, face=face,
               theta=theta, phi=phi, lon=lon, lat=lat)

    for order in (1, 3, 6):
        ring = torch.arange(12 * 4**order)
        s.case("ring2nest", f"ring2nest_o{order}", dict(order=order), ring=ring, nest=tg.ring2nest(ring, order))
    ring = torch.randperm(12 * 4**10, generator=gen)[:20000]
    s.case("ring2nest", "ring2nest_o10", dict(order=10), ring=ring, nest=tg.ring2nest(ring, 10))

    for nside in (1, 2, 8, 32, 128, 1024):
        vecs = torch.cat([special_vectors(), random_unit_vectors(gen, 3000)])
        theta = torch.acos(vecs[:, 2].clamp(-1.0, 1.0))
        phi = torch.atan2(vecs[:, 1], vecs[:, 0]) % (2.0 * math.pi)
        pix_ring, w_ring = tg.get_interpol_ring(theta, phi, nside)
        pix, w = tg.interp_weights_nested(vecs, order_of(nside))
        s.case("interp", f"interp_n{nside}", dict(nside=nside), vecs=vecs, theta=theta, phi=phi,
               pix_ring=pix_ring, w_ring=w_ring, pix_nest=pix, w_nest=w)

    th = torch.cat([torch.tensor([1e-6, math.pi / 2, math.pi - 1e-6]), torch.rand(300, generator=gen, dtype=torch.float64) * math.pi])
    ph = torch.rand(th.numel(), generator=gen, dtype=torch.float64) * 2 * math.pi
    for G in (1, 2, 4):
        for gauge in ("cosmo", "phi"):
            R = tg.rotation_matrices(th, ph, G, gauge, torch.float64)
            s.case("rotation", f"rotation_G{G}_{gauge}", dict(G=G, gauge=gauge), theta=th, phi=ph, R=R)

    for kernel_sz in (3, 5):
        for nside in (8, 1024):
            vec = tg.local_kernel_grid(kernel_sz, nside, "cpu", torch.float64)
            s.case("kernel_grid", f"kernel_grid_k{kernel_sz}_n{nside}", dict(kernel_sz=kernel_sz, nside=nside), vec=vec)

    # Convolution geometry and outputs on partial grids (block + conv halo).
    combos = [(3, 2, "cosmo"), (5, 4, "cosmo"), (5, 2, "phi")]
    conv_cases = [(8, 768, 0), (8, 64, 1), (32, 256, 2), (128, 256, 2), (1024, 256, 1)]
    for nside, block, n_random in conv_cases:
        npix = 12 * nside * nside
        for b, start in enumerate(block_starts(nside, block, gen, n_random)):
            core = torch.arange(start, start + block)
            for kernel_sz, G, gauge in combos:
                if nside == 1024 and (kernel_sz, G) == (5, 2):
                    continue
                rings = (kernel_sz - 1) // 2 + 2
                # Sorted domains only: with unsorted ids TorchConv (like the
                # library) returns its rows in neither the input nor the
                # sorted order; LocalGeometry always passes sorted ids.
                ids = core if block == npix else dilate(core, nside, rings)
                weights = torch.randn(1, 2, kernel_sz**2, generator=gen, dtype=torch.float64).numpy()
                conv = tg.TorchConv(ids, nside, kernel_sz, G, gauge, weights, torch.float64)
                assert torch.equal(conv._sort_order, torch.arange(ids.numel()))
                x = torch.randn(1, 1, ids.numel(), generator=gen, dtype=torch.float64)
                y = conv(x)
                # The raw binding buffers are large: keep them for the first
                # block of each resolution with the small stencil only.
                raw = {}
                if b == 0 and kernel_sz == 3:
                    raw = dict(pos_safe=conv._pos_safe, w_norm=conv._w_norm)
                s.case("conv", f"conv_n{nside}_b{block}_s{start}_k{kernel_sz}_G{G}_{gauge}",
                       dict(nside=nside, kernel_sz=kernel_sz, G=G, gauge=gauge, block_start=start,
                            block_pixels=block, halo_rings=0 if block == npix else rings,
                            raw_binding=bool(raw)),
                       ids=ids, weights=weights, weight_f32=conv.weight, complete=conv._complete,
                       x=x, y=y, **raw)

    # Smooth decimation on partial grids (block + 2 decimation halos), CSR as COO.
    for nside, block, n_random in [(4, 192, 0), (8, 64, 1), (32, 256, 2), (128, 256, 2), (1024, 1024, 1)]:
        # LocalGeometry.down_rings, same floating-point path.
        pix_deg = math.degrees(math.sqrt(4 * math.pi / (12 * nside**2)))
        down_rings = int(math.ceil(3.0 * pix_deg / pix_deg)) + 1
        for start in block_starts(nside, block, gen, n_random):
            core = torch.arange(start, start + block)
            ids = dilate(core, nside, 2 * down_rings)
            coarse = torch.unique(ids // 4)
            cand, nb = candidates(coarse, nside)
            coarse_ids, forward, transposed = tg.smooth_decimation_csr(ids, nside, cand, torch.float64)
            coo = forward.to_sparse_coo().coalesce()
            rows, cols = coo.indices()
            assert torch.equal(transposed.to_dense(), forward.to_dense().T)
            s.case("down", f"down_n{nside}_s{start}", dict(nside=nside, block_start=start, block_pixels=block,
                                                          halo_rings=2 * down_rings),
                   ids=ids, coarse_ids=coarse_ids,
                   rows=rows, cols=cols, values=coo.values())

    # Neighbourhoods (cdshealpix through healpix_geo) for the halo dilations.
    for nside, k in ((4, 1), (8, 2), (32, 3), (1024, 5)):
        ids = torch.cat([torch.tensor([0, 12 * nside * nside - 1, 4 * nside * nside]),
                         torch.randperm(12 * nside * nside, generator=gen)[:200]])
        nb = kth_neighbourhood(ids.numpy().astype(np.uint64), order_of(nside), k)
        s.case("neighbours", f"neighbours_n{nside}_k{k}", dict(nside=nside, k=k), ids=ids,
               neighbours=np.asarray(nb, dtype=np.int64))
    s.close()


def local_geometry(nside, J, L, kernel_sz, block_pixels):
    """``LocalGeometry`` of a float64 sphere operator (torch geometry, no caches)."""
    from STL_main.LocalGeometry import LocalGeometry
    from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch

    data = STL_Healpix_Kernel_Torch(torch.zeros(12 * nside**2, dtype=torch.float64), nside=nside)
    op = data.get_ST_op(J=J, L=L, norm="vanilla", compute_PS=False,
                        wavelet_op_kwargs=dict(kernel_size=kernel_sz, ellipsoid="sphere", conv_cache_dir=None))
    return LocalGeometry(op.wavelet_op, block_pixels=block_pixels, cache_downs=False)


def dump_block():
    """Milestone 2: wavelet on a block and its adjoint, block-wise decimation
    of a level and its transpose, chains of decimation pieces and their
    transposes (``LocalGeometry.wave``, ``wave_adjoint``, ``decimate``,
    ``decimate_adjoint``, ``piece_forward``, ``piece_adjoint``)."""
    from STL_main.AdjointOps import modulus_vjp

    s = Suite("block")
    gen = torch.Generator().manual_seed(20261017)
    B, C = 1, 2
    for nside, J, L, kernel_sz, block_pixels in [(16, 3, 2, 3, 256), (32, 4, 4, 5, 256), (128, 3, 4, 5, 1024)]:
        geo = local_geometry(nside, J, L, kernel_sz, block_pixels)
        w = geo.w
        tag = f"n{nside}_J{J}_L{L}_k{kernel_sz}"
        common = dict(nside=nside, J=J, L=L, kernel_sz=kernel_sz, block_pixels=block_pixels,
                      gauge=w.gauge_type, conv_rings=geo.conv_rings)
        s.case("wavelet", f"wavelet_{tag}", common, weights=w._wav_weights)
        maps = [torch.randn(B, C, geo.pixels(0), generator=gen, dtype=torch.float64)]
        for dg in range(J - 1):
            maps.append(geo.decimate(maps[-1], dg))
        for dg in range(J):
            meta = dict(common, dg=dg, down_rings=geo.down_rings(dg))
            s.case("level", f"level_{tag}_d{dg}", meta, x=maps[dg])
            blocks = geo.blocks(dg)
            for b in sorted({0, len(blocks) // 2, len(blocks) - 1}):
                block = blocks[b]
                z, domain, conv = geo.wave(maps[dg], block, dg)
                gz = torch.randn(z.shape, generator=gen, dtype=torch.complex128)
                gx = geo.wave_adjoint(gz, block, domain, conv)
                s.case("wave", f"wave_{tag}_d{dg}_b{b}", dict(meta, block=b, block_start=int(block[0])),
                       block=block, domain=domain, z_re=z.real, z_im=z.imag,
                       gz_re=gz.real, gz_im=gz.imag, gx=gx)
                # Chain of pieces of |z| from level dg to the coarsest level.
                a, support = z.abs(), block
                arrays = dict(block=block, a0=a)
                for level in range(dg, J - 1):
                    a, support = geo.piece_forward(a, support, level)
                    arrays[f"a{level - dg + 1}"] = a
                    arrays[f"support{level - dg + 1}"] = support
                steps = J - 1 - dg
                if steps > 0:
                    supports = [block] + [arrays[f"support{i}"] for i in range(1, steps + 1)]
                    g = torch.randn(a.shape, generator=gen, dtype=torch.float64)
                    arrays[f"g{steps}"] = g
                    for level in range(J - 2, dg - 1, -1):
                        i = level - dg
                        g = geo.piece_adjoint(g, supports[i], supports[i + 1], level)
                        arrays[f"g{i}"] = g
                    arrays["gz_re"], arrays["gz_im"] = (lambda v: (v.real, v.imag))(modulus_vjp(z, g))
                    s.case("pieces", f"pieces_{tag}_d{dg}_b{b}", dict(meta, block=b, steps=steps), **arrays)
            if dg < J - 1:
                g = torch.randn(B, C, geo.pixels(dg + 1), generator=gen, dtype=torch.float64)
                # Input: the "level" case of dg.
                s.case("decimate", f"decimate_{tag}_d{dg}", meta, y=maps[dg + 1],
                       g=g, gx=geo.decimate_adjoint(g, dg))
    s.close()


STATS_CASES = [
    # nside, J, L, kernel, channels, batch, fewer, block_pixels, j_to_dg
    (16, 3, 2, 3, 2, 2, False, 64, None),
    (16, 3, 2, 3, 2, 2, True, 64, None),
    (16, 3, 2, 3, 2, 1, False, 100000, None),
    (16, 3, 4, 5, 1, 1, False, 256, None),
    (32, 4, 4, 5, 1, 1, False, 256, None),
    (32, 4, 2, 3, 2, 1, True, 1024, None),
    (32, 4, 4, 5, 2, 1, False, 512, None),
    (32, 4, 2, 5, 1, 1, False, 256, (0, 1, 1, 2)),
]


def dump_stats():
    """Milestone 3: ``apply_local`` / ``vjp_local`` (``LocalEngine`` forward
    then reverse, as ``vjp_local`` runs them) on random maps."""
    from STL_main.LocalScattering import LocalEngine, _prepare, _values
    from STL_main.Pyramid import HealpixPyramid
    from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch

    s = Suite("stats")
    for nside, J, L, kernel_sz, channels, batch, fewer, block_pixels, j_to_dg in STATS_CASES:
        torch.manual_seed(1000 * nside + 100 * J + 10 * L + channels)
        data = STL_Healpix_Kernel_Torch(torch.randn(batch, channels, 12 * nside**2, dtype=torch.float64), nside=nside)
        op = data.get_ST_op(J=J, L=L, norm="vanilla", compute_PS=False, has_fewer_convolutions=fewer,
                            wavelet_op_kwargs=dict(kernel_size=kernel_sz, ellipsoid="sphere", conv_cache_dir=None))
        if j_to_dg is not None:
            op.wavelet_op.j_to_dg = list(j_to_dg)
        p = HealpixPyramid.from_map(data, op.wavelet_op)
        geo = LocalGeometry_for(op, block_pixels)
        plan, geo = _prepare(op, p, None, geo)
        engine = LocalEngine(op, p, geo)
        values = engine.forward(plan, _values(engine.maps, plan))
        gradient = torch.randn(values.shape, dtype=torch.complex128)
        active = (gradient != 0).any(dim=0).cpu().tolist()
        maps_gradient = engine.reverse(plan, gradient, active, [torch.zeros_like(x) for x in engine.maps])
        bands = geo.analysis_vjp(p, [g.clone() for g in maps_gradient])
        name = f"stats_n{nside}_J{J}_L{L}_k{kernel_sz}_C{channels}_B{batch}_{'fewer' if fewer else 'full'}_b{block_pixels}"
        if j_to_dg is not None:
            name += "_dg" + "".join(map(str, j_to_dg))
        arrays = dict(values_re=values.real, values_im=values.imag,
                      gradient_re=gradient.real, gradient_im=gradient.imag)
        for d, x in enumerate(engine.maps):
            arrays[f"map{d}"] = x
        for d, g in enumerate(maps_gradient):
            arrays[f"maps_gradient{d}"] = g
        for d, g in enumerate(bands):
            arrays[f"band_gradient{d}"] = g
        for d, row in engine.m.items():
            for n, m in row.items():
                arrays[f"m{d}_{n}"] = m
        meta = dict(nside=nside, J=J, L=L, kernel_sz=kernel_sz, channels=channels, batch=batch,
                    fewer=fewer, block_pixels=block_pixels, gauge=op.wavelet_op.gauge_type,
                    j_to_dg=list(op.wavelet_op.j_to_dg), levels=len(engine.maps), n_coefficients=len(plan))
        s.case("stats", name, meta, **arrays)
        fields = ("family", "native_dg", "c1", "c2", "j1", "j2", "j3", "l1", "l2", "l3")
        (s.dir / name / "plan.json").write_text(json.dumps([[getattr(c, f) for f in fields] for c in plan]))
        print(name, len(plan), "coefficients")
    s.close()


def LocalGeometry_for(op, block_pixels):
    from STL_main.LocalGeometry import LocalGeometry

    return LocalGeometry(op.wavelet_op, block_pixels=block_pixels, cache_downs=False)


def check_against_local_geometry():
    """The replicated candidate logic must give LocalGeometry.torch_down's matrix."""
    from STL_main.LocalGeometry import LocalGeometry
    from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch

    nside = 32
    data = STL_Healpix_Kernel_Torch(torch.randn(12 * nside**2, dtype=torch.float64), nside=nside)
    op = data.get_ST_op(J=2, L=2, norm="vanilla", compute_PS=False,
                        wavelet_op_kwargs=dict(kernel_size=3, ellipsoid="sphere", conv_cache_dir=None))
    geo = LocalGeometry(op.wavelet_op, block_pixels=256)
    ids = dilate(torch.arange(1024, 1280), nside, 8)
    ref = geo.torch_down(0, ids)._stl_csr_tensors[(False, torch.float64)].to_dense()
    cand, _ = candidates(torch.unique(ids // 4), nside)
    _, mine, _ = tg.smooth_decimation_csr(ids, nside, cand, torch.float64)
    assert torch.equal(mine.to_dense(), ref), "candidate replication differs from LocalGeometry"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suites", nargs="+", choices=["geometry", "block", "stats"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if "geometry" in args.suites:
        check_against_local_geometry()
        dump_geometry()
    if "block" in args.suites:
        dump_block()
    if "stats" in args.suites:
        dump_stats()
