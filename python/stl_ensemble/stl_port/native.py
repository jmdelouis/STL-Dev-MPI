"""Persistent ctypes binding to STL's native forward, VJP and L-BFGS.

The bridge is built from the pinned Rust source; this module contains no
scattering operator or optimizer implementation. Every handle owns exactly
one engine whose batch size is fixed at construction.
"""
from __future__ import annotations
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import numpy as np

SCRIPT_VERSION = "2026-11-06-v6"
print(f"SCRIPT VERSION: stl_port.native {SCRIPT_VERSION}", flush=True)

_DOUBLE = C.POINTER(C.c_double)
_INT64 = C.POINTER(C.c_int64)
_SIZE = C.POINTER(C.c_size_t)
_VOID = C.c_void_p
_CALLBACK = C.CFUNCTYPE(C.c_double, _DOUBLE, C.c_size_t, _DOUBLE, _VOID)
_LIBRARIES = {}


def _pointer(a):
    return a.ctypes.data_as(_DOUBLE)


def _library(path=None):
    if path is None:
        path = os.environ.get("STL_NATIVE_LIBRARY")
    if path is None:
        root = Path(__file__).resolve().parents[1]
        candidates = [root / "native/libstl_bridge.so",
                      root / "STL-Dev-MPI/target/release/libstl_bridge.so",
                      root.parent / "STL-Dev-MPI/target/release/libstl_bridge.so"]
        path = next((p for p in candidates if p.is_file()), candidates[0])
    path = str(Path(path).resolve())
    if path in _LIBRARIES:
        return _LIBRARIES[path]
    if not Path(path).is_file():
        raise RuntimeError(f"Native STL library is missing: {path}. Build stl-bridge and set STL_NATIVE_LIBRARY; no fallback scattering implementation is used.")
    lib = C.CDLL(path)
    binary_hash = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            binary_hash.update(chunk)
    lib._stlp_binary_sha256 = binary_hash.hexdigest()
    lib._stlp_binary_path = path
    signatures = {
        "stlp_last_error": ([], C.c_char_p),
        "stlp_string_free": ([_VOID], None),
        "stlp_new": ([C.c_char_p], _VOID),
        "stlp_free": ([_VOID], None),
        "stlp_info": ([_VOID], _VOID),
        "stlp_forward": ([_VOID, _DOUBLE, C.c_size_t, C.c_uint64, C.c_uint64, C.c_uint64, C.c_bool, _DOUBLE, _DOUBLE], C.c_int),
        "stlp_vjp": ([_VOID, _DOUBLE, C.c_size_t, C.c_uint64, C.c_uint64, C.c_uint64, _DOUBLE], C.c_int),
        "stlp_checkpoint": ([_VOID], _VOID),
        "stlp_checkpoint_free": ([_VOID], None),
        "stlp_restore": ([_VOID, _VOID, C.c_uint64, C.c_uint64, C.c_uint64], C.c_int),
        "stlp_set_masks": ([_VOID, _DOUBLE, C.c_size_t], C.c_int),
        "stlp_native_pixels": ([_VOID, _SIZE], C.c_int),
        "stlp_lbfgs": ([_DOUBLE, C.c_size_t, C.c_char_p, _CALLBACK, _VOID], C.c_int64),
        "stlp_patch_new": ([C.c_char_p], _VOID),
        "stlp_patch_free": ([_VOID], None),
        "stlp_patch_info": ([_VOID], _VOID),
        "stlp_pixel_vectors": ([C.c_size_t, _INT64, C.c_size_t, _DOUBLE], C.c_int),
        "stlp_patch_select": ([_VOID, _DOUBLE, C.c_size_t, C.c_double, C.c_double, _SIZE], C.c_int),
        "stlp_patch_ids": ([_VOID, C.c_size_t, _INT64], C.c_int),
        "stlp_patch_core_pixels": ([_VOID, _SIZE], C.c_int),
        "stlp_patch_forward": ([_VOID, _DOUBLE, C.c_size_t, C.c_uint64, C.c_bool, _DOUBLE, _DOUBLE], C.c_int),
        "stlp_patch_vjp": ([_VOID, _DOUBLE, C.c_size_t, C.c_uint64, _DOUBLE], C.c_int),
        "stlp_patch_forward_device": ([_VOID, C.c_uint64, C.c_size_t, C.c_uint64, C.c_bool, _DOUBLE, _DOUBLE], C.c_int),
        "stlp_patch_vjp_device": ([_VOID, _DOUBLE, C.c_size_t, C.c_uint64, C.c_uint64], C.c_int),
        "stlp_spec_new": ([C.c_char_p], _VOID),
        "stlp_spec_free": ([_VOID], None),
        "stlp_spec_info": ([_VOID], _VOID),
        "stlp_spec_set_masks": ([_VOID, _DOUBLE, C.c_size_t], C.c_int),
        "stlp_spec_forward": ([_VOID, _DOUBLE, C.c_size_t, C.c_uint64, C.c_uint64, C.c_uint64, _DOUBLE], C.c_int),
        "stlp_spec_vjp": ([_VOID, _DOUBLE, C.c_size_t, C.c_uint64, C.c_uint64, C.c_uint64, _DOUBLE], C.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, restype
    _LIBRARIES[path] = lib
    return lib


def _error(lib):
    return RuntimeError(lib.stlp_last_error().decode("utf-8", errors="replace"))


def _key(generation, member, objective):
    if isinstance(generation, bool) or not isinstance(generation, (int, np.integer)) or generation < 0 or generation >= 2**64:
        raise ValueError("generation must be an unsigned 64-bit evaluation number")
    if isinstance(member, bool) or not isinstance(member, (int, np.integer)) or member < 0 or member >= 2**64:
        raise ValueError("member must be an unsigned 64-bit member number")
    if not isinstance(objective, str) or not objective:
        raise ValueError("objective must be a nonempty name")
    return (int(generation), int(member), int.from_bytes(hashlib.sha256(objective.encode()).digest()[:8], "little"))


class Snapshot:
    """Owned complete host state; call close() when its reverse pass is done."""
    def __init__(self, lib, handle, key, owner):
        self._lib, self._handle, self.key, self.owner = lib, handle, key, owner

    def close(self):
        if self._handle:
            self._lib.stlp_checkpoint_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NativeEngine:
    """One persistent native workspace, with immutable plan/resolution/batch.

    forward returns (complex values, real within-map spatial variances). The
    latter is an empty (batch,0) array unless moments=True; it is NOT the
    TRAIN54 across-map variance and is NOT divided by native pixel count.

    cross=True: the maps are pairs [batch, 2, npix] and the plan holds the
    cross terms of the two channels only (S1, S2, S3, S4 with c1 != c2;
    S3/S4 in both orders); see forward_cross / vjp_cross.

    masks (or set_masks): K masks given as NESTED pixel ids of the full
    resolution or as [K, npix] indicators (nonzero: in). Every coefficient
    is then averaged over each mask at its native level (level n + 1 holds
    the parents of the level-n mask), the wavelets and moduli being those
    of the full sky; values, variances and cotangents are [batch, K,
    ncoeff], and native_pixels() gives the [K, ncoeff] pixel counts of the
    averages (for the ergodic weights).
    """
    def __init__(self, nside, j, batch=1, backend="cpu", kernel=5,
                 orientations=4, block_pixels=4096, cache_bytes=1 << 30,
                 threads=1, cache="transposition", device=0, library=None,
                 cross=False, masks=None):
        self._lib = _library(library)
        self.library_sha256 = self._lib._stlp_binary_sha256
        self._handle = None
        self._lock = threading.RLock()
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        if any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in (nside, j, batch, kernel, orientations, block_pixels, threads)):
            raise ValueError("engine sizes/thread count must be positive integers")
        if int(cache_bytes) < 0 or int(device) < 0:
            raise ValueError("cache size/device must be nonnegative")
        self.nside, self.j, self.batch = int(nside), int(j), int(batch)
        self.npix = 12 * self.nside**2
        self.backend = backend
        self.cross = bool(cross)
        self.channels = 2 if self.cross else 1
        self.config = dict(nside=self.nside, j=self.j, batch=self.batch,
                           backend=backend, kernel=int(kernel), orientations=int(orientations),
                           block_pixels=int(block_pixels), cache_bytes=int(cache_bytes),
                           threads=int(threads), cache=cache, device=int(device), cross=self.cross)
        self._handle = self._lib.stlp_new(json.dumps(self.config).encode())
        if not self._handle:
            raise _error(self._lib)
        self.plan = self.info()["plan"]
        self.plan_digest = hashlib.sha256(json.dumps(self.plan, sort_keys=True).encode()).hexdigest()
        self.ncoeff = len(self.plan)
        self._current_key = None
        self.nmasks = 0
        if masks is not None:
            self.set_masks(masks)

    def _open(self):
        if not self._handle:
            raise RuntimeError("native engine is closed")

    def set_masks(self, masks):
        """Average over the masks from now on (None: the whole sky). masks:
        a list of NESTED id arrays, or [K, npix] / [npix] indicators."""
        with self._lock:
            self._open()
            if masks is None:
                w = np.zeros((0, self.npix))
            elif isinstance(masks, np.ndarray) and masks.dtype != object and masks.shape[-1] == self.npix and masks.ndim in (1, 2):
                w = np.ascontiguousarray(np.atleast_2d(masks), dtype=np.float64)
            else:
                w = np.zeros((len(masks), self.npix))
                for k, ids in enumerate(masks):
                    ids = np.asarray(ids, dtype=np.int64)
                    if ids.size and (ids.min() < 0 or ids.max() >= self.npix):
                        raise ValueError(f"mask {k}: pixel ids out of range")
                    w[k, ids] = 1.0
            if not np.isfinite(w).all():
                raise ValueError("mask weights must be finite")
            self._current_key = None
            if self._lib.stlp_set_masks(self._handle, _pointer(w), w.shape[0]):
                raise _error(self._lib)
            self.nmasks = w.shape[0]

    def native_pixels(self):
        """Pixels each coefficient averages: [K, ncoeff] with masks, else [ncoeff]."""
        with self._lock:
            self._open()
            out = np.empty((max(self.nmasks, 1), self.ncoeff), dtype=np.uintp)
            if self._lib.stlp_native_pixels(self._handle, out.ctypes.data_as(_SIZE)):
                raise _error(self._lib)
            out = out.astype(np.int64)
            return out if self.nmasks else out[0]

    def _shape(self):
        return (self.batch, self.nmasks, self.ncoeff) if self.nmasks else (self.batch, self.ncoeff)

    def info(self):
        with self._lock:
            self._open()
            p = self._lib.stlp_info(self._handle)
            if not p:
                raise _error(self._lib)
            try:
                result = json.loads(C.string_at(p))
                result["library_sha256"] = self.library_sha256
                return result
            finally:
                self._lib.stlp_string_free(p)

    def forward(self, maps, generation=0, member=0, objective="gaussian", moments=False):
        with self._lock:
            self._open()
            key = _key(generation, member, objective)
            x = np.ascontiguousarray(maps, dtype=np.float64)
            shape = (self.batch, self.npix) if not self.cross else (self.batch, 2, self.npix)
            if x.shape == shape[1:] and self.batch == 1:
                x = x[None]
            if x.shape != shape:
                raise ValueError(f"expected maps {shape}, got {x.shape}")
            if not np.isfinite(x).all():
                raise ValueError("input map contains non-finite values")
            values = np.empty(self._shape(), dtype=np.complex128)
            variance = np.empty(self._shape() if moments else (self.batch, 0), dtype=np.float64)
            self._current_key = None
            status = self._lib.stlp_forward(self._handle, _pointer(x), x.size, *key,
                                           moments, _pointer(values.view(np.float64)), _pointer(variance))
            if status:
                raise _error(self._lib)
            self._current_key = key
            return values, variance

    def vjp(self, cotangents, generation=0, member=0, objective="gaussian"):
        with self._lock:
            self._open()
            key = _key(generation, member, objective)
            if key != self._current_key:
                raise RuntimeError("stale/wrong-member/objective forward state")
            c = np.ascontiguousarray(cotangents, dtype=np.complex128)
            shape = self._shape()
            if c.shape == shape[1:] and self.batch == 1:
                c = c[None]
            if c.shape != shape:
                raise ValueError(f"expected cotangents {shape}, got {c.shape}")
            if not np.isfinite(c).all():
                raise ValueError("non-finite coefficient cotangent")
            grad = np.empty((self.batch, self.npix) if not self.cross else (self.batch, 2, self.npix), dtype=np.float64)
            status = self._lib.stlp_vjp(self._handle, _pointer(c.view(np.float64)), c.size, *key, _pointer(grad))
            if status:
                raise _error(self._lib)
            return grad

    def forward_cross(self, a, b, generation=0, member=0, objective="gaussian", moments=False):
        """Cross statistics S(a, b) of the pairs a, b [batch, npix] (cross=True)."""
        if not self.cross:
            raise RuntimeError("forward_cross needs NativeEngine(..., cross=True)")
        a = np.asarray(a, dtype=np.float64).reshape(self.batch, self.npix)
        b = np.asarray(b, dtype=np.float64).reshape(self.batch, self.npix)
        return self.forward(np.stack([a, b], 1), generation, member, objective, moments)

    def vjp_cross(self, cotangents, generation=0, member=0, objective="gaussian"):
        """(grad_a, grad_b), each [batch, npix], after forward_cross."""
        g = self.vjp(cotangents, generation, member, objective)
        return g[:, 0], g[:, 1]

    def checkpoint(self):
        with self._lock:
            self._open()
            p = self._lib.stlp_checkpoint(self._handle)
            if not p:
                raise _error(self._lib)
            return Snapshot(self._lib, p, self._current_key, self._handle)

    def restore(self, snapshot, generation=0, member=0, objective="gaussian"):
        with self._lock:
            self._open()
            key = _key(generation, member, objective)
            if not isinstance(snapshot, Snapshot) or not snapshot._handle:
                raise ValueError("checkpoint is closed or invalid")
            if snapshot.key != key:
                raise RuntimeError("stale/wrong-member/objective checkpoint state")
            if snapshot._lib is not self._lib or snapshot.owner != self._handle:
                raise ValueError("checkpoint engine/plan owner mismatch")
            status = self._lib.stlp_restore(self._handle, snapshot._handle, *key)
            if status:
                raise _error(self._lib)
            self._current_key = key

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.stlp_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SpectrumEngine:
    """Differentiable anafast (iter=0, no ring weights, healpy conventions) of
    NESTED full-sky maps, CPU or GPU (backend="gpu": the transforms on the
    device, the spectra and their cotangents on the host).

    forward(maps [batch, npix], generation) -> C_l [batch, lmax + 1]; with
    masks (K mask weights, set_masks), every map is multiplied by each
    weight before the analysis and C_l is [batch, K, lmax + 1] (pseudo-C_l,
    not divided by f_sky; self.fsky gives mean(w^2) per mask).
    cross=True: maps [batch, 2, npix] and the cross spectra
    sum_m mult Re(a_lm conj(b_lm)) / (2l + 1) only (forward_cross(a, b)).
    spin=s >= 1: every field is a pair (Q, U) of a spin-s field, maps
    [batch, 2, npix] ([batch, 2, 2, npix] with cross), E and B as healpy's
    map2alm_spin (spin 2: anafast(pol=True)); C_l [batch, (K,) 3, lmax + 1]
    for EE, BB, EB, with cross [.., 4, ..] for E1E2, B1B2, E1B2, B1E2
    (self.components). The masks multiply Q and U alike. Spin >= 1 uses the
    host transforms with either backend.
    vjp(cotangents shaped like C_l, generation) -> map gradients shaped like
    maps, after the forward of the same (generation, member, objective).
    """
    def __init__(self, nside, lmax=None, batch=1, backend="cpu", device=0, cross=False,
                 masks=None, library=None, spin=0):
        self._lib = _library(library)
        self.library_sha256 = self._lib._stlp_binary_sha256
        self._handle = None
        self._lock = threading.RLock()
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        self.nside, self.batch = int(nside), int(batch)
        self.lmax = 3 * self.nside - 1 if lmax is None else int(lmax)
        if self.batch <= 0 or self.lmax < 0:
            raise ValueError("batch must be positive and lmax nonnegative")
        self.npix = 12 * self.nside**2
        self.backend, self.cross = backend, bool(cross)
        self.channels = 2 if self.cross else 1
        if isinstance(spin, bool) or int(spin) != spin or int(spin) < 0:
            raise ValueError("spin must be a nonnegative integer")
        self.spin = int(spin)
        self.config = dict(nside=self.nside, lmax=self.lmax, batch=self.batch, backend=backend,
                           device=int(device), cross=self.cross, spin=self.spin)
        self._handle = self._lib.stlp_spec_new(json.dumps(self.config).encode())
        if not self._handle:
            raise _error(self._lib)
        self.components = self.info()["components"]
        self._current_key = None
        self.nmasks, self.fsky = 0, None
        if masks is not None:
            self.set_masks(masks)

    def _open(self):
        if not self._handle:
            raise RuntimeError("spectrum engine is closed")

    def info(self):
        with self._lock:
            self._open()
            p = self._lib.stlp_spec_info(self._handle)
            if not p:
                raise _error(self._lib)
            try:
                result = json.loads(C.string_at(p))
                result["library_sha256"] = self.library_sha256
                return result
            finally:
                self._lib.stlp_string_free(p)

    def set_masks(self, masks):
        """Multiply the maps by each mask from now on (None: the whole sky).
        masks: [K, npix] / [npix] weights (any finite values, e.g. apodised)
        or a list of NESTED id arrays (weight 1 on the ids)."""
        with self._lock:
            self._open()
            if masks is None:
                w = np.zeros((0, self.npix))
            elif isinstance(masks, np.ndarray) and masks.dtype != object and masks.shape[-1] == self.npix and masks.ndim in (1, 2):
                w = np.ascontiguousarray(np.atleast_2d(masks), dtype=np.float64)
            else:
                w = np.zeros((len(masks), self.npix))
                for k, ids in enumerate(masks):
                    ids = np.asarray(ids, dtype=np.int64)
                    if ids.size and (ids.min() < 0 or ids.max() >= self.npix):
                        raise ValueError(f"mask {k}: pixel ids out of range")
                    w[k, ids] = 1.0
            if not np.isfinite(w).all():
                raise ValueError("mask weights must be finite")
            self._current_key = None
            if self._lib.stlp_spec_set_masks(self._handle, _pointer(w), w.shape[0]):
                raise _error(self._lib)
            self.nmasks = w.shape[0]
            self.fsky = np.mean(w**2, axis=1) if self.nmasks else None

    def _shape(self):
        inner = (self.lmax + 1,) if self.spin == 0 else (len(self.components), self.lmax + 1)
        return (self.batch, self.nmasks) + inner if self.nmasks else (self.batch,) + inner

    def _map_shape(self):
        field = (self.npix,) if self.spin == 0 else (2, self.npix)
        return (self.batch,) + ((2,) if self.cross else ()) + field

    def forward(self, maps, generation=0, member=0, objective="cl"):
        with self._lock:
            self._open()
            key = _key(generation, member, objective)
            x = np.ascontiguousarray(maps, dtype=np.float64)
            shape = self._map_shape()
            if x.shape == shape[1:] and self.batch == 1:
                x = x[None]
            if x.shape != shape:
                raise ValueError(f"expected maps {shape}, got {x.shape}")
            if not np.isfinite(x).all():
                raise ValueError("input map contains non-finite values")
            cl = np.empty(self._shape(), dtype=np.float64)
            self._current_key = None
            if self._lib.stlp_spec_forward(self._handle, _pointer(x), x.size, *key, _pointer(cl)):
                raise _error(self._lib)
            self._current_key = key
            return cl

    def vjp(self, cotangents, generation=0, member=0, objective="cl"):
        with self._lock:
            self._open()
            key = _key(generation, member, objective)
            if key != self._current_key:
                raise RuntimeError("stale/wrong-member/objective forward state")
            c = np.ascontiguousarray(cotangents, dtype=np.float64)
            shape = self._shape()
            if c.shape == shape[1:] and self.batch == 1:
                c = c[None]
            if c.shape != shape:
                raise ValueError(f"expected cotangents {shape}, got {c.shape}")
            if not np.isfinite(c).all():
                raise ValueError("non-finite spectrum cotangent")
            grad = np.empty(self._map_shape(), dtype=np.float64)
            if self._lib.stlp_spec_vjp(self._handle, _pointer(c), c.size, *key, _pointer(grad)):
                raise _error(self._lib)
            return grad

    def forward_cross(self, a, b, generation=0, member=0, objective="cl"):
        """Cross spectra of the pairs a, b [batch, npix] ([batch, 2, npix]
        (Q, U) with spin >= 1) (cross=True)."""
        if not self.cross:
            raise RuntimeError("forward_cross needs SpectrumEngine(..., cross=True)")
        field = self._map_shape()[2:]
        a = np.asarray(a, dtype=np.float64).reshape((self.batch,) + field)
        b = np.asarray(b, dtype=np.float64).reshape((self.batch,) + field)
        return self.forward(np.stack([a, b], 1), generation, member, objective)

    def vjp_cross(self, cotangents, generation=0, member=0, objective="cl"):
        """(grad_a, grad_b), each shaped like a, after forward_cross."""
        g = self.vjp(cotangents, generation, member, objective)
        return g[:, 0], g[:, 1]

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.stlp_spec_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def native_lbfgs(callback, x, options=None, library=None):
    """Run one joint native L-BFGS minimization, strong-Wolfe unchanged.

    The float64 C-contiguous input is updated in place; other inputs are
    converted. callback receives the same shape, as a read-only ephemeral
    view, and returns (scalar_loss, map_gradient). Returns (x, report).
    """
    lib = _library(library)
    resolved = dict(lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-12,
                    tolerance_change=1e-15, history_size=10)
    if options:
        unknown = set(options) - set(resolved)
        if unknown:
            raise ValueError(f"unknown native L-BFGS options: {sorted(unknown)}")
        resolved.update(options)
    for k in ("max_iter", "history_size"):
        if isinstance(resolved[k], bool) or int(resolved[k]) != resolved[k] or resolved[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if resolved["max_eval"] is not None and (int(resolved["max_eval"]) != resolved["max_eval"] or resolved["max_eval"] < 1):
        raise ValueError("max_eval must be a positive integer or None")
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty initial ensemble required")
    shape, history, errors = x.shape, [], []

    @_CALLBACK
    def evaluate(p, n, gp, _context):
        try:
            trial = np.ctypeslib.as_array(p, shape=(n,)).reshape(shape)
            trial.flags.writeable = False
            value, grad = callback(trial)
            grad = np.asarray(grad, dtype=np.float64)
            if grad.shape != shape:
                raise ValueError(f"gradient shape {grad.shape} != {shape}")
            np.ctypeslib.as_array(gp, shape=(n,))[:] = grad.ravel()
            history.append(float(value))
            return float(value)
        except BaseException as exc:
            errors.append(exc)
            return float("nan")

    count = lib.stlp_lbfgs(_pointer(x), x.size, json.dumps(resolved).encode(), evaluate, None)
    if errors:
        raise errors[0]
    if count < 0:
        raise _error(lib)
    return x, dict(evaluations=int(count), loss_history=history, options=resolved,
                   optimizer="native STL joint L-BFGS", line_search="strong_wolfe",
                   version=SCRIPT_VERSION)


def delouis_transform(values):
    """sign(Re) sqrt|Re| + i sign(Im) sqrt|Im| (Delouis et al. 2022, A&A 668,
    A122, Sect. 2.2), element-wise on complex statistics."""
    v = np.asarray(values, dtype=np.complex128)
    return np.sign(v.real) * np.sqrt(np.abs(v.real)) + 1j * np.sign(v.imag) * np.sqrt(np.abs(v.imag))


def delouis_transform_vjp(values, cotangents, eps=0.0):
    """Cotangent of the statistics from the cotangent of
    delouis_transform(values): d sqrt|x| / dx = 1 / (2 sqrt|x|), unbounded at
    x = 0 (eps > 0 bounds it)."""
    v = np.asarray(values, dtype=np.complex128)
    g = np.asarray(cotangents, dtype=np.complex128)
    return g.real / (2 * np.sqrt(np.abs(v.real)) + eps) + 1j * g.imag / (2 * np.sqrt(np.abs(v.imag)) + eps)


class PatchEngine:
    """Statistics on disc patches, CPU (the patches of a batch on host threads)
    or GPU (backend="gpu": the patches of a batch grouped by disc, the disc
    geometries cached on the device while it keeps reserve_bytes free (2 GiB,
    the free memory read when a disc is built, so that a network on the same
    GPU is accounted for), optionally capped at cache_bytes; same definitions
    and API).

    Same plan and formulas as NativeEngine (cross=True: the cross terms of
    two maps), averaged over a spherical disc. The support is every NESTED
    pixel of `nside` whose centre lies within `radius` degrees of the disc
    centre; a coarser level keeps the parents of the finer one; the wavelet
    is bound to the support with renormalised weights and the decimation is
    the smooth one restricted to the support. With `core` (degrees, <
    radius) the coefficients are means over the core pixels (and their
    parents at the coarser levels) only. A disc of radius 180 gives the
    full-sky statistics.

    select(centres, radius, core=None) -> ids, a list of sorted NESTED pixel
        arrays; centres are NESTED pixels [B] at nside or vectors [B, 3].
    forward(maps, centres=None, radius=None, core=None, moments=True,
        generation=0) -> (values [B, ncoeff] complex, variances [B, ncoeff]);
        maps[i] is the patch x[ids[i]] ([n_i], or [2, n_i] with cross=True),
        a list, or a dense array when every disc has the same size.
    vjp(cotangents [B, ncoeff], generation=0) -> gradients shaped like maps.
    native_pixels [B, ncoeff]: number of pixels each coefficient averages.
    """
    def __init__(self, nside, j, kernel=5, orientations=4, cross=False,
                 threads=0, cache=256, library=None, backend="cpu", device=0, cache_bytes=None,
                 reserve_bytes=None):
        self._lib = _library(library)
        self.library_sha256 = self._lib._stlp_binary_sha256
        self._handle = None
        self._lock = threading.RLock()
        self.nside, self.j, self.cross = int(nside), int(j), bool(cross)
        self.channels = 2 if self.cross else 1
        self.config = dict(nside=self.nside, j=self.j, kernel=int(kernel), orientations=int(orientations),
                           cross=self.cross, threads=int(threads), cache=int(cache), backend=str(backend))
        if backend == "gpu":
            self.config["device"] = int(device)
            if cache_bytes is not None:
                self.config["cache_bytes"] = int(cache_bytes)
            if reserve_bytes is not None:
                self.config["reserve_bytes"] = int(reserve_bytes)
        self._handle = self._lib.stlp_patch_new(json.dumps(self.config).encode())
        if not self._handle:
            raise _error(self._lib)
        info = self.info()
        self.plan, self.levels = info["plan"], info["levels"]
        self.ncoeff = len(self.plan)
        self._native = np.array([c["native_dg"] for c in self.plan], dtype=np.int64)
        self.ids, self.native_pixels = [], None
        self._selection, self._dense, self._generation = None, False, None

    def _open(self):
        if not self._handle:
            raise RuntimeError("patch engine is closed")

    def info(self):
        with self._lock:
            self._open()
            p = self._lib.stlp_patch_info(self._handle)
            if not p:
                raise _error(self._lib)
            try:
                result = json.loads(C.string_at(p))
                result["library_sha256"] = self.library_sha256
                return result
            finally:
                self._lib.stlp_string_free(p)

    def centre_vectors(self, centres):
        """Unit vectors [B, 3] of NESTED pixels [B] at nside; vectors [B, 3] as given."""
        c = np.asarray(centres)
        if np.issubdtype(c.dtype, np.integer):
            ids = np.ascontiguousarray(c.reshape(-1), dtype=np.int64)
            out = np.empty((ids.size, 3), dtype=np.float64)
            if self._lib.stlp_pixel_vectors(self.nside, ids.ctypes.data_as(_INT64), ids.size, _pointer(out)):
                raise _error(self._lib)
            return out
        c = np.ascontiguousarray(c, dtype=np.float64)
        if c.ndim == 1:
            c = c[None]
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError("centres must be NESTED pixel indices [B] or vectors [B, 3]")
        return c

    def select(self, centres, radius, core=None):
        """Selects the discs (radius, core in degrees); returns their sorted
        NESTED pixels (level 0). A repeated selection is free."""
        with self._lock:
            self._open()
            v = np.ascontiguousarray(self.centre_vectors(centres))
            key = (v.tobytes(), float(radius), None if core is None else float(core))
            if key == self._selection:
                return self.ids
            b = v.shape[0]
            counts = np.empty(b, dtype=np.uintp)
            rc = 0.0 if core is None else np.radians(float(core))
            self._generation, self._selection = None, None
            if self._lib.stlp_patch_select(self._handle, _pointer(v), b, np.radians(float(radius)), rc,
                                           counts.ctypes.data_as(_SIZE)):
                raise _error(self._lib)
            ids = []
            for i, n in enumerate(counts):
                a = np.empty(int(n), dtype=np.int64)
                if self._lib.stlp_patch_ids(self._handle, i, a.ctypes.data_as(_INT64)):
                    raise _error(self._lib)
                ids.append(a)
            core_pixels = np.empty((b, self.levels), dtype=np.uintp)
            if self._lib.stlp_patch_core_pixels(self._handle, core_pixels.ctypes.data_as(_SIZE)):
                raise _error(self._lib)
            self.ids, self._selection = ids, key
            self.native_pixels = core_pixels.astype(np.float64)[:, self._native]
            return ids

    def gather(self, maps):
        """Patches of the selected discs from full-sky maps: maps [npix]
        (the same map for every disc) or [B, npix] / [B, 2, npix] (map i
        for disc i); returns a list."""
        m = np.asarray(maps)
        if m.ndim == 1:
            return [m[i] for i in self.ids]
        return [m[k][..., i] for k, i in enumerate(self.ids)]

    def forward(self, maps, centres=None, radius=None, core=None, moments=True, generation=0):
        with self._lock:
            self._open()
            if centres is not None:
                if radius is None:
                    raise ValueError("radius (degrees) is required with centres")
                self.select(centres, radius, core)
            if not self.ids:
                raise RuntimeError("no disc selected")
            b = len(self.ids)
            if len(maps) != b:
                raise ValueError(f"expected {b} patches, got {len(maps)}")
            parts = []
            for i, (m, ids) in enumerate(zip(maps, self.ids)):
                m = np.asarray(m, dtype=np.float64)
                shape = (ids.size,) if not self.cross else (2, ids.size)
                if m.shape != shape:
                    raise ValueError(f"patch {i}: expected {shape}, got {m.shape}")
                parts.append(m.reshape(-1))
            x = np.ascontiguousarray(np.concatenate(parts))
            if not np.isfinite(x).all():
                raise ValueError("input patch contains non-finite values")
            self._dense = isinstance(maps, np.ndarray)
            values = np.empty((b, self.ncoeff), dtype=np.complex128)
            variance = np.empty((b, self.ncoeff if moments else 0), dtype=np.float64)
            self._generation = None
            if self._lib.stlp_patch_forward(self._handle, _pointer(x), x.size, int(generation), moments,
                                            _pointer(values.view(np.float64)), _pointer(variance)):
                raise _error(self._lib)
            self._generation = int(generation)
            return values, variance

    def vjp(self, cotangents, generation=0):
        with self._lock:
            self._open()
            if self._generation != int(generation):
                raise RuntimeError("stale forward state")
            b = len(self.ids)
            c = np.ascontiguousarray(cotangents, dtype=np.complex128)
            if c.shape != (b, self.ncoeff):
                raise ValueError(f"expected cotangents {(b, self.ncoeff)}, got {c.shape}")
            if not np.isfinite(c).all():
                raise ValueError("non-finite coefficient cotangent")
            sizes = [self.channels * i.size for i in self.ids]
            g = np.empty(sum(sizes), dtype=np.float64)
            if self._lib.stlp_patch_vjp(self._handle, _pointer(c.view(np.float64)), c.size, int(generation), _pointer(g)):
                raise _error(self._lib)
            out, at = [], 0
            for s, ids in zip(sizes, self.ids):
                out.append(g[at:at + s].reshape((ids.size,) if not self.cross else (2, ids.size)))
                at += s
            return np.stack(out) if self._dense else out

    def _device_input(self, x):
        n = sum(self.channels * i.size for i in self.ids)
        if not getattr(x, "is_cuda", False):
            raise ValueError("expected a CUDA tensor")
        if str(x.dtype) != "torch.float64" or not x.is_contiguous() or x.numel() != n:
            raise ValueError(f"expected a contiguous float64 CUDA tensor of {n} values (the patches concatenated)")
        ordinal = self.info()["device"]["ordinal"]
        if x.device.index != ordinal:
            raise ValueError(f"tensor on cuda:{x.device.index}, engine on device {ordinal}")
        return n

    def forward_device(self, x, moments=True, generation=0):
        """forward on device memory (backend="gpu"): x a contiguous float64
        CUDA tensor on the engine's device holding the patches of the
        selection concatenated (patch i: x[ids[i]], [2, n_i] flattened with
        cross=True), e.g. torch.cat([u[..., i].reshape(-1) for i in ids]).
        Values and variances are numpy arrays [B, ncoeff]."""
        import torch
        with self._lock:
            self._open()
            if not self.ids:
                raise RuntimeError("no disc selected")
            n = self._device_input(x)
            torch.cuda.synchronize(x.device)
            b = len(self.ids)
            values = np.empty((b, self.ncoeff), dtype=np.complex128)
            variance = np.empty((b, self.ncoeff if moments else 0), dtype=np.float64)
            self._generation = None
            if self._lib.stlp_patch_forward_device(self._handle, x.data_ptr(), n, int(generation), moments,
                                                   _pointer(values.view(np.float64)), _pointer(variance)):
                raise _error(self._lib)
            self._generation, self._device_shape = int(generation), (n, x.device)
            return values, variance

    def vjp_device(self, cotangents, generation=0):
        """vjp after forward_device: the patch cotangents as a float64 CUDA
        tensor laid out as the input of forward_device."""
        import torch
        with self._lock:
            self._open()
            if self._generation != int(generation) or getattr(self, "_device_shape", None) is None:
                raise RuntimeError("stale forward state")
            b = len(self.ids)
            c = np.ascontiguousarray(cotangents, dtype=np.complex128)
            if c.shape != (b, self.ncoeff):
                raise ValueError(f"expected cotangents {(b, self.ncoeff)}, got {c.shape}")
            if not np.isfinite(c).all():
                raise ValueError("non-finite coefficient cotangent")
            n, device = self._device_shape
            g = torch.empty(n, dtype=torch.float64, device=device)
            torch.cuda.synchronize(device)
            if self._lib.stlp_patch_vjp_device(self._handle, _pointer(c.view(np.float64)), c.size, int(generation), g.data_ptr()):
                raise _error(self._lib)
            return g

    def forward_cross(self, a, b, centres=None, radius=None, core=None, moments=True, generation=0):
        """Cross statistics S(a_i, b_i) of the patch pairs (cross=True)."""
        if not self.cross:
            raise RuntimeError("forward_cross needs PatchEngine(..., cross=True)")
        pairs = [np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]) for x, y in zip(a, b)]
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            pairs = np.stack(pairs)
        return self.forward(pairs, centres, radius, core, moments, generation)

    def vjp_cross(self, cotangents, generation=0):
        """(grad_a, grad_b) shaped like a and b."""
        g = self.vjp(cotangents, generation)
        if isinstance(g, np.ndarray):
            return g[:, 0], g[:, 1]
        return [x[0] for x in g], [x[1] for x in g]

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.stlp_patch_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
