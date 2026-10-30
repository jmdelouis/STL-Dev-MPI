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

SCRIPT_VERSION = "2026-10-28-v1"
print(f"SCRIPT VERSION: stl_port.native {SCRIPT_VERSION}", flush=True)

_DOUBLE = C.POINTER(C.c_double)
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
        "stlp_lbfgs": ([_DOUBLE, C.c_size_t, C.c_char_p, _CALLBACK, _VOID], C.c_int64),
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
    """
    def __init__(self, nside, j, batch=1, backend="cpu", kernel=5,
                 orientations=4, block_pixels=4096, cache_bytes=1 << 30,
                 threads=1, cache="transposition", device=0, library=None):
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
        self.config = dict(nside=self.nside, j=self.j, batch=self.batch,
                           backend=backend, kernel=int(kernel), orientations=int(orientations),
                           block_pixels=int(block_pixels), cache_bytes=int(cache_bytes),
                           threads=int(threads), cache=cache, device=int(device))
        self._handle = self._lib.stlp_new(json.dumps(self.config).encode())
        if not self._handle:
            raise _error(self._lib)
        self.plan = self.info()["plan"]
        self.plan_digest = hashlib.sha256(json.dumps(self.plan, sort_keys=True).encode()).hexdigest()
        self.ncoeff = len(self.plan)
        self._current_key = None

    def _open(self):
        if not self._handle:
            raise RuntimeError("native engine is closed")

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
            if x.shape == (self.npix,) and self.batch == 1:
                x = x[None, :]
            if x.shape != (self.batch, self.npix):
                raise ValueError(f"expected maps {(self.batch, self.npix)}, got {x.shape}")
            if not np.isfinite(x).all():
                raise ValueError("input map contains non-finite values")
            values = np.empty((self.batch, self.ncoeff), dtype=np.complex128)
            variance = np.empty((self.batch, self.ncoeff if moments else 0), dtype=np.float64)
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
            if c.shape == (self.ncoeff,) and self.batch == 1:
                c = c[None, :]
            if c.shape != (self.batch, self.ncoeff):
                raise ValueError(f"expected cotangents {(self.batch,self.ncoeff)}, got {c.shape}")
            if not np.isfinite(c).all():
                raise ValueError("non-finite coefficient cotangent")
            grad = np.empty((self.batch, self.npix), dtype=np.float64)
            status = self._lib.stlp_vjp(self._handle, _pointer(c.view(np.float64)), c.size, *key, _pointer(grad))
            if status:
                raise _error(self._lib)
            return grad

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
