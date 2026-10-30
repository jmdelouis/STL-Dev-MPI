"""Strict scientific configuration for the bounded STL ensemble port."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from .sc_variance import CONTRACT as SC_VARIANCE_CONTRACT

SCRIPT_VERSION = "2026-10-29-v2-variance-ratio"
print(f"SCRIPT VERSION: config.py {SCRIPT_VERSION}")
BASELINE_REVISION = "f083c4a753a244d1edf69fe4f944fa23fc5cdb76"
SOURCE_SHA256 = "1c507030ab95145c84405ac7e01426f0c9828a402d92d3a56ea7887b3f09b447"

@dataclasses.dataclass(frozen=True)
class Config:
    nside: int = 128
    members: int = 10
    kernel: int = 5
    orientations: int = 4
    mode: str = "stream-replay"
    backend: str = "gpu"
    j: int | None = None
    threads: int = 1
    block_pixels: int = 4096
    cache_bytes: int = 8 * 1024**3
    bootstrap_seed: int = 20260914
    initialization_seed: int = 61022026
    ps_variance_upper: float | None = 383.0
    dtype: str = "float64"
    optimizer: str = "native-lbfgs-strong-wolfe"
    history_size: int = 10
    learning_rate: float = 1.0
    tolerance_grad: float = 1e-12
    tolerance_change: float = 1e-15
    native_revision: str = BASELINE_REVISION

    @property
    def npix(self): return 12 * self.nside**2
    @property
    def lmax(self): return 3 * self.nside - 1
    @property
    def scales(self): return self.j if self.j is not None else int(math.log2(self.nside)) + 1
    @property
    def active_batch(self): return self.members if self.mode == "resident" else 1

    def validate(self, *, scientific=True):
        if self.nside < 2 or self.nside & (self.nside-1):
            raise ValueError("nside must be a positive HEALPix power of two >=2")
        if scientific and self.nside not in (128,2048):
            raise ValueError("Scientific configurations are native nside128 or nside2048; other grids are numerical tests only")
        if self.members != 10 or self.kernel != 5 or self.orientations != 4:
            raise ValueError("Fixed scientific configuration requires B=10, kernel=5, orientations=4")
        if self.dtype != "float64" or self.optimizer != "native-lbfgs-strong-wolfe":
            raise ValueError("The port uses native float64 and joint native L-BFGS/strong-Wolfe only")
        if self.native_revision != BASELINE_REVISION:
            raise ValueError("Unreviewed native revision; this package targets the pinned baseline plus bundled adapter patch")
        if self.mode == "resident" and self.nside > 128:
            raise ValueError("Resident B=10 is a numerical reference at manageable nside<=128 only")
        if self.mode not in ("resident","stream-replay","stream-checkpoint") or self.backend not in ("cpu","gpu"):
            raise ValueError("Invalid execution mode/backend")
        if not 1 <= self.scales <= int(math.log2(self.nside)) + 1:
            raise ValueError("Scattering scales exceed native grid")
        if self.nside == 128 and self.ps_variance_upper != 383:
            raise ValueError("nside128 physical-power variance cutoff must remain 383")
        if self.nside == 2048 and self.ps_variance_upper is None:
            raise ValueError("Resolve ps_variance_upper explicitly for native2048 before calibration or scientific evaluation; see METHOD_CONTRACT.md")
        if self.ps_variance_upper is not None and not 20 <= self.ps_variance_upper <= self.lmax:
            if scientific: raise ValueError("Power variance upper cutoff outside [20,lmax]")
        if self.threads < 1 or self.cache_bytes <= 0 or self.history_size < 1:
            raise ValueError("Positive threads/cache_bytes/history_size required")
        return self

    def manifest(self):
        result = dataclasses.asdict(self)
        result.update(sc_variance_contract=dict(SC_VARIANCE_CONTRACT), scales=self.scales, active_batch=self.active_batch,
                      optimization_nside=self.nside, finest_sc_nside=self.nside,
                      scattering_grid_nsides=[self.nside//(2**j) for j in range(self.scales)],
                      scale_selection="all native grids nside through 1; removes only source finest oversampled scale",
                      native_ordering="NEST", harmonic_ordering="RING",lmax=self.lmax,
                      train_ids=list(range(54)),heldout_ids=list(range(54,108)),
                      source_sha256=SOURCE_SHA256,script_version=SCRIPT_VERSION,
                      five_block_average=5,optimizer_family="L-BFGS",line_search="strong-Wolfe")
        return result

    @classmethod
    def read(cls, path, *, use_environment=True):
        data=json.loads(Path(path).read_text())
        unknown=set(data)-{f.name for f in dataclasses.fields(cls)}
        if unknown: raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if use_environment:
            for name,env in (("bootstrap_seed","V41_BOOTSTRAP_TARGET_SEED"),("initialization_seed","V41_LARGE_INIT_SEED")):
                if env in os.environ: data[name]=int(os.environ[env])
        return cls(**data)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""): h.update(b)
    return h.hexdigest()
