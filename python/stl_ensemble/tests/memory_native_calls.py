#!/usr/bin/env python3
"""Peak device and host memory of every trainer call of a training run:
runs train_native_unet.py (its arguments after `--`) with every NativeUNet
method that reaches the library followed by `memory(reset=True)`, and
prints, per method, the largest allocator high-water mark, the device's
total use at that mark and the host peak RSS, then the parts of the last
report (statistics engines: level fields, geometry cache; noise pools).

    python tests/memory_native_calls.py --out calls.json -- --data DIR --nside 256 ... (train_native_unet.py arguments)
"""
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stl_port.native import NativeUNet  # noqa: E402

argv = sys.argv[1:]
cut = argv.index("--")
out_path = argv[argv.index("--out") + 1] if "--out" in argv[:cut] else None
sys.argv = [str(Path(__file__).resolve().parents[1] / "train_native_unet.py")] + argv[cut + 1:]

calls = {}
order = []
last = {}


def wrap(name):
    f = getattr(NativeUNet, name)

    def g(self, *args, **kw):
        before = (self.memory(reset=True).get("device") or {}).get("pool_used_bytes", 0)
        out = f(self, *args, **kw)
        m = self.memory(reset=True)
        d = m.get("device") or {}
        rec = calls.setdefault(name, {"count": 0, "pool_used_high": 0, "pool_reserved_high": 0, "device_used_high": 0, "host_rss_high": 0,
                                      "resident_before": 0, "transient_high": 0})
        rec["count"] += 1
        rec["resident_before"] = max(rec["resident_before"], before)
        rec["transient_high"] = max(rec["transient_high"], d.get("pool_used_high_bytes", 0) - before)
        for k, v in (("pool_used_high", d.get("pool_used_high_bytes", 0)), ("pool_reserved_high", d.get("pool_reserved_high_bytes", 0)),
                     ("device_used_high", d.get("device_used_high_bytes", 0)), ("host_rss_high", m["host"]["rss_high_bytes"])):
            rec[k] = max(rec[k], v)
        if name not in order:
            order.append(name)
        last.update(m)
        return out
    setattr(NativeUNet, name, g)


for name in ("set_masks", "set_common", "set_joint", "set_field", "set_cond", "set_cl", "estimate_bias", "step", "live",
             "loss", "apply", "front"):
    if hasattr(NativeUNet, name):
        wrap(name)

try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    G = 2.0 ** 30
    print("call            count  pool_used  reserved  device_total  host_rss  resident  transient  (GiB, peaks during the call;"
          " resident: allocator use before the call, transient: peak above it)")
    for name in order:
        r = calls[name]
        print(f"{name:15s} {r['count']:5d} {r['pool_used_high'] / G:10.3f} {r['pool_reserved_high'] / G:9.3f} "
              f"{r['device_used_high'] / G:13.3f} {r['host_rss_high'] / G:9.3f} {r['resident_before'] / G:9.3f} {r['transient_high'] / G:10.3f}")
    d = last.get("device") or {}
    for k, e in d.get("engines", {}).items():
        print(f"engine {k}: fields now {e['field_bytes_now'] / G:.3f} GiB ({e['field_store']}), geometry cache {e['geometry_cache_used_bytes'] / G:.3f} "
              f"of {e['geometry_cache_budget_bytes'] / G:.3f} GiB, decimation {e['decimation_bytes'] / G:.3f} GiB, batches {e['batches']}")
    print("trainer", {k: round(v / G, 3) for k, v in (last.get("trainer") or {}).items()})
    if out_path:
        with open(out_path, "w") as fh:
            json.dump({"calls": calls, "order": order, "last": last}, fh, indent=1)
