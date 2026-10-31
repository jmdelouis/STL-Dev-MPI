#!/usr/bin/env python3
"""Run the optional unchanged-source MF autograd comparison with Torch installed."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_VERSION="2026-10-28-v1"
print(f"SCRIPT VERSION: validate_reference_mf.py {SCRIPT_VERSION}",flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,help="Unchanged supplied minkowski_sht_loss.py")
    p.add_argument("--output",required=True,help="JSON validation report path")
    args=p.parse_args(argv)
    source=Path(args.source).resolve()
    if not source.is_file(): p.error(f"Reference source is missing: {source}")
    output=Path(args.output)
    if output.exists(): p.error("Choose a new report path; existing reports are not overwritten")
    report={"script_version":SCRIPT_VERSION,"source":str(source),
            "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "scope":"Unchanged reference compute_curves and normalization/autograd with an independently validated dense harmonic derivative operator; not the legacy all-Legendre backend calibration."}
    if importlib.util.find_spec("torch") is None:
        report.update(status="not_run_missing_torch",passed=False)
        code=2
    else:
        env=os.environ.copy();env["STL_REFERENCE_MF"]=str(source)
        root=Path(__file__).resolve().parent
        env["PYTHONPATH"]=str(root)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        test=str(root/"tests/test_harmonics_mf.py")+"::test_original_torch_source_integrand_vjp_when_available"
        result=subprocess.run([sys.executable,"-m","pytest","-q","-s",test],env=env,text=True,capture_output=True)
        code=result.returncode
        report.update(status="passed" if code==0 else "failed",passed=code==0,
                      returncode=code,stdout=result.stdout,stderr=result.stderr)
        print(result.stdout,flush=True)
        if result.stderr: print(result.stderr,file=sys.stderr,flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(f"[REFERENCE MF] {report['status']}; report {output.resolve()}",flush=True)
    return code


if __name__=="__main__": raise SystemExit(main())
