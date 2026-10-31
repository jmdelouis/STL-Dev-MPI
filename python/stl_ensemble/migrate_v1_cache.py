#!/usr/bin/env python3
"""Migrate the approved v1 measurements into a separately identified v2 cache."""
from __future__ import annotations
import argparse
import os
from stl_port.config import Config
from stl_port.migrate_cache import migrate_v1

print("SCRIPT VERSION: migrate_v1_cache.py 2026-10-29-v2-variance-ratio", flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--v1-package',required=True)
    p.add_argument('--config',default='configs/native128.json')
    p.add_argument('--native-library',default=os.environ.get('STL_NATIVE_LIBRARY'))
    a=p.parse_args()
    if not a.native_library:p.error('--native-library or STL_NATIVE_LIBRARY is required')
    migrate_v1(Config.read(a.config),a.source,a.output,native_library=a.native_library,v1_package=a.v1_package)
    return 0

if __name__=='__main__':raise SystemExit(main())
