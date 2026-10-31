#!/usr/bin/env python3
"""Read-only host inventory and conservative capacity planning; no synthesis."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

SCRIPT_VERSION = '2026-10-28-v1'
print(f'SCRIPT VERSION: preflight.py {SCRIPT_VERSION}', flush=True)

def read(path):
    try: return Path(path).read_text().strip()
    except OSError: return None

def command(argv):
    if not shutil.which(argv[0]): return {'available':False}
    try:
        r=subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'available':True,'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    except (subprocess.TimeoutExpired,OSError) as exc: return {'available':True,'error':str(exc)}

def capacity(nside, batch=10, history=10):
    npix=12*nside*nside
    map_bytes=npix*8
    lmax=3*nside-1
    return {
        'kind':'planning estimates, not measurements', 'nside':nside, 'members':batch,
        'map_float64_bytes':map_bytes,
        'variables_gradient_history_pairs_bytes':map_bytes*batch*(2+2*history),
        'native_optimizer_additional_vectors':'Native L-BFGS also holds previous gradients, direction, line-search trial and callback copies; not included above',
        'checkpoint_bank_220_bytes_per_pixel_estimate':batch*npix*220,
        'packed_alms_complex128_bytes':batch*(lmax+1)*(lmax+2)//2*16,
        'training_physical_and_gaussian_float32_disk_bytes':2*54*npix*4,
        'exact_quantile_float64_workfile_disk_bytes':54*npix*8,
        'one_active_SC_fields_planning_bytes':10.3*1024**3*(nside/2048)**2,
        'additional_unbudgeted':'Geometry, native transient fields, one member MF derivatives, allocator peaks, OS/file cache, and exact quantile partition working set',
        'CPU_GHBM_counting':'NUMA node0 is CPU-local memory; do not add total free output to GPU memory',
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nside',type=int,default=2048)
    p.add_argument('--output',default='preflight.json')
    p.add_argument('--data-dir',default='.')
    a=p.parse_args()
    if a.nside<2 or a.nside&(a.nside-1): p.error('nside must be a power of two >=2')
    deps={}
    for name in ('numpy','scipy','healpy','ducc0','pytest','matplotlib'):
        try: deps[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: deps[name]=None
    node={}
    for d in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
        node[d.name]={'meminfo':read(d/'meminfo'),'cpulist':read(d/'cpulist')}
    cgroups={}
    for path in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current',
                 '/sys/fs/cgroup/memory.peak','/sys/fs/cgroup/cpuset.cpus.effective',
                 '/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        cgroups[path]=read(path)
    disk=shutil.disk_usage(a.data_dir)
    report={'version':SCRIPT_VERSION,'python':sys.version,'platform':platform.platform(),
            'machine':platform.machine(),'dependencies':deps,'NUMA_nodes':node,
            'process_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'process_status':read('/proc/self/status'),'cgroup_membership':read('/proc/self/cgroup'),
            'cgroup_limits':cgroups,'slurm':{k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','CUDA_VISIBLE_DEVICES')},
            'disk':dict(total=disk.total,used=disk.used,free=disk.free),
            'commands':{name:command(argv) for name,argv in {
                'nvidia_smi':['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv'],
                'numactl':['numactl','--show'],'nvcc':['nvcc','--version'],
                'rustc':['rustc','--version'],'cargo':['cargo','--version']}.items()},
            'capacity':capacity(a.nside),
            'acceptance':'Inventory only. A complete loss-plus-gradient benchmark with measured CPU/HBM peaks is required; this report does not authorize a production run.'}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(f'Preflight saved: {Path(a.output).resolve()}')
    return 0

if __name__=='__main__': raise SystemExit(main())
