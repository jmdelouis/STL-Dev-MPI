#!/usr/bin/env python3
"""Sample CPU RSS/HWM and per-process nvidia-smi memory during complete evaluations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCRIPT_VERSION='2026-10-29-v2-variance-ratio'
print(f'SCRIPT VERSION: benchmark_memory.py {SCRIPT_VERSION}',flush=True)

def cpu_bytes(pid):
    try:
        fields={s.split(':')[0]:s.split(':',1)[1].strip() for s in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in s}
        return {k:int(fields[k].split()[0])*1024 for k in ('VmRSS','VmHWM') if k in fields}
    except (OSError,ValueError): return {}

def gpu_bytes(pid):
    if not shutil.which('nvidia-smi'): return None
    try:
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
        if r.returncode: return None
        return sum(int(float(fields[1]))*1024**2 for line in r.stdout.splitlines() if len(fields:=line.split(','))==2 and fields[0].strip()==str(pid) and fields[1].strip().isdigit())
    except (subprocess.TimeoutExpired,OSError,ValueError): return None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/native128.json')
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['stream-replay','stream-checkpoint','resident'],default='stream-replay')
    p.add_argument('--backend',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--threads',type=int,default=int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    p.add_argument('--evaluations',type=int,default=2)
    p.add_argument('--numa-node',type=int,default=0)
    p.add_argument('--numa-policy',choices=['strict','cpu-only'],default='strict')
    p.add_argument('--no-numa',action='store_true',help='For CPU tests on non-GH200 hosts only')
    a=p.parse_args()
    if not 1<=a.evaluations<=10:p.error('1 through 10 complete evaluations only')
    out=Path(a.output).resolve()
    if out.exists():p.error('Output must not exist')
    out.mkdir(parents=True)
    cmd=[sys.executable,str(Path(__file__).with_name('synthesize.py')),'--config',a.config,'--mode',a.mode,'--backend',a.backend,'--threads',str(a.threads),'benchmark','--cache',a.cache,'--output',str(out/'run'),'--evaluations',str(a.evaluations)]
    if not a.no_numa:
        if not shutil.which('numactl'):p.error('numactl required for CPU-local placement; --no-numa is for non-GH200 CPU testing')
        prefix=['numactl',f'--cpunodebind={a.numa_node}']
        if a.numa_policy=='strict':prefix.append(f'--membind={a.numa_node}')
        cmd=prefix+cmd
    start=time.perf_counter()
    samples=[];cpu_peak=0;gpu_peak=None;next_gpu=0.;last_gpu=None
    with open(out/'process.log','w',buffering=1) as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            elapsed=time.perf_counter()-start
            cpu=cpu_bytes(proc.pid)
            cpu_peak=max(cpu_peak,cpu.get('VmRSS',0),cpu.get('VmHWM',0))
            if elapsed>=next_gpu:
                last_gpu=gpu_bytes(proc.pid);next_gpu=elapsed+1
                if last_gpu is not None:gpu_peak=max(gpu_peak or 0,last_gpu)
            samples.append({'seconds':elapsed,**cpu,'gpu_bytes_last_sample':last_gpu})
            time.sleep(.2)
    report={'version':SCRIPT_VERSION,'numa_policy':a.numa_policy if not a.no_numa else 'none',
            'memory_binding_enforced':not a.no_numa and a.numa_policy=='strict','command':cmd,'exit_code':proc.returncode,'elapsed_s':time.perf_counter()-start,
            'CPU_peak_RSS_or_kernel_HWM_bytes':cpu_peak,'GPU_sampled_peak_bytes':gpu_peak,
            'limitations':'GPU samples can miss short-lived allocations. CPU HWM read while process is alive can miss final allocations. Compare native phase-end peaks and CUDA profiling before capacity acceptance.',
            'samples':samples}
    (out/'memory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Memory report: {out / "memory.json"}; process exit {proc.returncode}',flush=True)
    return proc.returncode

if __name__=='__main__':raise SystemExit(main())
