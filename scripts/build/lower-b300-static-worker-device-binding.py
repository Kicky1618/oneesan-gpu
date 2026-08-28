#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ENSURE_OLD='void ensure(Code m,Code b,bool useMate,size_t im,size_t id){ck(cudaSetDevice(dev),"set ensure");'
ENSURE_NEW='void ensure(Code m,Code b,bool useMate,size_t im,size_t id){'
PROCESS_OLD='auto t0=std::chrono::steady_clock::now();ck(cudaSetDevice(c.dev),"set worker");'
PROCESS_NEW='auto t0=std::chrono::steady_clock::now();'
WORKER_OLD='for(int d=0;d<ng;++d)ths.emplace_back([&,d]{for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);});'
WORKER_NEW='for(int d=0;d<ng;++d)ths.emplace_back([&,d]{ck(cudaSetDevice(d),"set static worker device");for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);});'

def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1:raise SystemExit(f'{label}: expected exactly one static-LPT match, got {n}')
    return text.replace(old,new,1)

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();text=a.src.read_text()
    text=once(text,ENSURE_OLD,ENSURE_NEW,'remove per-group ensure cudaSetDevice')
    text=once(text,PROCESS_OLD,PROCESS_NEW,'remove per-group process cudaSetDevice')
    text=once(text,WORKER_OLD,WORKER_NEW,'bind worker device once')
    for stale in ('cudaSetDevice(c.dev),"set worker"','cudaSetDevice(dev),"set ensure"'):
        if stale in text:raise SystemExit(f'per-group device binding remains: {stale}')
    if text.count('cudaSetDevice(d),"set static worker device"')!=1:raise SystemExit('static worker device binding missing or duplicated')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} static_worker_device_binding=1 per_group_cudaSetDevice_calls_removed=2 expected_default_group_processings=458752 expected_old_worker_cudaSetDevice_calls=917504 expected_new_worker_cudaSetDevice_calls=448 call_reduction=2048x')

if __name__=='__main__':main()
