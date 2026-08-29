#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
G_OLD='''    ck(cudaGetLastError(),"doubleD gather");ck(cudaStreamSynchronize(c.sMain),"main gather sync");ck(cudaStreamSynchronize(c.sBlock),"block gather sync");'''
G_NEW='''    ck(cudaGetLastError(),"doubleD gather");'''
S_OLD='''    ck(cudaGetLastError(),"doubleD scatter");ck(cudaStreamSynchronize(c.sMain),"main scatter sync");ck(cudaStreamSynchronize(c.sBlock),"block scatter sync");'''
S_NEW='''    ck(cudaGetLastError(),"doubleD scatter");ck(cudaEventRecord(c.blockDone,c.sBlock),"record block scatter done");ck(cudaStreamWaitEvent(c.sMain,c.blockDone,0),"main wait block scatter");ck(cudaStreamSynchronize(c.sMain),"scatter join sync");'''
def once(t,o,n,l):
    c=t.count(o)
    if c!=1:raise SystemExit(f'{l}: expected one concurrent staged match, got {c}')
    return t.replace(o,n,1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();t=a.src.read_text();t=once(t,G_OLD,G_NEW,'remove gather host sync');t=once(t,S_OLD,S_NEW,'join scatter with one sync')
    for s in ('main gather sync','block gather sync','main scatter sync','block scatter sync'):
        if s in t:raise SystemExit(f'old host I/O sync remains: {s}')
    for r in ('record block scatter done','main wait block scatter','scatter join sync','cudaEventRecord(c.copyDone,c.sMain)','cudaEventRecord(c.clearDone,c.sBlock)','cudaStreamWaitEvent(c.sMain,c.clearDone,0)','cudaStreamWaitEvent(c.sBlock,c.copyDone,0)'):
        if r not in t:raise SystemExit(f'missing async-I/O dependency artifact: {r}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(t);print(f'lowered {a.out} gather_host_syncs_per_group=0 scatter_host_syncs_per_group=1 old_io_host_syncs_per_group=4 new_io_host_syncs_per_group=1 dependency_preserved_by_stream_order_and_events=1')
if __name__=='__main__':main()
