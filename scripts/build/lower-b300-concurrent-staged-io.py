#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

G_OLD='''    if(ms.size){if(pg.use_mi)interval_io_kernel<false><<<interval_blocks(pg.mi_stage_count,threads),threads>>>(c.dA,dmi,pg.mi_stage_count,c.authMain);else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size,c.authMain,group_meta);}
    if(ds.size){if(pg.use_di)interval_io_kernel<false><<<interval_blocks(pg.di_stage_count,threads),threads>>>(c.dD,ddi,pg.di_stage_count,c.authBlock);else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size,c.authBlock,group_meta);}
    ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");'''
G_NEW='''    if(ms.size){if(pg.use_mi)interval_io_kernel<false><<<interval_blocks(pg.mi_stage_count,threads),threads,0,c.sMain>>>(c.dA,dmi,pg.mi_stage_count,c.authMain);else gather_main_kernel<<<bm,threads,0,c.sMain>>>(c.dA,useMate?c.dMate:nullptr,ms.size,c.authMain,group_meta);}
    if(ds.size){if(pg.use_di)interval_io_kernel<false><<<interval_blocks(pg.di_stage_count,threads),threads,0,c.sBlock>>>(c.dD,ddi,pg.di_stage_count,c.authBlock);else gather_block_kernel<<<bd,threads,0,c.sBlock>>>(c.dD,ds.size,c.authBlock,group_meta);}
    ck(cudaGetLastError(),"doubleD gather");ck(cudaStreamSynchronize(c.sMain),"main gather sync");ck(cudaStreamSynchronize(c.sBlock),"block gather sync");'''
S_OLD='''    if(ms.size){if(pg.use_mi)interval_io_kernel<true><<<interval_blocks(pg.mi_stage_count,threads),threads>>>(cur,dmi,pg.mi_stage_count,c.authMain);else scatter_main_kernel<<<bm,threads>>>(cur,ms.size,c.authMain,group_meta);}
    if(ds.size){if(pg.use_di)interval_io_kernel<true><<<interval_blocks(pg.di_stage_count,threads),threads>>>(dcur,ddi,pg.di_stage_count,c.authBlock);else scatter_block_kernel<<<bd,threads>>>(dcur,ds.size,c.authBlock,group_meta);}
    ck(cudaGetLastError(),"doubleD scatter");ck(cudaDeviceSynchronize(),"group sync");'''
S_NEW='''    if(ms.size){if(pg.use_mi)interval_io_kernel<true><<<interval_blocks(pg.mi_stage_count,threads),threads,0,c.sMain>>>(cur,dmi,pg.mi_stage_count,c.authMain);else scatter_main_kernel<<<bm,threads,0,c.sMain>>>(cur,ms.size,c.authMain,group_meta);}
    if(ds.size){if(pg.use_di)interval_io_kernel<true><<<interval_blocks(pg.di_stage_count,threads),threads,0,c.sBlock>>>(dcur,ddi,pg.di_stage_count,c.authBlock);else scatter_block_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,c.authBlock,group_meta);}
    ck(cudaGetLastError(),"doubleD scatter");ck(cudaStreamSynchronize(c.sMain),"main scatter sync");ck(cudaStreamSynchronize(c.sBlock),"block scatter sync");'''

def once(t,o,n,l):
    c=t.count(o)
    if c!=1:raise SystemExit(f'{l}: expected one bundled match, got {c}')
    return t.replace(o,n,1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();t=a.src.read_text();t=once(t,G_OLD,G_NEW,'gather');t=once(t,S_OLD,S_NEW,'scatter')
    for s in ('cudaDeviceSynchronize(),"doubleD gather sync"','cudaDeviceSynchronize(),"group sync"'):
        if s in t:raise SystemExit(f'device-wide I/O sync remains: {s}')
    for r in ('pg.mi_stage_count,threads),threads,0,c.sMain','pg.di_stage_count,threads),threads,0,c.sBlock','main gather sync','block gather sync','main scatter sync','block scatter sync'):
        if r not in t:raise SystemExit(f'missing concurrent staged I/O artifact: {r}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(t);print(f'lowered {a.out} concurrent_staged_io=1 gather_streams=2 scatter_streams=2 staged_intervals_preserved=1 metadata_kernelarg_preserved=1 devicewide_io_sync=0')
if __name__=='__main__':main()
