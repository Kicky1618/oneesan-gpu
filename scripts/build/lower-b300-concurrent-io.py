#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

GATHER_OLD='''    if(ms.size){if(pg.use_mi)interval_io_kernel<false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size(),c.authMain);else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size,c.authMain);}
    if(ds.size){if(pg.use_di)interval_io_kernel<false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size(),c.authBlock);else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size,c.authBlock);}
    ck(cudaGetLastError(),"doubleD gather");ck(cudaDeviceSynchronize(),"doubleD gather sync");'''
GATHER_NEW='''    if(ms.size){if(pg.use_mi)interval_io_kernel<false><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>(c.dA,c.dIM,pg.mi.size(),c.authMain);else gather_main_kernel<<<bm,threads,0,c.sMain>>>(c.dA,useMate?c.dMate:nullptr,ms.size,c.authMain);}
    if(ds.size){if(pg.use_di)interval_io_kernel<false><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>(c.dD,c.dID,pg.di.size(),c.authBlock);else gather_block_kernel<<<bd,threads,0,c.sBlock>>>(c.dD,ds.size,c.authBlock);}
    ck(cudaGetLastError(),"doubleD gather");ck(cudaStreamSynchronize(c.sMain),"main gather sync");ck(cudaStreamSynchronize(c.sBlock),"block gather sync");'''

SCATTER_OLD='''    if(ms.size){if(pg.use_mi)interval_io_kernel<true><<<interval_blocks(pg.mi.size(),threads),threads>>>(cur,c.dIM,pg.mi.size(),c.authMain);else scatter_main_kernel<<<bm,threads>>>(cur,ms.size,c.authMain);}
    if(ds.size){if(pg.use_di)interval_io_kernel<true><<<interval_blocks(pg.di.size(),threads),threads>>>(dcur,c.dID,pg.di.size(),c.authBlock);else scatter_block_kernel<<<bd,threads>>>(dcur,ds.size,c.authBlock);}
    ck(cudaGetLastError(),"doubleD scatter");ck(cudaDeviceSynchronize(),"group sync");'''
SCATTER_NEW='''    if(ms.size){if(pg.use_mi)interval_io_kernel<true><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>(cur,c.dIM,pg.mi.size(),c.authMain);else scatter_main_kernel<<<bm,threads,0,c.sMain>>>(cur,ms.size,c.authMain);}
    if(ds.size){if(pg.use_di)interval_io_kernel<true><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>(dcur,c.dID,pg.di.size(),c.authBlock);else scatter_block_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,c.authBlock);}
    ck(cudaGetLastError(),"doubleD scatter");ck(cudaStreamSynchronize(c.sMain),"main scatter sync");ck(cudaStreamSynchronize(c.sBlock),"block scatter sync");'''


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1: raise SystemExit(f'{label}: expected exactly one basearg match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    text=a.src.read_text();text=once(text,GATHER_OLD,GATHER_NEW,'concurrent gather');text=once(text,SCATTER_OLD,SCATTER_NEW,'concurrent scatter')
    for stale in ('cudaDeviceSynchronize(),"doubleD gather sync"','cudaDeviceSynchronize(),"group sync"'):
        if stale in text: raise SystemExit(f'serial I/O synchronization remains: {stale}')
    for required in ('threads,0,c.sMain','threads,0,c.sBlock','main gather sync','block gather sync','main scatter sync','block scatter sync'):
        if required not in text: raise SystemExit(f'missing concurrent I/O artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} concurrent_authoritative_io=1 gather_streams=2 scatter_streams=2 devicewide_io_sync=0 main_block_overlap_candidate=1')

if __name__=='__main__':main()
