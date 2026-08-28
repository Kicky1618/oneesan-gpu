#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

NGPU_OLD='__constant__ int D_MAIN_W,D_BLOCK_W,D_NGPU;'
NGPU_NEW='__constant__ int D_MAIN_W,D_BLOCK_W;'

SHARD_SYMBOLS_OLD='''__constant__ Count* D_MAIN_PTR[MAXGPU];
__constant__ Count* D_BLOCK_PTR[MAXGPU];
__constant__ Count* D_MAIN_VBASE;
__constant__ Count* D_BLOCK_VBASE;
__constant__ Code D_MAIN_CHUNK,D_BLOCK_CHUNK;'''
SHARD_SYMBOLS_NEW='''__constant__ Count* D_MAIN_VBASE;
__constant__ Count* D_BLOCK_VBASE;'''

INIT_OLD='''void init(int d,Count mod,Count**mp,Count**bp,Code mc,Code bc,int ng){dev=d;ck(cudaSetDevice(dev),"set init");ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"main ptrs");ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"block ptrs");ck(cudaMemcpyToSymbol(D_MAIN_CHUNK,&mc,sizeof(mc)),"main chunk");ck(cudaMemcpyToSymbol(D_BLOCK_CHUNK,&bc,sizeof(bc)),"block chunk");ck(cudaMemcpyToSymbol(D_NGPU,&ng,sizeof(ng)),"ngpu");ck(cudaStreamCreateWithFlags(&sMain,cudaStreamNonBlocking),"stream main");ck(cudaStreamCreateWithFlags(&sBlock,cudaStreamNonBlocking),"stream block");ck(cudaEventCreateWithFlags(&copyDone,cudaEventDisableTiming),"event copy");ck(cudaEventCreateWithFlags(&clearDone,cudaEventDisableTiming),"event clear");ck(cudaEventCreateWithFlags(&mainDone,cudaEventDisableTiming),"event main");ck(cudaEventCreateWithFlags(&blockDone,cudaEventDisableTiming),"event block");}'''
INIT_NEW='''void init(int d,Count mod){dev=d;ck(cudaSetDevice(dev),"set init");ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");ck(cudaStreamCreateWithFlags(&sMain,cudaStreamNonBlocking),"stream main");ck(cudaStreamCreateWithFlags(&sBlock,cudaStreamNonBlocking),"stream block");ck(cudaEventCreateWithFlags(&copyDone,cudaEventDisableTiming),"event copy");ck(cudaEventCreateWithFlags(&clearDone,cudaEventDisableTiming),"event clear");ck(cudaEventCreateWithFlags(&mainDone,cudaEventDisableTiming),"event main");ck(cudaEventCreateWithFlags(&blockDone,cudaEventDisableTiming),"event block");}'''

INIT_CALL_OLD='ctx[d].init(d,mod,mp,bp,mc,bc,ng);'
INIT_CALL_NEW='ctx[d].init(d,mod);'


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1:
        raise SystemExit(f'{label}: expected exactly one generated-source match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('src',type=Path)
    ap.add_argument('out',type=Path)
    a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,NGPU_OLD,NGPU_NEW,'D_NGPU declaration')
    text=once(text,SHARD_SYMBOLS_OLD,SHARD_SYMBOLS_NEW,'stale shard symbol declarations')
    text=once(text,INIT_OLD,INIT_NEW,'DeviceCtx stale shard symbol copies')
    text=once(text,INIT_CALL_OLD,INIT_CALL_NEW,'DeviceCtx stale shard init arguments')
    for token in ('D_MAIN_PTR','D_BLOCK_PTR','D_MAIN_CHUNK','D_BLOCK_CHUNK','D_NGPU'):
        if token in text:
            raise SystemExit(f'stale VMM shard symbol remains after prune: {token}')
    if 'init(d,mod,mp,bp,mc,bc,ng)' in text:
        raise SystemExit('stale VMM DeviceCtx init arguments remain after prune')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(text)
    print(f'pruned {a.out} stale_shard_symbols=0 stale_shard_symbol_copies=0 stale_shard_init_args=0 direct_vmm_symbols=2')

if __name__=='__main__':
    main()
