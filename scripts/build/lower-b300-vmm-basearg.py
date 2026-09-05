#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

BASE_SYMBOLS_OLD='''__constant__ Count* D_MAIN_VBASE;
__constant__ Count* D_BLOCK_VBASE;'''
BASE_SYMBOLS_NEW=''

ACCESS_OLD='''__device__ __forceinline__ Count global_load_main(Code g){return D_MAIN_VBASE[g];}
__device__ __forceinline__ Count global_load_block(Code g){return D_BLOCK_VBASE[g];}
__device__ __forceinline__ void global_store_main(Code g,Count v){D_MAIN_VBASE[g]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){D_BLOCK_VBASE[g]=v;}'''
ACCESS_NEW='''__device__ __forceinline__ Count global_load_main(const Count* base,Code g){return base[g];}
__device__ __forceinline__ Count global_load_block(const Count* base,Code g){return base[g];}
__device__ __forceinline__ void global_store_main(Count* base,Code g,Count v){base[g]=v;}
__device__ __forceinline__ void global_store_block(Count* base,Code g,Count v){base[g]=v;}'''

GATHER_OLD='''__global__ void gather_main_kernel(Count*out,MateID*mates,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;MateID m=unrank_group_global_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP,g);out[i]=global_load_main(g);if(mates)mates[i]=m;}}
__global__ void gather_block_kernel(Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);out[i]=global_load_block(g);}}
__global__ void scatter_main_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP,g);global_store_main(g,in[i]);}}
__global__ void scatter_block_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);global_store_block(g,in[i]);}}'''
GATHER_NEW='''__global__ void gather_main_kernel(Count*out,MateID*mates,Code n,const Count* auth){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;MateID m=unrank_group_global_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP,g);out[i]=global_load_main(auth,g);if(mates)mates[i]=m;}}
__global__ void gather_block_kernel(Count*out,Code n,const Count* auth){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);out[i]=global_load_block(auth,g);}}
__global__ void scatter_main_kernel(const Count*in,Code n,Count* auth){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP,g);global_store_main(auth,g,in[i]);}}
__global__ void scatter_block_kernel(const Count*in,Code n,Count* auth){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);global_store_block(auth,g,in[i]);}}'''

INTERVAL_OLD='''template<bool BLOCK,bool SCATTER>
__global__ void interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv){
    for(size_t k=blockIdx.x;k<niv;k+=gridDim.x){
        PeerInterval x=iv[k];
        Count*peer=(BLOCK?D_BLOCK_VBASE:D_MAIN_VBASE)+x.remote;
        for(Code off=threadIdx.x;off<x.len;off+=blockDim.x){
            if constexpr(SCATTER) peer[off]=buf[x.local+off];
            else buf[x.local+off]=peer[off];
        }
    }
}'''
INTERVAL_NEW='''template<bool SCATTER>
__global__ void interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv,Count*auth){
    for(size_t k=blockIdx.x;k<niv;k+=gridDim.x){
        PeerInterval x=iv[k];
        Count*peer=auth+x.remote;
        for(Code off=threadIdx.x;off<x.len;off+=blockDim.x){
            if constexpr(SCATTER) peer[off]=buf[x.local+off];
            else buf[x.local+off]=peer[off];
        }
    }
}'''

CTX_FIELDS_OLD='''int dev=-1;uint8_t*arena=nullptr;'''
CTX_FIELDS_NEW='''int dev=-1;Count*authMain=nullptr,*authBlock=nullptr;uint8_t*arena=nullptr;'''

CTX_INIT_OLD='''void init(int d,Count mod){dev=d;ck(cudaSetDevice(dev),"set init");'''
CTX_INIT_NEW='''void init(int d,Count mod,Count*main_base,Count*block_base){dev=d;authMain=main_base;authBlock=block_base;ck(cudaSetDevice(dev),"set init");'''

CTX_BUILD_OLD='''std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set VMM symbols");ck(cudaMemcpyToSymbol(D_MAIN_VBASE,&main_base,sizeof(main_base)),"main VMM base");ck(cudaMemcpyToSymbol(D_BLOCK_VBASE,&block_base,sizeof(block_base)),"block VMM base");ctx[d].init(d,mod);}'''
CTX_BUILD_NEW='''std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d)ctx[d].init(d,mod,main_base,block_base);'''

MAIN_GATHER_OLD='''if(ms.size){if(pg.use_mi)interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size());else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size);}'''
MAIN_GATHER_NEW='''if(ms.size){if(pg.use_mi)interval_io_kernel<false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size(),c.authMain);else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size,c.authMain);}'''
BLOCK_GATHER_OLD='''if(ds.size){if(pg.use_di)interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size());else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size);}'''
BLOCK_GATHER_NEW='''if(ds.size){if(pg.use_di)interval_io_kernel<false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size(),c.authBlock);else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size,c.authBlock);}'''
MAIN_SCATTER_OLD='''if(ms.size){if(pg.use_mi)interval_io_kernel<false,true><<<interval_blocks(pg.mi.size(),threads),threads>>>(cur,c.dIM,pg.mi.size());else scatter_main_kernel<<<bm,threads>>>(cur,ms.size);}'''
MAIN_SCATTER_NEW='''if(ms.size){if(pg.use_mi)interval_io_kernel<true><<<interval_blocks(pg.mi.size(),threads),threads>>>(cur,c.dIM,pg.mi.size(),c.authMain);else scatter_main_kernel<<<bm,threads>>>(cur,ms.size,c.authMain);}'''
BLOCK_SCATTER_OLD='''if(ds.size){if(pg.use_di)interval_io_kernel<true,true><<<interval_blocks(pg.di.size(),threads),threads>>>(dcur,c.dID,pg.di.size());else scatter_block_kernel<<<bd,threads>>>(dcur,ds.size);}'''
BLOCK_SCATTER_NEW='''if(ds.size){if(pg.use_di)interval_io_kernel<true><<<interval_blocks(pg.di.size(),threads),threads>>>(dcur,c.dID,pg.di.size(),c.authBlock);else scatter_block_kernel<<<bd,threads>>>(dcur,ds.size,c.authBlock);}'''


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1: raise SystemExit(f'{label}: expected exactly one pruned-VMM match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,BASE_SYMBOLS_OLD,BASE_SYMBOLS_NEW,'VMM base symbols')
    text=once(text,ACCESS_OLD,ACCESS_NEW,'VMM access helpers')
    text=once(text,GATHER_OLD,GATHER_NEW,'gather/scatter base args')
    text=once(text,INTERVAL_OLD,INTERVAL_NEW,'interval base arg')
    text=once(text,CTX_FIELDS_OLD,CTX_FIELDS_NEW,'DeviceCtx auth fields')
    text=once(text,CTX_INIT_OLD,CTX_INIT_NEW,'DeviceCtx auth init')
    text=once(text,CTX_BUILD_OLD,CTX_BUILD_NEW,'remove VMM base symbol copies')
    text=once(text,MAIN_GATHER_OLD,MAIN_GATHER_NEW,'main gather base arg')
    text=once(text,BLOCK_GATHER_OLD,BLOCK_GATHER_NEW,'block gather base arg')
    text=once(text,MAIN_SCATTER_OLD,MAIN_SCATTER_NEW,'main scatter base arg')
    text=once(text,BLOCK_SCATTER_OLD,BLOCK_SCATTER_NEW,'block scatter base arg')
    for token in ('D_MAIN_VBASE','D_BLOCK_VBASE','cudaMemcpyToSymbol(D_MAIN_VBASE','cudaMemcpyToSymbol(D_BLOCK_VBASE','template<bool BLOCK,bool SCATTER>'):
        if token in text: raise SystemExit(f'base-symbol/unused-template artifact remains after kernel-arg lowering: {token}')
    for required in ('template<bool SCATTER>','global_load_main(const Count* base,Code g)','global_store_main(Count* base,Code g,Count v)','Count*authMain=nullptr,*authBlock=nullptr','ctx[d].init(d,mod,main_base,block_base)','c.authMain','c.authBlock'):
        if required not in text: raise SystemExit(f'missing base-arg production artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} vmm_base_source=kernel_param vmm_base_symbols=0 vmm_base_symbol_copies=0 interval_template_axes=1 direct_global_index=1 shard_free=1 compact_interval_bytes=24')

if __name__=='__main__':main()
