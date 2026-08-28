#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE/'gen-b300-shard-address-hi32-seed.py'
spec=importlib.util.spec_from_file_location('b300_hi32_seed_generator',PARENT)
if spec is None or spec.loader is None: raise SystemExit(f'cannot load {PARENT}')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
helper=p.helper
old='static_assert(B300_SHARD_ADDRESS_MODE>=0&&B300_SHARD_ADDRESS_MODE<=5,"B300_SHARD_ADDRESS_MODE must be 0..5");'
new='static_assert(B300_SHARD_ADDRESS_MODE>=0&&B300_SHARD_ADDRESS_MODE<=6,"B300_SHARD_ADDRESS_MODE must be 0..6");'
if helper.count(old)!=1: raise SystemExit('parent mode assertion changed')
helper=helper.replace(old,new,1)
marker='#if B300_SHARD_ADDRESS_MODE == 5\n'
mode6=r'''struct ShardAddrPair32{std::uint32_t lo,hi;};
__device__ __forceinline__ void shard_addr_add_pair32(ShardAddrPair32& p,std::uint32_t lo,std::uint32_t hi,std::uint32_t bit){std::uint32_t mask=0u-bit,xlo=lo&mask,xhi=hi&mask,old=p.lo;p.lo+=xlo;p.hi+=xhi+std::uint32_t(p.lo<old);}
__device__ __forceinline__ ShardAddrPair32 shard_addr_base_main_u32(std::uint32_t o){ShardAddrPair32 p{0u,0u};shard_addr_add_pair32(p,970298072u,11u,o&1u);shard_addr_add_pair32(p,1940596144u,22u,(o>>1)&1u);shard_addr_add_pair32(p,3881192288u,44u,(o>>2)&1u);return p;}
__device__ __forceinline__ ShardAddrPair32 shard_addr_base_block_u32(std::uint32_t o){ShardAddrPair32 p{0u,0u};shard_addr_add_pair32(p,3992036288u,3u,o&1u);shard_addr_add_pair32(p,3689105280u,7u,(o>>1)&1u);shard_addr_add_pair32(p,3083243264u,15u,(o>>2)&1u);return p;}
__device__ __forceinline__ ShardAddress8 shard_addr_finish_u32(Code g,std::uint32_t o,ShardAddrPair32 b,std::uint32_t clo,std::uint32_t chi){std::uint32_t glo=std::uint32_t(g),ghi=std::uint32_t(g>>32),rlo=glo-b.lo,borrow=std::uint32_t(glo<b.lo),rhi=ghi-b.hi-borrow;std::uint32_t corr=std::uint32_t(rhi>chi||(rhi==chi&&rlo>=clo)),mask=0u-corr,sublo=clo&mask,subhi=chi&mask,old=rlo;rlo-=sublo;rhi-=subhi+std::uint32_t(old<sublo);o+=corr;return{int(o),(Code(rhi)<<32)|Code(rlo)};}
__device__ __forceinline__ ShardAddress8 shard_address8_hi32_u32_main_w28_g8(Code g){return shard_addr_finish_u32(g,shard_hi32_seed_main(g),shard_addr_base_main_u32(shard_hi32_seed_main(g)),970298072u,11u);}
__device__ __forceinline__ ShardAddress8 shard_address8_hi32_u32_block_w28_g8(Code g){return shard_addr_finish_u32(g,shard_hi32_seed_block(g),shard_addr_base_block_u32(shard_hi32_seed_block(g)),3992036288u,3u);}
#if B300_SHARD_ADDRESS_MODE == 6
static_assert(TARGET_W==28,"fully-u32 hi32-seed shard address is specialized for TARGET_W=28");
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8_hi32_u32_main_w28_g8(g);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8_hi32_u32_block_w28_g8(g);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8_hi32_u32_main_w28_g8(g);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8_hi32_u32_block_w28_g8(g);D_BLOCK_PTR[a.owner][a.local]=v;}
#elif B300_SHARD_ADDRESS_MODE == 5
'''
if helper.count(marker)!=1: raise SystemExit('parent mode-5 marker changed')
helper=helper.replace(marker,mode6,1)

def once(text,old,new,label):
 c=text.count(old)
 if c!=1: raise SystemExit(f'{label}: expected one source match, got {c}')
 return text.replace(old,new,1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();text=a.src.read_text()
 text=once(text,p.base.HELPER_OLD,helper,'shard helper');text=once(text,p.base.NGPU_OLD,p.base.NGPU_NEW,'ngpu guard');text=once(text,p.base.CHUNK_OLD,p.base.CHUNK_NEW,'chunk guard')
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text);print(f'generated {a.out} shard_address_modes=0..6 fully_u32_hi32_mode=6 guarded_w28_ngpu8=1')
if __name__=='__main__': main()
