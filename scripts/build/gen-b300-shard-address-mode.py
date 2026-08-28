#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

HELPER_OLD = r'''struct ShardAddress8{int owner;Code local;};
__device__ __forceinline__ ShardAddress8 shard_address8(Code g,Code chunk){int o=0;Code c4=chunk<<2;if(g>=c4){g-=c4;o|=4;}Code c2=chunk<<1;if(g>=c2){g-=c2;o|=2;}if(g>=chunk){g-=chunk;o|=1;}return{o,g};}
#if B300_FAST_SHARD_ADDRESS8
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8(g,D_MAIN_CHUNK);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8(g,D_BLOCK_CHUNK);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8(g,D_MAIN_CHUNK);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8(g,D_BLOCK_CHUNK);D_BLOCK_PTR[a.owner][a.local]=v;}
#else
__device__ __forceinline__ Count global_load_main(Code g){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK];}
__device__ __forceinline__ Count global_load_block(Code g){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK];}
__device__ __forceinline__ void global_store_main(Code g,Count v){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK]=v;}
#endif'''

HELPER_NEW = r'''struct ShardAddress8{int owner;Code local;};
#ifndef B300_SHARD_ADDRESS_MODE
#define B300_SHARD_ADDRESS_MODE B300_FAST_SHARD_ADDRESS8
#endif
static_assert(B300_SHARD_ADDRESS_MODE>=0&&B300_SHARD_ADDRESS_MODE<=4,"B300_SHARD_ADDRESS_MODE must be 0..4");
__device__ __forceinline__ ShardAddress8 shard_address8(Code g,Code chunk){int o=0;Code c4=chunk<<2;if(g>=c4){g-=c4;o|=4;}Code c2=chunk<<1;if(g>=c2){g-=c2;o|=2;}if(g>=chunk){g-=chunk;o|=1;}return{o,g};}
__device__ __forceinline__ Code shard_base8_masked(int owner,Code chunk){Code u=Code(unsigned(owner));Code b0=(Code(0)-(u&1ULL))&chunk;Code b1=(Code(0)-((u>>1)&1ULL))&(chunk<<1);Code b2=(Code(0)-((u>>2)&1ULL))&(chunk<<2);return b0+b1+b2;}
template<std::uint32_t MLO,std::uint32_t MHI,unsigned HIGH_SHIFT>
__device__ __forceinline__ int shard_owner8_u32limb(Code g){std::uint32_t a0=std::uint32_t(g),a1=std::uint32_t(g>>32);std::uint32_t p00hi=__umulhi(a0,MLO);std::uint32_t p01lo=a0*MHI,p01hi=__umulhi(a0,MHI);std::uint32_t p10lo=a1*MLO,p10hi=__umulhi(a1,MLO);std::uint32_t p11=a1*MHI;std::uint32_t s0=p00hi+p01lo,carry=std::uint32_t(s0<p00hi);std::uint32_t s1=s0+p10lo;carry+=std::uint32_t(s1<s0);std::uint32_t high64=p01hi+p10hi+p11+carry;return int(high64>>HIGH_SHIFT);}
struct ShardProduct32{std::uint32_t lo,hi;};
__device__ __forceinline__ ShardProduct32 shard_mul45_shiftadd(std::uint32_t x){std::uint32_t lo=x<<5,hi=x>>27,add=x<<3,old=lo;lo+=add;hi+=(x>>29)+std::uint32_t(lo<old);add=x<<2;old=lo;lo+=add;hi+=(x>>30)+std::uint32_t(lo<old);old=lo;lo+=x;hi+=std::uint32_t(lo<old);return{lo,hi};}
__device__ __forceinline__ int shard_owner8_u32shift_main(Code g){std::uint32_t a0=std::uint32_t(g),a1=std::uint32_t(g>>32);std::uint32_t p00hi=__umulhi(a0,2614578007u);ShardProduct32 p01=shard_mul45_shiftadd(a0);std::uint32_t p10lo=a1*2614578007u,p10hi=__umulhi(a1,2614578007u);std::uint32_t p11=(a1<<5)+(a1<<3)+(a1<<2)+a1;std::uint32_t s0=p00hi+p01.lo,carry=std::uint32_t(s0<p00hi);std::uint32_t s1=s0+p10lo;carry+=std::uint32_t(s1<s0);return int((p01.hi+p10hi+p11+carry)>>9);}
__device__ __forceinline__ int shard_owner8_u32shift_block(Code g){std::uint32_t a0=std::uint32_t(g),a1=std::uint32_t(g>>32);std::uint32_t p00hi=__umulhi(a0,2466947517u),p01lo=a0<<5,p01hi=a0>>27;std::uint32_t p10lo=a1*2466947517u,p10hi=__umulhi(a1,2466947517u),p11=a1<<5;std::uint32_t s0=p00hi+p01lo,carry=std::uint32_t(s0<p00hi);std::uint32_t s1=s0+p10lo;carry+=std::uint32_t(s1<s0);return int((p01hi+p10hi+p11+carry)>>7);}
#if B300_SHARD_ADDRESS_MODE == 4
static_assert(TARGET_W==28,"shift-add pure-u32 shard address is specialized for TARGET_W=28");
__device__ __forceinline__ ShardAddress8 shard_address8_main_w28_g8(Code g){constexpr Code chunk=48214938328ULL;int o=shard_owner8_u32shift_main(g);return{o,g-shard_base8_masked(o,chunk)};}
__device__ __forceinline__ ShardAddress8 shard_address8_block_w28_g8(Code g){constexpr Code chunk=16876938176ULL;int o=shard_owner8_u32shift_block(g);return{o,g-shard_base8_masked(o,chunk)};}
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8_main_w28_g8(g);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8_block_w28_g8(g);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8_main_w28_g8(g);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8_block_w28_g8(g);D_BLOCK_PTR[a.owner][a.local]=v;}
#elif B300_SHARD_ADDRESS_MODE == 3
static_assert(TARGET_W==28,"pure-u32 shard address is specialized for TARGET_W=28");
__device__ __forceinline__ ShardAddress8 shard_address8_main_w28_g8(Code g){constexpr Code chunk=48214938328ULL;int o=shard_owner8_u32limb<2614578007u,45u,9>(g);return{o,g-shard_base8_masked(o,chunk)};}
__device__ __forceinline__ ShardAddress8 shard_address8_block_w28_g8(Code g){constexpr Code chunk=16876938176ULL;int o=shard_owner8_u32limb<2466947517u,32u,7>(g);return{o,g-shard_base8_masked(o,chunk)};}
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8_main_w28_g8(g);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8_block_w28_g8(g);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8_main_w28_g8(g);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8_block_w28_g8(g);D_BLOCK_PTR[a.owner][a.local]=v;}
#elif B300_SHARD_ADDRESS_MODE == 2
static_assert(TARGET_W==28,"mulhi masked shard address is specialized for TARGET_W=28");
__device__ __forceinline__ ShardAddress8 shard_address8_main_w28_g8(Code g){constexpr Code magic=195888106327ULL,chunk=48214938328ULL;int o=int(__umul64hi(g,magic)>>9);return{o,g-shard_base8_masked(o,chunk)};}
__device__ __forceinline__ ShardAddress8 shard_address8_block_w28_g8(Code g){constexpr Code magic=139905900989ULL,chunk=16876938176ULL;int o=int(__umul64hi(g,magic)>>7);return{o,g-shard_base8_masked(o,chunk)};}
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8_main_w28_g8(g);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8_block_w28_g8(g);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8_main_w28_g8(g);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8_block_w28_g8(g);D_BLOCK_PTR[a.owner][a.local]=v;}
#elif B300_SHARD_ADDRESS_MODE == 1
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8(g,D_MAIN_CHUNK);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8(g,D_BLOCK_CHUNK);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8(g,D_MAIN_CHUNK);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8(g,D_BLOCK_CHUNK);D_BLOCK_PTR[a.owner][a.local]=v;}
#else
__device__ __forceinline__ Count global_load_main(Code g){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK];}
__device__ __forceinline__ Count global_load_block(Code g){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK];}
__device__ __forceinline__ void global_store_main(Code g,Count v){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK]=v;}
#endif'''

NGPU_OLD = 'if(ng<1||ng>MAXGPU){std::cerr<<"need 1..8 GPUs\\n";return 2;}\n    int peers=0;'
NGPU_NEW = '''if(ng<1||ng>MAXGPU){std::cerr<<"need 1..8 GPUs\\n";return 2;}
#if B300_SHARD_ADDRESS_MODE >= 2
    if(ng!=8){std::cerr<<"B300_SHARD_ADDRESS_MODE>=2 requires exactly 8 GPUs\\n";return 2;}
#endif
    int peers=0;'''

CHUNK_OLD = 'Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{}'
CHUNK_NEW = '''Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;
#if B300_SHARD_ADDRESS_MODE >= 2
    if(mainN!=385719506620ULL||blockN!=135015505407ULL||mc!=48214938328ULL||bc!=16876938176ULL){std::cerr<<"B300_SHARD_ADDRESS_MODE>=2 W28x8 constants mismatch\\n";return 9;}
#endif
    Count*mp[MAXGPU]{}'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected exactly one source match, got {count}')
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    text = args.src.read_text()
    text = replace_once(text, HELPER_OLD, HELPER_NEW, 'shard helper')
    text = replace_once(text, NGPU_OLD, NGPU_NEW, 'ngpu guard')
    text = replace_once(text, CHUNK_OLD, CHUNK_NEW, 'chunk guard')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f'generated {args.out} from {args.src} shard_address_modes=0,1,2,3,4 guarded_w28_ngpu8=1')


if __name__ == '__main__':
    main()
