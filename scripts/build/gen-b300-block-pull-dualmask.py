#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-pull-dualmask.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('block_pull_endpoint_mask','while(left){','while(right){','const uint32_t endpoints=block_pull_endpoint_mask(d);'):
    if req not in s:raise SystemExit(f'dualmask transform requires artifact: {req}')

def once(old:str,new:str,label:str)->None:
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

marker='__device__ __forceinline__ uint32_t block_pull_endpoint_mask(MateID mate){'
p=s.find(marker)
if p<0:raise SystemExit('endpoint helper start not found')
brace=s.find('{',p);depth=0;end=-1
for i in range(brace,len(s)):
    if s[i]=='{':depth+=1
    elif s[i]=='}':
        depth-=1
        if depth==0:end=i+1;break
if end<0:raise SystemExit('endpoint helper end not found')
helper=r'''
struct B300BlockEndpointMasks{uint32_t r,l;};
__device__ __forceinline__ uint32_t b300_block_compact_even(uint64_t x){
    x&=0x5555555555555555ULL;
    x=(x|(x>>1))&0x3333333333333333ULL;
    x=(x|(x>>2))&0x0f0f0f0f0f0f0f0fULL;
    x=(x|(x>>4))&0x00ff00ff00ff00ffULL;
    x=(x|(x>>8))&0x0000ffff0000ffffULL;
    x=(x|(x>>16))&0x00000000ffffffffULL;
    uint32_t z=uint32_t(x);
    if constexpr(TARGET_W<32)z&=(uint32_t(1)<<TARGET_W)-1u;
    return z;
}
__device__ __forceinline__ B300BlockEndpointMasks b300_block_endpoint_masks(MateID mate){
    constexpr uint64_t EVEN=0x5555555555555555ULL;
    const uint64_t lo=uint64_t(mate)&EVEN,hi=(uint64_t(mate)>>1)&EVEN;
    // Exact for N/R/L production mates. X is deliberately excluded from both
    // directional masks rather than being misclassified as an endpoint.
    return{b300_block_compact_even(lo&~hi),b300_block_compact_even(hi&~lo)};
}
'''
s=s[:end]+helper+s[end:]

once('const uint32_t endpoints=block_pull_endpoint_mask(d);',
     'const B300BlockEndpointMasks ep=b300_block_endpoint_masks(d);const uint32_t endpoints=ep.r|ep.l;',
     'dual endpoint construction')
once('const int q=31-__clz(left);const MateValue v=mget(d,q);',
     'const int q=31-__clz(left);const uint32_t qb=uint32_t(1)<<q;const MateValue v=(ep.l&qb)?L:R;',
     'left bitplane decode')
once('const int q=__ffs(right)-1;const MateValue v=mget(d,q);',
     'const int q=__ffs(right)-1;const uint32_t qb=uint32_t(1)<<q;const MateValue v=(ep.l&qb)?L:R;',
     'right bitplane decode')

for stale in ('const uint32_t endpoints=block_pull_endpoint_mask(d);','const MateValue v=mget(d,q);'):
    if stale in s:raise SystemExit(f'stale endpoint decode remains: {stale}')
for req in ('B300BlockEndpointMasks','b300_block_endpoint_masks(d)','const MateValue v=(ep.l&qb)?L:R;'):
    if req not in s:raise SystemExit(f'dualmask artifact missing: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: block_pull_dualmask=1 endpoint_masks=R,L scan_mget_calls=0 extra_state_bytes=0 proof=b300-block-pull-dualmask-proof')
