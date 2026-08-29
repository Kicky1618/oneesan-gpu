#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-closure-warp-dualmask.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_block_closure_warp_kernel','block_pull_endpoint_mask','__ballot_sync','__shfl_down_sync'):
    if req not in s:raise SystemExit(f'closure-warp dualmask requires artifact: {req}')

# Insert the directional endpoint compaction helpers immediately after the
# existing endpoint-union helper. Keep the generic block-pull code untouched;
# only the warp lane-0 closure rank generator uses this experiment.
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
if 'B300ClosureWarpEndpointMasks' in s:raise SystemExit('closure-warp dualmask helper already present')
helper=r'''
struct B300ClosureWarpEndpointMasks{uint32_t r,l;};
__device__ __forceinline__ uint32_t b300_closure_warp_compact_even(uint64_t x){
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
__device__ __forceinline__ B300ClosureWarpEndpointMasks b300_closure_warp_endpoint_masks(MateID mate){
    constexpr uint64_t EVEN=0x5555555555555555ULL;
    const uint64_t lo=uint64_t(mate)&EVEN,hi=(uint64_t(mate)>>1)&EVEN;
    return{b300_closure_warp_compact_even(lo&~hi),b300_closure_warp_compact_even(hi&~lo)};
}
'''
s=s[:end]+helper+s[end:]

# Restrict all replacements to b300_block_closure_warp_kernel. The rank-state
# endpoint kernel and any generic fallback closure remain byte-for-byte intact.
name='b300_block_closure_warp_kernel'
p=s.find(name+'(')
if p<0:raise SystemExit('closure warp function not found')
start=s.rfind('\n',0,p)+1;brace=s.find('{',p);depth=0;finish=-1
for i in range(brace,len(s)):
    if s[i]=='{':depth+=1
    elif s[i]=='}':
        depth-=1
        if depth==0:finish=i+1;break
if finish<0:raise SystemExit('closure warp function end not found')
body=s[start:finish]
old0='const uint32_t endpoints=block_pull_endpoint_mask(d);'
old1='const int q=31-__clz(left);const MateValue v=mget(d,q);'
old2='const int q=__ffs(right)-1;const MateValue v=mget(d,q);const int hq=hbelow+(v==R)-(v==L);'
for old,label in ((old0,'endpoint union'),(old1,'left decode'),(old2,'right decode')):
    if body.count(old)!=1:raise SystemExit(f'{label}: expected one closure-warp match got {body.count(old)}')
body=body.replace(old0,'const B300ClosureWarpEndpointMasks ep=b300_closure_warp_endpoint_masks(d);const uint32_t endpoints=ep.r|ep.l;',1)
body=body.replace(old1,'const int q=31-__clz(left);const uint32_t qb=uint32_t(1)<<q;const MateValue v=(ep.l&qb)?L:R;',1)
body=body.replace(old2,'const int q=__ffs(right)-1;const uint32_t qb=uint32_t(1)<<q;const MateValue v=(ep.l&qb)?L:R;const int hq=hbelow+(v==R)-(v==L);',1)
s=s[:start]+body+s[finish:]

# Structural composition gates: the warp closure is dualmask-only, while the
# endpoint-only ILP4 kernel and warp reduction mechanism must remain present.
wp=s.find(name+'(');wend=s.find('\n\nstatic Code rank_full',wp)
warp=s[wp:wend if wend>=0 else len(s)]
for stale in ('const uint32_t endpoints=block_pull_endpoint_mask(d);','const MateValue v=mget(d,q);'):
    if stale in warp:raise SystemExit(f'stale closure-warp decode remains: {stale}')
for req in ('b300_closure_warp_endpoint_masks(d)','const MateValue v=(ep.l&qb)?L:R;','__ballot_sync','__shfl_down_sync','lane<unsigned(cnt)?in_main'):
    if req not in warp:raise SystemExit(f'closure-warp dualmask artifact missing: {req}')
if 'b300_block_pull_rankstate_ilp4_kernel' not in s:raise SystemExit('endpoint-only ILP4 kernel disappeared')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_closure_warp_dualmask=1 rank_generation_lane0_scan_mget_calls=0 endpoint_masks=R,L warp_reduction_preserved=1 extra_state_bytes=0 proof=b300-block-pull-dualmask-proof+closure-warp-proof')
