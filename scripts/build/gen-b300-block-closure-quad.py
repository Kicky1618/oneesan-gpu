#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-block-closure-quad.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('block_pull_add_rank','block_pull_kernel','else if(look==N){','out_block[i]=acc;'):
    if req not in s:raise SystemExit(f'closure-quad transform requires artifact: {req}')

# Insert explicit four-rank queue before the block-pull kernel. Keeping fields
# scalar rather than an indexed local array gives nvcc a chance to issue four
# independent HBM loads without local-memory spills.
marker='template<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel'
if marker not in s:
    # Support the uncached full-pull form for isolated experiments.
    marker='__global__ void block_pull_kernel(const Count*in_main,Code n,Count*out_block,int p)'
if marker not in s:raise SystemExit('block pull kernel marker not found')
helper=r'''
struct B300BlockClosureQuad{Code r0=0,r1=0,r2=0,r3=0;int n=0;};
__device__ __forceinline__ void b300_block_closure_quad_flush(
    Count& acc,const Count* __restrict__ in_main,B300BlockClosureQuad& q
){
    const Count v0=q.n>0?in_main[q.r0]:Count(0);
    const Count v1=q.n>1?in_main[q.r1]:Count(0);
    const Count v2=q.n>2?in_main[q.r2]:Count(0);
    const Count v3=q.n>3?in_main[q.r3]:Count(0);
    if(q.n>0)pull_add_mod(acc,v0);if(q.n>1)pull_add_mod(acc,v1);
    if(q.n>2)pull_add_mod(acc,v2);if(q.n>3)pull_add_mod(acc,v3);
    q.n=0;
}
__device__ __forceinline__ void b300_block_closure_quad_emit(
    Count& acc,const Count* __restrict__ in_main,B300BlockClosureQuad& q,Code r
){
    if(q.n==0)q.r0=r;else if(q.n==1)q.r1=r;else if(q.n==2)q.r2=r;else q.r3=r;
    ++q.n;if(q.n==4)b300_block_closure_quad_flush(acc,in_main,q);
}

'''
s=s.replace(marker,helper+marker,1)

# Restrict modifications to block_pull_kernel so main-pull helpers and any other
# rank consumers remain byte-for-byte unchanged.
p=s.find('block_pull_kernel')
brace=s.find('{',p);depth=0;end=-1
for i in range(brace,len(s)):
    if s[i]=='{':depth+=1
    elif s[i]=='}':
        depth-=1
        if depth==0:end=i+1;break
if end<0:raise SystemExit('block pull kernel end not found')
start=s.rfind('\n',0,p)+1
body=s[start:end]
cl=body.find('else if(look==N){')
if cl<0:raise SystemExit('closure branch not found inside block pull kernel')
outpos=body.rfind('out_block[i]=acc;')
if outpos<0 or outpos<=cl:raise SystemExit('block output not found after closure branch')
closure=body[cl:outpos]
adds=closure.count('block_pull_add_rank(acc,in_main,')
if adds!=3:raise SystemExit(f'expected three closure candidate source loads, got {adds}')
closure=closure.replace('else if(look==N){','else if(look==N){\n            B300BlockClosureQuad b300_quad{};',1)
closure=closure.replace('block_pull_add_rank(acc,in_main,','b300_block_closure_quad_emit(acc,in_main,b300_quad,')
# Every transformed call retains its original closing `);`, matching emit's 4th arg.
closure+='            b300_block_closure_quad_flush(acc,in_main,b300_quad);\n'
body=body[:cl]+closure+body[outpos:]
s=s[:start]+body+s[end:]

# Structural gates: the endpoint NR/NL load remains immediate, exactly three
# closure candidate sites are queued, and every closure state flushes its tail.
p=s.find('block_pull_kernel');wend=s.find('\n\nstatic Code rank_full',p)
hot=s[p:wend if wend>=0 else len(s)]
if hot.count('b300_block_closure_quad_emit(acc,in_main,b300_quad,')!=3:
    raise SystemExit('closure quad emit count mismatch')
if hot.count('b300_block_closure_quad_flush(acc,in_main,b300_quad);')<1:
    raise SystemExit('closure quad tail flush missing')
if 'block_pull_add_rank(acc,in_main,' not in hot:
    raise SystemExit('endpoint immediate source load unexpectedly removed')
for req in ('B300BlockClosureQuad','const Count v0=','const Count v3=','q.n==4'):
    if req not in s:raise SystemExit(f'closure quad artifact missing: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_closure_quad=1 batch=4 queued_candidate_sites=3 endpoint_load_immediate=1 closure_tail_flush=1 extra_state_bytes=0 production_default=off')
