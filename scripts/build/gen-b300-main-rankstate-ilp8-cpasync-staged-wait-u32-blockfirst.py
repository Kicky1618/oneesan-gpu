#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32-blockfirst.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_cpasync_count_ca','b300_cpasync_commit','b300_cpasync_wait_all','b300_main_pull_rankstate_ilp8_kernel','const Count self7=','const Count pair7=my_smem[7]','const Count block7=my_smem[15]'):
    if req not in s: raise SystemExit(f'block-first staged wait requires artifact: {req}')
if s.count('b300_cpasync_commit();')!=2 or s.count('b300_cpasync_wait_all();')!=1:
    raise SystemExit('block-first staged wait requires two groups and one wait-all')

anchor='''__device__ __forceinline__ void b300_cpasync_wait_all(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
'''
if s.count(anchor)!=1: raise SystemExit('wait helper anchor not unique')
s=s.replace(anchor,'''__device__ __forceinline__ void b300_cpasync_wait_oldest(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}
'''+anchor,1)

pair_lines='\n'.join([f'        b300_cpasync_count_ca(my_smem+{k},hp{k}?in+pj{k}:in,hp{k});' for k in range(8)])+'\n        b300_cpasync_commit();'
block_lines='\n'.join([f'        b300_cpasync_count_ca(my_smem+{8+k},(hb{k}&&bj{k}<nblock)?in_block+bj{k}:in,hb{k}&&bj{k}<nblock);' for k in range(8)])+'\n        b300_cpasync_commit();'
issue_old=pair_lines+'\n'+block_lines
issue_new=block_lines+'\n'+pair_lines
if s.count(issue_old)!=1: raise SystemExit('pair/block issue-order anchor not unique')
s=s.replace(issue_old,issue_new,1)

old=['        b300_cpasync_wait_all();']+[f'        const Count pair{k}=my_smem[{k}];' for k in range(8)]+[f'        const Count block{k}=my_smem[{8+k}];' for k in range(8)]+['        const uint64_t mod=D_MOD;']
for k in range(8):
    stmt=f'uint64_t a=uint64_t(self{k})+pair{k}+block{k};if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i{k}]=Count(a);'
    old.append(('        {' if k==0 else f'        if(v{k})'+'{')+stmt+'}')
old_text='\n'.join(old)
if s.count(old_text)!=1: raise SystemExit('consume anchor not unique')

new=['        b300_cpasync_wait_oldest();','        const uint64_t mod=D_MOD;']
for k in range(8):
    expr=f'uint64_t t=uint64_t(self{k})+uint64_t(my_smem[{8+k}]);if(t>=mod)t-=mod;partial{k}=Count(t);'
    new.append(f'        Count partial{k};{{{expr}}}' if k==0 else f'        Count partial{k}=0;if(v{k}){{{expr}}}')
new.append('        b300_cpasync_wait_all();')
for k in range(8):
    body=f'uint64_t a=uint64_t(partial{k})+uint64_t(my_smem[{k}]);if(a>=mod)a-=mod;out_main[i{k}]=Count(a);'
    new.append(('        {' if k==0 else f'        if(v{k})'+'{')+body+'}')
s=s.replace(old_text,'\n'.join(new),1)

for req in ('cp.async.wait_group 1','b300_cpasync_wait_oldest();','Count partial7=0','uint64_t(partial7)+uint64_t(my_smem[7])'):
    if req not in s: raise SystemExit(f'missing block-first artifact: {req}')
for stale in ('const Count pair7=my_smem[7]','const Count block7=my_smem[15]'):
    if stale in s: raise SystemExit(f'stale all-wait consume remains: {stale}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_ilp8_cpasync_staged_wait_u32_blockfirst=1 first_group=block second_group=pair wait_oldest_group=1 live_partial_bits=32 pair_reduce_overlap=0 block_reduce_overlap_pair=1 shared_bytes_unchanged=1 spill_gate_required=1')
