#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_cpasync_count_ca','b300_cpasync_commit','b300_cpasync_wait_all','cp.async.wait_group 0','b300_main_pull_rankstate_ilp8_kernel','const Count self7=','const Count pair7=my_smem[7]','const Count block7=my_smem[15]','const uint64_t mod=D_MOD;'):
    if req not in s: raise SystemExit(f'staged-u32 cp.async wait requires artifact: {req}')
if 'b300_cpasync_wait_pair' in s: raise SystemExit('source already contains staged cp.async wait')
if s.count('b300_cpasync_wait_all();') != 1 or s.count('b300_cpasync_commit();') != 2:
    raise SystemExit('staged-u32 requires exactly one wait-all and two committed groups')
anchor='''__device__ __forceinline__ void b300_cpasync_wait_all(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
'''
if s.count(anchor)!=1: raise SystemExit('wait-all helper anchor not unique')
s=s.replace(anchor,'''__device__ __forceinline__ void b300_cpasync_wait_pair(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}
'''+anchor,1)
old=['        b300_cpasync_wait_all();']+[f'        const Count pair{k}=my_smem[{k}];' for k in range(8)]+[f'        const Count block{k}=my_smem[{8+k}];' for k in range(8)]+['        const uint64_t mod=D_MOD;']
for k in range(8):
    stmt=f'uint64_t a=uint64_t(self{k})+pair{k}+block{k};if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i{k}]=Count(a);'
    old.append(('        {' if k==0 else f'        if(v{k})'+'{')+stmt+'}')
old_text='\n'.join(old)
if s.count(old_text)!=1: raise SystemExit('ILP8 consume anchor not unique')
new=['        b300_cpasync_wait_pair();','        const uint64_t mod=D_MOD;']
for k in range(8):
    expr=f'uint64_t t=uint64_t(self{k})+uint64_t(my_smem[{k}]);if(t>=mod)t-=mod;partial{k}=Count(t);'
    new.append(f'        Count partial{k};{{{expr}}}' if k==0 else f'        Count partial{k}=0;if(v{k}){{{expr}}}')
new.append('        b300_cpasync_wait_all();')
for k in range(8):
    body=f'uint64_t a=uint64_t(partial{k})+uint64_t(my_smem[{8+k}]);if(a>=mod)a-=mod;out_main[i{k}]=Count(a);'
    new.append(('        {' if k==0 else f'        if(v{k})'+'{')+body+'}')
s=s.replace(old_text,'\n'.join(new),1)
for req in ('cp.async.wait_group 1','b300_cpasync_wait_pair();','Count partial0;','Count partial7=0','uint64_t(partial7)+uint64_t(my_smem[15])'):
    if req not in s: raise SystemExit(f'missing staged-u32 artifact: {req}')
for stale in ('const Count pair7=my_smem[7]','const Count block7=my_smem[15]'):
    if stale in s: raise SystemExit(f'stale wait-all consume remains: {stale}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_ilp8_cpasync_staged_wait_u32=1 wait_pair_group=1 wait_block_group=0 pair_reduce_overlap_block=1 live_partial_bits=32 live_partials=8 pair_mod_subtractions=1 final_mod_subtractions=1 per_thread_shared_only=1 spill_gate_required=1')
