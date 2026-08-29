#!/usr/bin/env python3
from __future__ import annotations
import pathlib,re,sys

if len(sys.argv) not in (3,4):
    raise SystemExit('usage: gen-b300-mainrec-random-cg.py INPUT.cu OUTPUT.cu [L2_PREFETCH_BYTES=0|64|128|256]')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
try: l2_bytes=int(sys.argv[3],0) if len(sys.argv)==4 else 0
except ValueError: raise SystemExit('L2_PREFETCH_BYTES must be 0,64,128,256')
if l2_bytes not in (0,64,128,256): raise SystemExit('L2_PREFETCH_BYTES must be 0,64,128,256')
for req in ('main_pull_kernel_ilp2','const Count pair0=','const Count block0=','high_rec_groups='):
    if req not in s:raise SystemExit(f'mainrec CG requires artifact: {req}')
if 'b300_mainrec_random_load_cg' in s:raise SystemExit('mainrec source already contains random CG helper')

kernel_names=['main_pull_kernel_ilp2']
if 'main_pull_kernel_ilp8_hybrid(' in s:
    kernel_names.append('main_pull_kernel_ilp8_hybrid')

def function_span(text:str,name:str)->tuple[int,int]:
    token=name+'('
    if text.count(token)!=1:raise SystemExit(f'{name} definition expected once, got {text.count(token)}')
    p=text.find(token);start=text.rfind('\n',0,p)+1;brace=text.find('{',p);depth=0;end=-1
    if brace<0:raise SystemExit(f'{name} opening brace not found')
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:end=i+1;break
    if end<0:raise SystemExit(f'{name} end not found')
    return start,end

transforms=[];lane_counts={}
for name in kernel_names:
    start,end=function_span(s,name);body=s[start:end]
    ids=sorted({int(x) for x in re.findall(r'const Count pair(\d+)=',body)})
    if not ids or ids!=list(range(max(ids)+1)):raise SystemExit(f'{name}: unexpected pair lane ids {ids}')
    for k in ids:
        a=f'const Count pair{k}=hp{k}?in[pj{k}]:Count(0);'
        b=f'const Count block{k}=hb{k}?in_block[bj{k}]:Count(0);'
        if body.count(a)!=1 or body.count(b)!=1:raise SystemExit(f'{name}: lane {k} random load anchor mismatch')
        body=body.replace(a,f'const Count pair{k}=hp{k}?b300_mainrec_random_load_cg(in+pj{k}):Count(0);',1)
        body=body.replace(b,f'const Count block{k}=hb{k}?b300_mainrec_random_load_cg(in_block+bj{k}):Count(0);',1)
    if 'const Count self0=in[i0];' not in body:raise SystemExit(f'{name}: self stream unexpectedly rewritten')
    transforms.append((start,end,body,name));lane_counts[name]=len(ids)

# Replace from the end so source offsets remain valid, then inject one helper
# before the earliest recurrence kernel.  This keeps hybrid ILP2/ILP8 kernels
# on exactly the same cache policy instead of silently leaving ILP8 on default
# loads.
for start,end,body,_ in sorted(transforms,reverse=True):
    s=s[:start]+body+s[end:]
insert_at=min(x[0] for x in transforms)
qual='ld.global.cg.u32' if l2_bytes==0 else f'ld.global.cg.L2::{l2_bytes}B.u32'
helper=f'''static_assert(sizeof(Count)==4,"mainrec random CG assumes 32-bit Count");
__device__ __forceinline__ Count b300_mainrec_random_load_cg(const Count* p){{
#if __CUDA_ARCH__
    uint32_t v;const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("{qual} %0, [%1];" : "=r"(v) : "l"(a));
    return Count(v);
#else
    return *p;
#endif
}}

'''
s=s[:insert_at]+helper+s[insert_at:]

for name,nlanes in lane_counts.items():
    start,end=function_span(s,name);body=s[start:end]
    for k in range(nlanes):
        for req in (f'b300_mainrec_random_load_cg(in+pj{k})',f'b300_mainrec_random_load_cg(in_block+bj{k})'):
            if req not in body:raise SystemExit(f'{name}: missing CG lane artifact: {req}')
if qual not in s: raise SystemExit(f'missing load qualifier {qual}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
summary=','.join(f'{name}:{lane_counts[name]}' for name in kernel_names)
print(f'generated {out} from {src}: mainrec_random_cg=1 kernels={len(kernel_names)} kernel_lanes={summary} pair_cg_loads={sum(lane_counts.values())} block_cg_loads={sum(lane_counts.values())} self_load_policy=default semantic_load_width=32 l2_prefetch_bytes={l2_bytes} ptx_load={qual} hybrid_policy_consistent=1 extra_state_bytes=0 production_default=off')
