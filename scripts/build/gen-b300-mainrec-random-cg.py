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

p=s.find('main_pull_kernel_ilp2(');start=s.rfind('\n',0,p)+1;brace=s.find('{',p);depth=0;end=-1
for i in range(brace,len(s)):
    if s[i]=='{':depth+=1
    elif s[i]=='}':
        depth-=1
        if depth==0:end=i+1;break
if end<0:raise SystemExit('main recurrence kernel end not found')
body=s[start:end]
ids=sorted({int(x) for x in re.findall(r'const Count pair(\d+)=',body)})
if not ids or ids!=list(range(max(ids)+1)):raise SystemExit(f'unexpected pair lane ids {ids}')
for k in ids:
    a=f'const Count pair{k}=hp{k}?in[pj{k}]:Count(0);'
    b=f'const Count block{k}=hb{k}?in_block[bj{k}]:Count(0);'
    if body.count(a)!=1 or body.count(b)!=1:raise SystemExit(f'lane {k} random load anchor mismatch')
    body=body.replace(a,f'const Count pair{k}=hp{k}?b300_mainrec_random_load_cg(in+pj{k}):Count(0);',1)
    body=body.replace(b,f'const Count block{k}=hb{k}?b300_mainrec_random_load_cg(in_block+bj{k}):Count(0);',1)

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
s=s[:start]+helper+body+s[end:]
for k in ids:
    for req in (f'b300_mainrec_random_load_cg(in+pj{k})',f'b300_mainrec_random_load_cg(in_block+bj{k})'):
        if req not in s:raise SystemExit(f'missing CG lane artifact: {req}')
if 'const Count self0=in[i0];' not in s:raise SystemExit('self stream unexpectedly rewritten')
if qual not in s: raise SystemExit(f'missing load qualifier {qual}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: mainrec_random_cg=1 lanes={len(ids)} pair_cg_loads={len(ids)} block_cg_loads={len(ids)} self_load_policy=default semantic_load_width=32 l2_prefetch_bytes={l2_bytes} ptx_load={qual} extra_state_bytes=0 production_default=off')
