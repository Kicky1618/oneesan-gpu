#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys

if len(sys.argv) != 6:
    raise SystemExit('usage: gen-b300-mainrec-pair-block-load-policy.py INPUT.cu OUTPUT.cu PAIR_POLICY BLOCK_POLICY CG_L2_BYTES')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
pair_policy,block_policy=sys.argv[3],sys.argv[4]
try: cg_l2=int(sys.argv[5],0)
except ValueError: raise SystemExit('CG_L2_BYTES must be 0,64,128,256')
for name,p in (('PAIR_POLICY',pair_policy),('BLOCK_POLICY',block_policy)):
    if p not in ('default','cg','cs'): raise SystemExit(f'{name} must be default,cg,cs')
if cg_l2 not in (0,64,128,256): raise SystemExit('CG_L2_BYTES must be 0,64,128,256')
s=src.read_text()
if 'b300_mainrec_stagen_pair_block_policy=' in s:
    raise SystemExit('source already contains Stage-N pair/block policy')
for req in ('main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid','const Count pair0=','const Count block0=','const Count self0='):
    if req not in s: raise SystemExit(f'Stage-N requires artifact: {req}')


def function_span(text:str,name:str)->tuple[int,int]:
    token=name+'('
    # Ignore launch sites by locating the kernel definition token.
    m=re.search(r'__global__\s+void\s+'+re.escape(name)+r'\s*\(',text)
    if not m: raise SystemExit(f'{name} definition not found')
    start=text.rfind('\n',0,m.start())+1
    brace=text.find('{',m.end()); depth=0
    if brace<0: raise SystemExit(f'{name} opening brace not found')
    for i in range(brace,len(text)):
        if text[i]=='{': depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0: return start,i+1
    raise SystemExit(f'{name} closing brace not found')


def expr(kind:str,k:int,policy:str)->str:
    base=f'in+pj{k}' if kind=='pair' else f'in_block+bj{k}'
    direct=f'in[pj{k}]' if kind=='pair' else f'in_block[bj{k}]'
    if policy=='default': return direct
    if policy=='cg': return f'b300_mainrec_stagen_load_cg({base})'
    return f'b300_mainrec_stagen_load_cs({base})'

kernels=('main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid')
transforms=[]; lane_counts={}
for name in kernels:
    start,end=function_span(s,name); body=s[start:end]
    ids=sorted({int(x) for x in re.findall(r'const Count pair(\d+)=',body)})
    if not ids or ids!=list(range(max(ids)+1)):
        raise SystemExit(f'{name}: unexpected lane ids {ids}')
    for k in ids:
        pm=re.search(rf'const Count pair{k}=hp{k}\?([^;]+):Count\(0\);',body)
        bm=re.search(rf'const Count block{k}=hb{k}\?([^;]+):Count\(0\);',body)
        if not pm or not bm: raise SystemExit(f'{name}: lane {k} pair/block anchor missing')
        body=body[:pm.start()]+f'const Count pair{k}=hp{k}?{expr("pair",k,pair_policy)}:Count(0);'+body[pm.end():]
        # Re-search after pair replacement because offsets changed.
        bm=re.search(rf'const Count block{k}=hb{k}\?([^;]+):Count\(0\);',body)
        assert bm
        body=body[:bm.start()]+f'const Count block{k}=hb{k}?{expr("block",k,block_policy)}:Count(0);'+body[bm.end():]
    for req in ('const Count self0=','MateID'):
        if req not in body: raise SystemExit(f'{name}: Stage-N damaged required artifact {req}')
    transforms.append((start,end,body,name)); lane_counts[name]=len(ids)

for start,end,body,_ in sorted(transforms,reverse=True):
    s=s[:start]+body+s[end:]
insert_at=min(function_span(s,name)[0] for name in kernels)
helper=''
if 'cg' in (pair_policy,block_policy):
    qual='ld.global.cg.u32' if cg_l2==0 else f'ld.global.cg.L2::{cg_l2}B.u32'
    helper+=f'''static_assert(sizeof(Count)==4,"Stage-N CG assumes 32-bit Count");
__device__ __forceinline__ Count b300_mainrec_stagen_load_cg(const Count* p){{
#if __CUDA_ARCH__
    uint32_t v; const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("{qual} %0, [%1];" : "=r"(v) : "l"(a));
    return Count(v);
#else
    return *p;
#endif
}}

'''
if 'cs' in (pair_policy,block_policy):
    helper+='''__device__ __forceinline__ Count b300_mainrec_stagen_load_cs(const Count* p){
#if __CUDA_ARCH__
    return __ldcs(p);
#else
    return *p;
#endif
}

'''
s=s[:insert_at]+helper+s[insert_at:]

for name,nlanes in lane_counts.items():
    start,end=function_span(s,name); body=s[start:end]
    for k in range(nlanes):
        want_pair=f'const Count pair{k}=hp{k}?{expr("pair",k,pair_policy)}:Count(0);'
        want_block=f'const Count block{k}=hb{k}?{expr("block",k,block_policy)}:Count(0);'
        if body.count(want_pair)!=1 or body.count(want_block)!=1:
            raise SystemExit(f'{name}: final lane {k} policy mismatch')
    if 'const Count self0=' not in body: raise SystemExit(f'{name}: self stream changed')

# Source marker is a comment so a second transform is rejected without changing ABI.
s += f'\n// b300_mainrec_stagen_pair_block_policy=1 pair={pair_policy} block={block_policy} cg_l2_bytes={cg_l2}\n'
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
summary=','.join(f'{k}:{lane_counts[k]}' for k in kernels)
print(f'generated {out} from {src}: b300_mainrec_stagen_pair_block_policy=1 pair_policy={pair_policy} block_policy={block_policy} cg_l2_bytes={cg_l2} kernels={summary} self_policy_unchanged=1 mate_policy_unchanged=1 semantics_unchanged=1')
