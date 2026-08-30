#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys

if len(sys.argv) != 5:
    raise SystemExit('usage: gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py INPUT.cu OUTPUT.cu PAIR_L2_BYTES BLOCK_L2_BYTES')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
try:
    pair_l2=int(sys.argv[3],0); block_l2=int(sys.argv[4],0)
except ValueError:
    raise SystemExit('PAIR_L2_BYTES/BLOCK_L2_BYTES must be 0,64,128,256')
for name,v in (('PAIR_L2_BYTES',pair_l2),('BLOCK_L2_BYTES',block_l2)):
    if v not in (0,64,128,256): raise SystemExit(f'{name} must be 0,64,128,256')

s=src.read_text()
if 'b300_mainrec_stages_ilp2_pair_block_cg_l2=' in s:
    raise SystemExit('source already contains Stage-S ILP2 pair/block CG L2 policy')
mr=re.search(r'// b300_mainrec_stager_ilp2_pair_block_policy=1 pair=(default|cg|cs) block=(default|cg|cs) high_pair=(default|cg|cs) high_block=(default|cg|cs) high_pair_l2=(0|64|128|256) high_block_l2=(0|64|128|256) stageq_preserved=([01])',s)
if not mr: raise SystemExit('Stage S requires Stage-R ILP2 pair/block policy marker')
low_pair,low_block,high_pair,high_block=mr.group(1),mr.group(2),mr.group(3),mr.group(4)
high_pair_l2,high_block_l2=int(mr.group(5)),int(mr.group(6)); stageq=int(mr.group(7))
if low_pair!='cg' and pair_l2!=0: raise SystemExit('PAIR_L2_BYTES must be 0 when Stage-R pair policy is not cg')
if low_block!='cg' and block_l2!=0: raise SystemExit('BLOCK_L2_BYTES must be 0 when Stage-R block policy is not cg')
if low_pair!='cg' and low_block!='cg': raise SystemExit('Stage S is not applicable when neither Stage-R ILP2 axis uses cg')


def span(text:str,name:str)->tuple[int,int]:
    m=re.search(r'__global__\s+void\s+'+re.escape(name)+r'\s*\(',text)
    if not m: raise SystemExit(f'{name} definition not found')
    start=text.rfind('\n',0,m.start())+1
    brace=text.find('{',m.end())
    if brace<0: raise SystemExit(f'{name} opening brace not found')
    depth=0
    for i in range(brace,len(text)):
        if text[i]=='{': depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0: return start,i+1
    raise SystemExit(f'{name} closing brace not found')

for req in ('main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid','const Count self0=','const MateID m0='):
    if req not in s: raise SystemExit(f'Stage S requires artifact: {req}')
ilp2_start,ilp2_end=span(s,'main_pull_kernel_ilp2'); ilp8_start,ilp8_end=span(s,'main_pull_kernel_ilp8_hybrid')
ilp8_before=s[ilp8_start:ilp8_end]; ilp2=s[ilp2_start:ilp2_end]

def qual(v:int)->str:
    return 'ld.global.cg.u32' if v==0 else f'ld.global.cg.L2::{v}B.u32'
def helper(name:str,v:int)->str:
    return f'''__device__ __forceinline__ Count {name}(const Count* p){{
#if __CUDA_ARCH__
    uint32_t x; const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("{qual(v)} %0, [%1];" : "=r"(x) : "l"(a));
    return Count(x);
#else
    return *p;
#endif
}}

'''

helpers=''; changed=0
if low_pair=='cg':
    name='b300_mainrec_stages_ilp2_pair_load_cg'; helpers+=helper(name,pair_l2)
    pat=re.compile(r'b300_mainrec_stager_ilp2_load_cg\(in\+pj([01])\)')
    ilp2,nc=pat.subn(lambda m:f'{name}(in+pj{m.group(1)})',ilp2); changed+=nc
    if nc!=2: raise SystemExit(f'Stage S expected 2 ILP2 pair CG calls, got {nc}')
if low_block=='cg':
    name='b300_mainrec_stages_ilp2_block_load_cg'; helpers+=helper(name,block_l2)
    pat=re.compile(r'b300_mainrec_stager_ilp2_load_cg\(in_block\+bj([01])\)')
    ilp2,nc=pat.subn(lambda m:f'{name}(in_block+bj{m.group(1)})',ilp2); changed+=nc
    if nc!=2: raise SystemExit(f'Stage S expected 2 ILP2 block CG calls, got {nc}')
if changed==0: raise SystemExit('Stage S made no changes')

s=s[:ilp2_start]+ilp2+s[ilp2_end:]
ilp2_start,_=span(s,'main_pull_kernel_ilp2'); s=s[:ilp2_start]+helpers+s[ilp2_start:]
new8_start,new8_end=span(s,'main_pull_kernel_ilp8_hybrid')
if s[new8_start:new8_end] != ilp8_before: raise SystemExit('Stage S changed ILP8 high-state kernel')
new2_start,new2_end=span(s,'main_pull_kernel_ilp2'); final2=s[new2_start:new2_end]
if low_pair=='cg':
    for k in range(2):
        if final2.count(f'b300_mainrec_stages_ilp2_pair_load_cg(in+pj{k})')!=1: raise SystemExit(f'Stage S pair lane {k} mismatch')
if low_block=='cg':
    for k in range(2):
        if final2.count(f'b300_mainrec_stages_ilp2_block_load_cg(in_block+bj{k})')!=1: raise SystemExit(f'Stage S block lane {k} mismatch')
for req in ('const Count self0=','const MateID m0='):
    if req not in final2: raise SystemExit(f'Stage S damaged ILP2 artifact: {req}')
if stageq and 'b300_mainrec_stageq_ilp8_pair_block_cg_l2=1' not in s: raise SystemExit('Stage S lost Stage-Q marker')
s += f'\n// b300_mainrec_stages_ilp2_pair_block_cg_l2=1 pair_policy={low_pair} block_policy={low_block} pair_l2_bytes={pair_l2} block_l2_bytes={block_l2} upstream_pair_l2_bytes=0 upstream_block_l2_bytes=0 high_pair={high_pair} high_block={high_block} high_pair_l2={high_pair_l2} high_block_l2={high_block_l2} stageq_preserved={stageq}\n'
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_mainrec_stages_ilp2_pair_block_cg_l2=1 pair_policy={low_pair} block_policy={low_block} pair_l2_bytes={pair_l2} block_l2_bytes={block_l2} high_pair={high_pair} high_block={high_block} high_pair_l2={high_pair_l2} high_block_l2={high_block_l2} stageq_preserved={stageq} changed_calls={changed} ilp8_byte_identical=1 self_unchanged=1 mate_unchanged=1 semantics_unchanged=1')
