#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys

if len(sys.argv) != 5:
    raise SystemExit('usage: gen-b300-mainrec-ilp8-pair-block-cg-l2-policy.py INPUT.cu OUTPUT.cu PAIR_L2_BYTES BLOCK_L2_BYTES')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
try:
    pair_l2=int(sys.argv[3],0); block_l2=int(sys.argv[4],0)
except ValueError:
    raise SystemExit('PAIR_L2_BYTES/BLOCK_L2_BYTES must be 0,64,128,256')
for name,v in (('PAIR_L2_BYTES',pair_l2),('BLOCK_L2_BYTES',block_l2)):
    if v not in (0,64,128,256): raise SystemExit(f'{name} must be 0,64,128,256')
s=src.read_text()
if 'b300_mainrec_stageq_ilp8_pair_block_cg_l2=' in s:
    raise SystemExit('source already contains Stage-Q ILP8 pair/block CG L2 policy')

n=re.search(r'// b300_mainrec_stagen_pair_block_policy=1 pair=(default|cg|cs) block=(default|cg|cs) cg_l2_bytes=(0|64|128|256)',s)
if not n: raise SystemExit('Stage Q requires Stage-N pair/block policy marker')
pair_policy,block_policy,base_l2=n.group(1),n.group(2),int(n.group(3))
if pair_policy!='cg' and pair_l2!=0: raise SystemExit('PAIR_L2_BYTES must be 0 when pair policy is not cg')
if block_policy!='cg' and block_l2!=0: raise SystemExit('BLOCK_L2_BYTES must be 0 when block policy is not cg')
if pair_policy!='cg' and block_policy!='cg': raise SystemExit('Stage Q is not applicable when neither Stage-N axis uses cg')

o=re.search(r'// b300_mainrec_stageo_pair_block_cg_l2=1 pair_policy=(default|cg|cs) block_policy=(default|cg|cs) pair_l2_bytes=(0|64|128|256) block_l2_bytes=(0|64|128|256) base_l2_bytes=(0|64|128|256)',s)
if o:
    if (o.group(1),o.group(2))!=(pair_policy,block_policy):
        raise SystemExit('Stage-Q Stage-O policy drift from Stage N')
    upstream_pair_l2=int(o.group(3)); upstream_block_l2=int(o.group(4))
else:
    upstream_pair_l2=base_l2 if pair_policy=='cg' else 0
    upstream_block_l2=base_l2 if block_policy=='cg' else 0
stagep=1 if 'b300_mainrec_stagep_mate_cg_l2=' in s else 0


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
    if req not in s: raise SystemExit(f'Stage Q requires artifact: {req}')

ilp2_start,ilp2_end=span(s,'main_pull_kernel_ilp2')
ilp8_start,ilp8_end=span(s,'main_pull_kernel_ilp8_hybrid')
ilp2_before=s[ilp2_start:ilp2_end]
ilp8=s[ilp8_start:ilp8_end]

pair_current='b300_mainrec_stageo_pair_load_cg' if o and pair_policy=='cg' else 'b300_mainrec_stagen_load_cg'
block_current='b300_mainrec_stageo_block_load_cg' if o and block_policy=='cg' else 'b300_mainrec_stagen_load_cg'

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
if pair_policy=='cg':
    q='b300_mainrec_stageq_ilp8_pair_load_cg'
    helpers+=helper(q,pair_l2)
    pat=re.compile(re.escape(pair_current)+r'\(in\+pj([0-7])\)')
    ilp8,nc=pat.subn(lambda m:f'{q}(in+pj{m.group(1)})',ilp8); changed+=nc
    if nc!=8: raise SystemExit(f'Stage Q expected 8 ILP8 pair CG calls, got {nc}')
if block_policy=='cg':
    q='b300_mainrec_stageq_ilp8_block_load_cg'
    helpers+=helper(q,block_l2)
    pat=re.compile(re.escape(block_current)+r'\(in_block\+bj([0-7])\)')
    ilp8,nc=pat.subn(lambda m:f'{q}(in_block+bj{m.group(1)})',ilp8); changed+=nc
    if nc!=8: raise SystemExit(f'Stage Q expected 8 ILP8 block CG calls, got {nc}')
if changed==0: raise SystemExit('Stage Q made no changes')

# Replace the ILP8 body first, then insert helpers immediately before the ILP8 kernel.
s=s[:ilp8_start]+ilp8+s[ilp8_end:]
ilp8_start,_=span(s,'main_pull_kernel_ilp8_hybrid')
s=s[:ilp8_start]+helpers+s[ilp8_start:]

# ILP2 must stay byte-for-byte unchanged. It remains the exact Stage N/O control
# policy while only the large-group ILP8 stream receives the Stage-Q L2 hint.
ilp2_start2,ilp2_end2=span(s,'main_pull_kernel_ilp2')
if s[ilp2_start2:ilp2_end2] != ilp2_before:
    raise SystemExit('Stage Q changed ILP2 kernel')
ilp8_start2,ilp8_end2=span(s,'main_pull_kernel_ilp8_hybrid')
ilp8_after=s[ilp8_start2:ilp8_end2]
if pair_policy=='cg':
    for k in range(8):
        if ilp8_after.count(f'b300_mainrec_stageq_ilp8_pair_load_cg(in+pj{k})')!=1:
            raise SystemExit(f'Stage Q pair lane {k} mismatch')
if block_policy=='cg':
    for k in range(8):
        if ilp8_after.count(f'b300_mainrec_stageq_ilp8_block_load_cg(in_block+bj{k})')!=1:
            raise SystemExit(f'Stage Q block lane {k} mismatch')
for req in ('const Count self0=','const MateID m0=','mates[i7]=b300_high_state_advance'):
    if req not in ilp8_after: raise SystemExit(f'Stage Q damaged ILP8 artifact: {req}')
if stagep and 'b300_mainrec_stagep_mate_cg_l2=' not in s:
    raise SystemExit('Stage Q lost Stage-P marker')

s += f'\n// b300_mainrec_stageq_ilp8_pair_block_cg_l2=1 pair_policy={pair_policy} block_policy={block_policy} pair_l2_bytes={pair_l2} block_l2_bytes={block_l2} upstream_pair_l2_bytes={upstream_pair_l2} upstream_block_l2_bytes={upstream_block_l2} stagep_preserved={stagep}\n'
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_mainrec_stageq_ilp8_pair_block_cg_l2=1 pair_policy={pair_policy} block_policy={block_policy} pair_l2_bytes={pair_l2} block_l2_bytes={block_l2} upstream_pair_l2_bytes={upstream_pair_l2} upstream_block_l2_bytes={upstream_block_l2} changed_calls={changed} ilp2_unchanged=1 stagep_preserved={stagep} self_unchanged=1 mate_unchanged=1 semantics_unchanged=1')
