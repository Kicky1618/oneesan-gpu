#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys

if len(sys.argv) != 5:
    raise SystemExit('usage: gen-b300-mainrec-pair-block-cg-l2-policy.py INPUT.cu OUTPUT.cu PAIR_L2_BYTES BLOCK_L2_BYTES')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
try:
    pair_bytes=int(sys.argv[3],0); block_bytes=int(sys.argv[4],0)
except ValueError:
    raise SystemExit('PAIR_L2_BYTES/BLOCK_L2_BYTES must be 0,64,128,256')
for name,v in (('PAIR_L2_BYTES',pair_bytes),('BLOCK_L2_BYTES',block_bytes)):
    if v not in (0,64,128,256): raise SystemExit(f'{name} must be 0,64,128,256')
s=src.read_text()
if 'b300_mainrec_stageo_pair_block_cg_l2=' in s:
    raise SystemExit('source already contains Stage-O pair/block CG L2 policy')
m=re.search(r'// b300_mainrec_stagen_pair_block_policy=1 pair=(default|cg|cs) block=(default|cg|cs) cg_l2_bytes=(0|64|128|256)',s)
if not m: raise SystemExit('Stage O requires Stage-N pair/block policy marker')
pair_policy,block_policy,base_bytes=m.group(1),m.group(2),int(m.group(3))
if pair_policy!='cg' and pair_bytes!=0: raise SystemExit('PAIR_L2_BYTES must be 0 when pair policy is not cg')
if block_policy!='cg' and block_bytes!=0: raise SystemExit('BLOCK_L2_BYTES must be 0 when block policy is not cg')
if pair_policy!='cg' and block_policy!='cg': raise SystemExit('Stage O is not applicable when neither Stage-N axis uses cg')

# Insert axis-specific helpers immediately before the Stage-N shared CG helper.
helper_anchor='static_assert(sizeof(Count)==4,"Stage-N CG assumes 32-bit Count");'
if helper_anchor not in s: raise SystemExit('Stage-N CG helper anchor missing')
def qual(n:int)->str:
    return 'ld.global.cg.u32' if n==0 else f'ld.global.cg.L2::{n}B.u32'
def helper(name:str,n:int)->str:
    return f'''__device__ __forceinline__ Count {name}(const Count* p){{
#if __CUDA_ARCH__
    uint32_t v; const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("{qual(n)} %0, [%1];" : "=r"(v) : "l"(a));
    return Count(v);
#else
    return *p;
#endif
}}

'''
ins=''
if pair_policy=='cg': ins+=helper('b300_mainrec_stageo_pair_load_cg',pair_bytes)
if block_policy=='cg': ins+=helper('b300_mainrec_stageo_block_load_cg',block_bytes)
s=s.replace(helper_anchor,ins+helper_anchor,1)

# Stage N already constrained the target to ILP2 + ILP8 pair/block reads. Replace
# only the shared-CG calls on the corresponding axis; self and mate reads remain untouched.
changes=0
if pair_policy=='cg':
    pat=re.compile(r'b300_mainrec_stagen_load_cg\(in\+pj(\d+)\)')
    s,n=pat.subn(lambda x:f'b300_mainrec_stageo_pair_load_cg(in+pj{x.group(1)})',s); changes+=n
    if n!=10: raise SystemExit(f'expected 10 pair CG calls across ILP2+ILP8, got {n}')
if block_policy=='cg':
    pat=re.compile(r'b300_mainrec_stagen_load_cg\(in_block\+bj(\d+)\)')
    s,n=pat.subn(lambda x:f'b300_mainrec_stageo_block_load_cg(in_block+bj{x.group(1)})',s); changes+=n
    if n!=10: raise SystemExit(f'expected 10 block CG calls across ILP2+ILP8, got {n}')
if changes==0: raise SystemExit('Stage O made no changes')
for req in ('main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid','const Count self0=','const MateID m0='):
    if req not in s: raise SystemExit(f'Stage O damaged required artifact: {req}')
s += f'\n// b300_mainrec_stageo_pair_block_cg_l2=1 pair_policy={pair_policy} block_policy={block_policy} pair_l2_bytes={pair_bytes} block_l2_bytes={block_bytes} base_l2_bytes={base_bytes}\n'
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_mainrec_stageo_pair_block_cg_l2=1 pair_policy={pair_policy} block_policy={block_policy} pair_l2_bytes={pair_bytes} block_l2_bytes={block_bytes} base_l2_bytes={base_bytes} changed_calls={changes} self_unchanged=1 mate_unchanged=1 semantics_unchanged=1')
