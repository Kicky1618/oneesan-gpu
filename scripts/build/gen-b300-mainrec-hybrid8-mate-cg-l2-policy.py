#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py INPUT.cu OUTPUT.cu L2_BYTES')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
try:
    l2_bytes=int(sys.argv[3],0)
except ValueError:
    raise SystemExit('L2_BYTES must be 0,64,128,256')
if l2_bytes not in (0,64,128,256):
    raise SystemExit('L2_BYTES must be 0,64,128,256')
s=src.read_text()
if 'b300_mainrec_stagep_mate_cg_l2=' in s:
    raise SystemExit('source already contains Stage-P mate CG L2 policy')
helper='b300_mainrec_hybrid8_mate_load_policy_cg'
if helper not in s:
    if 'b300_mainrec_hybrid8_mate_load_policy_cs' in s:
        raise SystemExit('Stage P is not applicable to Stage-M cs mate-load policy')
    raise SystemExit('Stage P requires Stage-M cg mate-load policy helper')
for req in ('main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid','const Count self0=','const MateID m0='):
    if req not in s: raise SystemExit(f'Stage P requires artifact: {req}')

# Replace only the cg helper implementation. All eight ILP8 call sites remain
# byte-for-byte unchanged, so Stage P cannot perturb mate guards or recurrence order.
pat=re.compile(r'__device__\s+__forceinline__\s+MateID\s+'+re.escape(helper)+r'\s*\(const MateID\*\s+p\)\s*\{')
m=pat.search(s)
if not m: raise SystemExit('Stage-P mate cg helper definition not found')
brace=s.find('{',m.start()); depth=0; end=-1
for i in range(brace,len(s)):
    if s[i]=='{': depth+=1
    elif s[i]=='}':
        depth-=1
        if depth==0:
            end=i+1; break
if end<0: raise SystemExit('Stage-P mate cg helper closing brace not found')
qual='ld.global.cg.u64' if l2_bytes==0 else f'ld.global.cg.L2::{l2_bytes}B.u64'
new=f'''__device__ __forceinline__ MateID {helper}(const MateID* p){{
#if __CUDA_ARCH__
    unsigned long long v; const unsigned long long a=reinterpret_cast<unsigned long long>(p);
    asm volatile("{qual} %0, [%1];" : "=l"(v) : "l"(a));
    return MateID(v);
#else
    return *p;
#endif
}}'''
s=s[:m.start()]+new+s[end:]
call_pat=re.compile(re.escape(helper)+r'\(mates\+i[0-7]\)')
calls=call_pat.findall(s)
if len(calls)!=8: raise SystemExit(f'Stage P expected eight ILP8 mate policy calls, got {len(calls)}')
for k in range(8):
    if s.count(f'{helper}(mates+i{k})')!=1:
        raise SystemExit(f'Stage P damaged mate lane {k}')
for req in ('main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid','const Count self0=','mates[i7]=b300_high_state_advance'):
    if req not in s: raise SystemExit(f'Stage P damaged required artifact: {req}')
if '__ldcg(p)' in s[m.start():m.start()+len(new)+64]:
    raise SystemExit('Stage P left old __ldcg helper implementation')
s += f'\n// b300_mainrec_stagep_mate_cg_l2=1 l2_bytes={l2_bytes} scope=ilp8_mate_reads_only\n'
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_mainrec_stagep_mate_cg_l2=1 l2_bytes={l2_bytes} qualifier={qual} mate_calls=8 ilp2_unchanged=1 count_loads_unchanged=1 mate_writes_unchanged=1 semantics_unchanged=1')
