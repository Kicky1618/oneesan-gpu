#!/usr/bin/env python3
from __future__ import annotations
import pathlib,re,sys
if len(sys.argv)!=4:
    raise SystemExit('usage: gen-b300-mainrec-ilp2-mate-cg-l2-policy.py INPUT.cu OUTPUT.cu L2_BYTES')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
try:
    l2=int(sys.argv[3],0)
except ValueError:
    raise SystemExit('L2_BYTES must be 0,64,128,256')
if l2 not in (0,64,128,256):
    raise SystemExit('L2_BYTES must be 0,64,128,256')
s=src.read_text()
if 'b300_mainrec_stageu_ilp2_mate_cg_l2=' in s:
    raise SystemExit('source already contains Stage-U ILP2 mate CG L2 policy')
tm=re.search(r'// b300_mainrec_staget_ilp2_mate_load_policy=1 policy=(cg|cs) high_policy=(default|cg|cs) high_l2_bytes=(0|64|128|256) stages_preserved=([01]) scope=ilp2_mate_reads_only',s)
if not tm:
    raise SystemExit('Stage U requires Stage-T ILP2 mate policy marker')
low_policy,high_policy,high_l2,stages=tm.group(1),tm.group(2),int(tm.group(3)),int(tm.group(4))
if low_policy!='cg':
    raise SystemExit('Stage U is applicable only when Stage-T ILP2 mate policy is cg')
if high_policy!='cg' and high_l2!=0:
    raise SystemExit('Stage U found invalid high mate L2 provenance on non-cg policy')
helper='b300_mainrec_staget_ilp2_mate_load_policy_cg'
pat=re.compile(r'__device__\s+__forceinline__\s+MateID\s+'+re.escape(helper)+r'\s*\(const MateID\*\s+p\)\s*\{')
m=pat.search(s)
if not m:
    raise SystemExit('Stage-U low-state cg helper definition not found')
brace=s.find('{',m.start()); depth=0; end=-1
for i in range(brace,len(s)):
    if s[i]=='{': depth+=1
    elif s[i]=='}':
        depth-=1
        if depth==0:
            end=i+1; break
if end<0:
    raise SystemExit('Stage-U low-state cg helper closing brace not found')

def span(text:str,name:str)->tuple[int,int]:
    km=re.search(r'__global__\s+void\s+'+re.escape(name)+r'\s*\(',text)
    if not km: raise SystemExit(f'{name} definition not found')
    start=text.rfind('\n',0,km.start())+1; b=text.find('{',km.end())
    if b<0: raise SystemExit(f'{name} opening brace not found')
    d=0
    for j in range(b,len(text)):
        if text[j]=='{': d+=1
        elif text[j]=='}':
            d-=1
            if d==0: return start,j+1
    raise SystemExit(f'{name} closing brace not found')

p2a,p2b=span(s,'main_pull_kernel_ilp2'); p8a,p8b=span(s,'main_pull_kernel_ilp8_hybrid')
ilp2_before=s[p2a:p2b]; ilp8_before=s[p8a:p8b]
for k in (0,1):
    if ilp2_before.count(f'{helper}(mates+i{k})')!=1:
        raise SystemExit(f'Stage U expected one low mate helper call for lane {k}')
qual='ld.global.cg.u64' if l2==0 else f'ld.global.cg.L2::{l2}B.u64'
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
q2a,q2b=span(s,'main_pull_kernel_ilp2'); q8a,q8b=span(s,'main_pull_kernel_ilp8_hybrid')
if s[q8a:q8b]!=ilp8_before:
    raise SystemExit('Stage U changed ILP8 high-state kernel')
final2=s[q2a:q2b]
# Only helper implementation may change; call topology and all Count loads remain stable.
for k in (0,1):
    if final2.count(f'{helper}(mates+i{k})')!=1:
        raise SystemExit(f'Stage U damaged low mate lane {k}')
for req in ('const Count self0=','const Count pair0=','const Count block0='):
    if req not in final2:
        raise SystemExit(f'Stage U damaged ILP2 Count artifact: {req}')
for marker in ('b300_mainrec_stager_ilp2_pair_block_policy=1','b300_mainrec_staget_ilp2_mate_load_policy=1'):
    if marker not in s:
        raise SystemExit('Stage U lost upstream marker: '+marker)
if stages and 'b300_mainrec_stages_ilp2_pair_block_cg_l2=1' not in s:
    raise SystemExit('Stage U lost Stage-S marker')
s+=f'\n// b300_mainrec_stageu_ilp2_mate_cg_l2=1 l2_bytes={l2} high_policy={high_policy} high_l2_bytes={high_l2} stages_preserved={stages} scope=ilp2_mate_reads_only\n'
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_mainrec_stageu_ilp2_mate_cg_l2=1 l2_bytes={l2} qualifier={qual} low_policy=cg high_policy={high_policy} high_l2_bytes={high_l2} stages_preserved={stages} ilp8_byte_identical=1 count_loads_unchanged=1 semantics_unchanged=1')
