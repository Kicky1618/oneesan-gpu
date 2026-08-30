#!/usr/bin/env python3
from __future__ import annotations
import pathlib,re,sys
if len(sys.argv)!=4: raise SystemExit('usage: gen-b300-mainrec-ilp2-mate-load-policy.py INPUT.cu OUTPUT.cu POLICY')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); policy=sys.argv[3]
if policy not in ('cg','cs'): raise SystemExit('POLICY must be cg or cs')
s=src.read_text()
if 'b300_mainrec_staget_ilp2_mate_load_policy=' in s: raise SystemExit('source already contains Stage-T ILP2 mate-load policy')
if 'b300_mainrec_stager_ilp2_pair_block_policy=1' not in s: raise SystemExit('Stage T requires Stage-R ILP2 pair/block policy marker')
stages=1 if 'b300_mainrec_stages_ilp2_pair_block_cg_l2=1' in s else 0
high_cg='b300_mainrec_hybrid8_mate_load_policy_cg' in s
high_cs='b300_mainrec_hybrid8_mate_load_policy_cs' in s
if high_cg and high_cs: raise SystemExit('Stage T found conflicting high-state mate policy helpers')
high_policy='cg' if high_cg else ('cs' if high_cs else 'default')
pm=re.search(r'// b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=(0|64|128|256) scope=ilp8_mate_reads_only',s)
if pm and high_policy!='cg': raise SystemExit('Stage-P mate L2 marker requires high-state cg mate policy')
high_l2=int(pm.group(1)) if pm else 0

def span(text:str,name:str)->tuple[int,int]:
    m=re.search(r'__global__\s+void\s+'+re.escape(name)+r'\s*\(',text)
    if not m: raise SystemExit(f'{name} definition not found')
    start=text.rfind('\n',0,m.start())+1; brace=text.find('{',m.end())
    if brace<0: raise SystemExit(f'{name} opening brace not found')
    d=0
    for i in range(brace,len(text)):
        if text[i]=='{': d+=1
        elif text[i]=='}':
            d-=1
            if d==0: return start,i+1
    raise SystemExit(f'{name} closing brace not found')

ilp2_start,ilp2_end=span(s,'main_pull_kernel_ilp2'); ilp8_start,ilp8_end=span(s,'main_pull_kernel_ilp8_hybrid')
ilp8_before=s[ilp8_start:ilp8_end]; body=s[ilp2_start:ilp2_end]
found={}
for m in re.finditer(r'const\s+MateID\s+m([0-9]+)=([^;]+);',body):
    k=int(m.group(1))
    if k in (0,1): found[k]=(m.start(),m.end(),m.group(2))
if sorted(found)!=[0,1]: raise SystemExit(f'Stage T expects ILP2 mate lanes 0,1; got {sorted(found)}')
helper=f'b300_mainrec_staget_ilp2_mate_load_policy_{policy}'; intrinsic='__ldcg' if policy=='cg' else '__ldcs'
# Rewrite backwards so offsets remain valid; preserve any ternary guard around the load.
for k in (1,0):
    a,b,expr=found[k]; needle=f'mates[i{k}]'
    if expr.count(needle)!=1: raise SystemExit(f'Stage T lane {k} expected one {needle}, got {expr.count(needle)}')
    expr2=expr.replace(needle,f'{helper}(mates+i{k})',1)
    body=body[:a]+f'const MateID m{k}={expr2};'+body[b:]
s=s[:ilp2_start]+body+s[ilp2_end:]
ilp2_start,_=span(s,'main_pull_kernel_ilp2')
helper_src=f'''__device__ __forceinline__ MateID {helper}(const MateID* p){{
    return {intrinsic}(p);
}}

'''
s=s[:ilp2_start]+helper_src+s[ilp2_start:]
new8_start,new8_end=span(s,'main_pull_kernel_ilp8_hybrid')
if s[new8_start:new8_end]!=ilp8_before: raise SystemExit('Stage T changed ILP8 high-state kernel')
new2_start,new2_end=span(s,'main_pull_kernel_ilp2'); final2=s[new2_start:new2_end]
for k in (0,1):
    if final2.count(f'{helper}(mates+i{k})')!=1: raise SystemExit(f'Stage T final mate lane {k} mismatch')
for req in ('const Count self0=','const Count pair0=','const Count block0='):
    if req not in final2: raise SystemExit(f'Stage T damaged ILP2 artifact: {req}')
if 'b300_mainrec_stager_ilp2_pair_block_policy=1' not in s: raise SystemExit('Stage T lost Stage-R marker')
if stages and 'b300_mainrec_stages_ilp2_pair_block_cg_l2=1' not in s: raise SystemExit('Stage T lost Stage-S marker')
s+=f'\n// b300_mainrec_staget_ilp2_mate_load_policy=1 policy={policy} high_policy={high_policy} high_l2_bytes={high_l2} stages_preserved={stages} scope=ilp2_mate_reads_only\n'
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_mainrec_staget_ilp2_mate_load_policy=1 policy={policy} intrinsic={intrinsic} high_policy={high_policy} high_l2_bytes={high_l2} stages_preserved={stages} lanes=2 ilp8_byte_identical=1 count_loads_unchanged=1 mate_writes_unchanged=1 semantics_unchanged=1')
