#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-closure-contrib-table.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('block_pull_rank_contrib','D_MAIN_DP','allowed_host','cudaMemcpyToSymbol(D_MAIN_DP'):
    if req not in s:raise SystemExit(f'closure contrib table requires artifact: {req}')

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1;brace=text.find('{',p);depth=0;end=-1
    for k in range(brace,len(text)):
        if text[k]=='{':depth+=1
        elif text[k]=='}':
            depth-=1
            if depth==0:end=k+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

# block_pull_rank_contrib(v,pos,h) is the lexicographic rank mass of symbols
# strictly before v at this position, not the weight of v itself:
#   N -> 0
#   R -> N branch
#   L -> N branch + R branch
# Cache the R/L values once per group. 29*30*2*8 = 13,920 bytes at MAXW=28.
decl='__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];\n'
if s.count(decl)!=1:raise SystemExit(f'D_FULL_DP declaration expected once got {s.count(decl)}')
s=s.replace(decl,decl+'__constant__ Code D_BLOCK_CLOSURE_CONTRIB[MAXW+1][MAXW+2][2];\n',1)

s=replace_function(s,'block_pull_rank_contrib',r'''__device__ __forceinline__ Code block_pull_rank_contrib(MateValue v,int pos,int h){
    if(v==N)return 0;
    return D_BLOCK_CLOSURE_CONTRIB[pos][h][v==R?0:1];
}''')

anchor='''    ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main dp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block dp");'''
if s.count(anchor)!=1:raise SystemExit(f'group DP upload anchor expected once got {s.count(anchor)}')
prep=r'''    Code closureContrib[MAXW+1][MAXW+2][2]{};
    for(int pp=0;pp<W;++pp)for(int hh=0;hh<=MAXW+1;++hh){
        Code nbranch=0,rbranch=0;
        if(allowed_host(ms.fixed,ms.occ,pp,N))nbranch=ms.dp[pp][hh];
        if(hh>0&&allowed_host(ms.fixed,ms.occ,pp,R))rbranch=ms.dp[pp][hh-1];
        closureContrib[pp][hh][0]=nbranch;          // contribution before R
        closureContrib[pp][hh][1]=nbranch+rbranch; // contribution before L
    }
    ck(cudaMemcpyToSymbol(D_BLOCK_CLOSURE_CONTRIB,closureContrib,sizeof(closureContrib)),"block closure contrib");
'''
s=s.replace(anchor,prep+anchor,1)

for req in ('D_BLOCK_CLOSURE_CONTRIB','block closure contrib','contribution before R','contribution before L','return D_BLOCK_CLOSURE_CONTRIB[pos][h][v==R?0:1]'):
    if req not in s:raise SystemExit(f'missing closure-contrib artifact: {req}')
for stale in ('if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,N))','if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,R))'):
    p=s.find('block_pull_rank_contrib(');q=s.find('\n}',p)
    if stale in s[p:q]:raise SystemExit(f'stale closure contrib arithmetic: {stale}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_closure_contrib_table=1 constant_bytes_added=13920 values=rank_mass_before_R,rank_mass_before_L N_zero_fastpath=1 per_group_upload=1 closure_allowed_checks=0 closure_dp_sum_ops=0 exact_semantics=lexicographic_prefix_mass')
