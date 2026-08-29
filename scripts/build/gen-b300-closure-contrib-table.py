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

# One uint64 contribution for R and L at each (position,height). N is always
# zero. 29*30*2*8 = 13,920 bytes for MAXW=28, well below the 64-KiB constant
# budget when the separate HOT_DELTA_TABLE experiment is disabled.
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
        Code zr=0,zl=0;
        if(allowed_host(ms.fixed,ms.occ,pp,R)){
            if(hh>0)zr=ms.dp[pp][hh-1];
        }
        if(allowed_host(ms.fixed,ms.occ,pp,L)){
            zl=ms.dp[pp][hh];
            if(hh>0)zl+=ms.dp[pp][hh-1];
        }
        closureContrib[pp][hh][0]=zr;
        closureContrib[pp][hh][1]=zl;
    }
    ck(cudaMemcpyToSymbol(D_BLOCK_CLOSURE_CONTRIB,closureContrib,sizeof(closureContrib)),"block closure contrib");
'''
s=s.replace(anchor,prep+anchor,1)

for req in ('D_BLOCK_CLOSURE_CONTRIB','block closure contrib','return D_BLOCK_CLOSURE_CONTRIB[pos][h][v==R?0:1]'):
    if req not in s:raise SystemExit(f'missing closure-contrib artifact: {req}')
for stale in ('if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,N))','if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,R))'):
    # Those expressions may occur elsewhere in the source; only reject if they
    # survived inside block_pull_rank_contrib itself.
    p=s.find('block_pull_rank_contrib(');q=s.find('\n}',p)
    if stale in s[p:q]:raise SystemExit(f'stale closure contrib arithmetic: {stale}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_closure_contrib_table=1 constant_bytes_added=13920 values=R,L N_zero_fastpath=1 per_group_upload=1 closure_allowed_checks=0 closure_dp_sum_ops=0')
