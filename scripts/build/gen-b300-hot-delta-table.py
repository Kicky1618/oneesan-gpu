#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-hot-delta-table.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('b300_rank_delta_step(', 'main_pull_direct_pair_source_rank(', 'D_MAIN_DP', 'D_BLOCK_DP'):
    if req not in s: raise SystemExit(f'hot delta table requires artifact: {req}')

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1;brace=text.find('{',p)
    depth=0;end=-1
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:end=i+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

# The hot values are rank differences, not absolute ranks. Keep arithmetic in
# signed 64-bit on the host and reject a group if a value does not fit int32.
# Step N is identically zero and returns before touching the table, so store only
# the R/L entries.  The two hot tables add 17,400 bytes total, leaving generous
# headroom below CUDA's 64-KiB constant-memory limit.
decl='__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];\n'
if s.count(decl)!=1:raise SystemExit('D_FULL_DP declaration anchor not unique')
s=s.replace(decl,decl+'__constant__ int D_HOT_STEP_DELTA[MAXW+1][MAXW+2][2];\n__constant__ int D_HOT_PAIR_DELTA[MAXW+1][MAXW+2][3];\n',1)

s=replace_function(s,'b300_rank_delta_step',r'''__device__ __forceinline__ RankDelta b300_rank_delta_step(MateValue v,int p,int h){
    return v==N?RankDelta(0):RankDelta(D_HOT_STEP_DELTA[p][h][v==R?0:1]);
}''')

s=replace_function(s,'main_pull_direct_pair_source_rank',r'''__device__ __forceinline__ Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p,int h){
    const MateValuePair pair=mpair(m,p);int k=-1;
    if(pair==LR)k=0;else if(pair==NR)k=1;else if(pair==NL)k=2;else return dst_rank;
    const int d=D_HOT_PAIR_DELTA[p][h][k];
    return d>=0?dst_rank+Code(d):dst_rank-Code(-static_cast<long long>(d));
}''')

anchor='''    ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main dp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block dp");'''
if s.count(anchor)!=1:raise SystemExit(f'process_group DP copy anchor expected one got {s.count(anchor)}')
prep=r'''    int hotStep[MAXW+1][MAXW+2][2]{};
    int hotPair[MAXW+1][MAXW+2][3]{};
    auto hot_i32=[&](long long x,const char* what,int pp,int hh,int kk)->int{
        if(x < -2147483648LL || x > 2147483647LL){
            std::cerr<<"hot delta int32 overflow kind="<<what
                     <<" p="<<pp<<" h="<<hh<<" slot="<<kk<<" value="<<x<<'\n';
            std::exit(838);
        }
        return int(x);
    };
    long long hotAbsMax=0;
    for(int pp=1;pp<W;++pp)for(int hh=0;hh<=MAXW;++hh){
        const Code mh=ms.dp[pp][hh], bh=ds.dp[pp-1][hh];
        const long long stepR=static_cast<long long>(bh)-static_cast<long long>(mh);
        const Code mhm=hh?ms.dp[pp][hh-1]:0, bhm=hh?ds.dp[pp-1][hh-1]:0;
        const long long stepL=static_cast<long long>(bh+bhm)-static_cast<long long>(mh+mhm);
        hotStep[pp][hh][0]=hot_i32(stepR,"step",pp,hh,0);
        hotStep[pp][hh][1]=hot_i32(stepL,"step",pp,hh,1);

        const Code low_h=ms.dp[pp-1][hh];
        const Code low_hm=hh?ms.dp[pp-1][hh-1]:0;
        const Code low_hp=(hh<MAXW+1)?ms.dp[pp-1][hh+1]:0;
        // Indexed 0=LR, 1=NR, 2=NL; values are source_rank-destination_rank.
        const long long pairLR=-static_cast<long long>(mh+mhm+low_hp);
        const long long pairNR= static_cast<long long>(mh)-static_cast<long long>(low_h);
        const long long pairNL= static_cast<long long>(mh+mhm)-static_cast<long long>(low_h+low_hm);
        hotPair[pp][hh][0]=hot_i32(pairLR,"pair",pp,hh,0);
        hotPair[pp][hh][1]=hot_i32(pairNR,"pair",pp,hh,1);
        hotPair[pp][hh][2]=hot_i32(pairNL,"pair",pp,hh,2);
        const long long vals[5]={stepR,stepL,pairLR,pairNR,pairNL};
        for(long long x:vals){const long long a=x<0?-x:x;if(a>hotAbsMax)hotAbsMax=a;}
    }
    ck(cudaMemcpyToSymbol(D_HOT_STEP_DELTA,hotStep,sizeof(hotStep)),"hot step delta");
    ck(cudaMemcpyToSymbol(D_HOT_PAIR_DELTA,hotPair,sizeof(hotPair)),"hot pair delta");
    std::cerr<<"b300_hot_delta_table int_bits=32 bytes="
             <<(sizeof(hotStep)+sizeof(hotPair))
             <<" abs_max="<<hotAbsMax<<" int32_checked=1 step_n_stored=0\n";
'''
s=s.replace(anchor,prep+anchor,1)

for req in ('D_HOT_STEP_DELTA','D_HOT_PAIR_DELTA','hot step delta','hot pair delta',
            'return v==N?RankDelta(0)','const int d=D_HOT_PAIR_DELTA','hot delta int32 overflow','step_n_stored=0'):
    if req not in s:raise SystemExit(f'missing hot delta-table artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_hot_delta_table=1 delta_bits=32 constant_bytes_added=17400 known_dp_plus_hot_bytes=38280 rank_step_constant_loads=1 pair_rank_constant_loads=1 group_upload_bytes=17400 int32_runtime_checked=1 step_n_stored=0')
