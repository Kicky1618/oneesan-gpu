#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-closure-contrib-table.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('block_pull_rank_contrib','BlockClosureDelta','D_MAIN_DP','allowed_host','cudaMemcpyToSymbol(D_MAIN_DP'):
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

# For production n=27 / W=28 with the 13-bit window split, exhaustive occupancy
# enumeration gives max contributions R=821,588,872 and L=1,615,814,681. Keep a
# runtime range check for arbitrary planner shapes, then pack R/L into one u64.
# A second same-size packed table stores signed contrib(h+2)-contrib(h), replacing
# the two table loads + subtraction in every LL/RR endpoint-scan update.
decl='__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];\n'
if s.count(decl)!=1:raise SystemExit(f'D_FULL_DP declaration expected once got {s.count(decl)}')
s=s.replace(decl,decl+
'''__constant__ unsigned long long D_BLOCK_CLOSURE_CONTRIB32[MAXW+1][MAXW+2];
__constant__ unsigned long long D_BLOCK_CLOSURE_SHIFT232[MAXW+1][MAXW+2];
''',1)

s=replace_function(s,'block_pull_rank_contrib',r'''__device__ __forceinline__ Code block_pull_rank_contrib(MateValue v,int pos,int h){
    if(v==N)return 0;
    const unsigned long long z=D_BLOCK_CLOSURE_CONTRIB32[pos][h];
    return Code(v==R?uint32_t(z):uint32_t(z>>32));
}''')

# Add the signed h->h+2 helper immediately after the replaced rank contribution.
needle='''__device__ __forceinline__ Code block_pull_rank_contrib(MateValue v,int pos,int h){
    if(v==N)return 0;
    const unsigned long long z=D_BLOCK_CLOSURE_CONTRIB32[pos][h];
    return Code(v==R?uint32_t(z):uint32_t(z>>32));
}'''
if s.count(needle)!=1:raise SystemExit('replaced closure contrib helper not unique')
shift_helper=r'''
__device__ __forceinline__ BlockClosureDelta block_pull_rank_shift2(MateValue v,int pos,int h){
    if(v==N)return 0;
    const unsigned long long z=D_BLOCK_CLOSURE_SHIFT232[pos][h];
    const uint32_t raw=v==R?uint32_t(z):uint32_t(z>>32);
    return BlockClosureDelta(static_cast<int32_t>(raw));
}
'''
s=s.replace(needle,needle+shift_helper,1)

# Replace all exact recurring h+2 update expressions in generic, ILP4 and warp
# closure implementations. Candidate cross-symbol corrections stay on the
# absolute packed table because they use different symbols/heights.
patterns=(
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-\n                        BlockClosureDelta(block_pull_rank_contrib(v,q,hb))',
     'block_pull_rank_shift2(v,q,hb)'),
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-\n                         BlockClosureDelta(block_pull_rank_contrib(v,q,hq))',
     'block_pull_rank_shift2(v,q,hq)'),
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hb))',
     'block_pull_rank_shift2(v,q,hb)'),
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hq))',
     'block_pull_rank_shift2(v,q,hq)'),
)
repl=0
for old,new in patterns:
    n=s.count(old)
    if n:
        s=s.replace(old,new)
        repl+=n
if repl<2:raise SystemExit(f'closure shift2 update replacements too few: {repl}')

anchor='''    ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main dp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block dp");'''
if s.count(anchor)!=1:raise SystemExit(f'group DP upload anchor expected once got {s.count(anchor)}')
prep=r'''    unsigned long long closureContrib32[MAXW+1][MAXW+2]{};
    unsigned long long closureShift232[MAXW+1][MAXW+2]{};
    auto closure_contrib_host=[&](int pp,int hh,int which)->unsigned long long{
        if(hh<0||hh>MAXW+1)return 0;
        Code nbranch=0,rbranch=0;
        if(allowed_host(ms.fixed,ms.occ,pp,N))nbranch=ms.dp[pp][hh];
        if(hh>0&&allowed_host(ms.fixed,ms.occ,pp,R))rbranch=ms.dp[pp][hh-1];
        return which==0?unsigned long long(nbranch):unsigned long long(nbranch+rbranch);
    };
    unsigned long long closureAbsMax=0;long long closureShiftAbsMax=0;
    for(int pp=0;pp<W;++pp)for(int hh=0;hh<=MAXW+1;++hh){
        const unsigned long long cr=closure_contrib_host(pp,hh,0),cl=closure_contrib_host(pp,hh,1);
        if(cr>0xffffffffULL||cl>0xffffffffULL){
            std::cerr<<"closure contrib u32 overflow p="<<pp<<" h="<<hh<<" R="<<cr<<" L="<<cl<<'\n';std::exit(839);
        }
        closureContrib32[pp][hh]=(unsigned long long(uint32_t(cl))<<32)|uint32_t(cr);
        const long long dr=long long(closure_contrib_host(pp,hh+2,0))-long long(cr);
        const long long dl=long long(closure_contrib_host(pp,hh+2,1))-long long(cl);
        if(dr<-2147483648LL||dr>2147483647LL||dl<-2147483648LL||dl>2147483647LL){
            std::cerr<<"closure shift2 i32 overflow p="<<pp<<" h="<<hh<<" R="<<dr<<" L="<<dl<<'\n';std::exit(840);
        }
        closureShift232[pp][hh]=(unsigned long long(uint32_t(int32_t(dl)))<<32)|uint32_t(int32_t(dr));
        closureAbsMax=std::max(closureAbsMax,std::max(cr,cl));
        closureShiftAbsMax=std::max(closureShiftAbsMax,std::max(dr<0?-dr:dr,dl<0?-dl:dl));
    }
    ck(cudaMemcpyToSymbol(D_BLOCK_CLOSURE_CONTRIB32,closureContrib32,sizeof(closureContrib32)),"block closure contrib32");
    ck(cudaMemcpyToSymbol(D_BLOCK_CLOSURE_SHIFT232,closureShift232,sizeof(closureShift232)),"block closure shift232");
    std::cerr<<"b300_closure_contrib_table packed32=1 bytes="<<(sizeof(closureContrib32)+sizeof(closureShift232))
             <<" abs_max="<<closureAbsMax<<" shift_abs_max="<<closureShiftAbsMax
             <<" shift2_replacements="<<''' + str(repl) + r'''<<" u32_i32_checked=1\n";
'''
s=s.replace(anchor,prep+anchor,1)

for req in ('D_BLOCK_CLOSURE_CONTRIB32','D_BLOCK_CLOSURE_SHIFT232','block_pull_rank_shift2','block closure contrib32','block closure shift232','u32_i32_checked=1'):
    if req not in s:raise SystemExit(f'missing packed closure-table artifact: {req}')
for stale in ('D_BLOCK_CLOSURE_CONTRIB[MAXW+1][MAXW+2][2]','BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hb))','BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hq))'):
    if stale in s:raise SystemExit(f'stale closure table artifact remains: {stale}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_closure_contrib_table=packed32+shift2 constant_bytes_added=13920 contrib_bytes=6960 shift2_bytes=6960 values=rank_mass_before_R,rank_mass_before_L update_ops=one_lookup_per_endpoint u32_i32_runtime_checked=1 shift2_replacements={repl}')
