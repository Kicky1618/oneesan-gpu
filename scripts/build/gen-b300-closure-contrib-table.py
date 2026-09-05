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

# Three packed u64 tables, each 29*30*8 = 6,960 bytes at MAXW=28:
# - absolute lexicographic prefix mass before R/L (u32/u32)
# - same-symbol h->h+2 difference (i32/i32)
# - candidate cross-symbol correction used by LL/RR (i32/i32)
# Production W=28 / low-13 split bounds are build-gated independently.
decl='__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];\n'
if s.count(decl)!=1:raise SystemExit(f'D_FULL_DP declaration expected once got {s.count(decl)}')
s=s.replace(decl,decl+
'''__constant__ unsigned long long D_BLOCK_CLOSURE_CONTRIB32[MAXW+1][MAXW+2];
__constant__ unsigned long long D_BLOCK_CLOSURE_SHIFT232[MAXW+1][MAXW+2];
__constant__ unsigned long long D_BLOCK_CLOSURE_CROSS32[MAXW+1][MAXW+2];
''',1)

s=replace_function(s,'block_pull_rank_contrib',r'''__device__ __forceinline__ Code block_pull_rank_contrib(MateValue v,int pos,int h){
    if(v==N)return 0;
    const unsigned long long z=D_BLOCK_CLOSURE_CONTRIB32[pos][h];
    return Code(v==R?uint32_t(z):uint32_t(z>>32));
}''')

needle='''__device__ __forceinline__ Code block_pull_rank_contrib(MateValue v,int pos,int h){
    if(v==N)return 0;
    const unsigned long long z=D_BLOCK_CLOSURE_CONTRIB32[pos][h];
    return Code(v==R?uint32_t(z):uint32_t(z>>32));
}'''
if s.count(needle)!=1:raise SystemExit('replaced closure contrib helper not unique')
helpers=r'''
__device__ __forceinline__ BlockClosureDelta block_pull_rank_shift2(MateValue v,int pos,int h){
    if(v==N)return 0;
    const unsigned long long z=D_BLOCK_CLOSURE_SHIFT232[pos][h];
    const uint32_t raw=v==R?uint32_t(z):uint32_t(z>>32);
    return BlockClosureDelta(static_cast<int32_t>(raw));
}
__device__ __forceinline__ BlockClosureDelta block_pull_rank_cross_ll(int pos,int h){
    return BlockClosureDelta(static_cast<int32_t>(uint32_t(D_BLOCK_CLOSURE_CROSS32[pos][h])));
}
__device__ __forceinline__ BlockClosureDelta block_pull_rank_cross_rr(int pos,int h){
    return BlockClosureDelta(static_cast<int32_t>(uint32_t(D_BLOCK_CLOSURE_CROSS32[pos][h]>>32)));
}
'''
s=s.replace(needle,needle+helpers,1)

# Same-symbol running updates: contrib(v,h+2)-contrib(v,h).
shift_patterns=(
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-\n                        BlockClosureDelta(block_pull_rank_contrib(v,q,hb))','block_pull_rank_shift2(v,q,hb)'),
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-\n                         BlockClosureDelta(block_pull_rank_contrib(v,q,hq))','block_pull_rank_shift2(v,q,hq)'),
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hb))','block_pull_rank_shift2(v,q,hb)'),
    ('BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hq))','block_pull_rank_shift2(v,q,hq)'),
)
shift_repl=0
for old,new in shift_patterns:
    n=s.count(old)
    if n:s=s.replace(old,new);shift_repl+=n
if shift_repl<2:raise SystemExit(f'closure shift2 update replacements too few: {shift_repl}')

# Candidate corrections are also repeated in generic, ILP4 and warp closure
# generators. Cache the exact cross-symbol expressions:
#   LL = contrib(R,q,h+2) - contrib(L,q,h)
#   RR = contrib(L,q,h)   - contrib(R,q,h)
cross_patterns=(
    ('BlockClosureDelta(block_pull_rank_contrib(R,q,hb+2))-\n                        BlockClosureDelta(block_pull_rank_contrib(L,q,hb))','block_pull_rank_cross_ll(q,hb)'),
    ('BlockClosureDelta(block_pull_rank_contrib(R,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(L,q,hb))','block_pull_rank_cross_ll(q,hb)'),
    ('BlockClosureDelta(block_pull_rank_contrib(L,q,hq))-\n                        BlockClosureDelta(block_pull_rank_contrib(R,q,hq))','block_pull_rank_cross_rr(q,hq)'),
    ('BlockClosureDelta(block_pull_rank_contrib(L,q,hq))-BlockClosureDelta(block_pull_rank_contrib(R,q,hq))','block_pull_rank_cross_rr(q,hq)'),
)
cross_repl=0
for old,new in cross_patterns:
    n=s.count(old)
    if n:s=s.replace(old,new);cross_repl+=n
if cross_repl<2:raise SystemExit(f'closure cross correction replacements too few: {cross_repl}')

anchor='''    ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main dp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block dp");'''
if s.count(anchor)!=1:raise SystemExit(f'group DP upload anchor expected once got {s.count(anchor)}')
prep=r'''    unsigned long long closureContrib32[MAXW+1][MAXW+2]{};
    unsigned long long closureShift232[MAXW+1][MAXW+2]{};
    unsigned long long closureCross32[MAXW+1][MAXW+2]{};
    auto closure_contrib_host=[&](int pp,int hh,int which)->unsigned long long{
        if(hh<0||hh>MAXW+1)return 0;
        Code nbranch=0,rbranch=0;
        if(allowed_host(ms.fixed,ms.occ,pp,N))nbranch=ms.dp[pp][hh];
        if(hh>0&&allowed_host(ms.fixed,ms.occ,pp,R))rbranch=ms.dp[pp][hh-1];
        return which==0?static_cast<unsigned long long>(nbranch):static_cast<unsigned long long>(nbranch+rbranch);
    };
    unsigned long long closureAbsMax=0;long long closureShiftAbsMax=0,closureCrossAbsMax=0;
    for(int pp=0;pp<W;++pp)for(int hh=0;hh<=MAXW+1;++hh){
        const unsigned long long cr=closure_contrib_host(pp,hh,0),cl=closure_contrib_host(pp,hh,1);
        if(cr>0xffffffffULL||cl>0xffffffffULL){
            std::cerr<<"closure contrib u32 overflow p="<<pp<<" h="<<hh<<" R="<<cr<<" L="<<cl<<'\n';std::exit(839);
        }
        closureContrib32[pp][hh]=(static_cast<unsigned long long>(uint32_t(cl))<<32)|uint32_t(cr);
        const long long dr=static_cast<long long>(closure_contrib_host(pp,hh+2,0))-static_cast<long long>(cr);
        const long long dl=static_cast<long long>(closure_contrib_host(pp,hh+2,1))-static_cast<long long>(cl);
        if(dr<-2147483648LL||dr>2147483647LL||dl<-2147483648LL||dl>2147483647LL){
            std::cerr<<"closure shift2 i32 overflow p="<<pp<<" h="<<hh<<" R="<<dr<<" L="<<dl<<'\n';std::exit(840);
        }
        closureShift232[pp][hh]=(static_cast<unsigned long long>(uint32_t(int32_t(dl)))<<32)|uint32_t(int32_t(dr));
        const long long xll=static_cast<long long>(closure_contrib_host(pp,hh+2,0))-static_cast<long long>(cl);
        const long long xrr=static_cast<long long>(cl)-static_cast<long long>(cr);
        if(xll<-2147483648LL||xll>2147483647LL||xrr<-2147483648LL||xrr>2147483647LL){
            std::cerr<<"closure cross i32 overflow p="<<pp<<" h="<<hh<<" LL="<<xll<<" RR="<<xrr<<'\n';std::exit(841);
        }
        closureCross32[pp][hh]=(static_cast<unsigned long long>(uint32_t(int32_t(xrr)))<<32)|uint32_t(int32_t(xll));
        closureAbsMax=std::max(closureAbsMax,std::max(cr,cl));
        closureShiftAbsMax=std::max(closureShiftAbsMax,std::max(dr<0?-dr:dr,dl<0?-dl:dl));
        closureCrossAbsMax=std::max(closureCrossAbsMax,std::max(xll<0?-xll:xll,xrr<0?-xrr:xrr));
    }
    ck(cudaMemcpyToSymbol(D_BLOCK_CLOSURE_CONTRIB32,closureContrib32,sizeof(closureContrib32)),"block closure contrib32");
    ck(cudaMemcpyToSymbol(D_BLOCK_CLOSURE_SHIFT232,closureShift232,sizeof(closureShift232)),"block closure shift232");
    ck(cudaMemcpyToSymbol(D_BLOCK_CLOSURE_CROSS32,closureCross32,sizeof(closureCross32)),"block closure cross32");
    std::cerr<<"b300_closure_contrib_table packed32=1 bytes="<<(sizeof(closureContrib32)+sizeof(closureShift232)+sizeof(closureCross32))
             <<" abs_max="<<closureAbsMax<<" shift_abs_max="<<closureShiftAbsMax<<" cross_abs_max="<<closureCrossAbsMax
             <<" shift2_replacements="<<''' + str(shift_repl) + r'''<<" cross_replacements="<<''' + str(cross_repl) + r'''<<" u32_i32_checked=1\n";
'''
s=s.replace(anchor,prep+anchor,1)

for req in ('D_BLOCK_CLOSURE_CONTRIB32','D_BLOCK_CLOSURE_SHIFT232','D_BLOCK_CLOSURE_CROSS32','block_pull_rank_shift2','block_pull_rank_cross_ll','block_pull_rank_cross_rr','block closure contrib32','block closure shift232','block closure cross32','u32_i32_checked=1'):
    if req not in s:raise SystemExit(f'missing packed closure-table artifact: {req}')
for stale in (
    'D_BLOCK_CLOSURE_CONTRIB[MAXW+1][MAXW+2][2]',
    'BlockClosureDelta(block_pull_rank_contrib(v,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hb))',
    'BlockClosureDelta(block_pull_rank_contrib(v,q,hq+2))-BlockClosureDelta(block_pull_rank_contrib(v,q,hq))',
    'BlockClosureDelta(block_pull_rank_contrib(R,q,hb+2))-BlockClosureDelta(block_pull_rank_contrib(L,q,hb))',
    'BlockClosureDelta(block_pull_rank_contrib(L,q,hq))-BlockClosureDelta(block_pull_rank_contrib(R,q,hq))'
):
    if stale in s:raise SystemExit(f'stale closure table artifact remains: {stale}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_closure_contrib_table=packed32+shift2+cross constant_bytes_added=20880 contrib_bytes=6960 shift2_bytes=6960 cross_bytes=6960 values=rank_mass_before_R,rank_mass_before_L update_ops=one_lookup_per_endpoint candidate_cross_ops=one_lookup u32_i32_runtime_checked=1 shift2_replacements={shift_repl} cross_replacements={cross_repl}')
