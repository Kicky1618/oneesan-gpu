#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-main-pull-ilp3.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('template<bool CACHED_MATE>\n__global__ void main_pull_kernel','main_pull_direct_pair_source_rank','b300_low_cached_drop_rank'):
    if req not in s:raise SystemExit(f'ILP3 requires generated artifact: {req}')
if 'dBlockMate' not in s:raise SystemExit('ILP3 expects full-pull + block-mate-cache source')

high_rank='b300_high_chunk_drop_rank(i,m,p)' if 'b300_high_chunk_drop_rank' in s else 'rank_drop_n_t<TARGET_W>(i,m,p)'
marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:raise SystemExit('rank_full marker not found')
insert=f'''

__device__ __forceinline__ void b300_main_pull_prepare3(
    Code i,MateID m,int p,Code nblock,
    Code& pair_j,bool& has_pair,Code& block_j,bool& has_block
){{
    const MateValuePair pair=mpair(m,p);
    has_pair=(pair==LR||pair==NR||pair==NL);
    if(has_pair)pair_j=main_pull_direct_pair_source_rank(i,m,p);
    has_block=false;
    if(nblock&&mget(m,p)==N){{
        block_j=b300_low_window_cache_active()?b300_low_cached_drop_rank(i,m,p):{high_rank};
        has_block=block_j<nblock;
    }}
}}

// Three independent destinations per thread: intermediate MLP/register-pressure
// point between the ILP2 and ILP4 experiments.
__global__ void main_pull_kernel_ilp3(
    const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,
    const Count* __restrict__ in_block,Code nblock,
    Count* __restrict__ out_main,int p
){{
    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    const Code grid=Code(gridDim.x)*blockDim.x;
    for(Code base=tid;base<n;base+=3*grid){{
        const Code i0=base,i1=base+grid,i2=base+2*grid;
        const bool v1=i1<n,v2=i2<n;
        const MateID m0=mates[i0];
        const MateID m1=v1?mates[i1]:MateID(0);
        const MateID m2=v2?mates[i2]:MateID(0);
        const Count self0=in[i0];
        const Count self1=v1?in[i1]:Count(0);
        const Count self2=v2?in[i2]:Count(0);

        Code pj0=0,pj1=0,pj2=0,bj0=0,bj1=0,bj2=0;
        bool hp0=false,hp1=false,hp2=false,hb0=false,hb1=false,hb2=false;
        b300_main_pull_prepare3(i0,m0,p,nblock,pj0,hp0,bj0,hb0);
        if(v1)b300_main_pull_prepare3(i1,m1,p,nblock,pj1,hp1,bj1,hb1);
        if(v2)b300_main_pull_prepare3(i2,m2,p,nblock,pj2,hp2,bj2,hb2);

        const Count pair0=hp0?in[pj0]:Count(0);
        const Count pair1=hp1?in[pj1]:Count(0);
        const Count pair2=hp2?in[pj2]:Count(0);
        const Count block0=hb0?in_block[bj0]:Count(0);
        const Count block1=hb1?in_block[bj1]:Count(0);
        const Count block2=hb2?in_block[bj2]:Count(0);

        const uint64_t mod=D_MOD;
        uint64_t a0=uint64_t(self0)+pair0+block0;if(a0>=mod)a0-=mod;if(a0>=mod)a0-=mod;out_main[i0]=Count(a0);
        if(v1){{uint64_t a1=uint64_t(self1)+pair1+block1;if(a1>=mod)a1-=mod;if(a1>=mod)a1-=mod;out_main[i1]=Count(a1);}}
        if(v2){{uint64_t a2=uint64_t(self2)+pair2+block2;if(a2>=mod)a2-=mod;if(a2>=mod)a2-=mod;out_main[i2]=Count(a2);}}
    }}
}}
'''
s=s.replace(marker,insert+marker,1)
old='''if(ms.size){
                if(useMate)main_pull_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
                else main_pull_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p);
            }'''
new='''if(ms.size){
                if(useMate)main_pull_kernel_ilp3<<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
                else main_pull_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p);
            }'''
if s.count(old)!=1:raise SystemExit(f'main pull launch anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)
for required in ('main_pull_kernel_ilp3','base+=3*grid','const Count pair2=','const Count block2=','b300_low_cached_drop_rank'):
    if required not in s:raise SystemExit(f'missing ILP3 artifact: {required}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_pull_ilp3=1 destinations_per_thread=3 index_first=1 low_rank=chunked high_rank={"chunked" if "b300_high_chunk_drop_rank" in s else "generic"} register_pressure_requires_b300_ab=1')
