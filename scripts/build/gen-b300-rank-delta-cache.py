#!/usr/bin/env python3
from __future__ import annotations
import pathlib, sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-rank-delta-cache.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'template<bool CACHED_MATE>\n__global__ void main_pull_kernel' not in s:
    raise SystemExit('rank-delta cache requires main-pull transform')
if 'template<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel' not in s or 'dBlockMate' not in s:
    raise SystemExit('rank-delta cache requires block-pull + block-mate-cache transforms')

def once(old:str,new:str,label:str)->None:
    global s
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s: raise SystemExit('rank_full marker not found')
helpers=r'''

using RankDelta = long long;

__device__ __forceinline__ RankDelta b300_rank_delta_step(MateValue v,int p,int h){
    Code a=0,b=0;
    if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,p,N))a+=D_MAIN_DP[p][h];
    if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,p,R))a+=D_MAIN_DP[p][h-1];
    int q=p-1;
    if(v>N&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N))b+=D_BLOCK_DP[q][h];
    if(v>R&&h>0&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,R))b+=D_BLOCK_DP[q][h-1];
    return RankDelta(b)-RankDelta(a);
}

template<int WIDTH>
__device__ __forceinline__ RankDelta b300_rank_delta_prefix(MateID full,int p){
    Code a=0,b=0;int h=1;
#pragma unroll
    for(int pos=WIDTH-1;pos>p;--pos){
        MateValue v=mget(full,pos);
        if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,N))a+=D_MAIN_DP[pos][h];
        if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,R))a+=D_MAIN_DP[pos][h-1];
        int q=pos-1;
        if(v>N&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N))b+=D_BLOCK_DP[q][h];
        if(v>R&&h>0&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,R))b+=D_BLOCK_DP[q][h-1];
        if(v==R)--h;else if(v==L)++h;
    }
    return RankDelta(b)-RankDelta(a);
}

__device__ __forceinline__ Code b300_add_rank_delta(Code x,RankDelta d){
    return d>=0?x+Code(d):x-Code(-d);
}
__device__ __forceinline__ Code b300_sub_rank_delta(Code x,RankDelta d){
    return d>=0?x-Code(d):x+Code(-d);
}

__global__ void b300_init_main_rank_delta_kernel(const MateID*mates,Code n,RankDelta*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)out[i]=b300_rank_delta_prefix<TARGET_W>(mates[i],p);
}
__global__ void b300_init_block_rank_delta_kernel(const MateID*mates,Code n,RankDelta*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)out[i]=b300_rank_delta_prefix<TARGET_W>(minsert(mates[i],p,N),p);
}
'''
s=s.replace(marker,helpers+marker,1)

# Main pull: use cached b-a prefix delta for drop rank, then advance it one p.
once(
'''template<bool CACHED_MATE>
__global__ void main_pull_kernel(const Count*in,const MateID*mates,Code n,const Count*in_block,Code nblock,Count*out_main,int p){''',
'''template<bool CACHED_MATE,bool CACHED_RANK_DELTA>
__global__ void main_pull_kernel(const Count*in,const MateID*mates,Code n,const Count*in_block,Code nblock,Count*out_main,int p,RankDelta*rank_delta){''',
'main pull rank-delta template')
once(
'''        else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
        uint64_t acc=in[i];''',
'''        else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
        RankDelta rd=0;if constexpr(CACHED_RANK_DELTA)rd=rank_delta[i];
        uint64_t acc=in[i];''',
'main pull delta load')
once(
'''        if(nblock&&mget(m,p)==N){Code j=rank_drop_n_t<TARGET_W>(i,m,p);if(j<nblock)acc+=in_block[j];}''',
'''        if(nblock&&mget(m,p)==N){Code j;if constexpr(CACHED_RANK_DELTA)j=b300_add_rank_delta(i,rd);else j=rank_drop_n_t<TARGET_W>(i,m,p);if(j<nblock)acc+=in_block[j];}''',
'main pull cached drop rank')
once(
'''        uint64_t mod=D_MOD;if(acc>=mod)acc-=mod;if(acc>=mod)acc-=mod;out_main[i]=Count(acc);
    }
}''',
'''        uint64_t mod=D_MOD;if(acc>=mod)acc-=mod;if(acc>=mod)acc-=mod;out_main[i]=Count(acc);
        if constexpr(CACHED_RANK_DELTA){int h=height_before_rank_pos<TARGET_W>(m,p);rank_delta[i]=rd+b300_rank_delta_step(mget(m,p),p,h);}
    }
}''',
'main pull delta recurrence store')

# Block pull: rank_lift = block_rank - (b-a). The next-p delta is one O(1)
# contribution away and also serves the closure base rank at insertion p-1.
once(
'''template<bool CACHED_BLOCK_MATE>
__global__ void block_pull_kernel(const Count*in_main,const MateID*block_mates,Code n,Count*out_block,int p){''',
'''template<bool CACHED_BLOCK_MATE,bool CACHED_RANK_DELTA>
__global__ void block_pull_kernel(const Count*in_main,const MateID*block_mates,Code n,Count*out_block,int p,RankDelta*rank_delta){''',
'block pull rank-delta template')
once(
'''        MateID b;if constexpr(CACHED_BLOCK_MATE)b=block_mates[i];else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
        Count acc=0;''',
'''        MateID b;if constexpr(CACHED_BLOCK_MATE)b=block_mates[i];else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
        RankDelta rd=0,next_rd=0;if constexpr(CACHED_RANK_DELTA){rd=rank_delta[i];int h=height_before_rank_pos<TARGET_W-1>(b,p-1);next_rd=rd+b300_rank_delta_step(mget(b,p-1),p,h);}
        Count acc=0;''',
'block pull delta load/update')
once(
'''            Code j=rank_lift_n_t<TARGET_W>(i,b,p);
            block_pull_add_rank(acc,in_main,j);''',
'''            Code j;if constexpr(CACHED_RANK_DELTA)j=b300_sub_rank_delta(i,rd);else j=rank_lift_n_t<TARGET_W>(i,b,p);
            block_pull_add_rank(acc,in_main,j);''',
'block endpoint cached lift rank')
once(
'''            Code base_rank=rank_lift_n_t<TARGET_W>(i,b,p-1);''',
'''            Code base_rank;if constexpr(CACHED_RANK_DELTA)base_rank=b300_sub_rank_delta(i,next_rd);else base_rank=rank_lift_n_t<TARGET_W>(i,b,p-1);''',
'block closure cached base rank')
once(
'''        out_block[i]=acc;
    }
}''',
'''        out_block[i]=acc;
        if constexpr(CACHED_RANK_DELTA)rank_delta[i]=next_rd;
    }
}''',
'block delta recurrence store')

# Scratch: one signed 64-bit stream for main states and one for block states.
once(
'MateID*dMate=nullptr,*dBlockMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;',
'MateID*dMate=nullptr,*dBlockMate=nullptr;RankDelta*dMainRankDelta=nullptr,*dBlockRankDelta=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;',
'DeviceCtx rank-delta fields')
once(
'void ensure(Code m,Code b,bool useMate,bool useBlockMate,size_t im,size_t id){',
'void ensure(Code m,Code b,bool useMate,bool useBlockMate,bool useRankDelta,size_t im,size_t id){',
'DeviceCtx ensure rank-delta flag')
once(
'bmb=useBlockMate?al(size_t(b)*sizeof(MateID)):0,need=2*ab+2*db+mb+bmb;',
'bmb=useBlockMate?al(size_t(b)*sizeof(MateID)):0,mrb=useRankDelta?al(size_t(m)*sizeof(RankDelta)):0,brb=useRankDelta?al(size_t(b)*sizeof(RankDelta)):0,need=2*ab+2*db+mb+bmb+mrb+brb;',
'DeviceCtx rank-delta bytes')
once(
'dBlockMate=useBlockMate?(MateID*)(arena+off):nullptr;',
'dBlockMate=useBlockMate?(MateID*)(arena+off):nullptr;off+=bmb;dMainRankDelta=useRankDelta?(RankDelta*)(arena+off):nullptr;off+=mrb;dBlockRankDelta=useRankDelta?(RankDelta*)(arena+off):nullptr;',
'DeviceCtx rank-delta pointers')

old_size='''size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID),blockMateBytes=size_t(ds.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);bool useBlockMate=(countBytes+(useMate?mateBytes:0)+blockMateBytes<=target);
    c.ensure(ms.size,ds.size,useMate,useBlockMate,pg.mi.size(),pg.di.size());'''
new_size='''size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID),blockMateBytes=size_t(ds.size)*sizeof(MateID),rankDeltaBytes=size_t(ms.size+ds.size)*sizeof(RankDelta);bool useMate=(countBytes+mateBytes<=target);bool useBlockMate=(countBytes+(useMate?mateBytes:0)+blockMateBytes<=target);bool useRankDelta=useMate&&useBlockMate&&(countBytes+mateBytes+blockMateBytes+rankDeltaBytes<=target);
    c.ensure(ms.size,ds.size,useMate,useBlockMate,useRankDelta,pg.mi.size(),pg.di.size());'''
once(old_size,new_size,'process_group rank-delta sizing')

# Mate materialization/gather has completed and is synchronized here, so initialize
# the delta streams once per group/window before the p loop.
once(
'''    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;''',
'''    if(useRankDelta){if(ms.size)b300_init_main_rank_delta_kernel<<<bm,threads>>>(c.dMate,ms.size,c.dMainRankDelta,wp.p_hi);if(ds.size)b300_init_block_rank_delta_kernel<<<bd,threads>>>(c.dBlockMate,ds.size,c.dBlockRankDelta,wp.p_hi);ck(cudaGetLastError(),"rank delta init");ck(cudaDeviceSynchronize(),"rank delta init sync");}
    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;''',
'rank-delta initialization')

# Full-pull launches: use the delta variant only when both Mate caches and the
# extra 8B/state streams fit the existing scratch target. Otherwise exact fallback.
once(
'''if(useMate)main_pull_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
                else main_pull_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p);''',
'''if(useRankDelta)main_pull_kernel<true,true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,c.dMainRankDelta);
                else if(useMate)main_pull_kernel<true,false><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,nullptr);
                else main_pull_kernel<false,false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p,nullptr);''',
'main pull rank-delta launch')
once(
'''if(ds.size){if(useBlockMate)block_pull_kernel<true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p);else block_pull_kernel<false><<<bd,threads,0,c.sBlock>>>(cur,nullptr,ds.size,dnext,p);}''',
'''if(ds.size){if(useRankDelta)block_pull_kernel<true,true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockRankDelta);else if(useBlockMate)block_pull_kernel<true,false><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,nullptr);else block_pull_kernel<false,false><<<bd,threads,0,c.sBlock>>>(cur,nullptr,ds.size,dnext,p,nullptr);}''',
'block pull rank-delta launch')

for required in (
    'b300_rank_delta_step','b300_init_main_rank_delta_kernel','dMainRankDelta','useRankDelta=',
    'main_pull_kernel<true,true>','block_pull_kernel<true,true>','rank_delta[i]=next_rd'):
    if required not in s: raise SystemExit(f'missing rank-delta artifact: {required}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_rank_delta_cache=1 bytes_per_state=8 main_drop_prefix_walk=0 block_lift_prefix_walk=0 recurrence_step=O1 conditional_scratch=1 exact_fallback=1 hbm_readwrite_per_state_step=16')
