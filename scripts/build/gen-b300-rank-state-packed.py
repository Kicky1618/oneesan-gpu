#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-rank-state-packed.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'using RankDelta = long long;' not in s or 'dMainRankDelta' not in s or 'rank_delta_groups=' not in s:
    raise SystemExit('packed rank-state requires rank-delta + coverage-report source')

def once(old:str,new:str,label:str)->None:
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

once(
'using RankDelta = long long;',
'''using RankDelta = long long;
using RankState = unsigned long long;
static constexpr RankState B300_RANK_DELTA_MASK=(RankState(1)<<56)-1;
static constexpr RankState B300_RANK_DELTA_SIGN=RankState(1)<<55;
__device__ __forceinline__ RankState b300_pack_rank_state(RankDelta d,int h){
    return (RankState(d)&B300_RANK_DELTA_MASK)|(RankState(uint8_t(h))<<56);
}
__device__ __forceinline__ RankDelta b300_unpack_rank_delta(RankState s){
    RankState raw=s&B300_RANK_DELTA_MASK;
    if(raw&B300_RANK_DELTA_SIGN)raw|=~B300_RANK_DELTA_MASK;
    return RankDelta(raw);
}
__device__ __forceinline__ int b300_unpack_rank_height(RankState s){return int(uint8_t(s>>56));}
__device__ __forceinline__ int b300_rank_height_advance(int h,MateValue v){return h+(v==L)-(v==R);}''',
'rank-state helpers')

once(
'''__global__ void b300_init_main_rank_delta_kernel(const MateID*mates,Code n,RankDelta*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)out[i]=b300_rank_delta_prefix<TARGET_W>(mates[i],p);
}
__global__ void b300_init_block_rank_delta_kernel(const MateID*mates,Code n,RankDelta*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)out[i]=b300_rank_delta_prefix<TARGET_W>(minsert(mates[i],p,N),p);
}''',
'''__global__ void b300_init_main_rank_delta_kernel(const MateID*mates,Code n,RankState*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){MateID m=mates[i];out[i]=b300_pack_rank_state(b300_rank_delta_prefix<TARGET_W>(m,p),height_before_rank_pos<TARGET_W>(m,p));}
}
__global__ void b300_init_block_rank_delta_kernel(const MateID*mates,Code n,RankState*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){MateID b=mates[i];out[i]=b300_pack_rank_state(b300_rank_delta_prefix<TARGET_W>(minsert(b,p,N),p),height_before_rank_pos<TARGET_W-1>(b,p-1));}
}''',
'rank-state init kernels')

once(
'''__device__ __forceinline__ Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p){
    const int h=height_before_rank_pos<TARGET_W>(m,p);''',
'''__device__ __forceinline__ Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p,int h){''',
'main pair rank cached height')
once(
'int p,RankDelta*rank_delta){',
'int p,RankState*rank_state){',
'main rank-state pointer')
once(
'''        RankDelta rd=0;if constexpr(CACHED_RANK_DELTA)rd=rank_delta[i];
        uint64_t acc=in[i];''',
'''        RankDelta rd=0;int rank_h=0;
        if constexpr(CACHED_RANK_DELTA){RankState rs=rank_state[i];rd=b300_unpack_rank_delta(rs);rank_h=b300_unpack_rank_height(rs);}
        else rank_h=height_before_rank_pos<TARGET_W>(m,p);
        uint64_t acc=in[i];''',
'main rank-state load')
once(
'const Code j=main_pull_direct_pair_source_rank(i,m,p);',
'const Code j=main_pull_direct_pair_source_rank(i,m,p,rank_h);',
'main pair uses packed height')
once(
'''        if constexpr(CACHED_RANK_DELTA){int h=height_before_rank_pos<TARGET_W>(m,p);rank_delta[i]=rd+b300_rank_delta_step(mget(m,p),p,h);}''',
'''        if constexpr(CACHED_RANK_DELTA){RankDelta nd=rd+b300_rank_delta_step(mget(m,p),p,rank_h);rank_state[i]=b300_pack_rank_state(nd,b300_rank_height_advance(rank_h,mget(m,p)));}''',
'main rank-state store')

# There are two kernel signatures ending in RankDelta*: main was replaced above;
# replace the remaining block signature only.
once(
'int p,RankDelta*rank_delta){',
'int p,RankState*rank_state){',
'block rank-state pointer')
once(
'''        RankDelta rd=0,next_rd=0;if constexpr(CACHED_RANK_DELTA){rd=rank_delta[i];int h=height_before_rank_pos<TARGET_W-1>(b,p-1);next_rd=rd+b300_rank_delta_step(mget(b,p-1),p,h);}
        Count acc=0;''',
'''        RankDelta rd=0,next_rd=0;int rank_h=0;
        if constexpr(CACHED_RANK_DELTA){RankState rs=rank_state[i];rd=b300_unpack_rank_delta(rs);rank_h=b300_unpack_rank_height(rs);next_rd=rd+b300_rank_delta_step(mget(b,p-1),p,rank_h);}
        else rank_h=height_before_rank_pos<TARGET_W-1>(b,p-1);
        Count acc=0;''',
'block rank-state load')
once(
'const int H=height_before_rank_pos<TARGET_W>(d,p);',
'const int H=rank_h;',
'block closure uses packed height')
once(
'''        if constexpr(CACHED_RANK_DELTA)rank_delta[i]=next_rd;''',
'''        if constexpr(CACHED_RANK_DELTA)rank_state[i]=b300_pack_rank_state(next_rd,b300_rank_height_advance(rank_h,mget(b,p-1)));''',
'block rank-state store')

# Keep the existing field names so coverage-report code remains intact; only
# storage changes from signed 64-bit delta to packed delta+height state.
s=s.replace('RankDelta*dMainRankDelta=nullptr,*dBlockRankDelta=nullptr;',
            'RankState*dMainRankDelta=nullptr,*dBlockRankDelta=nullptr;',1)
s=s.replace('sizeof(RankDelta)', 'sizeof(RankState)')
s=s.replace('(RankDelta*)(arena+off)', '(RankState*)(arena+off)')

# W<=28 has at most 385,719,506,620 complete states. Any grouped local rank and
# therefore any prefix-rank difference is strictly smaller than that, far below
# signed 56-bit range. No group-size fallback is needed.
for stale in ('rankStateI32Safe','0x7fffffffULL','int32_t(uint32_t'):
    if stale in s:raise SystemExit(f'stale int32 rank-state artifact remains: {stale}')

for required in ('B300_RANK_DELTA_MASK','B300_RANK_DELTA_SIGN','b300_pack_rank_state','RankState*dMainRankDelta','main_pull_direct_pair_source_rank(i,m,p,rank_h)','const int H=rank_h'):
    if required not in s:raise SystemExit(f'missing packed rank-state artifact: {required}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_rank_state_packed=1 storage_bytes=8 delta_bits=56 height_bits=8 hbm_rw_per_state_step=16 prefix_rank_walk=0 prefix_height_popcount=0 width_max=28 full_state_bound=385719506620 fallback_required=0')
