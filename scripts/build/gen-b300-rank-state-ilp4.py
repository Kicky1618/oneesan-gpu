#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-rank-state-ilp4.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'using RankState = unsigned long long;',
    'b300_unpack_rank_delta','b300_unpack_rank_height','b300_pack_rank_state',
    'main_pull_direct_pair_source_rank(i,m,p,rank_h)',
    'main_pull_kernel<true,true>'
):
    if req not in s: raise SystemExit(f'rank-state ILP4 requires generated artifact: {req}')

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s: raise SystemExit('rank_full marker not found')
insert=r'''

__global__ void b300_main_pull_rankstate_ilp4_kernel(
    const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,
    const Count* __restrict__ in_block,Code nblock,
    Count* __restrict__ out_main,int p,RankState* __restrict__ rank_state
){
    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    const Code grid=Code(gridDim.x)*blockDim.x;
    for(Code base=tid;base<n;base+=4*grid){
        const Code i0=base,i1=base+grid,i2=base+2*grid,i3=base+3*grid;
        const bool v1=i1<n,v2=i2<n,v3=i3<n;
        const MateID m0=mates[i0],m1=v1?mates[i1]:MateID(0),m2=v2?mates[i2]:MateID(0),m3=v3?mates[i3]:MateID(0);
        const RankState s0=rank_state[i0],s1=v1?rank_state[i1]:RankState(0),s2=v2?rank_state[i2]:RankState(0),s3=v3?rank_state[i3]:RankState(0);
        const RankDelta rd0=b300_unpack_rank_delta(s0),rd1=v1?b300_unpack_rank_delta(s1):RankDelta(0),rd2=v2?b300_unpack_rank_delta(s2):RankDelta(0),rd3=v3?b300_unpack_rank_delta(s3):RankDelta(0);
        const int h0=b300_unpack_rank_height(s0),h1=v1?b300_unpack_rank_height(s1):0,h2=v2?b300_unpack_rank_height(s2):0,h3=v3?b300_unpack_rank_height(s3):0;
        const MateValuePair p0=mpair(m0,p),p1=v1?mpair(m1,p):NN,p2=v2?mpair(m2,p):NN,p3=v3?mpair(m3,p):NN;
        const bool hp0=(p0==LR||p0==NR||p0==NL),hp1=v1&&(p1==LR||p1==NR||p1==NL),hp2=v2&&(p2==LR||p2==NR||p2==NL),hp3=v3&&(p3==LR||p3==NR||p3==NL);
        const Code pj0=hp0?main_pull_direct_pair_source_rank(i0,m0,p,h0):Code(0),pj1=hp1?main_pull_direct_pair_source_rank(i1,m1,p,h1):Code(0),pj2=hp2?main_pull_direct_pair_source_rank(i2,m2,p,h2):Code(0),pj3=hp3?main_pull_direct_pair_source_rank(i3,m3,p,h3):Code(0);
        const MateValue mv0=mget(m0,p),mv1=v1?mget(m1,p):N,mv2=v2?mget(m2,p):N,mv3=v3?mget(m3,p):N;
        const bool hb0=nblock&&mv0==N,hb1=v1&&nblock&&mv1==N,hb2=v2&&nblock&&mv2==N,hb3=v3&&nblock&&mv3==N;
        const Code bj0=hb0?b300_add_rank_delta(i0,rd0):Code(0),bj1=hb1?b300_add_rank_delta(i1,rd1):Code(0),bj2=hb2?b300_add_rank_delta(i2,rd2):Code(0),bj3=hb3?b300_add_rank_delta(i3,rd3):Code(0);

        // The next packed rank state depends only on mate/rank metadata, not on
        // any Count gather. Commit it before the long-latency source reads so
        // mate, delta and height temporaries do not remain live across HBM wait.
        rank_state[i0]=b300_pack_rank_state(rd0+b300_rank_delta_step(mv0,p,h0),b300_rank_height_advance(h0,mv0));
        if(v1)rank_state[i1]=b300_pack_rank_state(rd1+b300_rank_delta_step(mv1,p,h1),b300_rank_height_advance(h1,mv1));
        if(v2)rank_state[i2]=b300_pack_rank_state(rd2+b300_rank_delta_step(mv2,p,h2),b300_rank_height_advance(h2,mv2));
        if(v3)rank_state[i3]=b300_pack_rank_state(rd3+b300_rank_delta_step(mv3,p,h3),b300_rank_height_advance(h3,mv3));

        // All random addresses are ready. Issue up to eight independent random
        // Count reads first, then the four coalesced self reads. The previous
        // implementation kept self[4] and all rank metadata live while these
        // random requests were outstanding.
        const Count pair0=hp0?in[pj0]:Count(0),pair1=hp1?in[pj1]:Count(0),pair2=hp2?in[pj2]:Count(0),pair3=hp3?in[pj3]:Count(0);
        const Count block0=(hb0&&bj0<nblock)?in_block[bj0]:Count(0),block1=(hb1&&bj1<nblock)?in_block[bj1]:Count(0),block2=(hb2&&bj2<nblock)?in_block[bj2]:Count(0),block3=(hb3&&bj3<nblock)?in_block[bj3]:Count(0);
        const Count self0=in[i0],self1=v1?in[i1]:Count(0),self2=v2?in[i2]:Count(0),self3=v3?in[i3]:Count(0);
        const uint64_t mod=D_MOD;
        uint64_t a0=uint64_t(self0)+pair0+block0;if(a0>=mod)a0-=mod;if(a0>=mod)a0-=mod;out_main[i0]=Count(a0);
        if(v1){uint64_t a=uint64_t(self1)+pair1+block1;if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i1]=Count(a);}
        if(v2){uint64_t a=uint64_t(self2)+pair2+block2;if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i2]=Count(a);}
        if(v3){uint64_t a=uint64_t(self3)+pair3+block3;if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i3]=Count(a);}
    }
}
'''
s=s.replace(marker,insert+marker,1)
old='''if(useRankDelta)main_pull_kernel<true,true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,c.dMainRankDelta);'''
new='''if(useRankDelta)b300_main_pull_rankstate_ilp4_kernel<<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,c.dMainRankDelta);'''
if s.count(old)!=1: raise SystemExit(f'rank-state ILP4 launch anchor expected one match got {s.count(old)}')
s=s.replace(old,new,1)
for required in (
    'b300_main_pull_rankstate_ilp4_kernel','base+=4*grid','pair3=','block3=',
    'rank_state[i3]=b300_pack_rank_state','Issue up to eight independent random',
    'const Count self0=in[i0]'
):
    if required not in s: raise SystemExit(f'missing rank-state ILP4 artifact: {required}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_rank_state_ilp4=1 destinations_per_thread=4 packed_rank_state=1 index_first=1 hbm_request_overlap=pair,block,four_destinations recurrence_exact=1 rank_state_store_before_count_gather=1 self_load_after_random_addresses=1 random_count_requests_up_to=8 register_live_range_reduced=1 register_pressure_requires_ab=1')
