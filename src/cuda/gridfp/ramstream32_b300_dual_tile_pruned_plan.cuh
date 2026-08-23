#pragma once

#include "ramstream32_b300_dual_tile_pruned_peer.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

struct B300DualPrunedRun {
    Code begin = 0;
    Code n = 0;
    uint8_t mode = 0;
};
struct B300DualPrunedPairPlan {
    int a = 0, b = 0;
    std::vector<B300DualPrunedRun> runs;
};
struct B300DualPrunedArrayPlan {
    bool blocked = false;
    bool low_to_high = false;
    std::vector<B300DualPrunedPairPlan> pairs;
    uint64_t logical_bytes = 0;   // minimum bytes if every raw mode boundary is kept
    uint64_t scheduled_bytes = 0; // bytes after launch-aware interval coalescing
    uint64_t full_bytes = 0;      // bytes of the unpruned whole-slot swap
    uint64_t launches = 0;
};
struct B300DualPrunedSchedulePlan {
    std::vector<B300DualPrunedArrayPlan> l2h_main;
    std::vector<B300DualPrunedArrayPlan> l2h_block;
    std::vector<B300DualPrunedArrayPlan> h2l_main;
    uint64_t launch_penalty_bytes = 0;

    uint64_t logical_bytes_per_residue() const {
        uint64_t x=0;for(const auto&p:l2h_main)x+=p.logical_bytes;
        for(const auto&p:l2h_block)x+=p.logical_bytes;
        for(const auto&p:h2l_main)x+=p.logical_bytes;return x;
    }
    uint64_t scheduled_bytes_per_residue() const {
        uint64_t x=0;for(const auto&p:l2h_main)x+=p.scheduled_bytes;
        for(const auto&p:l2h_block)x+=p.scheduled_bytes;
        for(const auto&p:h2l_main)x+=p.scheduled_bytes;return x;
    }
    uint64_t full_bytes_per_residue() const {
        uint64_t x=0;for(const auto&p:l2h_main)x+=p.full_bytes;
        for(const auto&p:l2h_block)x+=p.full_bytes;
        for(const auto&p:h2l_main)x+=p.full_bytes;return x;
    }
    uint64_t launches_per_residue() const {
        uint64_t x=0;for(const auto&p:l2h_main)x+=p.launches;
        for(const auto&p:l2h_block)x+=p.launches;
        for(const auto&p:h2l_main)x+=p.launches;return x;
    }
};

struct B300DualRawInterval { Code begin=0,end=0; uint8_t mode=0; };

static inline int b300_dt_mode_popcount(uint32_t mode) {
    return int(mode & 1u) + int((mode >> 1) & 1u);
}

static B300DualPrunedArrayPlan b300_dt_build_pruned_array_plan(
    const B300DualTileHost& z, const StorageLayout& l,
    bool blocked, bool low_to_high, const B300DualReachStage& stage,
    uint64_t launch_penalty_bytes
) {
    B300DualPrunedArrayPlan out;
    out.blocked=blocked;out.low_to_high=low_to_high;
    const auto&sz=blocked?z.pair_block_size:z.pair_main_size;
    std::vector<Code> bounds;

    for(int a=0;a<z.ngpu;++a)for(int b=a+1;b<z.ngpu;++b){
        int ahi,alo,bhi,blo;
        if(low_to_high){ahi=b;alo=a;bhi=a;blo=b;}
        else {ahi=a;alo=b;bhi=b;blo=a;}
        Code na=sz[ahi][alo],nb=sz[bhi][blo],limit=std::max(na,nb);
        out.full_bytes+=uint64_t(na+nb)*sizeof(Count);
        if(!limit)continue;
        b300_dt_pruned_boundaries(z,l,blocked,ahi,alo,bhi,blo,bounds);

        std::vector<B300DualRawInterval> e;
        for(size_t k=0;k+1<bounds.size();++k){
            Code x=bounds[k],y=std::min(bounds[k+1],limit);if(x>=y)continue;
            bool aa=x<na&&b300_dt_pruned_stream_active_at(z,l,stage,blocked,ahi,alo,x);
            bool bb=x<nb&&b300_dt_pruned_stream_active_at(z,l,stage,blocked,bhi,blo,x);
            uint8_t mode=uint8_t((aa?1:0)|(bb?2:0));
            if(!e.empty()&&e.back().mode==mode&&e.back().end==x)e.back().end=y;
            else e.push_back({x,y,mode});
            out.logical_bytes+=uint64_t(y-x)*sizeof(Count)*uint64_t(b300_dt_mode_popcount(mode));
        }
        if(e.empty())continue;

        struct Score {
            uint64_t objective=std::numeric_limits<uint64_t>::max();
            uint64_t bytes=std::numeric_limits<uint64_t>::max();
            uint32_t launches=std::numeric_limits<uint32_t>::max();
        };
        auto better=[](const Score&a,const Score&b){
            if(a.objective!=b.objective)return a.objective<b.objective;
            if(a.bytes!=b.bytes)return a.bytes<b.bytes;
            return a.launches<b.launches;
        };
        const int m=int(e.size());
        std::vector<Score> dp(m+1);
        std::vector<int> prev(m+1,-1);
        std::vector<uint8_t> pmode(m+1,0);
        dp[0]={0,0,0};
        for(int i=0;i<m;++i){
            if(dp[i].objective==std::numeric_limits<uint64_t>::max())continue;
            uint8_t mode=0;
            for(int j=i+1;j<=m;++j){
                mode=uint8_t(mode|e[j-1].mode);
                uint64_t n=uint64_t(e[j-1].end-e[i].begin);
                uint64_t bytes=n*sizeof(Count)*uint64_t(b300_dt_mode_popcount(mode));
                uint32_t launches=mode?1u:0u;
                Score cand;
                cand.bytes=dp[i].bytes+bytes;
                cand.launches=dp[i].launches+launches;
                cand.objective=cand.bytes+uint64_t(cand.launches)*launch_penalty_bytes;
                if(better(cand,dp[j])){dp[j]=cand;prev[j]=i;pmode[j]=mode;}
            }
        }
        if(prev[m]<0)std::exit(657);

        B300DualPrunedPairPlan pp;pp.a=a;pp.b=b;
        std::vector<B300DualPrunedRun> rev;
        for(int j=m;j>0;){int i=prev[j];uint8_t mode=pmode[j];
            if(mode)rev.push_back({e[i].begin,e[j-1].end-e[i].begin,mode});
            j=i;
        }
        std::reverse(rev.begin(),rev.end());pp.runs=std::move(rev);
        out.scheduled_bytes+=dp[m].bytes;out.launches+=dp[m].launches;
        if(!pp.runs.empty())out.pairs.push_back(std::move(pp));
    }
    return out;
}

static B300DualPrunedSchedulePlan build_b300_dual_pruned_schedule_plan(
    const B300DualTileHost&z,const StorageLayout&l,
    const B300DualReachSchedule&reach,uint64_t launch_penalty_bytes
){
    B300DualPrunedSchedulePlan p;p.launch_penalty_bytes=launch_penalty_bytes;
    p.l2h_main.reserve(reach.l2h.size());p.l2h_block.reserve(reach.l2h.size());
    p.h2l_main.reserve(reach.h2l.size());
    for(const auto&s:reach.l2h){
        p.l2h_main.push_back(b300_dt_build_pruned_array_plan(z,l,false,true,s,launch_penalty_bytes));
        p.l2h_block.push_back(b300_dt_build_pruned_array_plan(z,l,true,true,s,launch_penalty_bytes));
    }
    for(const auto&s:reach.h2l)
        p.h2l_main.push_back(b300_dt_build_pruned_array_plan(z,l,false,false,s,launch_penalty_bytes));
    return p;
}

static long double b300_dt_execute_pruned_array_plan(
    const B300DualTileHost&z,Count**ptrs,const B300DualPrunedArrayPlan&p,
    B300DualPrunedPeerContext&ctx,B300DualShuffleStats*stats=nullptr
){
    ctx.init(z.ngpu);
    for(const auto&pair:p.pairs)for(const auto&r:pair.runs)
        b300_dt_pruned_launch_run(z,ptrs,p.blocked,pair.a,pair.b,r.begin,r.n,r.mode,ctx);
    ctx.sync("dual planned pruned orientation sync");
    if(stats){
        if(p.blocked)stats->block_bytes+=(long double)p.scheduled_bytes;
        else stats->main_bytes+=(long double)p.scheduled_bytes;
        stats->rounds+=1;stats->chunk_barriers+=p.launches;
    }
    return (long double)p.scheduled_bytes;
}

static inline void b300_dt_execute_pruned_l2h(
    const B300DualTileHost&z,Count**mainp,Count**blockp,
    const B300DualPrunedArrayPlan&mp,const B300DualPrunedArrayPlan&bp,
    B300DualPrunedPeerContext&ctx,B300DualShuffleStats*stats=nullptr
){
    b300_dt_execute_pruned_array_plan(z,mainp,mp,ctx,stats);
    b300_dt_execute_pruned_array_plan(z,blockp,bp,ctx,stats);
}

static inline void b300_dt_execute_pruned_h2l_main(
    const B300DualTileHost&z,Count**mainp,const B300DualPrunedArrayPlan&mp,
    B300DualPrunedPeerContext&ctx,B300DualShuffleStats*stats=nullptr
){
    b300_dt_execute_pruned_array_plan(z,mainp,mp,ctx,stats);
}
