#pragma once

#include "ramstream32_b300_dual_tile_opt.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

struct B300DualJointOptStats {
    uint64_t logical_bytes=0;
    uint64_t retained_initial=0;
    uint64_t retained_final=0;
    uint64_t moves=0,swaps=0;
    double high_load_max_over_avg=0,low_load_max_over_avg=0;
    int iterations=0;
};

static void b300_dt_generic_assignment_refine(
    std::vector<uint8_t>&owner,const std::vector<uint64_t>&load,
    const std::vector<std::array<uint64_t,MAXGPU>>&benefit,
    int ngpu,double slack_gib,int pair_limit,
    uint64_t&moves,uint64_t&swaps
){
    std::array<uint64_t,MAXGPU> gl{};uint64_t total=0;
    for(size_t i=0;i<owner.size();++i){gl[owner[i]]+=load[i];total+=load[i];}
    long double avg=(long double)total/ngpu;uint64_t slack=uint64_t(slack_gib*(1ull<<30));
    uint64_t lo=avg>slack?uint64_t(avg-slack):0,hi=uint64_t(avg+slack);
    for(int pass=0;pass<12;++pass){bool changed=false;
        for(uint32_t m=0;m<owner.size();++m){int a=owner[m],best=a;int64_t gain=0;
            for(int b=0;b<ngpu;++b)if(b!=a&&gl[b]+load[m]<=hi&&gl[a]-load[m]>=lo){int64_t d=int64_t(benefit[m][b])-int64_t(benefit[m][a]);if(d>gain){gain=d;best=b;}}
            if(best!=a){gl[a]-=load[m];gl[best]+=load[m];owner[m]=uint8_t(best);++moves;changed=true;}}
        if(!changed)break;
    }
    if(pair_limit<=0)return;
    for(int pass=0;pass<8;++pass){std::array<std::vector<uint32_t>,MAXGPU>cand;
        for(uint32_t m=0;m<owner.size();++m)cand[owner[m]].push_back(m);
        for(int g=0;g<ngpu;++g){std::sort(cand[g].begin(),cand[g].end(),[&](uint32_t a,uint32_t b){
            auto regret=[&](uint32_t m){uint64_t z=benefit[m][owner[m]];for(int q=0;q<ngpu;++q)z=std::max(z,benefit[m][q]);return z-benefit[m][owner[m]];};
            uint64_t ra=regret(a),rb=regret(b);return ra!=rb?ra>rb:a<b;});if(int(cand[g].size())>pair_limit)cand[g].resize(pair_limit);}
        bool changed=false;
        for(int ga=0;ga<ngpu;++ga)for(int gb=ga+1;gb<ngpu;++gb){int64_t bg=0;uint32_t bu=0,bv=0;bool found=false;
            for(uint32_t u:cand[ga])for(uint32_t v:cand[gb]){uint64_t nla=gl[ga]-load[u]+load[v],nlb=gl[gb]-load[v]+load[u];if(nla<lo||nla>hi||nlb<lo||nlb>hi)continue;
                int64_t d=int64_t(benefit[u][gb])+int64_t(benefit[v][ga])-int64_t(benefit[u][ga])-int64_t(benefit[v][gb]);if(d>bg){bg=d;bu=u;bv=v;found=true;}}
            if(found){gl[ga]=gl[ga]-load[bu]+load[bv];gl[gb]=gl[gb]-load[bv]+load[bu];owner[bu]=uint8_t(gb);owner[bv]=uint8_t(ga);++swaps;changed=true;}}
        if(!changed)break;
    }
}

static uint64_t b300_dt_joint_retained(
    const StorageLayout&l,const std::vector<uint8_t>&ho,const std::vector<uint8_t>&lo,int ngpu
){
    constexpr int S=MAXW+2;uint64_t z=0;uint64_t wm=uint64_t(2*TARGET_W-1),wb=uint64_t(TARGET_W);
    std::array<std::array<uint32_t,MAXW+2>,MAXGPU> hc{},lc{};
    for(uint32_t hm=0;hm<ho.size();++hm){int g=ho[hm];for(int h=0;h<=MAXW;++h){size_t x=size_t(hm)*S+h;hc[g][h]+=G_FACTOR.high_mask_off[x+1]-G_FACTOR.high_mask_off[x];}}
    for(uint32_t lm=0;lm<lo.size();++lm){int g=lo[lm];for(int h=0;h<=MAXW;++h){size_t x=size_t(lm)*S+h;lc[g][h]+=G_FACTOR.low_mask_off[x+1]-G_FACTOR.low_mask_off[x];}}
    auto add=[&](const StorageBlock&b,uint64_t mult){if(!b.valid||!b.rows||!b.cols)return;for(int g=0;g<ngpu;++g)z+=mult*uint64_t(hc[g][b.he])*lc[g][b.hs]*sizeof(Count);};
    for(const auto&b:l.main_blocks)add(b,wm);for(const auto&b:l.block_blocks)add(b,wb);return z;
}

static B300DualTileHost build_b300_dual_tile_layout_joint_optimized(
    const StorageFactorHost&f,const StorageLayout&l,int ngpu,
    double slack_gib=4.0,int pair_limit=256,int outer_iters=4,
    B300DualJointOptStats*stats=nullptr
){
    constexpr int S=MAXW+2;constexpr uint32_t HN=1u<<HIGH_LUT_K,LN=1u<<LOW_LUT_K;
    B300DirectMaskShardHost h0=build_b300_direct_mask_shards(f,l,ngpu);
    std::vector<uint8_t> ho=h0.mask_owner,lo=b300_dual_low_mask_lpt(f,l,ngpu);
    std::vector<uint64_t> hload(HN),lload(LN);
    auto loadh=[&](const StorageBlock&b){if(!b.valid)return;for(uint32_t m=0;m<HN;++m){size_t x=size_t(m)*S+b.he;hload[m]+=uint64_t(G_FACTOR.high_mask_off[x+1]-G_FACTOR.high_mask_off[x])*b.cols*sizeof(Count);}};
    auto loadl=[&](const StorageBlock&b){if(!b.valid)return;for(uint32_t m=0;m<LN;++m){size_t x=size_t(m)*S+b.hs;lload[m]+=uint64_t(G_FACTOR.low_mask_off[x+1]-G_FACTOR.low_mask_off[x])*b.rows*sizeof(Count);}};
    for(const auto&b:l.main_blocks){loadh(b);loadl(b);}for(const auto&b:l.block_blocks){loadh(b);loadl(b);}
    uint64_t logical=uint64_t(2*TARGET_W-1)*uint64_t(l.main_size)*sizeof(Count)+uint64_t(TARGET_W)*uint64_t(l.block_size)*sizeof(Count);
    uint64_t initial=b300_dt_joint_retained(l,ho,lo,ngpu),moves=0,swaps=0;
    uint64_t wm=uint64_t(2*TARGET_W-1),wb=uint64_t(TARGET_W);

    int done=0;
    for(int it=0;it<outer_iters;++it){++done;uint64_t before=b300_dt_joint_retained(l,ho,lo,ngpu);
        // Optimize LOW owners against current HIGH owners.
        std::array<std::array<uint32_t,MAXW+2>,MAXGPU> hcnt{};
        for(uint32_t hm=0;hm<HN;++hm){int g=ho[hm];for(int h=0;h<=MAXW;++h){size_t x=size_t(hm)*S+h;hcnt[g][h]+=G_FACTOR.high_mask_off[x+1]-G_FACTOR.high_mask_off[x];}}
        std::vector<std::array<uint64_t,MAXGPU>> lb(LN);
        auto addlb=[&](const StorageBlock&b,uint64_t mult){if(!b.valid)return;for(uint32_t lm=0;lm<LN;++lm){size_t x=size_t(lm)*S+b.hs;uint64_t nc=G_FACTOR.low_mask_off[x+1]-G_FACTOR.low_mask_off[x];if(!nc)continue;for(int g=0;g<ngpu;++g)lb[lm][g]+=mult*nc*hcnt[g][b.he]*sizeof(Count);}};
        for(const auto&b:l.main_blocks)addlb(b,wm);for(const auto&b:l.block_blocks)addlb(b,wb);
        b300_dt_generic_assignment_refine(lo,lload,lb,ngpu,slack_gib,pair_limit,moves,swaps);

        // Optimize HIGH owners against the updated LOW owners.
        std::array<std::array<uint32_t,MAXW+2>,MAXGPU> lcnt{};
        for(uint32_t lm=0;lm<LN;++lm){int g=lo[lm];for(int h=0;h<=MAXW;++h){size_t x=size_t(lm)*S+h;lcnt[g][h]+=G_FACTOR.low_mask_off[x+1]-G_FACTOR.low_mask_off[x];}}
        std::vector<std::array<uint64_t,MAXGPU>> hb(HN);
        auto addhb=[&](const StorageBlock&b,uint64_t mult){if(!b.valid)return;for(uint32_t hm=0;hm<HN;++hm){size_t x=size_t(hm)*S+b.he;uint64_t nr=G_FACTOR.high_mask_off[x+1]-G_FACTOR.high_mask_off[x];if(!nr)continue;for(int g=0;g<ngpu;++g)hb[hm][g]+=mult*nr*lcnt[g][b.hs]*sizeof(Count);}};
        for(const auto&b:l.main_blocks)addhb(b,wm);for(const auto&b:l.block_blocks)addhb(b,wb);
        b300_dt_generic_assignment_refine(ho,hload,hb,ngpu,slack_gib,pair_limit,moves,swaps);
        uint64_t after=b300_dt_joint_retained(l,ho,lo,ngpu);if(after<=before)break;
    }

    B300DualTileHost z=build_b300_dual_tile_layout(f,l,ngpu);
    z.high=b300_build_mask_shard_from_owner(f,l,ngpu,ho);
    for(int g=0;g<ngpu;++g)for(int h=0;h<=MAXW;++h)z.high_count[g][h]=z.high.owned_off[g][h+1]-z.high.owned_off[g][h];
    b300_dt_rebuild_low_owner(z,lo,f,l);

    auto imbalance=[&](const std::vector<uint64_t>&load,const std::vector<uint8_t>&own){std::array<uint64_t,MAXGPU>a{};uint64_t t=0;for(size_t i=0;i<load.size();++i){a[own[i]]+=load[i];t+=load[i];}uint64_t mx=*std::max_element(a.begin(),a.begin()+ngpu);return t?double((long double)mx/((long double)t/ngpu)):0.0;};
    if(stats){stats->logical_bytes=logical;stats->retained_initial=initial;stats->retained_final=b300_dt_joint_retained(l,ho,lo,ngpu);stats->moves=moves;stats->swaps=swaps;stats->iterations=done;
        stats->high_load_max_over_avg=imbalance(hload,ho);stats->low_load_max_over_avg=imbalance(lload,lo);}
    return z;
}
