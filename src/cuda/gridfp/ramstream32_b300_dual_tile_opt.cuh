#pragma once

#include "ramstream32_b300_dual_tile_layout.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

struct B300DualLowOptStats {
    uint64_t logical_bytes = 0;
    uint64_t retained_before = 0;
    uint64_t retained_after = 0;
    uint64_t moves = 0;
    uint64_t swaps = 0;
    double load_max_over_avg = 0;
};

static std::vector<uint8_t> b300_dt_optimize_low_mask_owner(
    const StorageFactorHost& f,const StorageLayout& l,
    const B300DirectMaskShardHost& high,int ngpu,
    double slack_gib=4.0,int pair_limit=512,
    B300DualLowOptStats* stats=nullptr
){
    constexpr int S=MAXW+2;
    constexpr uint32_t NM=1u<<LOW_LUT_K;
    std::vector<uint64_t> load(NM,0);
    std::vector<std::array<uint64_t,MAXGPU>> benefit(NM);

    // Assignment-independent low-orientation load per LOW occupancy mask.
    auto add_load=[&](const StorageBlock&b){if(!b.valid||!b.rows||!b.cols)return;
        for(uint32_t m=0;m<NM;++m){size_t ix=size_t(m)*S+b.hs;
            uint64_t nc=G_FACTOR.low_mask_off[ix+1]-G_FACTOR.low_mask_off[ix];
            load[m]+=nc*uint64_t(b.rows)*sizeof(Count);}};
    for(const auto&b:l.main_blocks)add_load(b);for(const auto&b:l.block_blocks)add_load(b);

    // Bytes that stay on the same GPU if LOW mask m is assigned to g.  Main
    // survives both switches except the final H->L, while blocked moves only on
    // LOW->HIGH.  Weight by the exact number of switches in one residue.
    const uint64_t wm=uint64_t(2*TARGET_W-1),wb=uint64_t(TARGET_W);
    auto add_benefit=[&](const StorageBlock&b,uint64_t mult){if(!b.valid||!b.rows||!b.cols)return;
        std::array<uint32_t,MAXGPU> nr{};uint32_t ho=f.high_all_off[b.he];
        for(uint32_t hr=0;hr<b.rows;++hr)++nr[high.high_owner[ho+hr]];
        for(uint32_t m=0;m<NM;++m){size_t ix=size_t(m)*S+b.hs;
            uint64_t nc=G_FACTOR.low_mask_off[ix+1]-G_FACTOR.low_mask_off[ix];
            if(!nc)continue;for(int g=0;g<ngpu;++g)
                benefit[m][g]+=mult*nc*uint64_t(nr[g])*sizeof(Count);}};
    for(const auto&b:l.main_blocks)add_benefit(b,wm);
    for(const auto&b:l.block_blocks)add_benefit(b,wb);

    std::vector<uint8_t> owner=b300_dual_low_mask_lpt(f,l,ngpu);
    std::array<uint64_t,MAXGPU> gl{};uint64_t total_load=0;
    for(uint32_t m=0;m<NM;++m){gl[owner[m]]+=load[m];total_load+=load[m];}
    long double avg=(long double)total_load/ngpu;
    uint64_t slack=uint64_t(slack_gib*(1ull<<30));
    uint64_t lo=avg>slack?uint64_t(avg-slack):0,hi=uint64_t(avg+slack);
    auto retained=[&](){uint64_t z=0;for(uint32_t m=0;m<NM;++m)z+=benefit[m][owner[m]];return z;};
    uint64_t before=retained();

    // Single-mask improvements where the memory bounds have room.
    uint64_t moves=0;
    for(int pass=0;pass<16;++pass){bool changed=false;
        for(uint32_t m=0;m<NM;++m){int a=owner[m],best=a;int64_t gain=0;
            for(int b=0;b<ngpu;++b)if(b!=a&&gl[b]+load[m]<=hi&&gl[a]-load[m]>=lo){
                int64_t d=int64_t(benefit[m][b])-int64_t(benefit[m][a]);if(d>gain){gain=d;best=b;}}
            if(best!=a){gl[a]-=load[m];gl[best]+=load[m];owner[m]=uint8_t(best);++moves;changed=true;}}
        if(!changed)break;
    }

    // Capacity-preserving two-mask refinement.  Candidate priority is regret:
    // how much a mask could gain by living on its best other GPU.
    uint64_t swaps=0;
    pair_limit=std::max(0,pair_limit);
    for(int pass=0;pass<12&&pair_limit;++pass){
        std::array<std::vector<uint32_t>,MAXGPU> cand;
        for(uint32_t m=0;m<NM;++m)cand[owner[m]].push_back(m);
        for(int g=0;g<ngpu;++g){
            std::sort(cand[g].begin(),cand[g].end(),[&](uint32_t a,uint32_t b){
                auto regret=[&](uint32_t m){uint64_t best=benefit[m][owner[m]];for(int q=0;q<ngpu;++q)best=std::max(best,benefit[m][q]);return best-benefit[m][owner[m]];};
                uint64_t ra=regret(a),rb=regret(b);return ra!=rb?ra>rb:a<b;});
            if(int(cand[g].size())>pair_limit)cand[g].resize(pair_limit);
        }
        bool changed=false;
        for(int ga=0;ga<ngpu;++ga)for(int gb=ga+1;gb<ngpu;++gb){
            int64_t best_gain=0;uint32_t bu=0,bv=0;bool found=false;
            for(uint32_t u:cand[ga])for(uint32_t v:cand[gb]){
                uint64_t nla=gl[ga]-load[u]+load[v],nlb=gl[gb]-load[v]+load[u];
                if(nla<lo||nla>hi||nlb<lo||nlb>hi)continue;
                int64_t d=int64_t(benefit[u][gb])+int64_t(benefit[v][ga])
                         -int64_t(benefit[u][ga])-int64_t(benefit[v][gb]);
                if(d>best_gain){best_gain=d;bu=u;bv=v;found=true;}
            }
            if(found){gl[ga]=gl[ga]-load[bu]+load[bv];gl[gb]=gl[gb]-load[bv]+load[bu];
                owner[bu]=uint8_t(gb);owner[bv]=uint8_t(ga);++swaps;changed=true;}
        }
        if(!changed)break;
    }

    uint64_t after=retained();
    uint64_t total_main=uint64_t(l.main_size)*sizeof(Count),total_block=uint64_t(l.block_size)*sizeof(Count);
    uint64_t logical=wm*total_main+wb*total_block;
    uint64_t mx=*std::max_element(gl.begin(),gl.begin()+ngpu);
    if(stats){stats->logical_bytes=logical;stats->retained_before=before;stats->retained_after=after;
        stats->moves=moves;stats->swaps=swaps;stats->load_max_over_avg=avg?double((long double)mx/avg):0.0;}
    return owner;
}

static void b300_dt_rebuild_low_owner(
    B300DualTileHost&z,const std::vector<uint8_t>&owner,
    const StorageFactorHost&f,const StorageLayout&l
){
    if(owner.size()!=(size_t(1)<<LOW_LUT_K))std::exit(550);
    z.low_mask_owner=owner;
    z.low_owner.assign(f.low_all_codes.size(),0);z.low_local.assign(f.low_all_codes.size(),0);
    for(int g=0;g<z.ngpu;++g){z.owned_low_cols[g].clear();z.owned_low_off[g].fill(0);z.low_count[g].fill(0);}
    for(int h=0;h<=MAXW;++h){std::array<uint32_t,MAXGPU> local{};uint32_t n=f.low_all_off[h+1]-f.low_all_off[h];
        for(uint32_t lr=0;lr<n;++lr){uint32_t ai=f.low_all_off[h]+lr;uint32_t m=seg_occ(f.low_all_codes[ai],LOW_LUT_K);int g=owner[m];
            z.low_owner[ai]=uint8_t(g);z.low_local[ai]=local[g]++;}
        for(int g=0;g<z.ngpu;++g)z.low_count[g][h]=local[g];}
    for(int g=0;g<z.ngpu;++g){for(int h=0;h<=MAXW;++h){z.owned_low_off[g][h]=uint32_t(z.owned_low_cols[g].size());uint32_t n=f.low_all_off[h+1]-f.low_all_off[h];
        for(uint32_t lr=0;lr<n;++lr)if(z.low_owner[f.low_all_off[h]+lr]==g)z.owned_low_cols[g].push_back(lr);}z.owned_low_off[g][MAXW+1]=uint32_t(z.owned_low_cols[g].size());}

    int mn=int(l.main_blocks.size()),bn=int(l.block_blocks.size());
    z.pair_main_off.assign(size_t(z.ngpu)*z.ngpu*mn,0);z.pair_block_off.assign(size_t(z.ngpu)*z.ngpu*bn,0);
    z.pair_main_size={};z.pair_block_size={};z.main_slot_base={};z.block_slot_base={};z.main_slot_cap={};z.block_slot_cap={};z.main_count={};z.block_count={};
    for(int hi=0;hi<z.ngpu;++hi)for(int lo_=0;lo_<z.ngpu;++lo_){Code off=0;
        for(int bid=0;bid<mn;++bid){size_t ix=(size_t(hi)*z.ngpu+lo_)*mn+bid;z.pair_main_off[ix]=off;const auto&b=l.main_blocks[bid];off+=Code(z.high_count[hi][b.he])*z.low_count[lo_][b.hs];}z.pair_main_size[hi][lo_]=off;off=0;
        for(int bid=0;bid<bn;++bid){size_t ix=(size_t(hi)*z.ngpu+lo_)*bn+bid;z.pair_block_off[ix]=off;const auto&b=l.block_blocks[bid];off+=Code(z.high_count[hi][b.he])*z.low_count[lo_][b.hs];}z.pair_block_size[hi][lo_]=off;}
    for(int g=0;g<z.ngpu;++g){Code mo=0,bo=0;for(int p=0;p<z.ngpu;++p){z.main_slot_base[g][p]=mo;z.block_slot_base[g][p]=bo;
        z.main_slot_cap[g][p]=std::max(z.pair_main_size[g][p],z.pair_main_size[p][g]);z.block_slot_cap[g][p]=std::max(z.pair_block_size[g][p],z.pair_block_size[p][g]);
        mo+=z.main_slot_cap[g][p];bo+=z.block_slot_cap[g][p];}z.main_count[g]=mo;z.block_count[g]=bo;}
}
