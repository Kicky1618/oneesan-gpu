#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_graphselect.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

// v0.8: phase/peer-aware refinement.
// v0.7 minimizes total bulk-P2P bytes.  This layer also tracks the busiest
// peer pair and busiest GPU endpoint in each actual staging phase.  A candidate
// may replace v0.7 only when exact planners prove a Pareto improvement: total,
// max pair, max endpoint and max phase traffic never get worse.

struct GdptTraffic {
    unsigned long long total_bytes = 0;
    unsigned long long max_pair_phase_bytes = 0;
    unsigned long long max_endpoint_phase_bytes = 0;
    unsigned long long max_phase_bytes = 0;
    unsigned phases = 0;
};

static GdptTraffic gdpt_measure(
    const GdmsStagePlan& gather,const GdowOrbitPlan& orbit,
    const StorageLayout& layout,const GdmShardHost& shard,int ngpu
) {
    GdptTraffic out;
    using Pair = std::array<std::array<unsigned long long,GDM_MAX_GPU>,GDM_MAX_GPU>;

    auto commit=[&](const Pair& pair) {
        unsigned long long phase=0;
        std::array<unsigned long long,GDM_MAX_GPU> in{},outgoing{};
        bool any=false;
        for(int d=0;d<ngpu;++d) for(int s=0;s<ngpu;++s) if(d!=s && pair[d][s]) {
            unsigned long long z=pair[d][s]; any=true; phase+=z;
            in[d]+=z; outgoing[s]+=z;
            out.max_pair_phase_bytes=std::max(out.max_pair_phase_bytes,z);
        }
        if(!any) return;
        for(int d=0;d<ngpu;++d)
            out.max_endpoint_phase_bytes=std::max(out.max_endpoint_phase_bytes,in[d]+outgoing[d]);
        out.total_bytes+=phase;
        out.max_phase_bytes=std::max(out.max_phase_bytes,phase);
        ++out.phases;
    };

    auto main_stage=[&](const std::array<std::vector<std::uint8_t>,GDM_MAX_GPU>& srcs) {
        Pair pair{};
        for(int d=0;d<ngpu;++d) for(std::uint8_t id:srcs[d]) {
            std::uint32_t b=id; int s=shard.main_blocks[b].owner; if(s==d) continue;
            pair[d][s]+=gdpg_bytes(layout.main_blocks[b]);
        }
        commit(pair);
    };
    auto orbit_stage=[&](const GdpoOrbitPhaseDeps& ph) {
        Pair pair{};
        for(int d=0;d<ngpu;++d) {
            for(std::uint8_t id:ph.main_source[d]) {
                std::uint32_t b=id; int s=shard.main_blocks[b].owner; if(s==d) continue;
                pair[d][s]+=gdpg_bytes(layout.main_blocks[b]);
            }
            for(std::uint8_t id:ph.block_source[d]) {
                std::uint32_t b=id; int s=shard.block_blocks[b].owner; if(s==d) continue;
                pair[d][s]+=gdpg_bytes(layout.block_blocks[b]);
            }
        }
        commit(pair);
    };

    for(std::size_t i=0;i<orbit.high.size();++i) {
        orbit_stage(orbit.high[i].deps);
        main_stage(gather.high[i].source);
        main_stage(gather.high[i].cross_refresh);
    }
    for(std::size_t i=0;i<orbit.low.size();++i) {
        orbit_stage(orbit.low[i].deps);
        main_stage(gather.low[i].source);
        main_stage(gather.low[i].cross_refresh);
    }
    return out;
}

struct GdptApprox {
    unsigned long long total = 0;
    unsigned long long max_pair = 0;
    unsigned long long max_endpoint = 0;
};

static GdptApprox gdpt_approx_score(
    const GdpgGraph& g,const std::vector<std::uint8_t>& owner,int ngpu
) {
    std::array<std::array<unsigned long long,GDM_MAX_GPU>,GDM_MAX_GPU> pair{};
    for(std::uint32_t a=0;a<g.n;++a) for(std::uint32_t b=a+1;b<g.n;++b) {
        int x=owner[a],y=owner[b]; if(x==y) continue;
        unsigned long long z=g.e(a,b); if(!z) continue;
        if(x>y) std::swap(x,y);
        pair[x][y]+=z;
    }
    GdptApprox out;
    std::array<unsigned long long,GDM_MAX_GPU> endpoint{};
    for(int a=0;a<ngpu;++a) for(int b=a+1;b<ngpu;++b) if(pair[a][b]) {
        out.total+=pair[a][b];
        out.max_pair=std::max(out.max_pair,pair[a][b]);
        endpoint[a]+=pair[a][b]; endpoint[b]+=pair[a][b];
    }
    for(int d=0;d<ngpu;++d) out.max_endpoint=std::max(out.max_endpoint,endpoint[d]);
    return out;
}

static bool gdpt_approx_better(const GdptApprox&a,const GdptApprox&b) {
    if(a.max_pair!=b.max_pair) return a.max_pair<b.max_pair;
    if(a.max_endpoint!=b.max_endpoint) return a.max_endpoint<b.max_endpoint;
    return a.total<b.total;
}

static std::vector<std::uint8_t> gdpt_refine(
    const GdpgGraph& g,std::vector<std::uint8_t> owner,int ngpu
) {
    unsigned long long total_b=std::accumulate(g.bytes.begin(),g.bytes.end(),0ull);
    unsigned long long total_w=std::accumulate(g.work.begin(),g.work.end(),0ull);
    unsigned long long max_b=*std::max_element(g.bytes.begin(),g.bytes.end());
    unsigned long long max_w=*std::max_element(g.work.begin(),g.work.end());
    unsigned long long cap_b=std::max(max_b,((total_b+ngpu-1)/ngpu)*108/100);
    unsigned long long cap_w=std::max(max_w,((total_w+ngpu-1)/ngpu)*115/100);
    std::array<unsigned long long,GDM_MAX_GPU> lb{},lw{};
    for(std::uint32_t i=0;i<g.n;++i){lb[owner[i]]+=g.bytes[i];lw[owner[i]]+=g.work[i];}

    GdptApprox start=gdpt_approx_score(g,owner,ngpu),cur=start;
    unsigned long long cut_cap=start.total+(start.total/50); // at most +2% approximate cut
    std::vector<std::uint32_t> order(g.n); std::iota(order.begin(),order.end(),0u);
    std::vector<unsigned long long> degree(g.n,0);
    for(std::uint32_t i=0;i<g.n;++i) for(std::uint32_t j=0;j<g.n;++j) degree[i]+=g.e(i,j);
    std::sort(order.begin(),order.end(),[&](auto a,auto b){return degree[a]>degree[b];});

    for(int sweep=0;sweep<10;++sweep) {
        bool changed=false;
        for(std::uint32_t i:order) {
            if(!g.bytes[i] && !g.work[i]) continue;
            int src=owner[i],best=src; GdptApprox best_score=cur;
            for(int d=0;d<ngpu;++d) if(d!=src) {
                if(lb[d]+g.bytes[i]>cap_b || lw[d]+g.work[i]>cap_w) continue;
                owner[i]=std::uint8_t(d);
                GdptApprox s=gdpt_approx_score(g,owner,ngpu);
                owner[i]=std::uint8_t(src);
                if(s.total>cut_cap) continue;
                if(gdpt_approx_better(s,best_score)) { best=d; best_score=s; }
            }
            if(best!=src) {
                lb[src]-=g.bytes[i]; lw[src]-=g.work[i];
                lb[best]+=g.bytes[i]; lw[best]+=g.work[i];
                owner[i]=std::uint8_t(best); cur=best_score; changed=true;
            }
        }
        if(!changed) break;
    }
    return owner;
}

struct GdptCandidate {
    GdmShardHost shard;
    GdptApprox approx;
    double max_to_avg = 0.0;
};

static GdptCandidate build_gdpt_candidate(
    const StorageLayout& layout,const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect,const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,int ngpu
) {
    GdpgGraph g=build_gdpg_graph(layout,loworbit,highdirect,ordinary,cross);
    GdpgCandidate base=build_gdpg_candidate(layout,loworbit,highdirect,ordinary,cross,ngpu);
    auto owner=gdpt_refine(g,gdpg_owners(base.shard),ngpu);
    GdptCandidate out;
    out.approx=gdpt_approx_score(g,owner,ngpu);
    out.shard=gdpg_pack(layout,owner,ngpu);
    unsigned long long total=0;for(int d=0;d<ngpu;++d)total+=out.shard.total_elems[d];
    double avg=ngpu?double(total)/ngpu:0.0;
    out.max_to_avg=avg?double(out.shard.max_elems)/avg:0.0;
    return out;
}

struct GdptSelection {
    GdmShardHost shard;
    GdmsStagePlan gather;
    GdowOrbitPlan orbit;
    GdptTraffic base_traffic;
    GdptTraffic phase_traffic;
    bool phase_selected = false;
    bool base_graph_selected = false;
    double phase_max_to_avg = 0.0;
};

static GdptSelection gdpt_select_exact(
    const StorageLayout& layout,const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect,const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,int ngpu
) {
    GdptSelection out;
    GdpgSelection base=gdpg_select_exact(layout,loworbit,highdirect,ordinary,cross,ngpu);
    out.base_graph_selected=base.graph_selected;
    out.base_traffic=gdpt_measure(base.gather,base.orbit,layout,base.shard,ngpu);

    GdptCandidate cand=build_gdpt_candidate(layout,loworbit,highdirect,ordinary,cross,ngpu);
    GdmsStagePlan cg=build_gdms_stage_plan(layout,cand.shard,ordinary,cross,ngpu);
    GdowOrbitPlan co=build_gdow_orbit_plan(layout,cand.shard,loworbit,highdirect,ngpu);
    out.phase_traffic=gdpt_measure(cg,co,layout,cand.shard,ngpu);
    out.phase_max_to_avg=cand.max_to_avg;

    const auto&a=out.base_traffic; const auto&b=out.phase_traffic;
    bool nonworse=b.total_bytes<=a.total_bytes
        && b.max_pair_phase_bytes<=a.max_pair_phase_bytes
        && b.max_endpoint_phase_bytes<=a.max_endpoint_phase_bytes
        && b.max_phase_bytes<=a.max_phase_bytes;
    bool strict=b.total_bytes<a.total_bytes
        || b.max_pair_phase_bytes<a.max_pair_phase_bytes
        || b.max_endpoint_phase_bytes<a.max_endpoint_phase_bytes
        || b.max_phase_bytes<a.max_phase_bytes;
    out.phase_selected=nonworse && strict && cand.max_to_avg<=1.08;
    if(out.phase_selected) {
        out.shard=std::move(cand.shard); out.gather=std::move(cg); out.orbit=std::move(co);
    } else {
        out.shard=std::move(base.shard); out.gather=std::move(base.gather); out.orbit=std::move(base.orbit);
        out.phase_traffic=out.base_traffic;
    }
    return out;
}
