#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_phasepart.cuh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <vector>

// v0.9: map logical shard groups onto physical GPU IDs using a measured P2P
// bandwidth matrix. ONEESAN_P2P_GBPS is row-major [destination][source], with
// ngpu*ngpu numeric values in GB/s separated by commas/semicolons/whitespace.
// Diagonal entries are ignored. Without the variable, all links are uniform and
// identity mapping is retained. Group membership never changes here, so total
// bytes, HBM balance and exact state ownership are invariant under the mapping.

struct GdtpTopology {
    std::array<std::array<double,GDM_MAX_GPU>,GDM_MAX_GPU> gbps{};
    bool custom = false;
};

static GdtpTopology gdtp_topology_from_env(int ngpu) {
    GdtpTopology t;
    for(int d=0;d<ngpu;++d) for(int s=0;s<ngpu;++s) t.gbps[d][s]=d==s?1e300:1.0;
    const char* env=std::getenv("ONEESAN_P2P_GBPS");
    if(!env || !*env) return t;
    const char* p=env;
    for(int d=0;d<ngpu;++d) for(int s=0;s<ngpu;++s) {
        while(*p && (std::isspace(static_cast<unsigned char>(*p)) || *p==',' || *p==';' || *p==':')) ++p;
        char* end=nullptr; double v=std::strtod(p,&end);
        if(end==p || !std::isfinite(v)) { std::cerr<<"invalid ONEESAN_P2P_GBPS at ["<<d<<"]["<<s<<"]\n"; std::exit(190); }
        p=end;
        if(d!=s && v<=0.0) { std::cerr<<"nonpositive ONEESAN_P2P_GBPS at ["<<d<<"]["<<s<<"]\n"; std::exit(191); }
        if(d!=s) t.gbps[d][s]=v;
    }
    while(*p && (std::isspace(static_cast<unsigned char>(*p)) || *p==',' || *p==';' || *p==':')) ++p;
    if(*p) { std::cerr<<"extra values in ONEESAN_P2P_GBPS\n"; std::exit(192); }
    t.custom=true;
    return t;
}

using GdtpPair = std::array<std::array<unsigned long long,GDM_MAX_GPU>,GDM_MAX_GPU>;

static std::vector<GdtpPair> gdtp_collect_phases(
    const GdmsStagePlan& gather,const GdowOrbitPlan& orbit,
    const StorageLayout& layout,const GdmShardHost& shard,int ngpu
) {
    std::vector<GdtpPair> out;
    auto emit=[&](const GdtpPair& p){bool any=false;for(int d=0;d<ngpu;++d)for(int s=0;s<ngpu;++s)if(d!=s&&p[d][s])any=true;if(any)out.push_back(p);};
    auto main_stage=[&](const std::array<std::vector<std::uint8_t>,GDM_MAX_GPU>& srcs){GdtpPair p{};for(int d=0;d<ngpu;++d)for(std::uint8_t id:srcs[d]){std::uint32_t b=id;int s=shard.main_blocks[b].owner;if(s!=d)p[d][s]+=gdpg_bytes(layout.main_blocks[b]);}emit(p);};
    auto orbit_stage=[&](const GdpoOrbitPhaseDeps& ph){GdtpPair p{};for(int d=0;d<ngpu;++d){for(std::uint8_t id:ph.main_source[d]){std::uint32_t b=id;int s=shard.main_blocks[b].owner;if(s!=d)p[d][s]+=gdpg_bytes(layout.main_blocks[b]);}for(std::uint8_t id:ph.block_source[d]){std::uint32_t b=id;int s=shard.block_blocks[b].owner;if(s!=d)p[d][s]+=gdpg_bytes(layout.block_blocks[b]);}}emit(p);};
    for(std::size_t i=0;i<orbit.high.size();++i){orbit_stage(orbit.high[i].deps);main_stage(gather.high[i].source);main_stage(gather.high[i].cross_refresh);}
    for(std::size_t i=0;i<orbit.low.size();++i){orbit_stage(orbit.low[i].deps);main_stage(gather.low[i].source);main_stage(gather.low[i].cross_refresh);}
    return out;
}

struct GdtpScore {
    double max_link_ms = 0.0;
    double max_endpoint_ms = 0.0;
    double max_phase_linksum_ms = 0.0;
};

static GdtpScore gdtp_score(
    const std::vector<GdtpPair>& phases,const GdtpTopology& topo,
    const std::array<std::uint8_t,GDM_MAX_GPU>& map,int ngpu
) {
    GdtpScore out;
    for(const auto& ph:phases) {
        std::array<double,GDM_MAX_GPU> endpoint{}; double phase_sum=0.0;
        for(int ld=0;ld<ngpu;++ld) for(int ls=0;ls<ngpu;++ls) if(ld!=ls && ph[ld][ls]) {
            int d=map[ld],s=map[ls];
            double ms=double(ph[ld][ls])/(topo.gbps[d][s]*1.0e6);
            out.max_link_ms=std::max(out.max_link_ms,ms);
            endpoint[d]+=ms; endpoint[s]+=ms; phase_sum+=ms;
        }
        for(int d=0;d<ngpu;++d) out.max_endpoint_ms=std::max(out.max_endpoint_ms,endpoint[d]);
        out.max_phase_linksum_ms=std::max(out.max_phase_linksum_ms,phase_sum);
    }
    return out;
}

static bool gdtp_better(const GdtpScore&a,const GdtpScore&b) {
    constexpr double eps=1e-12;
    if(a.max_link_ms+eps<b.max_link_ms) return true;
    if(b.max_link_ms+eps<a.max_link_ms) return false;
    if(a.max_endpoint_ms+eps<b.max_endpoint_ms) return true;
    if(b.max_endpoint_ms+eps<a.max_endpoint_ms) return false;
    return a.max_phase_linksum_ms+eps<b.max_phase_linksum_ms;
}

struct GdtpMapping {
    std::array<std::uint8_t,GDM_MAX_GPU> logical_to_physical{};
    GdtpScore identity_score;
    GdtpScore selected_score;
    bool remapped = false;
};

static GdtpMapping gdtp_best_mapping(
    const std::vector<GdtpPair>& phases,const GdtpTopology& topo,int ngpu
) {
    GdtpMapping out; std::array<std::uint8_t,GDM_MAX_GPU> cur{};
    for(int d=0;d<ngpu;++d)cur[d]=std::uint8_t(d);
    out.logical_to_physical=cur; out.identity_score=gdtp_score(phases,topo,cur,ngpu); out.selected_score=out.identity_score;
    if(!topo.custom || ngpu<=1) return out;
    do { GdtpScore s=gdtp_score(phases,topo,cur,ngpu); if(gdtp_better(s,out.selected_score)){out.selected_score=s;out.logical_to_physical=cur;out.remapped=true;} } while(std::next_permutation(cur.begin(),cur.begin()+ngpu));
    return out;
}

static GdmShardHost gdtp_remap_shard(
    const StorageLayout& layout,const GdmShardHost& shard,
    const std::array<std::uint8_t,GDM_MAX_GPU>& map,int ngpu
) {
    auto owner=gdpg_owners(shard);for(auto&x:owner){if(x>=ngpu)std::exit(193);x=map[x];}return gdpg_pack(layout,owner,ngpu);
}

struct GdtpSelection {
    GdmShardHost shard;
    GdmsStagePlan gather;
    GdowOrbitPlan orbit;
    GdtpTopology topology;
    GdtpMapping mapping;
    GdptTraffic byte_traffic;
    bool phase_selected = false;
    bool graph_selected = false;
};

static GdtpSelection gdtp_select_exact(
    const StorageLayout& layout,const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect,const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,int ngpu
) {
    GdtpSelection out;
    GdptSelection base=gdpt_select_exact(layout,loworbit,highdirect,ordinary,cross,ngpu);
    out.phase_selected=base.phase_selected; out.graph_selected=base.base_graph_selected;
    out.topology=gdtp_topology_from_env(ngpu);
    auto phases=gdtp_collect_phases(base.gather,base.orbit,layout,base.shard,ngpu);
    out.mapping=gdtp_best_mapping(phases,out.topology,ngpu);
    if(out.mapping.remapped) {
        out.shard=gdtp_remap_shard(layout,base.shard,out.mapping.logical_to_physical,ngpu);
        out.gather=build_gdms_stage_plan(layout,out.shard,ordinary,cross,ngpu);
        out.orbit=build_gdow_orbit_plan(layout,out.shard,loworbit,highdirect,ngpu);
    } else {
        out.shard=std::move(base.shard); out.gather=std::move(base.gather); out.orbit=std::move(base.orbit);
    }
    out.byte_traffic=gdpt_measure(out.gather,out.orbit,layout,out.shard,ngpu);
    return out;
}
