#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_graphpart.cuh"

struct GdpgSelection {
    GdmShardHost shard;
    GdmsStagePlan gather;
    GdowOrbitPlan orbit;
    bool graph_selected = false;
    unsigned long long legacy_gather_bytes = 0;
    unsigned long long legacy_orbit_bytes = 0;
    unsigned long long graph_gather_bytes = 0;
    unsigned long long graph_orbit_bytes = 0;
    unsigned long long graph_cut_bytes = 0;
    double graph_max_to_avg = 0.0;
};

static GdpgSelection gdpg_select_exact(
    const StorageLayout& layout,const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect,const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,int ngpu
) {
    GdpgSelection out;
    GdmShardHost legacy=build_gdm_shards(layout,ngpu);
    GdmsStagePlan lg=build_gdms_stage_plan(layout,legacy,ordinary,cross,ngpu);
    GdowOrbitPlan lo=build_gdow_orbit_plan(layout,legacy,loworbit,highdirect,ngpu);
    out.legacy_gather_bytes=lg.copy_bytes_per_row;
    out.legacy_orbit_bytes=lo.copy_bytes_per_row;

    GdpgCandidate candidate=build_gdpg_candidate(layout,loworbit,highdirect,ordinary,cross,ngpu);
    GdmsStagePlan gg=build_gdms_stage_plan(layout,candidate.shard,ordinary,cross,ngpu);
    GdowOrbitPlan go=build_gdow_orbit_plan(layout,candidate.shard,loworbit,highdirect,ngpu);
    out.graph_gather_bytes=gg.copy_bytes_per_row;
    out.graph_orbit_bytes=go.copy_bytes_per_row;
    out.graph_cut_bytes=candidate.graph_cut;
    out.graph_max_to_avg=candidate.max_to_avg;

    unsigned long long legacy_bytes=out.legacy_gather_bytes+out.legacy_orbit_bytes;
    unsigned long long graph_bytes=out.graph_gather_bytes+out.graph_orbit_bytes;
    // Keep a hard runtime-memory guard in addition to the graph builder's
    // balance constraints. Exact communication must improve, not merely the
    // approximate graph cut.
    out.graph_selected=(graph_bytes<legacy_bytes && candidate.max_to_avg<=1.08);
    if (out.graph_selected) {
        out.shard=std::move(candidate.shard);
        out.gather=std::move(gg);
        out.orbit=std::move(go);
    } else {
        out.shard=std::move(legacy);
        out.gather=std::move(lg);
        out.orbit=std::move(lo);
    }
    return out;
}
