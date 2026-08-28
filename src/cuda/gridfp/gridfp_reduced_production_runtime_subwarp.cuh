#pragma once

#include "gridfp_reduced_production_runtime_small_step.cuh"
#include "gridfp_reduced_production_owner_component_plan_device.cuh"
#include "gridfp_reduced_production_discovery_device.cuh"
#include "gridfp_reduced_production_group_context_device.cuh"

namespace oneesan::gridfp::reducedprod {

static constexpr int RP_RUNTIME_WARPS_PER_BLOCK = 8;
static constexpr int RP_RUNTIME_SUBGROUPS_PER_WARP = 4;
static constexpr int RP_RUNTIME_SUBGROUP_WIDTH = 8;
static constexpr int RP_RUNTIME_MAX_PAIRS = 20;
static constexpr int RP_RUNTIME_THREADS = 32 * RP_RUNTIME_WARPS_PER_BLOCK;

// Production-path interior transfer.  There is deliberately no per-component
// accounting atomic: exactness probes can wrap this kernel externally, while
// the performance path performs only state traffic and local reconstruction.
__global__ void owner_component_runtime_subwarp_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 local_components,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int gpu_id,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ component_prefix,
    const Rank64* __restrict__ component_sr_begin,
    const Rank64* __restrict__ component_group,
    std::uint32_t mod,
    int* error
) {
    __shared__ DeviceKey sh_src[RP_RUNTIME_WARPS_PER_BLOCK]
                               [RP_RUNTIME_SUBGROUPS_PER_WARP]
                               [RP_RUNTIME_MAX_PAIRS];
    __shared__ DeviceKey sh_dst[RP_RUNTIME_WARPS_PER_BLOCK]
                               [RP_RUNTIME_SUBGROUPS_PER_WARP]
                               [RP_RUNTIME_MAX_PAIRS];
    __shared__ std::uint32_t sh_value[RP_RUNTIME_WARPS_PER_BLOCK]
                                     [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                     [RP_RUNTIME_MAX_PAIRS];
    __shared__ GroupedComponentContextDevice sh_ctx[RP_RUNTIME_WARPS_PER_BLOCK]
                                                   [RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_ns[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_nd[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int subgroup = lane / RP_RUNTIME_SUBGROUP_WIDTH;
    const int sublane = lane & (RP_RUNTIME_SUBGROUP_WIDTH - 1);
    const unsigned subgroup_mask = 0xffu << (subgroup * RP_RUNTIME_SUBGROUP_WIDTH);
    const Rank64 warp_global = Rank64(blockIdx.x) * RP_RUNTIME_WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 first = warp_global * RP_RUNTIME_SUBGROUPS_PER_WARP + Rank64(subgroup);
    const Rank64 stride = Rank64(gridDim.x) * RP_RUNTIME_WARPS_PER_BLOCK *
                          RP_RUNTIME_SUBGROUPS_PER_WARP;
    const int next = reverse ? q + 1 : q - 1;
    const OwnerComponentPlanDevice plan{
        component_prefix, component_sr_begin, component_group};

    for (Rank64 local_rank = first; local_rank < local_components; local_rank += stride) {
        if (sublane == 0) {
            sh_ns[warp][subgroup] = 0;
            sh_nd[warp][subgroup] = 0;
            const MateID label = owner_component_label_unrank_planned_device(
                W, q, reverse, tile_start, K, plan, local_rank);
            bool eligible = false;
            const DeviceKey seed = runtime_component_seed(label, W, q, reverse, eligible);
            if (!eligible) {
                runtime_set_error(error, 301);
            } else {
                const GroupedComponentContextDevice ctx = grouped_component_context_device(
                    seed, W, q, reverse, tile_start, K, ngpu, owner_begin);
                sh_ctx[warp][subgroup] = ctx;
                if (ctx.owner != gpu_id) {
                    runtime_set_error(error, 302);
                } else {
                    sh_src[warp][subgroup][0] = seed;
                    sh_ns[warp][subgroup] = 1;
                    int cursor = 0;
                    while (cursor < sh_ns[warp][subgroup]) {
                        RuntimeSmallTerms edge;
                        if (!runtime_small_step(
                                sh_src[warp][subgroup][cursor++], W, q, reverse, edge)) {
                            runtime_set_error(error, 303);
                            break;
                        }
                        for (int ei = 0; ei < edge.n; ++ei) {
                            if (!edge.v[ei].coef) continue;
                            const DeviceKey d = edge.v[ei].key;
                            if (runtime_find_key(
                                    sh_dst[warp][subgroup], sh_nd[warp][subgroup], d) >= 0)
                                continue;
                            if (sh_nd[warp][subgroup] >= RP_RUNTIME_MAX_PAIRS) {
                                runtime_set_error(error, 304);
                                break;
                            }
                            sh_dst[warp][subgroup][sh_nd[warp][subgroup]++] = d;
                            if (!discover_inverse_direction_to_set(
                                    d, W, q, reverse,
                                    sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                    RP_RUNTIME_MAX_PAIRS)) {
                                runtime_set_error(error, 305);
                                break;
                            }
                        }
                        if (error && *error) break;
                    }
                    if (sh_ns[warp][subgroup] != sh_nd[warp][subgroup])
                        runtime_set_error(error, 306);
                }
            }
        }
        __syncwarp(subgroup_mask);

        const int ns = sh_ns[warp][subgroup];
        const int nd = sh_nd[warp][subgroup];
        const GroupedComponentContextDevice ctx = sh_ctx[warp][subgroup];
        for (int i = sublane; i < ns; i += RP_RUNTIME_SUBGROUP_WIDTH) {
            const GroupedDeviceRank gr = grouped_rank_in_component_device(
                sh_src[warp][subgroup][i], W, q, reverse, ctx);
            if (gr.owner != gpu_id) {
                runtime_set_error(error, 307);
                sh_value[warp][subgroup][i] = 0;
            } else {
                sh_value[warp][subgroup][i] = state[gr.local];
            }
        }
        __syncwarp(subgroup_mask);

        for (int di = sublane; di < nd; di += RP_RUNTIME_SUBGROUP_WIDTH) {
            const DeviceKey mine = sh_dst[warp][subgroup][di];
            const GroupedDeviceRank dgr = grouped_rank_in_component_device(
                mine, W, next, reverse, ctx);
            if (dgr.owner != gpu_id) {
                runtime_set_error(error, 308);
                continue;
            }
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                RuntimeSmallTerms edge;
                if (!runtime_small_step(sh_src[warp][subgroup][si], W, q, reverse, edge)) {
                    runtime_set_error(error, 309);
                    continue;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    if (key_equal(edge.v[ei].key, mine))
                        acc += static_cast<long long>(edge.v[ei].coef) *
                               static_cast<long long>(sh_value[warp][subgroup][si]);
                }
            }
            long long z = acc % static_cast<long long>(mod);
            if (z < 0) z += mod;
            state[dgr.local] = static_cast<std::uint32_t>(z);
        }
        __syncwarp(subgroup_mask);
    }
}

} // namespace oneesan::gridfp::reducedprod
