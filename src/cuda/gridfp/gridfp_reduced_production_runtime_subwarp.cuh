#pragma once

#include "gridfp_reduced_production_runtime_small_step.cuh"
#include "gridfp_reduced_production_owner_component_plan_device.cuh"
#include "gridfp_reduced_production_runtime_shared_key.cuh"
#include "gridfp_reduced_production_group_context_device.cuh"

namespace oneesan::gridfp::reducedprod {

#ifndef RP_RUNTIME_CACHE_EDGES
#define RP_RUNTIME_CACHE_EDGES 1
#endif
#ifndef RP_RUNTIME_FAST_P32M5_MOD
#define RP_RUNTIME_FAST_P32M5_MOD 1
#endif
#ifndef RP_RUNTIME_POLL_GLOBAL_ERROR
#define RP_RUNTIME_POLL_GLOBAL_ERROR 0
#endif
static_assert(RP_RUNTIME_CACHE_EDGES == 0 || RP_RUNTIME_CACHE_EDGES == 1,
              "RP_RUNTIME_CACHE_EDGES must be 0 or 1");
static_assert(RP_RUNTIME_FAST_P32M5_MOD == 0 || RP_RUNTIME_FAST_P32M5_MOD == 1,
              "RP_RUNTIME_FAST_P32M5_MOD must be 0 or 1");
static_assert(RP_RUNTIME_POLL_GLOBAL_ERROR == 0 || RP_RUNTIME_POLL_GLOBAL_ERROR == 1,
              "RP_RUNTIME_POLL_GLOBAL_ERROR must be 0 or 1");

static constexpr int RP_RUNTIME_WARPS_PER_BLOCK = 8;
static constexpr int RP_RUNTIME_SUBGROUPS_PER_WARP = 4;
static constexpr int RP_RUNTIME_SUBGROUP_WIDTH = 8;
static constexpr int RP_RUNTIME_MAX_PAIRS = 20;
static constexpr int RP_RUNTIME_MAX_EDGE_TERMS = 3;
static constexpr int RP_RUNTIME_MAX_DESTINATIONS_PER_LANE =
    (RP_RUNTIME_MAX_PAIRS + RP_RUNTIME_SUBGROUP_WIDTH - 1) /
    RP_RUNTIME_SUBGROUP_WIDTH;
static constexpr int RP_RUNTIME_THREADS = 32 * RP_RUNTIME_WARPS_PER_BLOCK;
static constexpr int RP_RUNTIME_SHARED_KEY_ENTRIES_PER_BLOCK =
    2 * RP_RUNTIME_WARPS_PER_BLOCK * RP_RUNTIME_SUBGROUPS_PER_WARP *
    RP_RUNTIME_MAX_PAIRS;
static constexpr int RP_RUNTIME_SHARED_KEY_BYTES_PER_BLOCK =
    RP_RUNTIME_SHARED_KEY_ENTRIES_PER_BLOCK * int(sizeof(RuntimeSharedKey));
static_assert(RP_RUNTIME_SUBGROUP_WIDTH == 8);
static_assert(RP_RUNTIME_MAX_DESTINATIONS_PER_LANE == 3);
#if RP_RUNTIME_PACK_SHARED_KEYS
static_assert(sizeof(RuntimeSharedKey) == 8);
static_assert(RP_RUNTIME_SHARED_KEY_BYTES_PER_BLOCK == 10240);
#endif

// RuntimeSmallTerms is formed from at most three primitive contributions.
// Interior steps contain +1,+1,-1 and turn compression contains +1,+1, so after
// merging equal keys every emitted coefficient is in [-1,2]. Destination indices
// are in [0,19]. Pack both into one byte: bits 0..4 destination, bits 5..6 coef+1.
static constexpr int RP_RUNTIME_EDGE_DEST_BITS = 5;
static constexpr std::uint8_t RP_RUNTIME_EDGE_DEST_MASK =
    (std::uint8_t(1u) << RP_RUNTIME_EDGE_DEST_BITS) - 1u;
static constexpr int RP_RUNTIME_EDGE_COEF_MIN = -1;
static constexpr int RP_RUNTIME_EDGE_COEF_MAX = 2;
static constexpr int RP_RUNTIME_EDGE_COEF_BIAS = 1;
static_assert(RP_RUNTIME_MAX_PAIRS <= (1 << RP_RUNTIME_EDGE_DEST_BITS));

// For p=2^32-5, 2^32 == 5 (mod p). With at most 20 source terms and merged
// coefficient magnitude <=2, |acc| <= 40*(p-1). Its high 32-bit word is <=39,
// so lo+5*hi is <2p and one conditional subtraction completes the reduction.
static constexpr std::uint32_t RP_RUNTIME_P32M5_MODULUS = 4294967291u;
static constexpr std::uint64_t RP_RUNTIME_P32M5_MAX_ACC_MAG =
    std::uint64_t(RP_RUNTIME_MAX_PAIRS) * RP_RUNTIME_EDGE_COEF_MAX *
    (std::uint64_t(RP_RUNTIME_P32M5_MODULUS) - 1u);
static constexpr std::uint64_t RP_RUNTIME_P32M5_MAX_HI =
    RP_RUNTIME_P32M5_MAX_ACC_MAG >> 32;
static constexpr std::uint64_t RP_RUNTIME_P32M5_MAX_FOLDED =
    0xffffffffULL + 5ULL * RP_RUNTIME_P32M5_MAX_HI;
static_assert(RP_RUNTIME_P32M5_MAX_HI == 39u);
static_assert(RP_RUNTIME_P32M5_MAX_FOLDED <
              2ULL * std::uint64_t(RP_RUNTIME_P32M5_MODULUS));

__device__ __forceinline__ std::uint32_t runtime_reduce_accum(
    long long accum,
    std::uint32_t mod
) {
#if RP_RUNTIME_FAST_P32M5_MOD
    if (mod == RP_RUNTIME_P32M5_MODULUS) {
        const bool negative = accum < 0;
        const std::uint64_t magnitude = negative
            ? std::uint64_t(-(accum + 1)) + 1u
            : std::uint64_t(accum);
        std::uint64_t folded =
            std::uint64_t(std::uint32_t(magnitude)) + 5ULL * (magnitude >> 32);
        if (folded >= RP_RUNTIME_P32M5_MODULUS)
            folded -= RP_RUNTIME_P32M5_MODULUS;
        const std::uint32_t residue = std::uint32_t(folded);
        return negative && residue
            ? RP_RUNTIME_P32M5_MODULUS - residue
            : residue;
    }
#endif
    long long residue = accum % static_cast<long long>(mod);
    if (residue < 0) residue += mod;
    return static_cast<std::uint32_t>(residue);
}

// Compact source->destination topology recorded while the component is already
// being discovered. 20*(1 count byte + 3 packed edge bytes) = 80 bytes/subgroup,
// or 2560 bytes/block for 32 subgroups.
struct RuntimeEdgeCache {
    std::uint8_t count[RP_RUNTIME_MAX_PAIRS];
    std::uint8_t packed[RP_RUNTIME_MAX_PAIRS][RP_RUNTIME_MAX_EDGE_TERMS];
};
static_assert(sizeof(RuntimeEdgeCache) == 80,
              "runtime edge cache footprint regression");
static constexpr int RP_RUNTIME_EDGE_CACHE_BYTES_PER_BLOCK =
    int(sizeof(RuntimeEdgeCache)) * RP_RUNTIME_WARPS_PER_BLOCK *
    RP_RUNTIME_SUBGROUPS_PER_WARP;
static_assert(RP_RUNTIME_EDGE_CACHE_BYTES_PER_BLOCK == 2560);

__device__ __forceinline__ bool runtime_edge_cache_append(
    RuntimeEdgeCache& cache,
    int source_ix,
    int destination_ix,
    int coefficient
) {
    if (source_ix < 0 || source_ix >= RP_RUNTIME_MAX_PAIRS ||
        destination_ix < 0 || destination_ix >= RP_RUNTIME_MAX_PAIRS ||
        coefficient < RP_RUNTIME_EDGE_COEF_MIN ||
        coefficient > RP_RUNTIME_EDGE_COEF_MAX)
        return false;
    const std::uint8_t slot = cache.count[source_ix];
    if (slot >= RP_RUNTIME_MAX_EDGE_TERMS) return false;
    cache.count[source_ix] = std::uint8_t(slot + 1u);
    cache.packed[source_ix][slot] = std::uint8_t(
        std::uint8_t(destination_ix) |
        (std::uint8_t(coefficient + RP_RUNTIME_EDGE_COEF_BIAS)
         << RP_RUNTIME_EDGE_DEST_BITS));
    return true;
}

__device__ __forceinline__ std::uint8_t runtime_edge_destination(
    std::uint8_t packed
) {
    return packed & RP_RUNTIME_EDGE_DEST_MASK;
}

__device__ __forceinline__ int runtime_edge_coefficient(std::uint8_t packed) {
    return int(packed >> RP_RUNTIME_EDGE_DEST_BITS) - RP_RUNTIME_EDGE_COEF_BIAS;
}

// Production-path interior transfer. There is deliberately no per-component
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
    __shared__ RuntimeSharedKey sh_src[RP_RUNTIME_WARPS_PER_BLOCK]
                                      [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                      [RP_RUNTIME_MAX_PAIRS];
    __shared__ RuntimeSharedKey sh_dst[RP_RUNTIME_WARPS_PER_BLOCK]
                                      [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                      [RP_RUNTIME_MAX_PAIRS];
    __shared__ std::uint32_t sh_value[RP_RUNTIME_WARPS_PER_BLOCK]
                                     [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                     [RP_RUNTIME_MAX_PAIRS];
    __shared__ GroupedComponentContextDevice sh_ctx[RP_RUNTIME_WARPS_PER_BLOCK]
                                                   [RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_ns[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_nd[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];
#if RP_RUNTIME_CACHE_EDGES
    __shared__ RuntimeEdgeCache sh_edge[RP_RUNTIME_WARPS_PER_BLOCK]
                                       [RP_RUNTIME_SUBGROUPS_PER_WARP];
#endif

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
                    sh_src[warp][subgroup][0] = runtime_shared_key_encode(seed);
                    sh_ns[warp][subgroup] = 1;
                    int cursor = 0;
                    while (cursor < sh_ns[warp][subgroup]) {
                        const int source_ix = cursor++;
#if RP_RUNTIME_CACHE_EDGES
                        sh_edge[warp][subgroup].count[source_ix] = 0;
#endif
                        RuntimeSmallTerms edge;
                        if (!runtime_small_step(
                                runtime_shared_key_decode(sh_src[warp][subgroup][source_ix]),
                                W, q, reverse, edge)) {
                            runtime_set_error(error, 303);
                            break;
                        }
                        for (int ei = 0; ei < edge.n; ++ei) {
                            if (!edge.v[ei].coef) continue;
                            const DeviceKey d = edge.v[ei].key;
                            int destination_ix = runtime_find_shared_key(
                                sh_dst[warp][subgroup], sh_nd[warp][subgroup], d);
                            if (destination_ix < 0) {
                                if (sh_nd[warp][subgroup] >= RP_RUNTIME_MAX_PAIRS) {
                                    runtime_set_error(error, 304);
                                    cursor = RP_RUNTIME_MAX_PAIRS;
                                    break;
                                }
                                destination_ix = sh_nd[warp][subgroup]++;
                                sh_dst[warp][subgroup][destination_ix] =
                                    runtime_shared_key_encode(d);
                                if (!runtime_discover_inverse_direction_to_shared(
                                        d, W, q, reverse,
                                        sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                        RP_RUNTIME_MAX_PAIRS)) {
                                    runtime_set_error(error, 305);
                                    cursor = RP_RUNTIME_MAX_PAIRS;
                                    break;
                                }
                            }
#if RP_RUNTIME_CACHE_EDGES
                            if (!runtime_edge_cache_append(
                                    sh_edge[warp][subgroup], source_ix,
                                    destination_ix, int(edge.v[ei].coef))) {
                                runtime_set_error(error, 310);
                                cursor = RP_RUNTIME_MAX_PAIRS;
                                break;
                            }
#endif
                        }
#if RP_RUNTIME_POLL_GLOBAL_ERROR
                        if (error && *error) break;
#endif
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
                runtime_shared_key_decode(sh_src[warp][subgroup][i]),
                W, q, reverse, ctx);
            if (gr.owner != gpu_id) {
                runtime_set_error(error, 307);
                sh_value[warp][subgroup][i] = 0;
            } else {
                sh_value[warp][subgroup][i] = state[gr.local];
            }
        }
        __syncwarp(subgroup_mask);

#if RP_RUNTIME_CACHE_EDGES
        // All eight lanes walk the same compact edge stream once. Destination
        // index low bits select the owning lane; high bits select one of that
        // lane's at most three destination accumulators. This avoids rescanning
        // the edge stream for destination indices 8..15 and 16..19.
        long long accum[RP_RUNTIME_MAX_DESTINATIONS_PER_LANE] = {0, 0, 0};
        for (int si = 0; si < ns; ++si) {
            const long long value = static_cast<long long>(sh_value[warp][subgroup][si]);
            const std::uint8_t nedge = sh_edge[warp][subgroup].count[si];
            for (std::uint8_t ei = 0; ei < nedge; ++ei) {
                const std::uint8_t packed = sh_edge[warp][subgroup].packed[si][ei];
                const std::uint8_t destination_ix = runtime_edge_destination(packed);
                if ((destination_ix & (RP_RUNTIME_SUBGROUP_WIDTH - 1)) == sublane) {
                    const int slot = destination_ix >> 3;
                    accum[slot] +=
                        static_cast<long long>(runtime_edge_coefficient(packed)) * value;
                }
            }
        }
        int accumulator_ix = 0;
        for (int di = sublane; di < nd;
             di += RP_RUNTIME_SUBGROUP_WIDTH, ++accumulator_ix) {
            const DeviceKey mine =
                runtime_shared_key_decode(sh_dst[warp][subgroup][di]);
            const GroupedDeviceRank dgr = grouped_rank_in_component_device(
                mine, W, next, reverse, ctx);
            if (dgr.owner != gpu_id) {
                runtime_set_error(error, 308);
                continue;
            }
            state[dgr.local] = runtime_reduce_accum(accum[accumulator_ix], mod);
        }
#else
        for (int di = sublane; di < nd; di += RP_RUNTIME_SUBGROUP_WIDTH) {
            const DeviceKey mine =
                runtime_shared_key_decode(sh_dst[warp][subgroup][di]);
            const GroupedDeviceRank dgr = grouped_rank_in_component_device(
                mine, W, next, reverse, ctx);
            if (dgr.owner != gpu_id) {
                runtime_set_error(error, 308);
                continue;
            }
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                RuntimeSmallTerms edge;
                if (!runtime_small_step(
                        runtime_shared_key_decode(sh_src[warp][subgroup][si]),
                        W, q, reverse, edge)) {
                    runtime_set_error(error, 309);
                    continue;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    if (key_equal(edge.v[ei].key, mine))
                        acc += static_cast<long long>(edge.v[ei].coef) *
                               static_cast<long long>(sh_value[warp][subgroup][si]);
                }
            }
            state[dgr.local] = runtime_reduce_accum(acc, mod);
        }
#endif
        __syncwarp(subgroup_mask);
    }
}

} // namespace oneesan::gridfp::reducedprod