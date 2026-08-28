#pragma once

#include "ramstream32_bucket_low_rankformula_nometa4.cuh"

#ifndef P10DC_RANKFORMULA_NOMETA_WARPSHARE
#define P10DC_RANKFORMULA_NOMETA_WARPSHARE 0
#endif
#ifndef P10DC_RANKFORMULA_NOMETA_COOPGROUP
#define P10DC_RANKFORMULA_NOMETA_COOPGROUP 0
#endif
static_assert(P10DC_RANKFORMULA_NOMETA_WARPSHARE == 0 ||
              P10DC_RANKFORMULA_NOMETA_WARPSHARE == 1,
              "P10DC_RANKFORMULA_NOMETA_WARPSHARE must be 0 or 1");
static_assert(P10DC_RANKFORMULA_NOMETA_COOPGROUP == 0 ||
              P10DC_RANKFORMULA_NOMETA_COOPGROUP == 1,
              "P10DC_RANKFORMULA_NOMETA_COOPGROUP must be 0 or 1");
static_assert(!P10DC_RANKFORMULA_NOMETA_COOPGROUP ||
              P10DC_RANKFORMULA_NOMETA_WARPSHARE,
              "cooperative nometa successor loading requires warp sharing");
static_assert(32u % P10DC_RANKFORMULA_NOMETA4_BLOCK == 0u,
              "warp-shared nometa locator requires block size dividing 32");

// Warp-striped HIGH kernels enumerate lr as base32+lane, and every later stripe
// advances by gridDim.x*32. Therefore rank/B is shared by exactly B adjacent
// lanes for B in {4,8,16}. The subgroup leader loads block16 and the initial
// group64, but only the 64-bit group entry is broadcast. group64 embeds its own
// 14-bit global index, so lanes crossing a group boundary can load the successor
// without a third shuffle for gi.
__device__ __forceinline__ P10DCRankFormulaNometa4Resolved
p10dc_low_rankformula_nometa_resolve_warpshare(uint32_t h, uint32_t rank) {
    constexpr uint32_t B = P10DC_RANKFORMULA_NOMETA4_BLOCK;
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t src_lane = lane & ~(B - 1u);
    const unsigned active = __activemask();

    uint32_t elo = 0u, ehi = 0u;
    if (lane == src_lane) {
        const uint32_t bi = D_P10DC_LOW_RANKFORMULA_NOMETA4_BOFF[h] + rank / B;
        const uint32_t gi = uint32_t(D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16[bi]);
        const uint64_t e0 = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[gi];
        elo = uint32_t(e0);
        ehi = uint32_t(e0 >> 32);
    }
    elo = __shfl_sync(active, elo, int(src_lane));
    ehi = __shfl_sync(active, ehi, int(src_lane));
    uint64_t e = uint64_t(elo) | (uint64_t(ehi) << 32);

#pragma unroll
    for (int k = 0; k < int(B - 1u); ++k) {
        const uint32_t start = p10dc_rankformula_nometa4_group_start(e);
        const uint32_t count = p10dc_rankformula_nometa4_group_count(e);
        if (rank < start + count) break;
        const uint32_t next = p10dc_rankformula_nometa4_group_index(e) + 1u;
        e = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[next];
    }
    return P10DCRankFormulaNometa4Resolved{
        p10dc_rankformula_nometa4_group_n(e),
        p10dc_rankformula_nometa4_group_start(e),
        p10dc_rankformula_nometa4_group_delta(e)};
}

// Successor groups are also shared. Every active lane executes the same fixed
// B rounds so __ballot_sync(active, ...) remains legal. Within each B-lane
// subgroup, need_sub is uniform: the subgroup leader performs one successor
// load and broadcasts it with a subgroup mask. Lanes remember the first cursor
// that covers their rank while continuing to help advance the shared cursor for
// higher lanes. This changes successor global loads from the sum of per-lane
// locator steps to the number of distinct group boundaries crossed per block.
__device__ __forceinline__ P10DCRankFormulaNometa4Resolved
p10dc_low_rankformula_nometa_resolve_coopgroup(uint32_t h, uint32_t rank) {
    constexpr uint32_t B = P10DC_RANKFORMULA_NOMETA4_BLOCK;
    constexpr uint32_t BITS = (1u << B) - 1u;
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t src_lane = lane & ~(B - 1u);
    const unsigned active = __activemask();
    const unsigned submask = active & (BITS << src_lane);

    uint32_t elo = 0u, ehi = 0u;
    if (lane == src_lane) {
        const uint32_t bi = D_P10DC_LOW_RANKFORMULA_NOMETA4_BOFF[h] + rank / B;
        const uint32_t gi = uint32_t(D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16[bi]);
        const uint64_t e0 = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[gi];
        elo = uint32_t(e0);
        ehi = uint32_t(e0 >> 32);
    }
    elo = __shfl_sync(active, elo, int(src_lane));
    ehi = __shfl_sync(active, ehi, int(src_lane));
    uint64_t cursor = uint64_t(elo) | (uint64_t(ehi) << 32);

    uint32_t result_n = 0u, result_start = 0u;
    int result_delta = 0;
    bool resolved = false;
#pragma unroll
    for (int k = 0; k < int(B); ++k) {
        const uint32_t start = p10dc_rankformula_nometa4_group_start(cursor);
        const uint32_t count = p10dc_rankformula_nometa4_group_count(cursor);
        const bool need = rank >= start + count;
        if (!resolved && !need) {
            result_n = p10dc_rankformula_nometa4_group_n(cursor);
            result_start = start;
            result_delta = p10dc_rankformula_nometa4_group_delta(cursor);
            resolved = true;
        }
        if (k + 1 < int(B)) {
            const unsigned needs = __ballot_sync(active, need) & submask;
            if (needs) {
                uint32_t nlo = 0u, nhi = 0u;
                if (lane == src_lane) {
                    const uint32_t next = p10dc_rankformula_nometa4_group_index(cursor) + 1u;
                    const uint64_t en = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[next];
                    nlo = uint32_t(en);
                    nhi = uint32_t(en >> 32);
                }
                nlo = __shfl_sync(submask, nlo, int(src_lane));
                nhi = __shfl_sync(submask, nhi, int(src_lane));
                cursor = uint64_t(nlo) | (uint64_t(nhi) << 32);
            }
        }
    }
    return P10DCRankFormulaNometa4Resolved{result_n, result_start, result_delta};
}

__device__ __forceinline__ P10DCRankFormulaNometa4Resolved
p10dc_low_rankformula_nometa_resolve_active(uint32_t h, uint32_t rank) {
#if P10DC_RANKFORMULA_NOMETA_COOPGROUP
    return p10dc_low_rankformula_nometa_resolve_coopgroup(h, rank);
#elif P10DC_RANKFORMULA_NOMETA_WARPSHARE
    return p10dc_low_rankformula_nometa_resolve_warpshare(h, rank);
#else
    return p10dc_low_rankformula_nometa4_resolve(h, rank);
#endif
}
