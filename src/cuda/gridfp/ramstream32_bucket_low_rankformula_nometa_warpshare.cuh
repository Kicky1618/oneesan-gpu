#pragma once

#include "ramstream32_bucket_low_rankformula_nometa4.cuh"

#ifndef P10DC_RANKFORMULA_NOMETA_WARPSHARE
#define P10DC_RANKFORMULA_NOMETA_WARPSHARE 0
#endif
static_assert(P10DC_RANKFORMULA_NOMETA_WARPSHARE == 0 ||
              P10DC_RANKFORMULA_NOMETA_WARPSHARE == 1,
              "P10DC_RANKFORMULA_NOMETA_WARPSHARE must be 0 or 1");
static_assert(32u % P10DC_RANKFORMULA_NOMETA4_BLOCK == 0u,
              "warp-shared nometa locator requires block size dividing 32");

// Warp-striped HIGH kernels enumerate lr as base32+lane, and every later stripe
// advances by gridDim.x*32.  Therefore rank/B is shared by exactly B adjacent
// lanes for B in {4,8,16}.  The first lane of each B-lane subgroup loads both
// the block index and the initial group entry, then broadcasts them.  Each lane
// only performs additional group loads when its rank crosses a group boundary.
__device__ __forceinline__ P10DCRankFormulaNometa4Resolved
p10dc_low_rankformula_nometa_resolve_warpshare(uint32_t h, uint32_t rank) {
    constexpr uint32_t B = P10DC_RANKFORMULA_NOMETA4_BLOCK;
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const uint32_t src_lane = lane & ~(B - 1u);
    const unsigned active = __activemask();

    uint32_t gi = 0u, elo = 0u, ehi = 0u;
    if (lane == src_lane) {
        const uint32_t bi = D_P10DC_LOW_RANKFORMULA_NOMETA4_BOFF[h] + rank / B;
        gi = uint32_t(D_P10DC_LOW_RANKFORMULA_NOMETA4_BLOCK16[bi]);
        const uint64_t e0 = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[gi];
        elo = uint32_t(e0);
        ehi = uint32_t(e0 >> 32);
    }
    gi = __shfl_sync(active, gi, int(src_lane));
    elo = __shfl_sync(active, elo, int(src_lane));
    ehi = __shfl_sync(active, ehi, int(src_lane));
    uint64_t e = uint64_t(elo) | (uint64_t(ehi) << 32);

#pragma unroll
    for (int k = 0; k < int(B - 1u); ++k) {
        const uint32_t start = p10dc_rankformula_nometa4_group_start(e);
        const uint32_t count = p10dc_rankformula_nometa4_group_count(e);
        if (rank < start + count) break;
        ++gi;
        e = D_P10DC_LOW_RANKFORMULA_NOMETA4_GROUP64[gi];
    }
    return P10DCRankFormulaNometa4Resolved{
        p10dc_rankformula_nometa4_group_n(e),
        p10dc_rankformula_nometa4_group_start(e),
        p10dc_rankformula_nometa4_group_delta(e)};
}

__device__ __forceinline__ P10DCRankFormulaNometa4Resolved
p10dc_low_rankformula_nometa_resolve_active(uint32_t h, uint32_t rank) {
#if P10DC_RANKFORMULA_NOMETA_WARPSHARE
    return p10dc_low_rankformula_nometa_resolve_warpshare(h, rank);
#else
    return p10dc_low_rankformula_nometa4_resolve(h, rank);
#endif
}
