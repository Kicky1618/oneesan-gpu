#pragma once

#include "ramstream32_b300_dual_tile_opt.cuh"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

// Exact W28 / LOW14 / HIGH13 / 8-GPU occupancy-popcount quotas obtained from
// the aggregate MILP with a 0.01 GiB load slack.  For a fixed popcount all
// occupancy masks have identical height/cardinality signatures, so choosing
// arbitrary numeric masks within the quota is equivalent for memory and
// shuffle-volume accounting.  This replaces 24,576 explicit owner bytes with
// 8*(14+15)=232 small integers and reconstructs the assignment deterministically.
//
// Exact aggregate metrics for this quota solution:
//   HIGH orientation load 242.476822 .. 242.492994 GiB/GPU
//   LOW  orientation load 242.476352 .. 242.493787 GiB/GPU
//   LOW->HIGH off-GPU      1704.288849376 GiB/row (main+blocked)
//   HIGH->LOW off-GPU      1201.316896942 GiB/row (main)
//   W=28, final H->L elided: 78.281909373 TiB/residue
//   max GPU port volume:      9.799945773 TiB/residue
//   ideal 1.8 TB/s floor:     5.986197 s/residue
//   max pair-slot arena:      242.804958 GiB/GPU
static constexpr uint16_t B300_W28_HIGH_POP_QUOTA[8][14] = {
    {0,0,1,0,0,2,0,2,0,119,0,78,0,1},
    {0,0,0,2,0,0,0,726,0,193,0,0,0,0},
    {0,0,0,0,0,0,0,4,0,403,0,0,0,0},
    {0,0,0,0,1,0,3,0,750,0,0,0,0,0},
    {0,0,2,0,2,0,1,0,352,0,115,0,0,0},
    {0,2,0,0,1,0,1,0,0,0,171,0,13,0},
    {1,0,75,0,711,0,1711,0,185,0,0,0,0,0},
    {0,11,0,284,0,1285,0,984,0,0,0,0,0,0},
};

static constexpr uint16_t B300_W28_LOW_POP_QUOTA[8][15] = {
    {0,0,0,1,0,1,0,0,0,2,0,283,0,14,0},
    {0,0,0,0,0,0,0,103,0,1126,0,0,0,0,0},
    {0,0,0,0,0,2,0,2,0,874,0,81,0,0,0},
    {0,0,0,0,0,0,2,0,1832,0,90,0,0,0,0},
    {0,0,0,0,0,0,1,0,1,0,621,0,0,0,0},
    {1,0,0,0,0,0,0,0,3,0,290,0,91,0,1},
    {0,0,91,0,1001,0,3000,0,1167,0,0,0,0,0,0},
    {0,14,0,363,0,1999,0,3327,0,0,0,0,0,0,0},
};

static inline uint32_t b300_dt_popcount(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return uint32_t(__builtin_popcount(x));
#else
    uint32_t n=0; while(x){n+=x&1u;x>>=1;} return n;
#endif
}

template<int K, int P>
static std::vector<uint8_t> b300_dt_owner_from_pop_quota(
    const uint16_t (&quota)[8][P]
) {
    static_assert(P == K + 1, "quota column count must be K+1");
    constexpr uint32_t NM = 1u << K;
    std::array<std::array<uint32_t, P>, 8> used{};
    std::array<uint32_t, P> total{};
    for (int g=0; g<8; ++g) for (int k=0; k<P; ++k) total[k] += quota[g][k];

    // Verify each popcount bucket contains exactly C(K,k) masks.
    uint64_t choose=1;
    for (int k=0; k<P; ++k) {
        if (total[k] != choose) std::exit(580 + k);
        if (k < K) choose = choose * uint64_t(K-k) / uint64_t(k+1);
    }

    std::vector<uint8_t> owner(NM, 0xffu);
    for (uint32_t mask=0; mask<NM; ++mask) {
        uint32_t k=b300_dt_popcount(mask);
        int g=0;
        while (g<8 && used[g][k] >= quota[g][k]) ++g;
        if (g==8) std::exit(600);
        owner[mask]=uint8_t(g);
        ++used[g][k];
    }
    for (int g=0; g<8; ++g) for (int k=0; k<P; ++k)
        if (used[g][k] != quota[g][k]) std::exit(601);
    return owner;
}

static B300DualTileHost build_b300_dual_tile_layout_w28_precomputed(
    const StorageFactorHost& f,const StorageLayout& l,int ngpu
) {
    if constexpr (TARGET_W==28 && HIGH_LUT_K==13 && LOW_LUT_K==14) {
        if (ngpu==8) {
            std::vector<uint8_t> ho=b300_dt_owner_from_pop_quota<13>(B300_W28_HIGH_POP_QUOTA);
            std::vector<uint8_t> lo=b300_dt_owner_from_pop_quota<14>(B300_W28_LOW_POP_QUOTA);
            B300DualTileHost z=build_b300_dual_tile_layout(f,l,ngpu);
            z.high=b300_build_mask_shard_from_owner(f,l,ngpu,ho);
            for(int g=0;g<ngpu;++g)for(int h=0;h<=MAXW;++h)
                z.high_count[g][h]=z.high.owned_off[g][h+1]-z.high.owned_off[g][h];
            b300_dt_rebuild_low_owner(z,lo,f,l);
            return z;
        }
    }
    return build_b300_dual_tile_layout(f,l,ngpu);
}
