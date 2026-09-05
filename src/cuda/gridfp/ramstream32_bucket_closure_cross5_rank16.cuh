#pragma once

#include "ramstream32_bucket_closure_cross5.cuh"
#include "ramstream32_bucket_low_prekey_rank16.cuh"

template<int START, int LEN>
__device__ __forceinline__ uint32_t p10dc_cross5_apply_chunk_rank16(
    uint32_t full_key, uint32_t& state, const Count* source_row,
    const uint16_t* rank_row, BkczCrossAccum& sum
) {
    static_assert(START >= 0 && LEN >= 1 && LEN <= P10DC_CROSS5_CHUNK,
                  "invalid cross5 rank16 chunk");
    constexpr uint32_t DIV = bkcz_pow3_const(START);
    constexpr uint32_t MOD = bkcz_pow3_const(LEN);
    uint32_t chunk = (full_key / DIV) % MOD;
    if (state >= P10DC_CROSS5_STATES) return 2u;
    uint8_t e = D_P10DC_CROSS5[size_t(state) * P10DC_CROSS5_KEYS + chunk];
    uint8_t mask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    while (mask) {
        int i = __ffs(int(mask)) - 1;
        mask = uint8_t(mask & uint8_t(mask - 1));
        uint32_t pos = uint32_t(START + i);
        uint16_t rank = rank_row[pos];
        if (rank != P10DC_LOW_RANK16_INVALID)
            sum = bkcz_cross_add(sum, source_row[uint32_t(rank)]);
    }
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;
    state = uint32_t(int(state) + int(D_P10DC_CROSS5_DELTA[chunk]));
    return 0u;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rank16_nofallback(
    uint32_t key, uint32_t depth, const Count* source_row,
    const uint16_t* rank_row, bool& overflow
) {
    overflow = false;
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth;
    if (state >= P10DC_CROSS5_STATES) {
        overflow = true;
        return BkczCrossAccum(0);
    }
    BkczCrossAccum sum = 0;
    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    uint32_t st = p10dc_cross5_apply_chunk_rank16<S0, L0>(
        key, state, source_row, rank_row, sum);
    if (st == 1u) return sum;
    if (st == 2u) { overflow = true; return BkczCrossAccum(0); }

    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        st = p10dc_cross5_apply_chunk_rank16<S1, L1>(
            key, state, source_row, rank_row, sum);
        if (st == 1u) return sum;
        if (st == 2u) { overflow = true; return BkczCrossAccum(0); }
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= 5 ? 5 : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "K<=14 must fit in three rank16 CROSS5 chunks");
            st = p10dc_cross5_apply_chunk_rank16<S2, L2>(
                key, state, source_row, rank_row, sum);
            if (st == 2u) { overflow = true; return BkczCrossAccum(0); }
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rank16_fixed(
    uint32_t h, uint32_t rank, uint32_t key, uint32_t depth, const Count* source_row
) {
    const uint16_t* rank_row = p10dc_low_rank16_row(h, rank);
    bool overflow = false;
    BkczCrossAccum sum = p10dc_resolved_low_preimages_cross5_rank16_nofallback(
        key, depth, source_row, rank_row, overflow);
    if (!overflow) return sum;

    // Defensive path only. The 26-state proof establishes that depth4/K<=14
    // cannot reach it, but preserve the original direct-table scalar walker.
    size_t ix = D_BKF_LOW_CODE_OFF[
        size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + h] + rank;
    uint32_t dc = D_BKF_LOW_CODES[ix];
    return p10dc_resolved_low_preimages_cross5_fallback_prekey(
        dc, key, depth, source_row);
}
