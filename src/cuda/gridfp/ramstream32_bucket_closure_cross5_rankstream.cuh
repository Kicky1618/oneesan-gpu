#pragma once

#include "ramstream32_bucket_closure_cross5.cuh"
#include "ramstream32_bucket_low_prekey_rankstream.cuh"

__constant__ uint8_t D_P10DC_RANKSTREAM_LMASK[P10DC_CROSS5_KEYS];

static constexpr uint8_t p10dc_rankstream_lmask_host(uint32_t key) {
    uint8_t mask = 0;
    for (int pos = 0; pos < P10DC_CROSS5_CHUNK; ++pos) {
        uint32_t v = (key / p10dc_cross5_pow3_host(pos)) % 3u;
        if (v == uint32_t(::L)) mask = uint8_t(mask | uint8_t(1u << pos));
    }
    return mask;
}

static std::array<uint8_t, P10DC_CROSS5_KEYS> p10dc_rankstream_lmask_table() {
    std::array<uint8_t, P10DC_CROSS5_KEYS> out{};
    for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k)
        out[k] = p10dc_rankstream_lmask_host(k);
    return out;
}

static void p10dc_install_rankstream_lmask() {
    static const auto table = p10dc_rankstream_lmask_table();
    ck(cudaMemcpyToSymbol(D_P10DC_RANKSTREAM_LMASK, table.data(),
                          table.size() * sizeof(uint8_t)),
       "p10dc rankstream L-mask table");
}

template<int START, int LEN>
__device__ __forceinline__ uint32_t p10dc_cross5_apply_chunk_rankstream(
    uint32_t full_key, uint32_t& state, const Count* source_row,
    const uint16_t* rank_row, uint32_t& lbase, BkczCrossAccum& sum
) {
    static_assert(START >= 0 && LEN >= 1 && LEN <= P10DC_CROSS5_CHUNK,
                  "invalid cross5 rankstream chunk");
    constexpr uint32_t DIV = bkcz_pow3_const(START);
    constexpr uint32_t MOD = bkcz_pow3_const(LEN);
    uint32_t chunk = (full_key / DIV) % MOD;
    if (state >= P10DC_CROSS5_STATES) return 2u;
    uint8_t e = D_P10DC_CROSS5[size_t(state) * P10DC_CROSS5_KEYS + chunk];
    uint8_t mask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    uint8_t lmask = D_P10DC_RANKSTREAM_LMASK[chunk];
    while (mask) {
        int i = __ffs(int(mask)) - 1;
        mask = uint8_t(mask & uint8_t(mask - 1));
        uint32_t higher_mask = uint32_t(lmask) & ~((1u << uint32_t(i + 1)) - 1u);
        uint32_t ordinal = lbase + uint32_t(__popc(higher_mask));
        uint16_t rank = rank_row[ordinal];
        sum = bkcz_cross_add(sum, source_row[uint32_t(rank)]);
    }
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;
    lbase += uint32_t(__popc(uint32_t(lmask)));
    state = uint32_t(int(state) + int(D_P10DC_CROSS5_DELTA[chunk]));
    return 0u;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream_nofallback(
    uint32_t key, uint32_t depth, const Count* source_row,
    const uint16_t* rank_row, bool& overflow
) {
    overflow = false;
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth, lbase = 0;
    if (state >= P10DC_CROSS5_STATES) {
        overflow = true;
        return BkczCrossAccum(0);
    }
    BkczCrossAccum sum = 0;
    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    uint32_t st = p10dc_cross5_apply_chunk_rankstream<S0, L0>(
        key, state, source_row, rank_row, lbase, sum);
    if (st == 1u) return sum;
    if (st == 2u) { overflow = true; return BkczCrossAccum(0); }

    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        st = p10dc_cross5_apply_chunk_rankstream<S1, L1>(
            key, state, source_row, rank_row, lbase, sum);
        if (st == 1u) return sum;
        if (st == 2u) { overflow = true; return BkczCrossAccum(0); }
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= 5 ? 5 : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "K<=14 must fit in three rankstream CROSS5 chunks");
            st = p10dc_cross5_apply_chunk_rankstream<S2, L2>(
                key, state, source_row, rank_row, lbase, sum);
            if (st == 2u) { overflow = true; return BkczCrossAccum(0); }
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream_fixed(
    uint32_t h, uint32_t rank, uint32_t key, uint32_t depth, const Count* source_row
) {
    const uint16_t* rank_row = p10dc_low_rankstream_row(h, rank);
    bool overflow = false;
    BkczCrossAccum sum = p10dc_resolved_low_preimages_cross5_rankstream_nofallback(
        key, depth, source_row, rank_row, overflow);
    if (!overflow) return sum;
    size_t ix = D_BKF_LOW_CODE_OFF[
        size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + h] + rank;
    uint32_t dc = D_BKF_LOW_CODES[ix];
    return p10dc_resolved_low_preimages_cross5_fallback_prekey(
        dc, key, depth, source_row);
}
