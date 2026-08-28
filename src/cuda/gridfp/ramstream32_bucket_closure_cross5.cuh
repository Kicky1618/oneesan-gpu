#pragma once

#include "ramstream32_bucket_closure_cross5_common.cuh"

static_assert(
    P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS * sizeof(uint8_t) +
        P10DC_CROSS5_KEYS * sizeof(int8_t) == 6561,
    "compact cross5 table size regression");

__constant__ uint8_t D_P10DC_CROSS5[P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS];
__constant__ int8_t D_P10DC_CROSS5_DELTA[P10DC_CROSS5_KEYS];

static std::array<uint8_t, P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS>
p10dc_cross5_host_table() {
    std::array<uint8_t, P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS> out{};
    for (uint32_t s = 0; s < P10DC_CROSS5_STATES; ++s)
        for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k)
            out[size_t(s) * P10DC_CROSS5_KEYS + k] = p10dc_cross5_host_entry(k, s);
    return out;
}

static std::array<int8_t, P10DC_CROSS5_KEYS> p10dc_cross5_host_delta_table() {
    std::array<int8_t, P10DC_CROSS5_KEYS> out{};
    for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k)
        out[k] = p10dc_cross5_host_delta(k);
    return out;
}

static void p10dc_install_cross5_lut() {
    static const auto table = p10dc_cross5_host_table();
    static const auto delta = p10dc_cross5_host_delta_table();
    ck(cudaMemcpyToSymbol(D_P10DC_CROSS5, table.data(),
                          table.size() * sizeof(uint8_t)),
       "p10dc cross5 constant mask table");
    ck(cudaMemcpyToSymbol(D_P10DC_CROSS5_DELTA, delta.data(),
                          delta.size() * sizeof(int8_t)),
       "p10dc cross5 constant delta table");
}

template<int START, int LEN, bool CHECK_STATE = true>
__device__ __forceinline__ uint32_t p10dc_cross5_apply_chunk(
    uint32_t full_key, uint32_t& state, const Count* source_row, BkczCrossAccum& sum
) {
    static_assert(START >= 0 && LEN >= 1 && LEN <= P10DC_CROSS5_CHUNK,
                  "invalid cross5 chunk");
    constexpr uint32_t DIV = bkcz_pow3_const(START);
    constexpr uint32_t MOD = bkcz_pow3_const(LEN);
    uint32_t chunk = (full_key / DIV) % MOD;
    if constexpr (CHECK_STATE) {
        if (state >= P10DC_CROSS5_STATES) return 2u;
    }
    uint8_t e = D_P10DC_CROSS5[size_t(state) * P10DC_CROSS5_KEYS + chunk];
    uint8_t mask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    while (mask) {
        int i = __ffs(int(mask)) - 1;
        mask = uint8_t(mask & uint8_t(mask - 1));
        uint32_t pos = uint32_t(START + i);
        uint32_t x = D_BKF_LOW_DIRECT[full_key - p10dc_pow3(pos)];
        if (x != BKF_DIRECT_INVALID)
            sum = bkcz_cross_add(sum, source_row[bkf_loc_rank(x)]);
    }
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;
    state = uint32_t(int(state) + int(D_P10DC_CROSS5_DELTA[chunk]));
    return 0u;
}

// Execute the compact automaton without a fallback. overflow=true means the
// caller must discard the partial sum and recompute with the scalar walker.
// For valid depth4 input it is structurally unreachable: chunk start states are
// bounded by 15,20,25 and state zero cannot be reached because down at one halts.
__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_key_nofallback(
    uint32_t key, uint32_t depth, const Count* source_row, bool& overflow
) {
    overflow = false;
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth;
    if (state >= P10DC_CROSS5_STATES) {
        overflow = true;
        return 0;
    }
    BkczCrossAccum sum = 0;
    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    uint32_t st = p10dc_cross5_apply_chunk<S0, L0>(key, state, source_row, sum);
    if (st == 1u) return sum;
    if (st == 2u) { overflow = true; return 0; }
    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        st = p10dc_cross5_apply_chunk<S1, L1>(key, state, source_row, sum);
        if (st == 1u) return sum;
        if (st == 2u) { overflow = true; return 0; }
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= 5 ? 5 : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "K<=14 must fit in three cross5 chunks");
            st = p10dc_cross5_apply_chunk<S2, L2>(key, state, source_row, sum);
            if (st == 2u) { overflow = true; return 0; }
        }
    }
    return sum;
}

// Production depthcode path. depth is decoded from four bits (1..15), and
// LOW_LUT_K<=14 means chunk-start states are bounded by 15,20,25. Therefore all
// three table accesses are in [0,25] and the overflow checks above are dead.
__device__ __forceinline__ BkczCrossAccum p10dc_resolved_low_preimages_cross5_key_fast(
    uint32_t key, uint32_t depth, const Count* source_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth;
    BkczCrossAccum sum = 0;
    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    uint32_t st = p10dc_cross5_apply_chunk<S0, L0, false>(key, state, source_row, sum);
    if (st == 1u) return sum;
    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        st = p10dc_cross5_apply_chunk<S1, L1, false>(key, state, source_row, sum);
        if (st == 1u) return sum;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= 5 ? 5 : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "K<=14 must fit in three cross5 chunks");
            p10dc_cross5_apply_chunk<S2, L2, false>(key, state, source_row, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum p10dc_resolved_low_preimages_cross5_prekey(
    uint32_t dest_code, uint32_t key, uint32_t depth, const Count* source_row
) {
    bool overflow = false;
    BkczCrossAccum sum = p10dc_resolved_low_preimages_cross5_key_nofallback(
        key, depth, source_row, overflow);
    return overflow
        ? p10dc_resolved_low_preimages_cross5_fallback_prekey(
              dest_code, key, depth, source_row)
        : sum;
}

// Production prekey path: the packed LOW code and the overflow/fallback branch
// are both unnecessary under the depth4 + K<=14 invariant. Keep code_ix in the
// signature so callers do not need a separate source-level variant.
__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_prekey_indexed(
    size_t code_ix, uint32_t key, uint32_t depth, const Count* source_row
) {
    (void)code_ix;
    return p10dc_resolved_low_preimages_cross5_key_fast(key, depth, source_row);
}

__device__ __forceinline__ BkczCrossAccum p10dc_resolved_low_preimages_cross5(
    uint32_t dest_code, uint32_t depth, const Count* source_row
) {
    return p10dc_resolved_low_preimages_cross5_prekey(
        dest_code, bkcz_ternary_key<LOW_LUT_K>(dest_code), depth, source_row);
}
