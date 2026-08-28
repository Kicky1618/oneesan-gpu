#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_plan.cuh"
#include <array>

static constexpr int P10DC_CROSS5_CHUNK = 5;
static constexpr int P10DC_CROSS5_KEYS = 243;
static constexpr int P10DC_CROSS5_STATES = 32;
static constexpr uint16_t P10DC_CROSS5_MASK_MASK = 0x1fu;
static constexpr int P10DC_CROSS5_STATE_SHIFT = 5;
static constexpr uint16_t P10DC_CROSS5_STATE_MASK = 0x3fu;
static constexpr int P10DC_CROSS5_HALT_SHIFT = 11;
static_assert(P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS * sizeof(uint16_t) == 15552,
              "cross5 table size regression");

__constant__ uint16_t D_P10DC_CROSS5[P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS];

static constexpr uint32_t p10dc_cross5_pow3_host(int n) {
    return n <= 0 ? 1u : 3u * p10dc_cross5_pow3_host(n - 1);
}

static constexpr uint16_t p10dc_cross5_host_entry(uint32_t key, uint32_t input_state) {
    uint32_t s = input_state;
    uint16_t mask = 0;
    bool halt = false;
    for (int pos = P10DC_CROSS5_CHUNK - 1; pos >= 0; --pos) {
        uint32_t v = (key / p10dc_cross5_pow3_host(pos)) % 3u;
        // LOW inactive factor normalization: R(1)=down, L(2)=up.
        if (v == uint32_t(R)) {
            if (s == 1u) { halt = true; break; }
            --s;
        } else if (v == uint32_t(::L)) {
            if (s == 1u) mask |= uint16_t(1u << pos);
            ++s;
        }
    }
    return uint16_t(mask |
                    (uint16_t(s) << P10DC_CROSS5_STATE_SHIFT) |
                    (uint16_t(halt) << P10DC_CROSS5_HALT_SHIFT));
}

static std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS>
p10dc_cross5_host_table() {
    std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS> out{};
    for (uint32_t s = 0; s < P10DC_CROSS5_STATES; ++s)
        for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k)
            out[size_t(s) * P10DC_CROSS5_KEYS + k] = p10dc_cross5_host_entry(k, s);
    return out;
}

static void p10dc_install_cross5_lut() {
    static const auto table = p10dc_cross5_host_table();
    ck(cudaMemcpyToSymbol(D_P10DC_CROSS5, table.data(), table.size() * sizeof(uint16_t)),
       "p10dc cross5 constant table");
}

__device__ __forceinline__ BkczCrossAccum p10dc_resolved_low_preimages_cross5_fallback(
    uint32_t dest_code, uint32_t depth, const Count* source_row
) {
    BkczCrossAccum sum = 0;
    int s = int(depth);
    uint32_t key = bkcz_ternary_key<LOW_LUT_K>(dest_code);
    uint32_t weight = bkcz_pow3_const(LOW_LUT_K - 1);
#pragma unroll
    for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
        uint32_t v = (dest_code >> (2 * pos)) & 3u;
        if (v == uint32_t(R)) {
            if (s == 1) break;
            --s;
        } else if (v == uint32_t(::L)) {
            if (s == 1) {
                uint32_t x = D_BKF_LOW_DIRECT[key - weight];
                if (x != BKF_DIRECT_INVALID)
                    sum = bkcz_cross_add(sum, source_row[bkf_loc_rank(x)]);
            }
            ++s;
        }
        if (pos) weight /= 3u;
    }
    return sum;
}

template<int START, int LEN>
__device__ __forceinline__ bool p10dc_cross5_apply_chunk(
    uint32_t full_key, uint32_t& state, const Count* source_row, BkczCrossAccum& sum
) {
    static_assert(START >= 0 && LEN >= 1 && LEN <= P10DC_CROSS5_CHUNK,
                  "invalid cross5 chunk");
    constexpr uint32_t DIV = bkcz_pow3_const(START);
    constexpr uint32_t MOD = bkcz_pow3_const(LEN);
    uint32_t chunk = (full_key / DIV) % MOD;
    if (state >= P10DC_CROSS5_STATES) return true;
    uint16_t e = D_P10DC_CROSS5[size_t(state) * P10DC_CROSS5_KEYS + chunk];
    uint16_t mask = e & P10DC_CROSS5_MASK_MASK;
    while (mask) {
        int i = __ffs(int(mask)) - 1;
        mask = uint16_t(mask & (mask - 1));
        uint32_t pos = uint32_t(START + i);
        uint32_t x = D_BKF_LOW_DIRECT[full_key - p10dc_pow3(pos)];
        if (x != BKF_DIRECT_INVALID)
            sum = bkcz_cross_add(sum, source_row[bkf_loc_rank(x)]);
    }
    state = uint32_t((e >> P10DC_CROSS5_STATE_SHIFT) & P10DC_CROSS5_STATE_MASK);
    return ((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0;
}

__device__ __forceinline__ BkczCrossAccum p10dc_resolved_low_preimages_cross5(
    uint32_t dest_code, uint32_t depth, const Count* source_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth;
    if (state >= P10DC_CROSS5_STATES)
        return p10dc_resolved_low_preimages_cross5_fallback(dest_code, depth, source_row);
    uint32_t key = bkcz_ternary_key<LOW_LUT_K>(dest_code);
    BkczCrossAccum sum = 0;

    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    if (p10dc_cross5_apply_chunk<S0, L0>(key, state, source_row, sum)) return sum;

    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        if (state >= P10DC_CROSS5_STATES)
            return p10dc_resolved_low_preimages_cross5_fallback(dest_code, depth, source_row);
        if (p10dc_cross5_apply_chunk<S1, L1>(key, state, source_row, sum)) return sum;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= 5 ? 5 : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "K<=14 must fit in three cross5 chunks");
            if (state >= P10DC_CROSS5_STATES)
                return p10dc_resolved_low_preimages_cross5_fallback(dest_code, depth, source_row);
            (void)p10dc_cross5_apply_chunk<S2, L2>(key, state, source_row, sum);
        }
    }
    return sum;
}
