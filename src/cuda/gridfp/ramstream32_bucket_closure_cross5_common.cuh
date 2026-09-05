#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_plan.cuh"
#include <array>

static constexpr int P10DC_CROSS5_CHUNK = 5;
static constexpr int P10DC_CROSS5_KEYS = 243;
static constexpr int P10DC_CROSS5_STATES = 26;
static constexpr uint8_t P10DC_CROSS5_MASK_MASK = 0x1fu;
static constexpr int P10DC_CROSS5_HALT_SHIFT = 5;
static constexpr int P10DC_CROSS5_MAX_RUNTIME_DEPTH = 15;
static constexpr int P10DC_CROSS5_MAX_TABLE_INPUT_STATE =
    P10DC_CROSS5_MAX_RUNTIME_DEPTH + 2 * P10DC_CROSS5_CHUNK;
static_assert(LOW_LUT_K <= 14, "CROSS5 fast path assumes at most three chunks");
static_assert(P10DC_CROSS5_MAX_TABLE_INPUT_STATE == 25);
static_assert(P10DC_CROSS5_STATES == P10DC_CROSS5_MAX_TABLE_INPUT_STATE + 1);

static constexpr uint32_t p10dc_cross5_pow3_host(int n) {
    return n <= 0 ? 1u : 3u * p10dc_cross5_pow3_host(n - 1);
}

static constexpr uint8_t p10dc_cross5_host_entry(
    uint32_t key, uint32_t input_state
) {
    uint32_t s = input_state;
    uint8_t mask = 0;
    bool halt = false;
    for (int pos = P10DC_CROSS5_CHUNK - 1; pos >= 0; --pos) {
        uint32_t v = (key / p10dc_cross5_pow3_host(pos)) % 3u;
        if (v == uint32_t(R)) {
            if (s == 1u) {
                halt = true;
                break;
            }
            --s;
        } else if (v == uint32_t(::L)) {
            if (s == 1u) mask = uint8_t(mask | uint8_t(1u << pos));
            ++s;
        }
    }
    return uint8_t(mask | (uint8_t(halt) << P10DC_CROSS5_HALT_SHIFT));
}

static constexpr int8_t p10dc_cross5_host_delta(uint32_t key) {
    int d = 0;
    for (int pos = 0; pos < P10DC_CROSS5_CHUNK; ++pos) {
        uint32_t v = (key / p10dc_cross5_pow3_host(pos)) % 3u;
        if (v == uint32_t(R)) --d;
        else if (v == uint32_t(::L)) ++d;
    }
    return int8_t(d);
}

// Scalar fallback shared by ordinary CROSS5 and sparse-rank CROSS5.  It does
// not depend on either compact automaton LUT; only the fixed-owner direct map
// is consulted.  Valid depth4 production input never reaches this path.
__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_fallback_prekey(
    uint32_t dest_code, uint32_t key, uint32_t depth, const Count* source_row
) {
    BkczCrossAccum sum = 0;
    int s = int(depth);
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

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_fallback(
    uint32_t dest_code, uint32_t depth, const Count* source_row
) {
    return p10dc_resolved_low_preimages_cross5_fallback_prekey(
        dest_code, bkcz_ternary_key<LOW_LUT_K>(dest_code), depth, source_row);
}
