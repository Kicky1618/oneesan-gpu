#pragma once

#include "ramstream32_bucket_closure_cross5_common.cuh"
#include "ramstream32_bucket_low_prekey_rankstream.cuh"

#ifndef P10DC_RANKSTREAM_LUT_LDG
#define P10DC_RANKSTREAM_LUT_LDG 0
#endif
#ifndef P10DC_RANKSTREAM_LUT_PAD256
#define P10DC_RANKSTREAM_LUT_PAD256 0
#endif
static_assert(P10DC_RANKSTREAM_LUT_LDG == 0 || P10DC_RANKSTREAM_LUT_LDG == 1,
              "P10DC_RANKSTREAM_LUT_LDG must be 0 or 1");
static_assert(P10DC_RANKSTREAM_LUT_PAD256 == 0 || P10DC_RANKSTREAM_LUT_PAD256 == 1,
              "P10DC_RANKSTREAM_LUT_PAD256 must be 0 or 1");
static_assert(!P10DC_RANKSTREAM_LUT_PAD256 || P10DC_RANKSTREAM_LUT_LDG,
              "256-byte rankstream row padding is only meaningful for LDG mode");

// Warp-striped execution gives every lane a different LOW rank/key. Constant
// memory is therefore not always a broadcast. Keep the storage mode behind a
// compile-time switch so B300 can compare constant-cache serialization with
// ordinary read-only/L1 loads. In the padded LDG experiment every 243-byte
// state row gets a 256-byte stride, making every row 128-byte aligned and
// limiting a complete state row to exactly two 128-byte cache lines.
static constexpr int P10DC_RANKSTREAM_LUT_STRIDE =
    P10DC_RANKSTREAM_LUT_PAD256 ? 256 : P10DC_CROSS5_KEYS;
static_assert(P10DC_CROSS5_KEYS == 243);
static_assert(P10DC_RANKSTREAM_LUT_STRIDE >= P10DC_CROSS5_KEYS);

#if P10DC_RANKSTREAM_LUT_LDG
__device__ __align__(128) uint8_t D_P10DC_RANKSTREAM_META[P10DC_CROSS5_KEYS];
__device__ __align__(128) uint8_t
    D_P10DC_RANKSTREAM_CROSS5[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#else
__constant__ uint8_t D_P10DC_RANKSTREAM_META[P10DC_CROSS5_KEYS];
__constant__ uint8_t
    D_P10DC_RANKSTREAM_CROSS5[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#endif

static constexpr uint8_t P10DC_RANKSTREAM_META_LCOUNT_MASK = 0x07u;
static constexpr int P10DC_RANKSTREAM_META_DELTA_SHIFT = 3;
static constexpr int P10DC_RANKSTREAM_META_DELTA_BIAS = 5;
static_assert(P10DC_CROSS5_CHUNK == 5, "rankstream metadata packing assumes CROSS5");
static_assert(P10DC_CROSS5_STATES * P10DC_CROSS5_KEYS + P10DC_CROSS5_KEYS == 6561,
              "rankstream CROSS5 logical LUT size regression");

static constexpr uint8_t p10dc_rankstream_lmask_host(uint32_t key) {
    uint8_t mask = 0;
    for (int pos = 0; pos < P10DC_CROSS5_CHUNK; ++pos) {
        uint32_t v = (key / p10dc_cross5_pow3_host(pos)) % 3u;
        if (v == uint32_t(::L)) mask = uint8_t(mask | uint8_t(1u << pos));
    }
    return mask;
}

static constexpr uint8_t p10dc_rankstream_popcount5_host(uint8_t x) {
    uint8_t n = 0;
    for (int i = 0; i < P10DC_CROSS5_CHUNK; ++i)
        n = uint8_t(n + ((x >> i) & 1u));
    return n;
}

static constexpr uint8_t p10dc_rankstream_host_entry(
    uint32_t key, uint32_t input_state
) {
    const uint8_t e = p10dc_cross5_host_entry(key, input_state);
    const uint8_t mask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    const uint8_t lmask = p10dc_rankstream_lmask_host(key);
    uint8_t rankmask = 0;
    for (int pos = 0; pos < P10DC_CROSS5_CHUNK; ++pos) {
        if (((mask >> pos) & 1u) == 0u) continue;
        const uint8_t higher = uint8_t(
            lmask & uint8_t(~uint8_t((uint8_t(1u << (pos + 1))) - 1u)));
        const uint8_t ordinal = p10dc_rankstream_popcount5_host(higher);
        rankmask = uint8_t(rankmask | uint8_t(1u << ordinal));
    }
    return uint8_t(
        rankmask | (e & uint8_t(1u << P10DC_CROSS5_HALT_SHIFT)));
}

static constexpr uint8_t p10dc_rankstream_meta_host(uint32_t key) {
    const uint8_t lcount = p10dc_rankstream_popcount5_host(
        p10dc_rankstream_lmask_host(key));
    const int delta = int(p10dc_cross5_host_delta(key));
    return uint8_t(
        lcount |
        (uint8_t(delta + P10DC_RANKSTREAM_META_DELTA_BIAS)
         << P10DC_RANKSTREAM_META_DELTA_SHIFT));
}

static std::array<uint8_t, P10DC_CROSS5_KEYS>
p10dc_rankstream_meta_table() {
    std::array<uint8_t, P10DC_CROSS5_KEYS> out{};
    for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k)
        out[k] = p10dc_rankstream_meta_host(k);
    return out;
}

static std::array<uint8_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE>
p10dc_rankstream_cross5_table() {
    std::array<uint8_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE> out{};
    for (uint32_t s = 0; s < P10DC_CROSS5_STATES; ++s) {
        for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k) {
            out[size_t(s) * P10DC_RANKSTREAM_LUT_STRIDE + k] =
                p10dc_rankstream_host_entry(k, s);
        }
    }
    return out;
}

static void p10dc_install_rankstream_lut() {
    static const auto meta = p10dc_rankstream_meta_table();
    static const auto table = p10dc_rankstream_cross5_table();
    ck(cudaMemcpyToSymbol(D_P10DC_RANKSTREAM_META, meta.data(),
                          meta.size() * sizeof(uint8_t)),
       "p10dc rankstream chunk metadata table");
    ck(cudaMemcpyToSymbol(D_P10DC_RANKSTREAM_CROSS5, table.data(),
                          table.size() * sizeof(uint8_t)),
       "p10dc rankstream CROSS5 rank-mask table");
}

// Compatibility for existing experimental callers.
static void p10dc_install_rankstream_lmask() {
    p10dc_install_rankstream_lut();
}

__device__ __forceinline__ uint8_t p10dc_rankstream_entry_load(size_t ix) {
#if P10DC_RANKSTREAM_LUT_LDG
    return __ldg(D_P10DC_RANKSTREAM_CROSS5 + ix);
#else
    return D_P10DC_RANKSTREAM_CROSS5[ix];
#endif
}

__device__ __forceinline__ uint8_t p10dc_rankstream_meta_load(uint32_t chunk) {
#if P10DC_RANKSTREAM_LUT_LDG
    return __ldg(D_P10DC_RANKSTREAM_META + chunk);
#else
    return D_P10DC_RANKSTREAM_META[chunk];
#endif
}

template<int START, int LEN, bool CHECK_STATE = true>
__device__ __forceinline__ uint32_t p10dc_cross5_apply_chunk_rankstream(
    uint32_t full_key, uint32_t& state, const Count* source_row,
    const uint16_t* rank_row, uint32_t& lbase, BkczCrossAccum& sum
) {
    static_assert(START >= 0 && LEN >= 1 && LEN <= P10DC_CROSS5_CHUNK,
                  "invalid cross5 rankstream chunk");
    constexpr uint32_t DIV = bkcz_pow3_const(START);
    constexpr uint32_t MOD = bkcz_pow3_const(LEN);
    const uint32_t chunk = (full_key / DIV) % MOD;
    if constexpr (CHECK_STATE) {
        if (state >= P10DC_CROSS5_STATES) return 2u;
    }

    const uint8_t e = p10dc_rankstream_entry_load(
        size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk);
    uint8_t rankmask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    while (rankmask) {
        const int ordinal = __ffs(int(rankmask)) - 1;
        rankmask = uint8_t(rankmask & uint8_t(rankmask - 1));
        const uint16_t rank = rank_row[lbase + uint32_t(ordinal)];
        sum = bkcz_cross_add(sum, source_row[uint32_t(rank)]);
    }
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;

    const uint8_t meta = p10dc_rankstream_meta_load(chunk);
    lbase += uint32_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
    state = uint32_t(
        int(state) + int(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT)
        - P10DC_RANKSTREAM_META_DELTA_BIAS);
    return 0u;
}

// Checked form retained for reference tests and non-depth4 callers.
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

// Production depthcode form. depth is four bits (1..15) and LOW_LUT_K<=14,
// hence chunk-start states are bounded by 15,20,25 exactly as in ordinary
// CROSS5. Bounds checks and the scalar fallback are dead and are omitted here.
__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream_key_fast(
    uint32_t key, uint32_t depth, const Count* source_row, const uint16_t* rank_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth, lbase = 0;
    BkczCrossAccum sum = 0;
    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    uint32_t st = p10dc_cross5_apply_chunk_rankstream<S0, L0, false>(
        key, state, source_row, rank_row, lbase, sum);
    if (st == 1u) return sum;

    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        st = p10dc_cross5_apply_chunk_rankstream<S1, L1, false>(
            key, state, source_row, rank_row, lbase, sum);
        if (st == 1u) return sum;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= 5 ? 5 : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "K<=14 must fit in three rankstream CROSS5 chunks");
            p10dc_cross5_apply_chunk_rankstream<S2, L2, false>(
                key, state, source_row, rank_row, lbase, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream_fixed(
    uint32_t h, uint32_t rank, uint32_t key, uint32_t depth, const Count* source_row
) {
    const uint16_t* rank_row = p10dc_low_rankstream_row(h, rank);
    return p10dc_resolved_low_preimages_cross5_rankstream_key_fast(
        key, depth, source_row, rank_row);
}
