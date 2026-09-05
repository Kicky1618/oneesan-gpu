#pragma once

#include "ramstream32_bucket_closure_cross5_rankstream.cuh"
#include "ramstream32_bucket_low_rankdelta8.cuh"

#ifndef P10DC_RANKDELTA8_FUSED13
#ifdef P10DC_RANKCHUNK32_FUSED16
#define P10DC_RANKDELTA8_FUSED13 P10DC_RANKCHUNK32_FUSED16
#else
#define P10DC_RANKDELTA8_FUSED13 1
#endif
#endif
static_assert(P10DC_RANKDELTA8_FUSED13 == 0 || P10DC_RANKDELTA8_FUSED13 == 1,
              "P10DC_RANKDELTA8_FUSED13 must be 0 or 1");

static constexpr uint32_t P10DC_RANKDELTA8_PAIR_CONSUME_SHIFT = 6u;
static constexpr uint32_t P10DC_RANKDELTA8_PAIR_CONSUME_MASK = 0x7u;
static constexpr uint32_t P10DC_RANKDELTA8_PAIR_DELTA_SHIFT = 9u;
static constexpr uint32_t P10DC_RANKDELTA8_PAIR_DELTA_MASK = 0xfu;

static constexpr uint8_t p10dc_rankdelta8_consume_host(uint8_t e, uint8_t meta) {
    const bool halt = ((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0;
    if (!halt) return uint8_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
    const uint8_t rankmask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    uint8_t consume = 0;
    for (uint8_t i = 0; i < P10DC_CROSS5_CHUNK; ++i)
        if (rankmask & uint8_t(1u << i)) consume = uint8_t(i + 1u);
    return consume;
}

static constexpr uint16_t p10dc_rankdelta8_pair_host(uint32_t key, uint32_t state) {
    const uint8_t e = p10dc_rankstream_host_entry(key, state);
    const uint8_t meta = p10dc_rankstream_meta_host(key);
    const uint32_t consume = p10dc_rankdelta8_consume_host(e, meta);
    const uint32_t delta_bias = uint32_t(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT);
    return uint16_t(
        uint32_t(e & uint8_t(P10DC_CROSS5_MASK_MASK | (1u << P10DC_CROSS5_HALT_SHIFT))) |
        (consume << P10DC_RANKDELTA8_PAIR_CONSUME_SHIFT) |
        (delta_bias << P10DC_RANKDELTA8_PAIR_DELTA_SHIFT));
}
static_assert(P10DC_CROSS5_MASK_MASK == 0x1fu && P10DC_CROSS5_HALT_SHIFT == 5);
static_assert(P10DC_RANKDELTA8_PAIR_CONSUME_SHIFT == 6u);
static_assert(P10DC_RANKDELTA8_PAIR_DELTA_SHIFT == 9u);
static_assert(P10DC_RANKSTREAM_META_DELTA_BIAS == 5);

#if P10DC_RANKDELTA8_FUSED13
#if P10DC_RANKSTREAM_LUT_LDG
__device__ __align__(128) uint16_t
    D_P10DC_RANKDELTA8_FUSED13[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#else
__constant__ uint16_t
    D_P10DC_RANKDELTA8_FUSED13[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#endif

static std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE>
p10dc_rankdelta8_fused13_table() {
    std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE> out{};
    for (uint32_t s = 0; s < P10DC_CROSS5_STATES; ++s)
        for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k)
            out[size_t(s) * P10DC_RANKSTREAM_LUT_STRIDE + k] =
                p10dc_rankdelta8_pair_host(k, s);
    return out;
}

static void p10dc_install_rankdelta8_lut() {
    static const auto table = p10dc_rankdelta8_fused13_table();
    ck(cudaMemcpyToSymbol(D_P10DC_RANKDELTA8_FUSED13, table.data(),
                          table.size() * sizeof(uint16_t)),
       "p10dc rankdelta8 fused13 CROSS5 table");
}

__device__ __forceinline__ uint16_t p10dc_rankdelta8_pair_load(
    uint32_t state, uint32_t chunk
) {
    const size_t ix = size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk;
#if P10DC_RANKSTREAM_LUT_LDG
    return __ldg(D_P10DC_RANKDELTA8_FUSED13 + ix);
#else
    return D_P10DC_RANKDELTA8_FUSED13[ix];
#endif
}
#else
static void p10dc_install_rankdelta8_lut() { p10dc_install_rankstream_lut(); }
#endif

struct P10DCRankDelta8Cursor {
    const uint8_t* p = nullptr;
    uint32_t rank = 0;
    uint32_t started = 0;
};

__device__ __forceinline__ uint32_t p10dc_rankdelta8_next(P10DCRankDelta8Cursor& c) {
    if (!c.started) {
        c.rank = uint32_t(c.p[0]) | (uint32_t(c.p[1]) << 8);
        c.p += 2;
        c.started = 1;
        return c.rank;
    }
    const uint32_t b0 = uint32_t(*c.p++);
    uint32_t delta = b0;
    if (b0 == 0u) {
        delta = uint32_t(c.p[0]) | (uint32_t(c.p[1]) << 8);
        c.p += 2;
    }
    c.rank += delta;
    return c.rank;
}

__device__ __forceinline__ uint32_t p10dc_rankdelta8_chunk_device(
    uint32_t packed_chunks, uint32_t slot
) {
    if (slot == 0u) return packed_chunks & 0xffu;
    if (slot == 1u) return (packed_chunks >> 8) & 0xffu;
    return (packed_chunks >> 16) & 0x7fu;
}

__device__ __forceinline__ uint32_t p10dc_cross5_apply_rankdelta8(
    uint32_t chunk, uint32_t& state, const Count* source_row,
    P10DCRankDelta8Cursor& cursor, BkczCrossAccum& sum
) {
    uint32_t rankmask, consume, delta_bias;
    bool halt;
#if P10DC_RANKDELTA8_FUSED13
    const uint32_t pair = uint32_t(p10dc_rankdelta8_pair_load(state, chunk));
    rankmask = pair & P10DC_CROSS5_MASK_MASK;
    halt = ((pair >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0;
    consume = (pair >> P10DC_RANKDELTA8_PAIR_CONSUME_SHIFT) &
              P10DC_RANKDELTA8_PAIR_CONSUME_MASK;
    delta_bias = (pair >> P10DC_RANKDELTA8_PAIR_DELTA_SHIFT) &
                 P10DC_RANKDELTA8_PAIR_DELTA_MASK;
#else
    const uint8_t e = p10dc_rankstream_entry_load(
        size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk);
    const uint8_t meta = p10dc_rankstream_meta_load(chunk);
    rankmask = uint32_t(e & P10DC_CROSS5_MASK_MASK);
    halt = ((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0;
    consume = halt
        ? (rankmask ? 32u - uint32_t(__clz(rankmask)) : 0u)
        : uint32_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
    delta_bias = uint32_t(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT);
#endif
#pragma unroll
    for (uint32_t ordinal = 0; ordinal < P10DC_CROSS5_CHUNK; ++ordinal) {
        if (ordinal < consume) {
            const uint32_t source_rank = p10dc_rankdelta8_next(cursor);
            if (rankmask & (1u << ordinal))
                sum = bkcz_cross_add(sum, source_row[source_rank]);
        }
    }
    if (halt) return 1u;
    state = uint32_t(
        int(state) + int(delta_bias) - P10DC_RANKSTREAM_META_DELTA_BIAS);
    return 0u;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankdelta8_fast(
    uint32_t packed_chunks, uint32_t depth, const Count* source_row,
    const uint8_t* rank_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth;
    BkczCrossAccum sum = 0;
    P10DCRankDelta8Cursor cursor{rank_row, 0u, 0u};

    uint32_t st = p10dc_cross5_apply_rankdelta8(
        p10dc_rankdelta8_chunk_device(packed_chunks, 0u),
        state, source_row, cursor, sum);
    if (st == 1u) return sum;

    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    if constexpr (S0 > 0) {
        st = p10dc_cross5_apply_rankdelta8(
            p10dc_rankdelta8_chunk_device(packed_chunks, 1u),
            state, source_row, cursor, sum);
        if (st == 1u) return sum;
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "rankdelta8 K<=14 must fit in three chunks");
            p10dc_cross5_apply_rankdelta8(
                p10dc_rankdelta8_chunk_device(packed_chunks, 2u),
                state, source_row, cursor, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankdelta8_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    uint32_t packed_chunks = 0;
    const uint8_t* rank_row = nullptr;
    p10dc_low_rankdelta8_row_warpstripe(h, rank, packed_chunks, rank_row);
    return p10dc_resolved_low_preimages_cross5_rankdelta8_fast(
        packed_chunks, depth, source_row, rank_row);
}
