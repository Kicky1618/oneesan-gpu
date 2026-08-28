#pragma once

#include "ramstream32_bucket_closure_cross5_rankchunk32.cuh"
#include "ramstream32_bucket_low_rankdelta8.cuh"

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
    uint32_t delta = b0 & 0x7fu;
    if (b0 & 0x80u) delta |= uint32_t(*c.p++) << 7;
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
#if P10DC_RANKCHUNK32_FUSED16
    const uint16_t pair = p10dc_rankchunk32_pair_load(state, chunk);
    const uint8_t e = uint8_t(pair);
    const uint8_t meta = uint8_t(pair >> 8);
#else
    const uint8_t e = p10dc_rankstream_entry_load(
        size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk);
    const uint8_t meta = p10dc_rankstream_meta_load(chunk);
#endif
    const uint32_t lcount = uint32_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
    const uint32_t rankmask = uint32_t(e & P10DC_CROSS5_MASK_MASK);
    const bool halt = ((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0;
    const uint32_t consume = halt
        ? (rankmask ? 32u - uint32_t(__clz(rankmask)) : 0u)
        : lcount;
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
        int(state) + int(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT)
        - P10DC_RANKSTREAM_META_DELTA_BIAS);
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
