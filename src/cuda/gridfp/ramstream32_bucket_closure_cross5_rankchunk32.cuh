#pragma once

#include "ramstream32_bucket_closure_cross5_rankstream.cuh"
#include "ramstream32_bucket_low_rankchunk32.cuh"

// Height-local rankchunk metadata is 32-entry aligned.  A warp stripe therefore
// covers at most the two fixed 16-entry blocks sourced by lanes 0 and 16.
// Each source lane loads one block base and every active lane selects its half
// with a single variable-source shuffle.
__device__ __forceinline__ void p10dc_low_rankchunk32_row_warpstripe_oneshfl(
    uint32_t h, uint32_t rank, uint32_t& packed_chunks, const uint16_t*& row
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t compact = D_P10DC_LOW_RANKCHUNK_HOFF[h] + rank;
    const uint32_t meta = D_P10DC_LOW_RANKCHUNKMETA32[compact];
    packed_chunks = meta & P10DC_RANKCHUNK32_CHUNK_MASK;
    const uint32_t prefix = meta >> P10DC_RANKCHUNK32_CHUNK_BITS;

    const uint32_t first_block = (compact - lane) >> P10DC_RANKCHUNK32_BLOCK_LOG2;
    const uint32_t source_lane = lane & P10DC_RANKCHUNK32_BLOCK;
    uint32_t local_base = 0;
    if ((lane & (P10DC_RANKCHUNK32_BLOCK - 1u)) == 0u)
        local_base = D_P10DC_LOW_RANKCHUNKBLOCK16[
            first_block + (lane >> P10DC_RANKCHUNK32_BLOCK_LOG2)];
    const uint32_t block_base = __shfl_sync(active, local_base, int(source_lane));
    row = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
}

// Chunk-value form of the sparse-rank automaton.  The caller supplies an
// already-decoded base-243 chunk, so there is no integer division or modulo in
// the device hot path.  Valid depth4 input and K<=14 keep state in [1,25].
__device__ __forceinline__ uint32_t p10dc_cross5_apply_rankchunk32(
    uint32_t chunk, uint32_t& state, const Count* source_row,
    const uint16_t* rank_row, uint32_t& lbase, BkczCrossAccum& sum
) {
    const uint8_t e = p10dc_rankstream_entry_load(
        size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk);
    uint8_t rankmask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    while (rankmask) {
        const int ordinal = __ffs(int(rankmask)) - 1;
        rankmask = uint8_t(rankmask & uint8_t(rankmask - 1));
        const uint16_t source_rank = rank_row[lbase + uint32_t(ordinal)];
        sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
    }
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;

    const uint8_t meta = p10dc_rankstream_meta_load(chunk);
    lbase += uint32_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
    state = uint32_t(
        int(state) + int(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT)
        - P10DC_RANKSTREAM_META_DELTA_BIAS);
    return 0u;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankchunk32_fast(
    uint32_t packed_chunks, uint32_t depth, const Count* source_row,
    const uint16_t* rank_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth, lbase = 0;
    BkczCrossAccum sum = 0;

    uint32_t st = p10dc_cross5_apply_rankchunk32(
        packed_chunks & 0xffu, state, source_row, rank_row, lbase, sum);
    if (st == 1u) return sum;

    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    if constexpr (S0 > 0) {
        st = p10dc_cross5_apply_rankchunk32(
            (packed_chunks >> 8) & 0xffu, state, source_row, rank_row, lbase, sum);
        if (st == 1u) return sum;
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "rankchunk32 K<=14 must fit in three chunks");
            p10dc_cross5_apply_rankchunk32(
                (packed_chunks >> 16) & 0xffu, state, source_row, rank_row, lbase, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankchunk32_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    uint32_t packed_chunks = 0;
    const uint16_t* rank_row = nullptr;
    p10dc_low_rankchunk32_row_warpstripe_oneshfl(h, rank, packed_chunks, rank_row);
    return p10dc_resolved_low_preimages_cross5_rankchunk32_fast(
        packed_chunks, depth, source_row, rank_row);
}
