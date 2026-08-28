#pragma once

#include "ramstream32_bucket_closure_cross5_rankstream.cuh"
#include "ramstream32_bucket_low_rankchunk32.cuh"

#ifndef P10DC_RANKCHUNK32_FUSED16
#define P10DC_RANKCHUNK32_FUSED16 0
#endif
static_assert(P10DC_RANKCHUNK32_FUSED16 == 0 || P10DC_RANKCHUNK32_FUSED16 == 1,
              "P10DC_RANKCHUNK32_FUSED16 must be 0 or 1");

#if P10DC_RANKCHUNK32_FUSED16
#if P10DC_RANKSTREAM_LUT_LDG
__device__ __align__(128) uint16_t
    D_P10DC_RANKCHUNK32_FUSED16[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#else
__constant__ uint16_t
    D_P10DC_RANKCHUNK32_FUSED16[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#endif

static std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE>
p10dc_rankchunk32_fused16_table() {
    std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE> out{};
    for (uint32_t s = 0; s < P10DC_CROSS5_STATES; ++s) {
        for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k) {
            const uint16_t e = uint16_t(p10dc_rankstream_host_entry(k, s));
            const uint16_t meta = uint16_t(p10dc_rankstream_meta_host(k));
            out[size_t(s) * P10DC_RANKSTREAM_LUT_STRIDE + k] = e | (meta << 8);
        }
    }
    return out;
}

static void p10dc_install_rankchunk32_lut() {
    static const auto table = p10dc_rankchunk32_fused16_table();
    ck(cudaMemcpyToSymbol(D_P10DC_RANKCHUNK32_FUSED16, table.data(),
                          table.size() * sizeof(uint16_t)),
       "p10dc rankchunk32 fused16 CROSS5 table");
}

__device__ __forceinline__ uint16_t p10dc_rankchunk32_pair_load(
    uint32_t state, uint32_t chunk
) {
    const size_t ix = size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk;
#if P10DC_RANKSTREAM_LUT_LDG
    return __ldg(D_P10DC_RANKCHUNK32_FUSED16 + ix);
#else
    return D_P10DC_RANKCHUNK32_FUSED16[ix];
#endif
}
#else
static void p10dc_install_rankchunk32_lut() {
    p10dc_install_rankstream_lut();
}
#endif

// Centralized production decoder. The third CROSS5 chunk occupies only bits
// 16..22; bit 23 is already the low bit of the 9-bit rankstream prefix.
__device__ __forceinline__ uint32_t p10dc_rankchunk32_chunk_device(
    uint32_t packed_chunks, uint32_t slot
) {
    if (slot == 0u) return packed_chunks & 0xffu;
    if (slot == 1u) return (packed_chunks >> 8) & 0xffu;
    return (packed_chunks >> 16) & 0x7fu;
}

__device__ __forceinline__ uint32_t p10dc_cross5_apply_rankchunk32(
    uint32_t chunk, uint32_t& state, const Count* source_row,
    const uint16_t* rank_row, uint32_t& lbase, BkczCrossAccum& sum
) {
#if P10DC_RANKCHUNK32_FUSED16
    const uint16_t pair = p10dc_rankchunk32_pair_load(state, chunk);
    const uint8_t e = uint8_t(pair);
#else
    const uint8_t e = p10dc_rankstream_entry_load(
        size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk);
#endif
    uint8_t rankmask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    while (rankmask) {
        const int ordinal = __ffs(int(rankmask)) - 1;
        rankmask = uint8_t(rankmask & uint8_t(rankmask - 1));
        const uint16_t source_rank = rank_row[lbase + uint32_t(ordinal)];
        sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
    }
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;

#if P10DC_RANKCHUNK32_FUSED16
    const uint8_t meta = uint8_t(pair >> 8);
#else
    const uint8_t meta = p10dc_rankstream_meta_load(chunk);
#endif
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
        p10dc_rankchunk32_chunk_device(packed_chunks, 0u),
        state, source_row, rank_row, lbase, sum);
    if (st == 1u) return sum;

    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    if constexpr (S0 > 0) {
        st = p10dc_cross5_apply_rankchunk32(
            p10dc_rankchunk32_chunk_device(packed_chunks, 1u),
            state, source_row, rank_row, lbase, sum);
        if (st == 1u) return sum;
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "rankchunk32 K<=14 must fit in three chunks");
            p10dc_cross5_apply_rankchunk32(
                p10dc_rankchunk32_chunk_device(packed_chunks, 2u),
                state, source_row, rank_row, lbase, sum);
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
    // The low-rankchunk header owns the warp-striped block mapping and the
    // P10DC_RANKCHUNK32_ONESHFL A/B switch. Keep exactly one implementation so
    // packing/block-layout changes cannot drift between the two hot paths.
    p10dc_low_rankchunk32_row_warpstripe(h, rank, packed_chunks, rank_row);
    return p10dc_resolved_low_preimages_cross5_rankchunk32_fast(
        packed_chunks, depth, source_row, rank_row);
}
