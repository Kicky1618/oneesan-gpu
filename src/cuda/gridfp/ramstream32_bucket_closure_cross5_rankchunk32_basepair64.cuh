#pragma once

#include "ramstream32_bucket_closure_cross5_rankchunk32.cuh"
#include "ramstream32_bucket_low_rankchunk32_basepair64.cuh"

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankchunk32_basepair64_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    uint32_t packed_chunks = 0;
    const uint16_t* rank_row = nullptr;
    p10dc_low_rankchunk32_basepair64_row_warpstripe(
        h, rank, packed_chunks, rank_row);
    return p10dc_resolved_low_preimages_cross5_rankchunk32_fast(
        packed_chunks, depth, source_row, rank_row);
}
