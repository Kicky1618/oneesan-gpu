#pragma once

#include "ramstream32_bucket_closure_cross5_rankstream.cuh"
#include "ramstream32_bucket_low_rankstream32.cuh"

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream32_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    uint32_t key = 0;
    const uint16_t* rank_row = nullptr;
    p10dc_low_rankstream32_row_warpstripe(h, rank, key, rank_row);
    // Production depthcode is four bits and LOW_LUT_K<=14.  The shared
    // rankstream proof bounds chunk-start states by 15,20,25, so the checked
    // executor and scalar fallback are dead here just as in prekey rankstream.
    return p10dc_resolved_low_preimages_cross5_rankstream_key_fast(
        key, depth, source_row, rank_row);
}
