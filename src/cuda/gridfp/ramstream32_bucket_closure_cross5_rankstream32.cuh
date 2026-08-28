#pragma once

#include "ramstream32_bucket_closure_cross5_rankstream.cuh"
#include "ramstream32_bucket_low_rankstream32.cuh"

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream32_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    uint32_t key = 0;
    const uint16_t* rank_row = nullptr;
    p10dc_low_rankstream32_row(h, rank, key, rank_row);
    bool overflow = false;
    BkczCrossAccum sum = p10dc_resolved_low_preimages_cross5_rankstream_nofallback(
        key, depth, source_row, rank_row, overflow);
    if (!overflow) return sum;
    size_t ix = D_BKF_LOW_CODE_OFF[
        size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + h] + rank;
    uint32_t dc = D_BKF_LOW_CODES[ix];
    return p10dc_resolved_low_preimages_cross5_fallback_prekey(
        dc, key, depth, source_row);
}
