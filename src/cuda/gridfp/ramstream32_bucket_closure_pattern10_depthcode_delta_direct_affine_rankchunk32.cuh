#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#include "ramstream32_bucket_closure_cross5_rankchunk32.cuh"

__device__ __forceinline__ Count p10dc_direct_resolved_high_plan_sum_cross5_rankchunk32(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db, uint32_t lr
) {
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum = 0;
#else
    Count sum = 0;
#endif
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) {
        if (i < c.local_n) {
            const Count v = c.local_base[i][lr];
#if GPU_DIRECT_PM_ACCUM
            sum += uint64_t(v);
#else
            sum = gpu_direct_add(sum, v);
#endif
        }
    }
    if (c.cross_depth) {
#if GPU_DIRECT_PM_ACCUM
        sum += p10dc_resolved_low_preimages_cross5_rankchunk32_fixed(
            db.hs, lr, c.cross_depth, c.cross_base);
#else
        sum = gpu_direct_add(
            sum, p10dc_resolved_low_preimages_cross5_rankchunk32_fixed(
                     db.hs, lr, c.cross_depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}
