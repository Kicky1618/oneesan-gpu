#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#include "ramstream32_bucket_low_prekey.cuh"

__device__ __forceinline__ Count p10dc_direct_resolved_high_plan_sum_cross5_prekey(
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
            Count v = c.local_base[i][lr];
#if GPU_DIRECT_PM_ACCUM
            sum += uint64_t(v);
#else
            sum = gpu_direct_add(sum, v);
#endif
        }
    }
    uint32_t depth = c.cross_depth;
    if (depth) {
        size_t ix = D_BKF_LOW_CODE_OFF[
            size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + db.hs] + lr;
        uint32_t key = D_P10DC_LOW_PREKEY[ix];
#if GPU_DIRECT_PM_ACCUM
        sum += p10dc_resolved_low_preimages_cross5_prekey_indexed(
            ix, key, depth, c.cross_base);
#else
        sum = gpu_direct_add(
            sum, p10dc_resolved_low_preimages_cross5_prekey_indexed(
                     ix, key, depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}
