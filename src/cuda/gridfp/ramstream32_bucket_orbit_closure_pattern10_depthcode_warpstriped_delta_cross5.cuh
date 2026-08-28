#pragma once

// Load the canonical resolved context first. The experimental warp kernels are
// instantiated afterwards under macro-selected plan/cross helpers, so the
// stable resolved kernels remain unchanged in the same translation unit.
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_resolved.cuh"
#include "ramstream32_bucket_closure_cross5.cuh"

__device__ __forceinline__ Count p10dc_resolved_high_plan_sum_cross5(
    const P10DCHighResolvedCtx& c,
    const BucketPhysicalBlock& db,
    uint32_t lr
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
    uint32_t depth = bkcz_plan_cross_depth(c.plan);
    if (depth) {
        uint32_t dc = D_BKF_LOW_CODES[
            D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + db.hs] + lr];
#if GPU_DIRECT_PM_ACCUM
        sum += p10dc_resolved_low_preimages_cross5(dc, depth, c.cross_base);
#else
        sum = gpu_direct_add(
            sum, p10dc_resolved_low_preimages_cross5(dc, depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}

#define p10dc_forward_high p10dc_forward_high_delta
#define p10dc_reverse_high p10dc_reverse_high_delta
#define p10dc_resolved_high_plan_sum p10dc_resolved_high_plan_sum_cross5
#define bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel \
    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_cross5_kernel
#define bucket_reverse_high_pattern10_depthcode_warpstriped_kernel \
    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_cross5_kernel
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped.cuh"
#undef bucket_reverse_high_pattern10_depthcode_warpstriped_kernel
#undef bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel
#undef p10dc_resolved_high_plan_sum
#undef p10dc_reverse_high
#undef p10dc_forward_high
