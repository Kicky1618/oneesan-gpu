#pragma once

// rankchunk32 uses the sparse-rank CROSS5 executor and predecoded base-243
// chunks.  Suppress the ordinary CROSS5 dependency while instantiating the
// shared direct/affine resolver so only one automaton LUT is resident.
#define P10DC_DIRECT_RESOLVED_NO_CROSS5 1
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#undef P10DC_DIRECT_RESOLVED_NO_CROSS5
#include "ramstream32_bucket_closure_cross5_rankchunk32.cuh"

#ifdef P10DC_CROSS5_ORDINARY_LUT_DEFINED
#error "rankchunk32 CROSS5 variant must not pull in the ordinary CROSS5 device LUT"
#endif

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
