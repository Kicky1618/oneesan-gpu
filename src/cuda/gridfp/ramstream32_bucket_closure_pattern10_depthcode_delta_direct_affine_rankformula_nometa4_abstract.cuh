#pragma once

#define P10DC_DIRECT_RESOLVED_NO_CROSS5 1
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#undef P10DC_DIRECT_RESOLVED_NO_CROSS5

#ifndef P10DC_RANKFORMULA_GATHER_MLP
#define P10DC_RANKFORMULA_GATHER_MLP 0
#endif
static_assert(P10DC_RANKFORMULA_GATHER_MLP == 0 || P10DC_RANKFORMULA_GATHER_MLP == 1,
              "P10DC_RANKFORMULA_GATHER_MLP must be 0 or 1");
#if P10DC_RANKFORMULA_GATHER_MLP
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_mlp.cuh"
#else
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract.cuh"
#endif

#ifdef P10DC_CROSS5_ORDINARY_LUT_DEFINED
#error "rankformula nometa4 abstract variant must not pull in ordinary CROSS5 device LUT"
#endif

#ifndef P10DC_SPARSE_CROSS5_INSTALL_COMPAT_DEFINED
#define P10DC_SPARSE_CROSS5_INSTALL_COMPAT_DEFINED 1
static inline void p10dc_install_cross5_lut() { p10dc_install_rankformula_abstract_lut(); }
#endif

__device__ __forceinline__ Count p10dc_direct_resolved_high_plan_sum_cross5_rankformula_nometa4_abstract(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db, uint32_t lr
) {
#if P10DC_RANKFORMULA_GATHER_MLP
    // First issue all ordinary source-row reads. Do not immediately consume
    // them: run the independent cross resolver/gather while those HBM requests
    // are outstanding, then reduce everything at the end.
    BkczCrossAccum v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    BkczCrossAccum v4 = 0, v5 = 0, v6 = 0, v7 = 0;
    if constexpr (BKCZ_MAX_LOCAL > 0) if (c.local_n > 0) v0 = BkczCrossAccum(c.local_base[0][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 1) if (c.local_n > 1) v1 = BkczCrossAccum(c.local_base[1][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 2) if (c.local_n > 2) v2 = BkczCrossAccum(c.local_base[2][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 3) if (c.local_n > 3) v3 = BkczCrossAccum(c.local_base[3][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 4) if (c.local_n > 4) v4 = BkczCrossAccum(c.local_base[4][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 5) if (c.local_n > 5) v5 = BkczCrossAccum(c.local_base[5][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 6) if (c.local_n > 6) v6 = BkczCrossAccum(c.local_base[6][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 7) if (c.local_n > 7) v7 = BkczCrossAccum(c.local_base[7][lr]);

    BkczCrossAccum cross = 0;
    if (c.cross_depth) {
        cross = p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
            db.hs, lr, c.cross_depth, c.cross_base);
    }

    const BkczCrossAccum a01 = p10dc_rankformula_accum_add(v0, v1);
    const BkczCrossAccum a23 = p10dc_rankformula_accum_add(v2, v3);
    const BkczCrossAccum a45 = p10dc_rankformula_accum_add(v4, v5);
    const BkczCrossAccum a67 = p10dc_rankformula_accum_add(v6, v7);
    const BkczCrossAccum local = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(a01, a23),
        p10dc_rankformula_accum_add(a45, a67));
    const BkczCrossAccum sum = p10dc_rankformula_accum_add(local, cross);
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
#else
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
        sum += p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_fixed(
            db.hs, lr, c.cross_depth, c.cross_base);
#else
        sum = gpu_direct_add(sum,
            p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_fixed(
                db.hs, lr, c.cross_depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
#endif
}
