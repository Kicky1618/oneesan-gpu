#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract.cuh"
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64_quad.cuh"

#ifndef P10DC_RANKFORMULA_QUAD_MLP
#define P10DC_RANKFORMULA_QUAD_MLP 0
#endif

#if P10DC_RANKFORMULA_QUAD_MLP
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64,
              "quad HIGH plan sum requires DIRECTGATHER64");
static_assert(P10DC_RANKFORMULA_PAIR_MLP,
              "quad HIGH plan sum requires PAIR_MLP fallback for tails");
static_assert(P10DC_WARPSTRIPED_COL_ILP == 4,
              "quad HIGH plan sum requires COL_ILP=4");

__device__ __forceinline__ void
p10dc_direct_resolved_high_plan_sum_quad_cross5_rankformula_nometa4_abstract(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db,
    uint32_t lr0, uint32_t lr1, uint32_t lr2, uint32_t lr3,
    Count& out0, Count& out1, Count& out2, Count& out3
) {
    BkczCrossAccum s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    if (c.cross_depth) {
        const auto cross =
            p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather64_quad_fixed(
                db.hs, lr0, lr1, lr2, lr3, c.cross_depth, c.cross_base);
        s0 = cross.a; s1 = cross.b; s2 = cross.c; s3 = cross.d;
    }

    // Four independent ordinary-source loads are issued for each row before
    // accumulation. This keeps register pressure bounded while exposing more
    // memory-level parallelism than two serial pair helpers.
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) {
        if (i < uint32_t(c.local_n)) {
            const Count* row = c.local_base[i];
            const BkczCrossAccum a = BkczCrossAccum(__ldg(row + lr0));
            const BkczCrossAccum b = BkczCrossAccum(__ldg(row + lr1));
            const BkczCrossAccum d = BkczCrossAccum(__ldg(row + lr2));
            const BkczCrossAccum e = BkczCrossAccum(__ldg(row + lr3));
            s0 = p10dc_rankformula_accum_add(s0, a);
            s1 = p10dc_rankformula_accum_add(s1, b);
            s2 = p10dc_rankformula_accum_add(s2, d);
            s3 = p10dc_rankformula_accum_add(s3, e);
        }
    }

#if GPU_DIRECT_PM_ACCUM
    out0 = gpu_direct_pm_reduce_u64(s0);
    out1 = gpu_direct_pm_reduce_u64(s1);
    out2 = gpu_direct_pm_reduce_u64(s2);
    out3 = gpu_direct_pm_reduce_u64(s3);
#else
    out0 = Count(s0); out1 = Count(s1); out2 = Count(s2); out3 = Count(s3);
#endif
}
#endif
