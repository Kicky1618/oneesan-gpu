#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract.cuh"

#ifndef P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR
#define P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR 0
#endif
static_assert(P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR == 0 ||
              P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR == 1,
              "P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR must be 0 or 1");
#if P10DC_RANKFORMULA_CPASYNC_LOCAL_PAIR
static_assert(P10DC_RANKFORMULA_PAIR_MLP,
              "local cp.async pair requires PAIR_MLP");
static_assert(P10DC_RANKFORMULA_CPASYNC_PAIR,
              "local cp.async pair reuses the fourteen-slot cross scratch");
static_assert(BKCZ_MAX_LOCAL <= 8,
              "local cp.async pair assumes at most eight ordinary source rows");

__device__ __forceinline__ const Count* p10dc_rankformula_local_pair_src(
    const P10DCDirectHighResolvedCtx& c, uint32_t row, uint32_t rank
) {
    // cpasync_u32 ignores src when valid=false, but avoid forming a pointer from
    // a null local_base entry. ip_base is valid for every live HIGH context.
    return row < uint32_t(c.local_n) ? c.local_base[row] + rank : c.ip_base;
}

__device__ __forceinline__ void p10dc_rankformula_cpasync_wait_one_pending() {
#if __CUDA_ARCH__ >= 800
    // With exactly two committed groups this guarantees the older (column A)
    // group is complete while allowing the newer column-B group to remain in
    // flight during the A reduction below.
    asm volatile("cp.async.wait_group 1;");
#endif
}

__device__ __forceinline__ void
p10dc_direct_resolved_high_plan_sum_pair_local_cpasync(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db,
    uint32_t lr0, uint32_t lr1, Count& out0, Count& out1
) {
    BkczCrossAccum cross0 = 0, cross1 = 0;
    if (c.cross_depth) {
#if P10DC_RANKFORMULA_DIRECTGATHER64
        const auto cross =
            p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather64_pair_fixed(
                db.hs, lr0, lr1, c.cross_depth, c.cross_base);
#else
        const auto cross =
            p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_pair_fixed(
                db.hs, lr0, lr1, c.cross_depth, c.cross_base);
#endif
        cross0 = cross.a;
        cross1 = cross.b;
    }

    // The cross pair path has already waited for its cp.async groups, so its
    // fourteen shared slots can be recycled. Commit column A and B separately.
    // We then wait only for A, reduce it while B is still in flight, and wait for
    // B immediately before reading its shared slots. This turns the previous
    // wait-all bubble into useful integer reduction work.
#define P10DC_LOCAL_CPASYNC_ROW(slot,row,rank) \
    p10dc_rankformula_cpasync_u32( \
        p10dc_rankformula_cpasync_slot(slot), \
        p10dc_rankformula_local_pair_src(c, row, rank), \
        uint32_t(c.local_n) > uint32_t(row))

    P10DC_LOCAL_CPASYNC_ROW(0, 0, lr0);
    P10DC_LOCAL_CPASYNC_ROW(1, 1, lr0);
    P10DC_LOCAL_CPASYNC_ROW(2, 2, lr0);
    P10DC_LOCAL_CPASYNC_ROW(3, 3, lr0);
    P10DC_LOCAL_CPASYNC_ROW(4, 4, lr0);
    P10DC_LOCAL_CPASYNC_ROW(5, 5, lr0);
    P10DC_LOCAL_CPASYNC_ROW(6, 6, lr0);
    p10dc_rankformula_cpasync_commit();

    P10DC_LOCAL_CPASYNC_ROW(7, 0, lr1);
    P10DC_LOCAL_CPASYNC_ROW(8, 1, lr1);
    P10DC_LOCAL_CPASYNC_ROW(9, 2, lr1);
    P10DC_LOCAL_CPASYNC_ROW(10, 3, lr1);
    P10DC_LOCAL_CPASYNC_ROW(11, 4, lr1);
    P10DC_LOCAL_CPASYNC_ROW(12, 5, lr1);
    P10DC_LOCAL_CPASYNC_ROW(13, 6, lr1);
    p10dc_rankformula_cpasync_commit();
#undef P10DC_LOCAL_CPASYNC_ROW

    BkczCrossAccum a7 = 0;
    if constexpr (BKCZ_MAX_LOCAL > 7) {
        if (c.local_n > 7)
            a7 = BkczCrossAccum(__ldg(c.local_base[7] + lr0));
    }

    p10dc_rankformula_cpasync_wait_one_pending();
    const BkczCrossAccum a01 = p10dc_rankformula_accum_add(
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(0)),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(1)));
    const BkczCrossAccum a23 = p10dc_rankformula_accum_add(
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(2)),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(3)));
    const BkczCrossAccum a45 = p10dc_rankformula_accum_add(
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(4)),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(5)));
    const BkczCrossAccum alocal = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(a01, a23),
        p10dc_rankformula_accum_add(
            p10dc_rankformula_accum_add(
                BkczCrossAccum(*p10dc_rankformula_cpasync_slot(6)), a7),
            a45));

    BkczCrossAccum b7 = 0;
    if constexpr (BKCZ_MAX_LOCAL > 7) {
        if (c.local_n > 7)
            b7 = BkczCrossAccum(__ldg(c.local_base[7] + lr1));
    }
    p10dc_rankformula_cpasync_wait_all();

    const BkczCrossAccum b01 = p10dc_rankformula_accum_add(
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(7)),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(8)));
    const BkczCrossAccum b23 = p10dc_rankformula_accum_add(
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(9)),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(10)));
    const BkczCrossAccum b45 = p10dc_rankformula_accum_add(
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(11)),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(12)));
    const BkczCrossAccum blocal = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(b01, b23),
        p10dc_rankformula_accum_add(
            p10dc_rankformula_accum_add(
                BkczCrossAccum(*p10dc_rankformula_cpasync_slot(13)), b7),
            b45));

    const BkczCrossAccum sum0 = p10dc_rankformula_accum_add(alocal, cross0);
    const BkczCrossAccum sum1 = p10dc_rankformula_accum_add(blocal, cross1);
#if GPU_DIRECT_PM_ACCUM
    out0 = gpu_direct_pm_reduce_u64(sum0);
    out1 = gpu_direct_pm_reduce_u64(sum1);
#else
    out0 = Count(sum0);
    out1 = Count(sum1);
#endif
}
#endif
