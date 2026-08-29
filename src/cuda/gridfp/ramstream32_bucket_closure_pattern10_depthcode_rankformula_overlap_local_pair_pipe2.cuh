#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_rankformula_overlap_local_pair.cuh"

#ifndef P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2
#define P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 0
#endif
static_assert(P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 == 0 ||
              P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 == 1,
              "P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2 must be 0 or 1");

#if P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PIPE2
static_assert(P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR,
              "overlap-local pipe2 requires overlap-local pair mode");
static_assert(P10DC_RANKFORMULA_CPASYNC_PAIR,
              "overlap-local pipe2 requires cp.async pair support");
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64,
              "overlap-local pipe2 currently targets DIRECTGATHER64");
static_assert(BKCZ_MAX_LOCAL <= 8,
              "overlap-local pipe2 assumes at most eight ordinary source rows");

__device__ __forceinline__ void p10dc_overlap_local_wait_one_pending() {
#if __CUDA_ARCH__ >= 800
    // Two groups are committed in A-then-B order. Leave B outstanding while
    // consuming A so arithmetic covers part of the second group's latency.
    asm volatile("cp.async.wait_group 1;");
#endif
}

__device__ __forceinline__ void
p10dc_direct_resolved_high_plan_sum_pair_overlap_local_pipe2(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db,
    uint32_t lr0, uint32_t lr1, Count& out0, Count& out1
) {
    P10DCDirectGatherPairRanks r{};
    const bool have_cross = c.cross_depth != 0u;
    if (have_cross)
        r = p10dc_rankformula_directgather64_pair_ranks(
            db.hs, lr0, lr1, c.cross_depth);

    if (have_cross) {
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(0), c.cross_base + r.a0, r.n0 > 0u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(1), c.cross_base + r.a1, r.n0 > 1u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(2), c.cross_base + r.a2, r.n0 > 2u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(3), c.cross_base + r.a3, r.n0 > 3u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(4), c.cross_base + r.a4, r.n0 > 4u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(5), c.cross_base + r.a5, r.n0 > 5u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(6), c.cross_base + r.a6, r.n0 > 6u);
        p10dc_rankformula_cpasync_commit();

        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(7), c.cross_base + r.b0, r.n1 > 0u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(8), c.cross_base + r.b1, r.n1 > 1u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(9), c.cross_base + r.b2, r.n1 > 2u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(10), c.cross_base + r.b3, r.n1 > 3u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(11), c.cross_base + r.b4, r.n1 > 4u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(12), c.cross_base + r.b5, r.n1 > 5u);
        p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(13), c.cross_base + r.b6, r.n1 > 6u);
        p10dc_rankformula_cpasync_commit();
    }

    // Keep only one column's ordinary values live at a time. Compared with the
    // 30-request overlap-local mode this deliberately trades some peak MLP for a
    // much shorter register live range, which can admit more resident warps.
    BkczCrossAccum a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    if constexpr (BKCZ_MAX_LOCAL > 0) if (c.local_n > 0) a0=__ldg(c.local_base[0]+lr0);
    if constexpr (BKCZ_MAX_LOCAL > 1) if (c.local_n > 1) a1=__ldg(c.local_base[1]+lr0);
    if constexpr (BKCZ_MAX_LOCAL > 2) if (c.local_n > 2) a2=__ldg(c.local_base[2]+lr0);
    if constexpr (BKCZ_MAX_LOCAL > 3) if (c.local_n > 3) a3=__ldg(c.local_base[3]+lr0);
    if constexpr (BKCZ_MAX_LOCAL > 4) if (c.local_n > 4) a4=__ldg(c.local_base[4]+lr0);
    if constexpr (BKCZ_MAX_LOCAL > 5) if (c.local_n > 5) a5=__ldg(c.local_base[5]+lr0);
    if constexpr (BKCZ_MAX_LOCAL > 6) if (c.local_n > 6) a6=__ldg(c.local_base[6]+lr0);
    if constexpr (BKCZ_MAX_LOCAL > 7) if (c.local_n > 7) a7=__ldg(c.local_base[7]+lr0);

    if (have_cross) p10dc_overlap_local_wait_one_pending();

    BkczCrossAccum cross0=0;
    if (have_cross) {
        const BkczCrossAccum ca01=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(0),*p10dc_rankformula_cpasync_slot(1));
        const BkczCrossAccum ca23=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(2),*p10dc_rankformula_cpasync_slot(3));
        const BkczCrossAccum ca45=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(4),*p10dc_rankformula_cpasync_slot(5));
        cross0=p10dc_rankformula_accum_add(
            p10dc_rankformula_accum_add(ca01,ca23),
            p10dc_rankformula_accum_add(ca45,*p10dc_rankformula_cpasync_slot(6)));
    }
    const BkczCrossAccum la01=p10dc_rankformula_accum_add(a0,a1);
    const BkczCrossAccum la23=p10dc_rankformula_accum_add(a2,a3);
    const BkczCrossAccum la45=p10dc_rankformula_accum_add(a4,a5);
    const BkczCrossAccum la67=p10dc_rankformula_accum_add(a6,a7);
    const BkczCrossAccum sum0=p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(la01,la23),
        p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(la45,la67),cross0));

    BkczCrossAccum b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0,b7=0;
    if constexpr (BKCZ_MAX_LOCAL > 0) if (c.local_n > 0) b0=__ldg(c.local_base[0]+lr1);
    if constexpr (BKCZ_MAX_LOCAL > 1) if (c.local_n > 1) b1=__ldg(c.local_base[1]+lr1);
    if constexpr (BKCZ_MAX_LOCAL > 2) if (c.local_n > 2) b2=__ldg(c.local_base[2]+lr1);
    if constexpr (BKCZ_MAX_LOCAL > 3) if (c.local_n > 3) b3=__ldg(c.local_base[3]+lr1);
    if constexpr (BKCZ_MAX_LOCAL > 4) if (c.local_n > 4) b4=__ldg(c.local_base[4]+lr1);
    if constexpr (BKCZ_MAX_LOCAL > 5) if (c.local_n > 5) b5=__ldg(c.local_base[5]+lr1);
    if constexpr (BKCZ_MAX_LOCAL > 6) if (c.local_n > 6) b6=__ldg(c.local_base[6]+lr1);
    if constexpr (BKCZ_MAX_LOCAL > 7) if (c.local_n > 7) b7=__ldg(c.local_base[7]+lr1);

    if (have_cross) p10dc_rankformula_cpasync_wait_all();

    BkczCrossAccum cross1=0;
    if (have_cross) {
        const BkczCrossAccum cb01=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(7),*p10dc_rankformula_cpasync_slot(8));
        const BkczCrossAccum cb23=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(9),*p10dc_rankformula_cpasync_slot(10));
        const BkczCrossAccum cb45=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(11),*p10dc_rankformula_cpasync_slot(12));
        cross1=p10dc_rankformula_accum_add(
            p10dc_rankformula_accum_add(cb01,cb23),
            p10dc_rankformula_accum_add(cb45,*p10dc_rankformula_cpasync_slot(13)));
    }
    const BkczCrossAccum lb01=p10dc_rankformula_accum_add(b0,b1);
    const BkczCrossAccum lb23=p10dc_rankformula_accum_add(b2,b3);
    const BkczCrossAccum lb45=p10dc_rankformula_accum_add(b4,b5);
    const BkczCrossAccum lb67=p10dc_rankformula_accum_add(b6,b7);
    const BkczCrossAccum sum1=p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(lb01,lb23),
        p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(lb45,lb67),cross1));

#if GPU_DIRECT_PM_ACCUM
    out0=gpu_direct_pm_reduce_u64(sum0);
    out1=gpu_direct_pm_reduce_u64(sum1);
#else
    out0=Count(sum0);
    out1=Count(sum1);
#endif
}
#endif
