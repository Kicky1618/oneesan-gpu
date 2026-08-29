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
#if P10DC_RANKFORMULA_CPASYNC_PAIR
static_assert(P10DC_RANKFORMULA_CPASYNC_VALUES_PER_THREAD >= 28,
              "quad local cp.async requires 28 source slots per thread");
#endif

__device__ __forceinline__ const Count* p10dc_rankformula_quad_local_src(
    const P10DCDirectHighResolvedCtx& c, uint32_t row, uint32_t rank
) {
    return row < uint32_t(c.local_n) ? c.local_base[row] + rank : c.ip_base;
}

#if P10DC_RANKFORMULA_CPASYNC_PAIR
__device__ __forceinline__ void p10dc_rankformula_quad_wait3() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 3;");
#endif
}
__device__ __forceinline__ void p10dc_rankformula_quad_wait2() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 2;");
#endif
}
__device__ __forceinline__ void p10dc_rankformula_quad_wait1() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;");
#endif
}
#endif

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

#if P10DC_RANKFORMULA_CPASYNC_PAIR
    // CROSS quad has already waited, so recycle all 28 slots. Commit one group
    // per column. wait_group 3/2/1 lets each completed column be reduced while
    // the remaining columns are still in flight instead of paying a wait-all
    // bubble before doing any useful arithmetic.
#define P10DC_QUAD_LOCAL_CPASYNC(slot,row,rank) \
    p10dc_rankformula_cpasync_u32( \
        p10dc_rankformula_cpasync_slot(slot), \
        p10dc_rankformula_quad_local_src(c, row, rank), \
        uint32_t(c.local_n) > uint32_t(row))

    P10DC_QUAD_LOCAL_CPASYNC(0,0,lr0); P10DC_QUAD_LOCAL_CPASYNC(1,1,lr0); P10DC_QUAD_LOCAL_CPASYNC(2,2,lr0); P10DC_QUAD_LOCAL_CPASYNC(3,3,lr0); P10DC_QUAD_LOCAL_CPASYNC(4,4,lr0); P10DC_QUAD_LOCAL_CPASYNC(5,5,lr0); P10DC_QUAD_LOCAL_CPASYNC(6,6,lr0); p10dc_rankformula_cpasync_commit();
    P10DC_QUAD_LOCAL_CPASYNC(7,0,lr1); P10DC_QUAD_LOCAL_CPASYNC(8,1,lr1); P10DC_QUAD_LOCAL_CPASYNC(9,2,lr1); P10DC_QUAD_LOCAL_CPASYNC(10,3,lr1); P10DC_QUAD_LOCAL_CPASYNC(11,4,lr1); P10DC_QUAD_LOCAL_CPASYNC(12,5,lr1); P10DC_QUAD_LOCAL_CPASYNC(13,6,lr1); p10dc_rankformula_cpasync_commit();
    P10DC_QUAD_LOCAL_CPASYNC(14,0,lr2); P10DC_QUAD_LOCAL_CPASYNC(15,1,lr2); P10DC_QUAD_LOCAL_CPASYNC(16,2,lr2); P10DC_QUAD_LOCAL_CPASYNC(17,3,lr2); P10DC_QUAD_LOCAL_CPASYNC(18,4,lr2); P10DC_QUAD_LOCAL_CPASYNC(19,5,lr2); P10DC_QUAD_LOCAL_CPASYNC(20,6,lr2); p10dc_rankformula_cpasync_commit();
    P10DC_QUAD_LOCAL_CPASYNC(21,0,lr3); P10DC_QUAD_LOCAL_CPASYNC(22,1,lr3); P10DC_QUAD_LOCAL_CPASYNC(23,2,lr3); P10DC_QUAD_LOCAL_CPASYNC(24,3,lr3); P10DC_QUAD_LOCAL_CPASYNC(25,4,lr3); P10DC_QUAD_LOCAL_CPASYNC(26,5,lr3); P10DC_QUAD_LOCAL_CPASYNC(27,6,lr3); p10dc_rankformula_cpasync_commit();
#undef P10DC_QUAD_LOCAL_CPASYNC

    BkczCrossAccum r70=0,r71=0,r72=0,r73=0;
    if constexpr (BKCZ_MAX_LOCAL > 7) {
        if (c.local_n > 7) {
            r70=BkczCrossAccum(__ldg(c.local_base[7]+lr0));
            r71=BkczCrossAccum(__ldg(c.local_base[7]+lr1));
            r72=BkczCrossAccum(__ldg(c.local_base[7]+lr2));
            r73=BkczCrossAccum(__ldg(c.local_base[7]+lr3));
        }
    }
#define P10DC_QUAD_LOCAL_SLOT(i) BkczCrossAccum(*p10dc_rankformula_cpasync_slot(i))
#define P10DC_QUAD_LOCAL_REDUCE(base,r7) \
    p10dc_rankformula_accum_add( \
        p10dc_rankformula_accum_add( \
            p10dc_rankformula_accum_add(P10DC_QUAD_LOCAL_SLOT(base),P10DC_QUAD_LOCAL_SLOT((base)+1)), \
            p10dc_rankformula_accum_add(P10DC_QUAD_LOCAL_SLOT((base)+2),P10DC_QUAD_LOCAL_SLOT((base)+3))), \
        p10dc_rankformula_accum_add( \
            p10dc_rankformula_accum_add(P10DC_QUAD_LOCAL_SLOT((base)+4),P10DC_QUAD_LOCAL_SLOT((base)+5)), \
            p10dc_rankformula_accum_add(P10DC_QUAD_LOCAL_SLOT((base)+6),(r7))))

    p10dc_rankformula_quad_wait3();
    s0=p10dc_rankformula_accum_add(s0,P10DC_QUAD_LOCAL_REDUCE(0,r70));
    p10dc_rankformula_quad_wait2();
    s1=p10dc_rankformula_accum_add(s1,P10DC_QUAD_LOCAL_REDUCE(7,r71));
    p10dc_rankformula_quad_wait1();
    s2=p10dc_rankformula_accum_add(s2,P10DC_QUAD_LOCAL_REDUCE(14,r72));
    p10dc_rankformula_cpasync_wait_all();
    s3=p10dc_rankformula_accum_add(s3,P10DC_QUAD_LOCAL_REDUCE(21,r73));
#undef P10DC_QUAD_LOCAL_REDUCE
#undef P10DC_QUAD_LOCAL_SLOT
#else
    // Register quad: issue all four ordinary-source loads for each row before
    // consuming any of them.
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
#endif

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
