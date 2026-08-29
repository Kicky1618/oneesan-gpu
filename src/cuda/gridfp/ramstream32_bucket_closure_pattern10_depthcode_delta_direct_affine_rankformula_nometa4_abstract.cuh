#pragma once

#define P10DC_DIRECT_RESOLVED_NO_CROSS5 1
#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#undef P10DC_DIRECT_RESOLVED_NO_CROSS5

#ifndef P10DC_RANKFORMULA_GATHER_MLP
#define P10DC_RANKFORMULA_GATHER_MLP 0
#endif
#ifndef P10DC_RANKFORMULA_PREFETCH_NEXT
#define P10DC_RANKFORMULA_PREFETCH_NEXT 0
#endif
#ifndef P10DC_RANKFORMULA_PAIR_MLP
#define P10DC_RANKFORMULA_PAIR_MLP 0
#endif
#ifndef P10DC_RANKFORMULA_DIRECTGATHER64
#define P10DC_RANKFORMULA_DIRECTGATHER64 0
#endif
#ifndef P10DC_WARPSTRIPED_COL_ILP
#define P10DC_WARPSTRIPED_COL_ILP 1
#endif
static_assert(P10DC_RANKFORMULA_GATHER_MLP == 0 || P10DC_RANKFORMULA_GATHER_MLP == 1,
              "P10DC_RANKFORMULA_GATHER_MLP must be 0 or 1");
static_assert(P10DC_RANKFORMULA_PREFETCH_NEXT == 0 || P10DC_RANKFORMULA_PREFETCH_NEXT == 1,
              "P10DC_RANKFORMULA_PREFETCH_NEXT must be 0 or 1");
static_assert(P10DC_RANKFORMULA_PAIR_MLP == 0 || P10DC_RANKFORMULA_PAIR_MLP == 1,
              "P10DC_RANKFORMULA_PAIR_MLP must be 0 or 1");
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64 == 0 || P10DC_RANKFORMULA_DIRECTGATHER64 == 1,
              "P10DC_RANKFORMULA_DIRECTGATHER64 must be 0 or 1");
static_assert(!P10DC_RANKFORMULA_PAIR_MLP || P10DC_RANKFORMULA_GATHER_MLP,
              "PAIR_MLP requires GATHER_MLP");
static_assert(!P10DC_RANKFORMULA_DIRECTGATHER64 || P10DC_RANKFORMULA_GATHER_MLP,
              "DIRECTGATHER64 requires GATHER_MLP");
static_assert(!(P10DC_RANKFORMULA_DIRECTGATHER64 && P10DC_RANKFORMULA_PAIR_MLP),
              "DIRECTGATHER64 pair path is intentionally isolated for A/B");
static_assert(!(P10DC_RANKFORMULA_DIRECTGATHER64 && P10DC_RANKFORMULA_PREFETCH_NEXT),
              "DIRECTGATHER64 prefetch path is intentionally isolated for A/B");
#if P10DC_RANKFORMULA_GATHER_MLP
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_mlp.cuh"
#if P10DC_RANKFORMULA_DIRECTGATHER64
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64.cuh"
#endif
#if P10DC_RANKFORMULA_PAIR_MLP
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_pair.cuh"
#endif
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

#if P10DC_RANKFORMULA_GATHER_MLP
__device__ __forceinline__ BkczCrossAccum p10dc_rankformula_cross_mlp_dispatch(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
#if P10DC_RANKFORMULA_DIRECTGATHER64
    return p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather64_fixed(
        h, rank, depth, source_row);
#else
    return p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
        h, rank, depth, source_row);
#endif
}
#endif

#if P10DC_RANKFORMULA_PREFETCH_NEXT
__device__ __forceinline__ void p10dc_rankformula_prefetch_l2(const void* p) {
    const unsigned long long a = reinterpret_cast<unsigned long long>(p);
    asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
}

__device__ __forceinline__ void p10dc_rankformula_prefetch_next_cross(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db, uint32_t rank
) {
#if P10DC_RANKFORMULA_DIRECTGATHER
    if (!c.cross_depth || c.cross_depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS || !c.cross_base)
        return;
#if P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
    const size_t gi = size_t(D_P10DC_RANKFORMULA_DIRECTGATHER_DEPTH_OFF[
        db.hs * P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS + (c.cross_depth - 1u)]) + size_t(rank);
#else
    const size_t gi =
        (size_t(D_P10DC_RANKFORMULA_DIRECTGATHER_OFF[db.hs]) + size_t(rank)) *
            P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS +
        size_t(c.cross_depth - 1u);
#endif
    const uint4 d = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER4 + gi);
    const uint32_t count = d.w >> 16;
    const uint32_t r0 = d.x & 0xffffu, r1 = d.x >> 16;
    const uint32_t r2 = d.y & 0xffffu, r3 = d.y >> 16;
    const uint32_t r4 = d.z & 0xffffu, r5 = d.z >> 16;
    const uint32_t r6 = d.w & 0xffffu;
#if P10DC_RANKFORMULA_DIRECTGATHER_FORCE7
    p10dc_rankformula_prefetch_l2(c.cross_base + r0);
    p10dc_rankformula_prefetch_l2(c.cross_base + r1);
    p10dc_rankformula_prefetch_l2(c.cross_base + r2);
    p10dc_rankformula_prefetch_l2(c.cross_base + r3);
    p10dc_rankformula_prefetch_l2(c.cross_base + r4);
    p10dc_rankformula_prefetch_l2(c.cross_base + r5);
    p10dc_rankformula_prefetch_l2(c.cross_base + r6);
#else
    if (count > 0) p10dc_rankformula_prefetch_l2(c.cross_base + r0);
    if (count > 1) p10dc_rankformula_prefetch_l2(c.cross_base + r1);
    if (count > 2) p10dc_rankformula_prefetch_l2(c.cross_base + r2);
    if (count > 3) p10dc_rankformula_prefetch_l2(c.cross_base + r3);
    if (count > 4) p10dc_rankformula_prefetch_l2(c.cross_base + r4);
    if (count > 5) p10dc_rankformula_prefetch_l2(c.cross_base + r5);
    if (count > 6) p10dc_rankformula_prefetch_l2(c.cross_base + r6);
#endif
#else
    (void)c; (void)db; (void)rank;
#endif
}

__device__ __forceinline__ void p10dc_rankformula_prefetch_next_high(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db, uint32_t lr
) {
    const uint32_t step = uint32_t(gridDim.x) * 32u * uint32_t(P10DC_WARPSTRIPED_COL_ILP);
    const uint32_t next = lr + step;
    if (next >= c.xb.cols) return;
    p10dc_rankformula_prefetch_l2(c.ip_base + next);
    p10dc_rankformula_prefetch_l2(c.dp_base + next);
    p10dc_rankformula_prefetch_l2(c.jp_base + next);
    if constexpr (BKCZ_MAX_LOCAL > 0) if (c.local_n > 0) p10dc_rankformula_prefetch_l2(c.local_base[0] + next);
    if constexpr (BKCZ_MAX_LOCAL > 1) if (c.local_n > 1) p10dc_rankformula_prefetch_l2(c.local_base[1] + next);
    if constexpr (BKCZ_MAX_LOCAL > 2) if (c.local_n > 2) p10dc_rankformula_prefetch_l2(c.local_base[2] + next);
    if constexpr (BKCZ_MAX_LOCAL > 3) if (c.local_n > 3) p10dc_rankformula_prefetch_l2(c.local_base[3] + next);
    if constexpr (BKCZ_MAX_LOCAL > 4) if (c.local_n > 4) p10dc_rankformula_prefetch_l2(c.local_base[4] + next);
    if constexpr (BKCZ_MAX_LOCAL > 5) if (c.local_n > 5) p10dc_rankformula_prefetch_l2(c.local_base[5] + next);
    if constexpr (BKCZ_MAX_LOCAL > 6) if (c.local_n > 6) p10dc_rankformula_prefetch_l2(c.local_base[6] + next);
    if constexpr (BKCZ_MAX_LOCAL > 7) if (c.local_n > 7) p10dc_rankformula_prefetch_l2(c.local_base[7] + next);
    p10dc_rankformula_prefetch_next_cross(c, db, next);
}
#endif

__device__ __forceinline__ Count p10dc_direct_resolved_high_plan_sum_cross5_rankformula_nometa4_abstract(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db, uint32_t lr
) {
#if P10DC_RANKFORMULA_PREFETCH_NEXT
    p10dc_rankformula_prefetch_next_high(c, db, lr);
#endif
#if P10DC_RANKFORMULA_GATHER_MLP
#if P10DC_RANKFORMULA_MLP_WINDOW4
    BkczCrossAccum v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    if constexpr (BKCZ_MAX_LOCAL > 0) if (c.local_n > 0) v0 = BkczCrossAccum(c.local_base[0][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 1) if (c.local_n > 1) v1 = BkczCrossAccum(c.local_base[1][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 2) if (c.local_n > 2) v2 = BkczCrossAccum(c.local_base[2][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 3) if (c.local_n > 3) v3 = BkczCrossAccum(c.local_base[3][lr]);

    BkczCrossAccum cross = 0;
    if (c.cross_depth) {
        cross = p10dc_rankformula_cross_mlp_dispatch(
            db.hs, lr, c.cross_depth, c.cross_base);
    }
    const BkczCrossAccum local03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(v0, v1),
        p10dc_rankformula_accum_add(v2, v3));

    BkczCrossAccum v4 = 0, v5 = 0, v6 = 0, v7 = 0;
    if constexpr (BKCZ_MAX_LOCAL > 4) if (c.local_n > 4) v4 = BkczCrossAccum(c.local_base[4][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 5) if (c.local_n > 5) v5 = BkczCrossAccum(c.local_base[5][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 6) if (c.local_n > 6) v6 = BkczCrossAccum(c.local_base[6][lr]);
    if constexpr (BKCZ_MAX_LOCAL > 7) if (c.local_n > 7) v7 = BkczCrossAccum(c.local_base[7][lr]);
    const BkczCrossAccum local47 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(v4, v5),
        p10dc_rankformula_accum_add(v6, v7));
    const BkczCrossAccum sum = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(local03, local47), cross);
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
#else
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
        cross = p10dc_rankformula_cross_mlp_dispatch(
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

#if P10DC_RANKFORMULA_PAIR_MLP
__device__ __forceinline__ void
p10dc_direct_resolved_high_plan_sum_pair_cross5_rankformula_nometa4_abstract(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db,
    uint32_t lr0, uint32_t lr1, Count& out0, Count& out1
) {
    BkczCrossAccum cross0 = 0, cross1 = 0;
    if (c.cross_depth) {
        const auto cross =
            p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_pair_fixed(
                db.hs, lr0, lr1, c.cross_depth, c.cross_base);
        cross0 = cross.a;
        cross1 = cross.b;
    }

    BkczCrossAccum a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    BkczCrossAccum b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    if constexpr (BKCZ_MAX_LOCAL > 0) if (c.local_n > 0) { a0 = BkczCrossAccum(c.local_base[0][lr0]); b0 = BkczCrossAccum(c.local_base[0][lr1]); }
    if constexpr (BKCZ_MAX_LOCAL > 1) if (c.local_n > 1) { a1 = BkczCrossAccum(c.local_base[1][lr0]); b1 = BkczCrossAccum(c.local_base[1][lr1]); }
    if constexpr (BKCZ_MAX_LOCAL > 2) if (c.local_n > 2) { a2 = BkczCrossAccum(c.local_base[2][lr0]); b2 = BkczCrossAccum(c.local_base[2][lr1]); }
    if constexpr (BKCZ_MAX_LOCAL > 3) if (c.local_n > 3) { a3 = BkczCrossAccum(c.local_base[3][lr0]); b3 = BkczCrossAccum(c.local_base[3][lr1]); }
    const BkczCrossAccum local03_0 = p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(a0, a1), p10dc_rankformula_accum_add(a2, a3));
    const BkczCrossAccum local03_1 = p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(b0, b1), p10dc_rankformula_accum_add(b2, b3));

    BkczCrossAccum a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    BkczCrossAccum b4 = 0, b5 = 0, b6 = 0, b7 = 0;
    if constexpr (BKCZ_MAX_LOCAL > 4) if (c.local_n > 4) { a4 = BkczCrossAccum(c.local_base[4][lr0]); b4 = BkczCrossAccum(c.local_base[4][lr1]); }
    if constexpr (BKCZ_MAX_LOCAL > 5) if (c.local_n > 5) { a5 = BkczCrossAccum(c.local_base[5][lr0]); b5 = BkczCrossAccum(c.local_base[5][lr1]); }
    if constexpr (BKCZ_MAX_LOCAL > 6) if (c.local_n > 6) { a6 = BkczCrossAccum(c.local_base[6][lr0]); b6 = BkczCrossAccum(c.local_base[6][lr1]); }
    if constexpr (BKCZ_MAX_LOCAL > 7) if (c.local_n > 7) { a7 = BkczCrossAccum(c.local_base[7][lr0]); b7 = BkczCrossAccum(c.local_base[7][lr1]); }
    const BkczCrossAccum local47_0 = p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(a4, a5), p10dc_rankformula_accum_add(a6, a7));
    const BkczCrossAccum local47_1 = p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(b4, b5), p10dc_rankformula_accum_add(b6, b7));
    const BkczCrossAccum sum0 = p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(local03_0, local47_0), cross0);
    const BkczCrossAccum sum1 = p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(local03_1, local47_1), cross1);
#if GPU_DIRECT_PM_ACCUM
    out0 = gpu_direct_pm_reduce_u64(sum0);
    out1 = gpu_direct_pm_reduce_u64(sum1);
#else
    out0 = sum0;
    out1 = sum1;
#endif
}
#endif
