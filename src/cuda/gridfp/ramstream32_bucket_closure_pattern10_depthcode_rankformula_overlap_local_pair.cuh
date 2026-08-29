#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankformula_nometa4_abstract.cuh"

#ifndef P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR
#define P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR 0
#endif
static_assert(P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR == 0 ||
              P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR == 1,
              "P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR must be 0 or 1");
#if P10DC_RANKFORMULA_CPASYNC_OVERLAP_LOCAL_PAIR
static_assert(P10DC_RANKFORMULA_PAIR_MLP,
              "overlap-local pair requires PAIR_MLP");
static_assert(P10DC_RANKFORMULA_CPASYNC_PAIR,
              "overlap-local pair requires cross cp.async support");
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64,
              "overlap-local pair currently targets DIRECTGATHER64");
static_assert(BKCZ_MAX_LOCAL <= 8,
              "overlap-local pair assumes at most eight ordinary source rows");

struct P10DCDirectGatherPairRanks {
    uint32_t n0 = 0, n1 = 0;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0;
    uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
};

__device__ __forceinline__ P10DCDirectGatherPairRanks
p10dc_rankformula_directgather64_pair_ranks(
    uint32_t h, uint32_t rank0, uint32_t rank1, uint32_t depth
) {
    P10DCDirectGatherPairRanks z{};
    const size_t gi0 = p10dc_rankformula_directgather_index(h, rank0, depth);
    const size_t gi1 = p10dc_rankformula_directgather_index(h, rank1, depth);
    P10DCDirectGather64Word p0 = 0, p1 = 0;

#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    const size_t wi0 = gi0 >> 5, wi1 = gi1 >> 5;
    const uint32_t bit0 = uint32_t(gi0) & 31u, bit1 = uint32_t(gi1) & 31u;
    const P10DCDirectGather64Word ix0 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi0);
    const P10DCDirectGather64Word ix1 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi1);
    const uint32_t bits0 = uint32_t(ix0), bits1 = uint32_t(ix1);
    const uint32_t flag0 = 1u << bit0, flag1 = 1u << bit1;
    if (bits0 & flag0) {
        const uint32_t lower0 = bit0 ? (bits0 & (flag0 - 1u)) : 0u;
        const uint32_t ci0 = uint32_t(ix0 >> 32) + uint32_t(__popc(lower0));
        p0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci0);
    }
    if (bits1 & flag1) {
        const uint32_t lower1 = bit1 ? (bits1 & (flag1 - 1u)) : 0u;
        const uint32_t ci1 = uint32_t(ix1 >> 32) + uint32_t(__popc(lower1));
        p1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci1);
    }
#else
    p0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi0);
    p1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi1);
#endif

    z.n0 = uint32_t((p0 >> 45) & 7u);
    z.n1 = uint32_t((p1 >> 45) & 7u);
    z.a0 = uint32_t(p0 & 0x7fffu);
    z.a1 = uint32_t((p0 >> 15) & 0x7fffu);
    z.a2 = uint32_t((p0 >> 30) & 0x7fffu);
    z.b0 = uint32_t(p1 & 0x7fffu);
    z.b1 = uint32_t((p1 >> 15) & 0x7fffu);
    z.b2 = uint32_t((p1 >> 30) & 0x7fffu);

    P10DCDirectGather64Word q0 = 0, q1 = 0;
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    if (z.n0 > 3u) q0 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE + uint32_t(p0 >> 48));
    if (z.n1 > 3u) q1 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE + uint32_t(p1 >> 48));
#else
    if (z.n0 > 3u) q0 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE + uint32_t(p0 >> 48));
    if (z.n1 > 3u) q1 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE + uint32_t(p1 >> 48));
#endif
    if (z.n0 > 3u) {
        z.a3 = uint32_t(q0 & 0x7fffu);
        z.a4 = uint32_t((q0 >> 15) & 0x7fffu);
        z.a5 = uint32_t((q0 >> 30) & 0x7fffu);
        z.a6 = uint32_t((q0 >> 45) & 0x7fffu);
    }
    if (z.n1 > 3u) {
        z.b3 = uint32_t(q1 & 0x7fffu);
        z.b4 = uint32_t((q1 >> 15) & 0x7fffu);
        z.b5 = uint32_t((q1 >> 30) & 0x7fffu);
        z.b6 = uint32_t((q1 >> 45) & 0x7fffu);
    }
    return z;
}

__device__ __forceinline__ void
p10dc_direct_resolved_high_plan_sum_pair_overlap_local(
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

    // Issue all sixteen ordinary source loads before consuming either column.
    // They remain independent of the outstanding cross cp.async groups, giving
    // one lane up to thirty source-memory requests in flight without enlarging
    // the dynamic shared scratch.
    BkczCrossAccum a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    BkczCrossAccum b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0,b7=0;
    if constexpr (BKCZ_MAX_LOCAL > 0) if (c.local_n > 0) { a0=__ldg(c.local_base[0]+lr0); b0=__ldg(c.local_base[0]+lr1); }
    if constexpr (BKCZ_MAX_LOCAL > 1) if (c.local_n > 1) { a1=__ldg(c.local_base[1]+lr0); b1=__ldg(c.local_base[1]+lr1); }
    if constexpr (BKCZ_MAX_LOCAL > 2) if (c.local_n > 2) { a2=__ldg(c.local_base[2]+lr0); b2=__ldg(c.local_base[2]+lr1); }
    if constexpr (BKCZ_MAX_LOCAL > 3) if (c.local_n > 3) { a3=__ldg(c.local_base[3]+lr0); b3=__ldg(c.local_base[3]+lr1); }
    if constexpr (BKCZ_MAX_LOCAL > 4) if (c.local_n > 4) { a4=__ldg(c.local_base[4]+lr0); b4=__ldg(c.local_base[4]+lr1); }
    if constexpr (BKCZ_MAX_LOCAL > 5) if (c.local_n > 5) { a5=__ldg(c.local_base[5]+lr0); b5=__ldg(c.local_base[5]+lr1); }
    if constexpr (BKCZ_MAX_LOCAL > 6) if (c.local_n > 6) { a6=__ldg(c.local_base[6]+lr0); b6=__ldg(c.local_base[6]+lr1); }
    if constexpr (BKCZ_MAX_LOCAL > 7) if (c.local_n > 7) { a7=__ldg(c.local_base[7]+lr0); b7=__ldg(c.local_base[7]+lr1); }

    if (have_cross) p10dc_rankformula_cpasync_wait_all();

    BkczCrossAccum cross0 = 0, cross1 = 0;
    if (have_cross) {
        const BkczCrossAccum ca01=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(0),*p10dc_rankformula_cpasync_slot(1));
        const BkczCrossAccum ca23=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(2),*p10dc_rankformula_cpasync_slot(3));
        const BkczCrossAccum ca45=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(4),*p10dc_rankformula_cpasync_slot(5));
        cross0=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(ca01,ca23),p10dc_rankformula_accum_add(ca45,*p10dc_rankformula_cpasync_slot(6)));
        const BkczCrossAccum cb01=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(7),*p10dc_rankformula_cpasync_slot(8));
        const BkczCrossAccum cb23=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(9),*p10dc_rankformula_cpasync_slot(10));
        const BkczCrossAccum cb45=p10dc_rankformula_accum_add(*p10dc_rankformula_cpasync_slot(11),*p10dc_rankformula_cpasync_slot(12));
        cross1=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(cb01,cb23),p10dc_rankformula_accum_add(cb45,*p10dc_rankformula_cpasync_slot(13)));
    }

    const BkczCrossAccum la01=p10dc_rankformula_accum_add(a0,a1), la23=p10dc_rankformula_accum_add(a2,a3);
    const BkczCrossAccum la45=p10dc_rankformula_accum_add(a4,a5), la67=p10dc_rankformula_accum_add(a6,a7);
    const BkczCrossAccum lb01=p10dc_rankformula_accum_add(b0,b1), lb23=p10dc_rankformula_accum_add(b2,b3);
    const BkczCrossAccum lb45=p10dc_rankformula_accum_add(b4,b5), lb67=p10dc_rankformula_accum_add(b6,b7);
    const BkczCrossAccum sum0=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(la01,la23),p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(la45,la67),cross0));
    const BkczCrossAccum sum1=p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(lb01,lb23),p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(lb45,lb67),cross1));
#if GPU_DIRECT_PM_ACCUM
    out0=gpu_direct_pm_reduce_u64(sum0); out1=gpu_direct_pm_reduce_u64(sum1);
#else
    out0=Count(sum0); out1=Count(sum1);
#endif
}
#endif
