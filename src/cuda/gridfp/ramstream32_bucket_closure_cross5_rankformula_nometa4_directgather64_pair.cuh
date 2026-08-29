#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_directgather64.cuh"
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_pair.cuh"

#if P10DC_RANKFORMULA_DIRECTGATHER64 && P10DC_RANKFORMULA_PAIR_MLP

__device__ __forceinline__ P10DCRankFormulaPairAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather64_pair_fixed(
    uint32_t h, uint32_t rank0, uint32_t rank1, uint32_t depth,
    const Count* source_row
) {
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return P10DCRankFormulaPairAccum{0, 0};

    const size_t gi0 = p10dc_rankformula_directgather_index(h, rank0, depth);
    const size_t gi1 = p10dc_rankformula_directgather_index(h, rank1, depth);
    P10DCDirectGather64Word p0 = 0, p1 = 0;

#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    // Issue both sparse-index words before consuming either result. They are
    // independent even when they land in different 32-entry groups.
    const size_t wi0 = gi0 >> 5;
    const size_t wi1 = gi1 >> 5;
    const uint32_t bit0 = uint32_t(gi0) & 31u;
    const uint32_t bit1 = uint32_t(gi1) & 31u;
    const P10DCDirectGather64Word ix0 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi0);
    const P10DCDirectGather64Word ix1 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi1);
    const uint32_t bits0 = uint32_t(ix0), bits1 = uint32_t(ix1);
    const uint32_t flag0 = 1u << bit0, flag1 = 1u << bit1;
    const bool live0 = (bits0 & flag0) != 0u;
    const bool live1 = (bits1 & flag1) != 0u;
    if (!live0 && !live1) return P10DCRankFormulaPairAccum{0, 0};

    uint32_t ci0 = 0, ci1 = 0;
    if (live0) {
        const uint32_t lower0 = bit0 ? (bits0 & (flag0 - 1u)) : 0u;
        ci0 = uint32_t(ix0 >> 32) + uint32_t(__popc(lower0));
    }
    if (live1) {
        const uint32_t lower1 = bit1 ? (bits1 & (flag1 - 1u)) : 0u;
        ci1 = uint32_t(ix1 >> 32) + uint32_t(__popc(lower1));
    }
    if (live0) p0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci0);
    if (live1) p1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci1);
#else
    // Dense directgather64: issue both primary descriptor loads up front.
    p0 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi0);
    p1 = __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER64 + gi1);
#endif

    const uint32_t n0 = uint32_t((p0 >> 45) & 7u);
    const uint32_t n1 = uint32_t((p1 >> 45) & 7u);
    if (!(n0 | n1)) return P10DCRankFormulaPairAccum{0, 0};

    const uint32_t a0 = uint32_t(p0 & 0x7fffu);
    const uint32_t a1 = uint32_t((p0 >> 15) & 0x7fffu);
    const uint32_t a2 = uint32_t((p0 >> 30) & 0x7fffu);
    const uint32_t b0 = uint32_t(p1 & 0x7fffu);
    const uint32_t b1 = uint32_t((p1 >> 15) & 0x7fffu);
    const uint32_t b2 = uint32_t((p1 >> 30) & 0x7fffu);
    uint32_t a3 = 0, a4 = 0, a5 = 0, a6 = 0;
    uint32_t b3 = 0, b4 = 0, b5 = 0, b6 = 0;

    P10DCDirectGather64Word q0 = 0, q1 = 0;
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    if (n0 > 3u) q0 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE + uint32_t(p0 >> 48));
    if (n1 > 3u) q1 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE + uint32_t(p1 >> 48));
#else
    if (n0 > 3u) q0 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE + uint32_t(p0 >> 48));
    if (n1 > 3u) q1 = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE + uint32_t(p1 >> 48));
#endif
    if (n0 > 3u) {
        a3 = uint32_t(q0 & 0x7fffu);
        a4 = uint32_t((q0 >> 15) & 0x7fffu);
        a5 = uint32_t((q0 >> 30) & 0x7fffu);
        a6 = uint32_t((q0 >> 45) & 0x7fffu);
    }
    if (n1 > 3u) {
        b3 = uint32_t(q1 & 0x7fffu);
        b4 = uint32_t((q1 >> 15) & 0x7fffu);
        b5 = uint32_t((q1 >> 30) & 0x7fffu);
        b6 = uint32_t((q1 >> 45) & 0x7fffu);
    }

#if P10DC_RANKFORMULA_CPASYNC_PAIR
    // Keep descriptor compression while moving all fourteen potentially remote
    // Count requests out of the register dependency chain. Two async groups are
    // committed so one lane can expose up to 14 source reads before reduction.
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(0), source_row + a0, n0 > 0u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(1), source_row + a1, n0 > 1u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(2), source_row + a2, n0 > 2u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(3), source_row + a3, n0 > 3u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(4), source_row + a4, n0 > 4u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(5), source_row + a5, n0 > 5u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(6), source_row + a6, n0 > 6u);
    p10dc_rankformula_cpasync_commit();

    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(7), source_row + b0, n1 > 0u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(8), source_row + b1, n1 > 1u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(9), source_row + b2, n1 > 2u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(10), source_row + b3, n1 > 3u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(11), source_row + b4, n1 > 4u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(12), source_row + b5, n1 > 5u);
    p10dc_rankformula_cpasync_u32(p10dc_rankformula_cpasync_slot(13), source_row + b6, n1 > 6u);
    p10dc_rankformula_cpasync_commit();
    p10dc_rankformula_cpasync_wait_all();

    const BkczCrossAccum as03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(0)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(1))),
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(2)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(3))));
    const BkczCrossAccum as46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(4)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(5))),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(6)));
    const BkczCrossAccum bs03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(7)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(8))),
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(9)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(10))));
    const BkczCrossAccum bs46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(11)),
            BkczCrossAccum(*p10dc_rankformula_cpasync_slot(12))),
        BkczCrossAccum(*p10dc_rankformula_cpasync_slot(13)));
    return P10DCRankFormulaPairAccum{
        p10dc_rankformula_accum_add(as03, as46),
        p10dc_rankformula_accum_add(bs03, bs46)};
#else
    // Register path: first source window issues A and B before either reduction,
    // creating up to eight independent 32-bit source requests per lane.
    BkczCrossAccum av0 = 0, av1 = 0, av2 = 0, av3 = 0;
    BkczCrossAccum bv0 = 0, bv1 = 0, bv2 = 0, bv3 = 0;
    if (n0 > 0u) av0 = BkczCrossAccum(__ldg(source_row + a0));
    if (n0 > 1u) av1 = BkczCrossAccum(__ldg(source_row + a1));
    if (n0 > 2u) av2 = BkczCrossAccum(__ldg(source_row + a2));
    if (n0 > 3u) av3 = BkczCrossAccum(__ldg(source_row + a3));
    if (n1 > 0u) bv0 = BkczCrossAccum(__ldg(source_row + b0));
    if (n1 > 1u) bv1 = BkczCrossAccum(__ldg(source_row + b1));
    if (n1 > 2u) bv2 = BkczCrossAccum(__ldg(source_row + b2));
    if (n1 > 3u) bv3 = BkczCrossAccum(__ldg(source_row + b3));
    const BkczCrossAccum as03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(av0, av1),
        p10dc_rankformula_accum_add(av2, av3));
    const BkczCrossAccum bs03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(bv0, bv1),
        p10dc_rankformula_accum_add(bv2, bv3));

    BkczCrossAccum av4 = 0, av5 = 0, av6 = 0;
    BkczCrossAccum bv4 = 0, bv5 = 0, bv6 = 0;
    if (n0 > 4u) av4 = BkczCrossAccum(__ldg(source_row + a4));
    if (n0 > 5u) av5 = BkczCrossAccum(__ldg(source_row + a5));
    if (n0 > 6u) av6 = BkczCrossAccum(__ldg(source_row + a6));
    if (n1 > 4u) bv4 = BkczCrossAccum(__ldg(source_row + b4));
    if (n1 > 5u) bv5 = BkczCrossAccum(__ldg(source_row + b5));
    if (n1 > 6u) bv6 = BkczCrossAccum(__ldg(source_row + b6));
    const BkczCrossAccum as46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(av4, av5), av6);
    const BkczCrossAccum bs46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(bv4, bv5), bv6);

    return P10DCRankFormulaPairAccum{
        p10dc_rankformula_accum_add(as03, as46),
        p10dc_rankformula_accum_add(bs03, bs46)};
#endif
}

#endif
