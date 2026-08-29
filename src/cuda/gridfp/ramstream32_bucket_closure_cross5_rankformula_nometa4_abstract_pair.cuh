#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_mlp.cuh"

#ifndef P10DC_RANKFORMULA_PAIR_MLP
#define P10DC_RANKFORMULA_PAIR_MLP 0
#endif
static_assert(P10DC_RANKFORMULA_PAIR_MLP == 0 || P10DC_RANKFORMULA_PAIR_MLP == 1,
              "P10DC_RANKFORMULA_PAIR_MLP must be 0 or 1");
#if P10DC_RANKFORMULA_PAIR_MLP
static_assert(P10DC_RANKFORMULA_DIRECTGATHER,
              "pair MLP currently requires DIRECTGATHER");
static_assert(P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR,
              "pair MLP currently requires depth-major DIRECTGATHER");
static_assert(P10DC_RANKFORMULA_MLP_WINDOW4,
              "pair MLP requires WINDOW4 to bound register pressure");
#endif

struct P10DCRankFormulaPairAccum {
    BkczCrossAccum a;
    BkczCrossAccum b;
};

__device__ __forceinline__ uint4 p10dc_rankformula_directgather_depth_desc(
    uint32_t h, uint32_t rank, uint32_t depth
) {
    const size_t gi = size_t(D_P10DC_RANKFORMULA_DIRECTGATHER_DEPTH_OFF[
        h * P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS + (depth - 1u)]) + size_t(rank);
    return __ldg(D_P10DC_RANKFORMULA_DIRECTGATHER4 + gi);
}

__device__ __forceinline__ P10DCRankFormulaPairAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_pair_fixed(
    uint32_t h, uint32_t rank0, uint32_t rank1, uint32_t depth,
    const Count* source_row
) {
#if !P10DC_RANKFORMULA_PAIR_MLP
    return P10DCRankFormulaPairAccum{
        p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
            h, rank0, depth, source_row),
        p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
            h, rank1, depth, source_row)};
#else
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return P10DCRankFormulaPairAccum{0, 0};

    // Issue both 16-byte descriptors before consuming either one.  With the
    // depth-major table, every warp still sees contiguous descriptor accesses,
    // while each lane now has two independent descriptor requests in flight.
    const uint4 d0 = p10dc_rankformula_directgather_depth_desc(h, rank0, depth);
    const uint4 d1 = p10dc_rankformula_directgather_depth_desc(h, rank1, depth);
    const uint32_t n0 = d0.w >> 16;
    const uint32_t n1 = d1.w >> 16;
    if (!(n0 | n1)) return P10DCRankFormulaPairAccum{0, 0};

    const uint32_t a0 = d0.x & 0xffffu, a1 = d0.x >> 16;
    const uint32_t a2 = d0.y & 0xffffu, a3 = d0.y >> 16;
    const uint32_t a4 = d0.z & 0xffffu, a5 = d0.z >> 16;
    const uint32_t a6 = d0.w & 0xffffu;
    const uint32_t b0 = d1.x & 0xffffu, b1 = d1.x >> 16;
    const uint32_t b2 = d1.y & 0xffffu, b3 = d1.y >> 16;
    const uint32_t b4 = d1.z & 0xffffu, b5 = d1.z >> 16;
    const uint32_t b6 = d1.w & 0xffffu;

    // Four-wide window for both columns.  The loads for column B are issued
    // before column A is reduced, increasing outstanding requests without
    // keeping fourteen source values live simultaneously.
    BkczCrossAccum a03_0 = 0, a03_1 = 0, a03_2 = 0, a03_3 = 0;
    BkczCrossAccum b03_0 = 0, b03_1 = 0, b03_2 = 0, b03_3 = 0;
    if (n0 > 0) a03_0 = BkczCrossAccum(__ldg(source_row + a0));
    if (n0 > 1) a03_1 = BkczCrossAccum(__ldg(source_row + a1));
    if (n0 > 2) a03_2 = BkczCrossAccum(__ldg(source_row + a2));
    if (n0 > 3) a03_3 = BkczCrossAccum(__ldg(source_row + a3));
    if (n1 > 0) b03_0 = BkczCrossAccum(__ldg(source_row + b0));
    if (n1 > 1) b03_1 = BkczCrossAccum(__ldg(source_row + b1));
    if (n1 > 2) b03_2 = BkczCrossAccum(__ldg(source_row + b2));
    if (n1 > 3) b03_3 = BkczCrossAccum(__ldg(source_row + b3));
    const BkczCrossAccum sa03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(a03_0, a03_1),
        p10dc_rankformula_accum_add(a03_2, a03_3));
    const BkczCrossAccum sb03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(b03_0, b03_1),
        p10dc_rankformula_accum_add(b03_2, b03_3));

    BkczCrossAccum a46_4 = 0, a46_5 = 0, a46_6 = 0;
    BkczCrossAccum b46_4 = 0, b46_5 = 0, b46_6 = 0;
    if (n0 > 4) a46_4 = BkczCrossAccum(__ldg(source_row + a4));
    if (n0 > 5) a46_5 = BkczCrossAccum(__ldg(source_row + a5));
    if (n0 > 6) a46_6 = BkczCrossAccum(__ldg(source_row + a6));
    if (n1 > 4) b46_4 = BkczCrossAccum(__ldg(source_row + b4));
    if (n1 > 5) b46_5 = BkczCrossAccum(__ldg(source_row + b5));
    if (n1 > 6) b46_6 = BkczCrossAccum(__ldg(source_row + b6));
    const BkczCrossAccum sa46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(a46_4, a46_5), a46_6);
    const BkczCrossAccum sb46 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(b46_4, b46_5), b46_6);

    return P10DCRankFormulaPairAccum{
        p10dc_rankformula_accum_add(sa03, sa46),
        p10dc_rankformula_accum_add(sb03, sb46)};
#endif
}
