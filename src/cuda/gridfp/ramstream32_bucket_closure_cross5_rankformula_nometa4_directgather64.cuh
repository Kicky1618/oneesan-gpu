#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract_mlp.cuh"

#ifndef P10DC_RANKFORMULA_DIRECTGATHER64
#define P10DC_RANKFORMULA_DIRECTGATHER64 0
#endif
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64 == 0 || P10DC_RANKFORMULA_DIRECTGATHER64 == 1,
              "P10DC_RANKFORMULA_DIRECTGATHER64 must be 0 or 1");
#if P10DC_RANKFORMULA_DIRECTGATHER64
static_assert(P10DC_RANKFORMULA_DIRECTGATHER,
              "DIRECTGATHER64 requires DIRECTGATHER");
static_assert(!P10DC_RANKFORMULA_DIRECTGATHER_FORCE7,
              "DIRECTGATHER64 does not use FORCE7");
#endif

using P10DCDirectGather64Word = unsigned long long;
static_assert(sizeof(P10DCDirectGather64Word) == 8,
              "directgather64 word must be exactly 64 bits");

#if P10DC_RANKFORMULA_DIRECTGATHER64
__constant__ P10DCDirectGather64Word* D_P10DC_RANKFORMULA_DIRECTGATHER64;
__constant__ P10DCDirectGather64Word* D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE;
#endif

__device__ __forceinline__ size_t p10dc_rankformula_directgather_index(
    uint32_t h, uint32_t rank, uint32_t depth
) {
#if P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
    return size_t(D_P10DC_RANKFORMULA_DIRECTGATHER_DEPTH_OFF[
        h * P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS + (depth - 1u)]) + size_t(rank);
#else
    return (size_t(D_P10DC_RANKFORMULA_DIRECTGATHER_OFF[h]) + size_t(rank)) *
               P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS +
           size_t(depth - 1u);
#endif
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather64_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
#if !P10DC_RANKFORMULA_DIRECTGATHER64
    return p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
        h, rank, depth, source_row);
#else
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return BkczCrossAccum(0);
    const P10DCDirectGather64Word p = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER64 +
        p10dc_rankformula_directgather_index(h, rank, depth));
    const uint32_t count = uint32_t((p >> 45) & 7u);
    if (!count) return BkczCrossAccum(0);

    const uint32_t r0 = uint32_t(p & 0x7fffu);
    const uint32_t r1 = uint32_t((p >> 15) & 0x7fffu);
    const uint32_t r2 = uint32_t((p >> 30) & 0x7fffu);
    uint32_t r3 = 0, r4 = 0, r5 = 0, r6 = 0;
    if (count > 3) {
        const uint32_t rare_ix = uint32_t(p >> 48);
        const P10DCDirectGather64Word q = __ldg(
            D_P10DC_RANKFORMULA_DIRECTGATHER64_RARE + rare_ix);
        r3 = uint32_t(q & 0x7fffu);
        r4 = uint32_t((q >> 15) & 0x7fffu);
        r5 = uint32_t((q >> 30) & 0x7fffu);
        r6 = uint32_t((q >> 45) & 0x7fffu);
    }

#if P10DC_RANKFORMULA_MLP_WINDOW4
    BkczCrossAccum v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    if (count > 0) v0 = BkczCrossAccum(__ldg(source_row + r0));
    if (count > 1) v1 = BkczCrossAccum(__ldg(source_row + r1));
    if (count > 2) v2 = BkczCrossAccum(__ldg(source_row + r2));
    if (count > 3) v3 = BkczCrossAccum(__ldg(source_row + r3));
    const BkczCrossAccum s03 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(v0, v1),
        p10dc_rankformula_accum_add(v2, v3));
    BkczCrossAccum v4 = 0, v5 = 0, v6 = 0;
    if (count > 4) v4 = BkczCrossAccum(__ldg(source_row + r4));
    if (count > 5) v5 = BkczCrossAccum(__ldg(source_row + r5));
    if (count > 6) v6 = BkczCrossAccum(__ldg(source_row + r6));
    return p10dc_rankformula_accum_add(
        s03,
        p10dc_rankformula_accum_add(p10dc_rankformula_accum_add(v4, v5), v6));
#else
    BkczCrossAccum v0 = 0, v1 = 0, v2 = 0;
    if (count > 0) v0 = BkczCrossAccum(__ldg(source_row + r0));
    if (count > 1) v1 = BkczCrossAccum(__ldg(source_row + r1));
    if (count > 2) v2 = BkczCrossAccum(__ldg(source_row + r2));
    const BkczCrossAccum s02 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(v0, v1), v2);
    if (count <= 3) return s02;
    BkczCrossAccum v3 = BkczCrossAccum(__ldg(source_row + r3));
    BkczCrossAccum v4 = 0, v5 = 0, v6 = 0;
    if (count > 4) v4 = BkczCrossAccum(__ldg(source_row + r4));
    if (count > 5) v5 = BkczCrossAccum(__ldg(source_row + r5));
    if (count > 6) v6 = BkczCrossAccum(__ldg(source_row + r6));
    const BkczCrossAccum s36 = p10dc_rankformula_accum_add(
        p10dc_rankformula_accum_add(v3, v4),
        p10dc_rankformula_accum_add(v5, v6));
    return p10dc_rankformula_accum_add(s02, s36);
#endif
#endif
}
