#pragma once

#ifndef P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#define P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64 0
#endif
static_assert(P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64 == 0 ||
              P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64 == 1,
              "P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64 must be 0 or 1");
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
static_assert(P10DC_RANKFORMULA_DIRECTGATHER64,
              "SPARSE64 uses the DIRECTGATHER64 descriptor encoding");
#endif

#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
__constant__ P10DCDirectGather64Word* D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX;
__constant__ P10DCDirectGather64Word* D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY;
__constant__ P10DCDirectGather64Word* D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE;
#endif

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather_sparse64_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
#if !P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
    return p10dc_resolved_low_preimages_cross5_rankformula_nometa4_directgather64_fixed(
        h, rank, depth, source_row);
#else
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return BkczCrossAccum(0);
    const size_t gi = p10dc_rankformula_directgather_index(h, rank, depth);
    const size_t wi = gi >> 5;
    const uint32_t bit = uint32_t(gi) & 31u;
    const P10DCDirectGather64Word ix = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_INDEX + wi);
    const uint32_t bits = uint32_t(ix);
    const uint32_t flag = 1u << bit;
    if (!(bits & flag)) return BkczCrossAccum(0);
    const uint32_t prefix = uint32_t(ix >> 32);
    const uint32_t lower = bit ? (bits & (flag - 1u)) : 0u;
    const uint32_t ci = prefix + uint32_t(__popc(lower));

    const P10DCDirectGather64Word p = __ldg(
        D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_PRIMARY + ci);
    const uint32_t count = uint32_t((p >> 45) & 7u);
    if (!count) return BkczCrossAccum(0);

    const uint32_t r0 = uint32_t(p & 0x7fffu);
    const uint32_t r1 = uint32_t((p >> 15) & 0x7fffu);
    const uint32_t r2 = uint32_t((p >> 30) & 0x7fffu);
    uint32_t r3 = 0, r4 = 0, r5 = 0, r6 = 0;
    if (count > 3u) {
        const uint32_t rare_ix = uint32_t(p >> 48);
        const P10DCDirectGather64Word q = __ldg(
            D_P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64_RARE + rare_ix);
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
    const BkczCrossAccum v3 = BkczCrossAccum(__ldg(source_row + r3));
    BkczCrossAccum v4 = 0, v5 = 0, v6 = 0;
    if (count > 4) v4 = BkczCrossAccum(__ldg(source_row + r4));
    if (count > 5) v5 = BkczCrossAccum(__ldg(source_row + r5));
    if (count > 6) v6 = BkczCrossAccum(__ldg(source_row + r6));
    return p10dc_rankformula_accum_add(
        s02,
        p10dc_rankformula_accum_add(
            p10dc_rankformula_accum_add(v3, v4),
            p10dc_rankformula_accum_add(v5, v6)));
#endif
#endif
}
