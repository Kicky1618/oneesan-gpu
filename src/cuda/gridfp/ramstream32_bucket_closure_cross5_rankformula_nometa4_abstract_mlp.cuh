#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract.cuh"

// B300-oriented memory-level parallelism for the compact abstract CROSS path.
// The legacy walker reduces one source load before issuing the next one.  This
// helper computes all selected source addresses first, issues up to seven
// independent reads, then combines them as a balanced tree.  The arithmetic is
// exactly associative in both modes: modulo Count when PM accumulation is off,
// and uint64 accumulation when it is on.
__device__ __forceinline__ BkczCrossAccum p10dc_rankformula_accum_add(
    BkczCrossAccum a, BkczCrossAccum b
) {
#if GPU_DIRECT_PM_ACCUM
    return a + b;
#else
    return gpu_direct_add(a, b);
#endif
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_mlp_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
#if !(P10DC_RANKFORMULA_ABSTRACT_SELECT8 && P10DC_RANKFORMULA_ABSTRACT_SRCPACK10)
    return p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_fixed(
        h, rank, depth, source_row);
#else
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return BkczCrossAccum(0);
    const auto z = p10dc_low_rankformula_nometa_resolve_active(h, rank);
    if (z.n <= h) return BkczCrossAccum(0);

    const uint32_t lcount = (z.n - h) >> 1;
    const uint32_t local = rank - z.start;
#if P10DC_RANKFORMULA_NOMETA_GROUP56 || P10DC_RANKFORMULA_NOMETA_GROUP61
    const uint32_t di = z.abstract_off + local;
#else
    const uint32_t di =
        uint32_t(D_P10DC_RANKFORMULA_ABSTRACT_OFF[z.n * 16u + h]) + local;
#endif

    uint32_t select;
#if P10DC_RANKFORMULA_ABSTRACT_DEPTH4
    uint32_t dpack = p10dc_rankformula_abstract_depth03_load(di);
    if (lcount > 4u) dpack |= p10dc_rankformula_abstract_depth46_load(di) << 16;
    select = p10dc_rankformula_abstract_depth4_select(dpack, depth);
#else
    select = p10dc_rankformula_abstract_select_load(
        (depth - 1u) * P10DC_RANKFORMULA_ABSTRACT_DESC_N + di);
#endif
    if (!select) return BkczCrossAccum(0);

    const uint32_t source_base = z.source_base;
    uint32_t src03 = 0, src36 = 0;
    if (select & 0x07u) src03 = p10dc_rankformula_abstract_src03_load(di);
    if (select & 0x38u) src36 = p10dc_rankformula_abstract_src36_load(di);

    const uint32_t s0 = src03 & 1023u;
    const uint32_t s1 = (src03 >> 10) & 1023u;
    const uint32_t s2 = (src03 >> 20) & 1023u;
    const uint32_t s3 = src36 & 1023u;
    const uint32_t s4 = (src36 >> 10) & 1023u;
    const uint32_t s5 = (src36 >> 20) & 1023u;
    uint32_t s6 = 0;
    if (select & 0x40u) s6 = p10dc_rankformula_abstract_src7_load(local);

    BkczCrossAccum v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    BkczCrossAccum v4 = 0, v5 = 0, v6 = 0;
    if (select & 0x01u) v0 = BkczCrossAccum(source_row[source_base + s0]);
    if (select & 0x02u) v1 = BkczCrossAccum(source_row[source_base + s1]);
    if (select & 0x04u) v2 = BkczCrossAccum(source_row[source_base + s2]);
    if (select & 0x08u) v3 = BkczCrossAccum(source_row[source_base + s3]);
    if (select & 0x10u) v4 = BkczCrossAccum(source_row[source_base + s4]);
    if (select & 0x20u) v5 = BkczCrossAccum(source_row[source_base + s5]);
    if (select & 0x40u) v6 = BkczCrossAccum(source_row[source_base + s6]);

    const BkczCrossAccum a01 = p10dc_rankformula_accum_add(v0, v1);
    const BkczCrossAccum a23 = p10dc_rankformula_accum_add(v2, v3);
    const BkczCrossAccum a45 = p10dc_rankformula_accum_add(v4, v5);
    const BkczCrossAccum a0123 = p10dc_rankformula_accum_add(a01, a23);
    const BkczCrossAccum a456 = p10dc_rankformula_accum_add(a45, v6);
    return p10dc_rankformula_accum_add(a0123, a456);
#endif
}
