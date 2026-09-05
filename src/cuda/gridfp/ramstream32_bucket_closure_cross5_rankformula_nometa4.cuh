#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula.cuh"
#include "ramstream32_bucket_low_rankformula_nometa4.cuh"

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    if (!depth) return BkczCrossAccum(0);
    const P10DCRankFormulaNometa4Resolved z =
        p10dc_low_rankformula_nometa4_resolve(h, rank);
    uint32_t rem = z.n;
    uint32_t local = rank - z.start;
    uint32_t state = depth;
    int factor_h = int(h);
    int prefix_corr = 0;
    const int source_rank_origin = int(rank) + z.base_delta;
    BkczCrossAccum sum = 0;

    // Physical support positions do not affect the ballot word inside a mask
    // group.  Iterate the n occupied symbols directly: no popcount/clz scan.
    for (uint32_t ordinal = 0; ordinal < z.n; ++ordinal) {
        const uint32_t bp = p10dc_rankformula_ballot_load(rem, uint32_t(factor_h));
        const uint32_t dest_contrib = bp & 0xffffu;
        const int diff = int(int16_t(bp >> 16));
        const bool take_r = factor_h > 0 && local < dest_contrib;
        if (take_r) {
            if (state == 1u) return sum;
            --state;
            --factor_h;
        } else {
            local -= dest_contrib;
            if (state == 1u) {
                const int source_rank =
                    source_rank_origin + prefix_corr - int(dest_contrib);
                sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
            }
            prefix_corr += diff;
            ++state;
            ++factor_h;
        }
        --rem;
    }
    return sum;
}
