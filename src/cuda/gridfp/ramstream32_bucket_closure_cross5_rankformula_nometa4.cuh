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
    uint32_t remaining = z.mask;
    uint32_t rem = uint32_t(__popc(z.mask));
    uint32_t local = rank - z.start;
    uint32_t state = depth;
    int factor_h = int(h);
    int prefix_corr = 0;
    const int source_rank_origin = int(rank) + z.base_delta;
    BkczCrossAccum sum = 0;

    // The group-local rank is the ballot-lexicographic rank.  At each occupied
    // position dest_contrib is exactly the size of the R-first subtree, so the
    // same LUT load both unranks the symbol and supplies the L->R rank formula.
    while (remaining) {
        const int pos = 31 - __clz(remaining);
        remaining ^= 1u << pos;
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
