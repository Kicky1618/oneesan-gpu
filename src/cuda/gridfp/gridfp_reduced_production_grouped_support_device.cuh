#pragma once

#include "gridfp_reduced_production_grouped_device.cuh"

namespace oneesan::gridfp::reducedprod {

// Owner lookup when the physical support mask is already known.  This avoids
// materializing a MateID and is sufficient for traffic-only analysis.
__device__ __forceinline__ int grouped_support_owner_device(
    std::uint32_t full_support,
    int W,
    bool reverse,
    int tile_start,
    int K,
    int ngpu
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;
    const std::uint32_t outer = compact_outside_window_device(
        full_support, W, lo, hi);
    return weighted_outer_owner_device(outer, L, O, ngpu);
}

// Rank the beginning of the primitive slab for one physical support mask.
// equal_run_key0_device() always chooses primitive rank zero, so computing a
// MateID and then running primitive_rank_device() is redundant.  This helper
// reproduces grouped_rank_device(equal_run_key0_device(...)) using support
// combinadics only.
__device__ __forceinline__ GroupedDeviceRank grouped_support_slab_rank_device(
    std::uint32_t full_support,
    bool blocked,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const Rank64* owner_begin
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;

    const std::uint32_t outer = compact_outside_window_device(
        full_support, W, lo, hi);
    const int outer_ones = __popc(outer);
    const Rank64 group = outer_group_size_device(L, outer_ones);
    const Rank64 sr_outer = compact_support_rank_device(outer, O, outer_ones);
    Rank64 prefix = 0;
    for (int t = 0; t < outer_ones; ++t)
        prefix += choose_device(O, t) * outer_group_size_device(L, t);
    const Rank64 group_base = prefix + sr_outer * group;
    const int owner = weighted_outer_owner_device(outer, L, O, ngpu);

    const std::uint32_t local_mask = local_window_support_device(
        full_support, lo, L);
    const int local_ones = __popc(local_mask);
    const int occupied = outer_ones + local_ones;
    const Rank64 pc = RP_PRIMITIVE[occupied][1];
    Rank64 within = group_local_sector_offset_device(
        L, outer_ones, local_ones);

    if (!blocked) {
        const Rank64 sr = compact_support_rank_device(
            local_mask, L, local_ones);
        within += sr * pc;
    } else {
        const int missing_bit = reverse ? q - 1 : q;
        const int fixed_bit = reverse ? q : q - 1;
        const int missing_pos = missing_bit - lo;
        const int fixed_pos = fixed_bit - lo;
        const std::uint32_t compact = erase_two_local_bits_device(
            local_mask, L, missing_pos, fixed_pos);
        const Rank64 sr = compact_support_rank_device(
            compact, L - 2, local_ones - 1);
        within += choose_device(L, local_ones) * pc + sr * pc;
    }

    return GroupedDeviceRank{
        owner, group_base - owner_begin[owner] + within};
}

} // namespace oneesan::gridfp::reducedprod
