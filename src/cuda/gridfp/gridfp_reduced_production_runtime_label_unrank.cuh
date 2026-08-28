#pragma once

#include "gridfp_reduced_production_owner_component_plan_device.cuh"
#include "gridfp_reduced_production_runtime_fastdiv64.cuh"

namespace oneesan::gridfp::reducedprod {

// Runtime-only counterpart of owner_component_label_unrank_planned_device.
// The outer-support group division stays in the legacy form for now; the
// universal primitive-count div/mod pair uses exact reciprocal multiply-high.
__device__ __forceinline__ MateID runtime_owner_component_label_unrank_planned_device(
    int W,
    int p,
    bool reverse,
    int tile_start,
    int K,
    const OwnerComponentPlanDevice& plan,
    Rank64 rank
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? tile_start - 1 : tile_start - K - 1;
    const int hi = lo + L - 1;

    int outer_ones = -1;
    Rank64 local = 0;
    for (int r = 0; r <= O; ++r) {
        const Rank64 next = plan.prefix[r + 1];
        if (rank < next) {
            outer_ones = r;
            local = rank - plan.prefix[r];
            break;
        }
    }
    if (outer_ones < 0) return 0;

    const Rank64 component_group = plan.component_group[outer_ones];
    if (!component_group) return 0;
    const Rank64 outer_sr = plan.sr_begin[outer_ones] + local / component_group;
    Rank64 within = local % component_group;
    const std::uint32_t outer = support_unrank_mask_device(O, outer_ones, outer_sr);

    int local_ones = -1;
    Rank64 local_sr = 0;
    Rank64 primitive_rank = 0;
    for (int l = 0; l <= L - 1; ++l) {
        const int occupied = outer_ones + l;
        if (!(occupied & 1)) continue;
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        const Rank64 supports = choose_device(L - 1, l) - choose_device(L - 3, l);
        const Rank64 n = supports * pc;
        if (within < n) {
            local_ones = l;
            runtime_fastdivmod64_magic(
                within, pc, RP_RUNTIME_PRIMITIVE1_MAGIC[occupied],
                local_sr, primitive_rank);
            break;
        }
        within -= n;
    }
    if (local_ones < 0) return 0;

    const int missing = reverse ? p - 1 : p;
    const int mark_a = reverse ? p : p - 1;
    const int mark_b = reverse ? p + 1 : p - 2;
    if (missing < lo || missing > hi || mark_a < lo || mark_a > hi ||
        mark_b < lo || mark_b > hi) return 0;

    const int mark0 = owner_local_index_without_missing_device(mark_a, lo, missing);
    const int mark1 = owner_local_index_without_missing_device(mark_b, lo, missing);
    if (mark0 < 0 || mark0 >= L - 1 || mark1 < 0 || mark1 >= L - 1 || mark0 == mark1)
        return 0;

    const std::uint32_t local_support = component_support_unrank_device(
        L - 1, local_ones, mark0, mark1, local_sr);

    std::uint32_t full = 0;
    owner_expand_outer_support_device(outer, W, lo, hi, full);
    int cp = 0;
    for (int bit = lo; bit <= hi; ++bit) {
        if (bit == missing) continue;
        if ((local_support >> cp) & 1u) full |= std::uint32_t(1) << bit;
        ++cp;
    }
    if ((full >> missing) & 1u) return 0;

    const int occupied = outer_ones + local_ones;
    const std::uint32_t label_support = owner_label_lr_support_device(full, W, missing);
    return materialize_primitive_device(label_support, W - 1, occupied, primitive_rank);
}

} // namespace oneesan::gridfp::reducedprod
