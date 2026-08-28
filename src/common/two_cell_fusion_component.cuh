#pragma once

#include "two_cell_component_device.cuh"
#include "two_cell_fusion_rank.hpp"

#if defined(__CUDACC__)
#define ONEESAN_TC_FCOMP_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_FCOMP_HD inline
#endif

namespace oneesan::twocell {

struct FusionComponentLabel {
    PackedWord word{};
    Rank primitive = 0;
    std::uint32_t local_support = 0;
    bool valid = false;
};

// A k-step union block removes k+1 positions from the width-(W-2) component
// label.  The remaining outer support is invariant for every component of all
// k transfers in the segment.  For fixed outer popcount o, enumerate the
// 2^(k+1) local support patterns and then the primitive connectivity rank.
ONEESAN_TC_FCOMP_HD Rank fusion_component_count(
    int steps,
    int outer_ones,
    const RankTables& t
) {
    Rank z = 0;
    const std::uint32_t limit = std::uint32_t(1) << (steps + 1);
    for (std::uint32_t code = 0; code < limit; ++code)
        z += primitive_count_for_occupied(
            outer_ones + popcount32(code), t);
    return z;
}

ONEESAN_TC_FCOMP_HD FusionComponentLabel fusion_component_unrank(
    Rank rank,
    std::uint32_t outer_mask,
    int W,
    int start,
    int steps,
    const RankTables& t
) {
    FusionComponentLabel out{};
    const int outer_ones = popcount32(outer_mask);
    const std::uint32_t limit = std::uint32_t(1) << (steps + 1);
    for (std::uint32_t code = 0; code < limit; ++code) {
        const int occupied = outer_ones + popcount32(code);
        const Rank pc = primitive_count_for_occupied(occupied, t);
        if (rank >= pc) {
            rank -= pc;
            continue;
        }
        const std::uint32_t support = insert_support_window(
            outer_mask, start, steps + 1, code);
        const std::uint32_t left = primitive_left_unrank(
            support, W - 2, occupied, rank, t);
        out.word = PackedWord{
            support, left, static_cast<std::uint8_t>(W - 2)};
        out.primitive = rank;
        out.local_support = code;
        out.valid = true;
        return out;
    }
    return out;
}

ONEESAN_TC_FCOMP_HD std::uint32_t fusion_component_outer_mask(
    PackedWord label,
    int start,
    int steps
) {
    return remove_support_window(label.support, start, steps + 1);
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_FCOMP_HD
