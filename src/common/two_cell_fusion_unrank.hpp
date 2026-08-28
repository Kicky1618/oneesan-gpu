#pragma once

#include "two_cell_fusion_rank.hpp"

#if defined(__CUDACC__)
#define ONEESAN_TC_FUSE_UNRANK_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_FUSE_UNRANK_HD inline
#endif

namespace oneesan::twocell {

struct FusionDecodedState {
    PackedKey key{};
    Rank primitive = 0;
    bool valid = false;
};

// Inverse of fusion_local_rank_at. `outer_mask` is the invariant support mask
// outside the fused segment. C coordinates are first constructed in Q_start
// and then rebased to the requested active position without changing the
// compact primitive connectivity word.
ONEESAN_TC_FUSE_UNRANK_HD FusionDecodedState fusion_local_unrank_at(
    Rank rank,
    std::uint32_t outer_mask,
    int W,
    int start,
    int steps,
    int active,
    const RankTables& t
) {
    FusionDecodedState out{};
    const int outer_ones = popcount32(outer_mask);
    const Rank block_size = fusion_block_size(steps, outer_ones, t);
    if (rank >= block_size) return out;

    const std::uint32_t a_codes = std::uint32_t(1) << (steps + 2);
    for (std::uint32_t code = 0; code < a_codes; ++code) {
        const int occupied = outer_ones + popcount32(code);
        const Rank pc = primitive_count_for_occupied(occupied, t);
        if (rank >= pc) {
            rank -= pc;
            continue;
        }
        const std::uint32_t support = fusion_A_support(
            outer_mask, start, steps, code);
        const std::uint32_t left = primitive_left_unrank(
            support, W - 1, occupied, rank, t);
        out.key = PackedKey{support, left, 0};
        out.primitive = rank;
        out.valid = true;
        return out;
    }

    const std::uint32_t c_codes = std::uint32_t(1) << steps;
    for (std::uint32_t code = 0; code < c_codes; ++code) {
        const int occupied = outer_ones + 1 + popcount32(code);
        const Rank pc = primitive_count_for_occupied(occupied, t);
        if (rank >= pc) {
            rank -= pc;
            continue;
        }
        const std::uint32_t reference_support = fusion_C_support(
            outer_mask, start, steps, code);
        const std::uint32_t support = stationary_c_rebase_support(
            reference_support, start, active);
        const std::uint32_t left = primitive_left_unrank(
            support, W - 2, occupied, rank, t);
        out.key = PackedKey{support, left, 1};
        out.primitive = rank;
        out.valid = true;
        return out;
    }
    return out;
}

ONEESAN_TC_FUSE_UNRANK_HD FusionDecodedState fusion_local_unrank(
    Rank rank,
    std::uint32_t outer_mask,
    int W,
    int start,
    int steps,
    const RankTables& t
) {
    return fusion_local_unrank_at(rank, outer_mask, W, start, steps, start, t);
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_FUSE_UNRANK_HD
