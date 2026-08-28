#pragma once

#include "two_cell_stationary_rank.hpp"

#if defined(__CUDACC__)
#define ONEESAN_TC_FUSE_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_FUSE_HD inline
#endif

namespace oneesan::twocell {

constexpr int kMaxFusionSteps = 6;

ONEESAN_TC_FUSE_HD std::uint32_t insert_support_window(
    std::uint32_t outer,
    int start,
    int count,
    std::uint32_t local
) {
    const std::uint32_t lo = outer & low_mask(start);
    const std::uint32_t hi = outer >> start;
    return lo | ((local & low_mask(count)) << start) | (hi << (start + count));
}

ONEESAN_TC_FUSE_HD Rank primitive_count_for_occupied(
    int occupied,
    const RankTables& t
) {
    if (occupied <= 0 || occupied > kMaxWidth || !(occupied & 1)) return 0;
    return t.primitive[occupied][1];
}

ONEESAN_TC_FUSE_HD Rank fusion_block_size(
    int steps,
    int outer_ones,
    const RankTables& t
) {
    Rank z = 0;
    for (int local = 0; local <= steps + 2; ++local)
        z += t.choose[steps + 2][local] *
             primitive_count_for_occupied(outer_ones + local, t);
    for (int local = 0; local <= steps; ++local)
        z += t.choose[steps][local] *
             primitive_count_for_occupied(outer_ones + 1 + local, t);
    return z;
}

ONEESAN_TC_FUSE_HD std::uint32_t fusion_outer_mask(
    PackedKey key,
    int start,
    int steps
) {
    return key.type == 0
        ? remove_support_window(key.support, start, steps + 2)
        : remove_support_window(key.support, start, steps + 1);
}

ONEESAN_TC_FUSE_HD Rank fusion_A_prefix(
    int steps,
    int outer_ones,
    std::uint32_t code,
    const RankTables& t
) {
    Rank base = 0;
    const std::uint32_t limit = std::uint32_t(1) << (steps + 2);
    if (code > limit) code = limit;
    for (std::uint32_t c = 0; c < code; ++c)
        base += primitive_count_for_occupied(outer_ones + popcount32(c), t);
    return base;
}

ONEESAN_TC_FUSE_HD Rank fusion_A_total(
    int steps,
    int outer_ones,
    const RankTables& t
) {
    return fusion_A_prefix(
        steps, outer_ones, std::uint32_t(1) << (steps + 2), t);
}

ONEESAN_TC_FUSE_HD Rank fusion_C_prefix(
    int steps,
    int outer_ones,
    std::uint32_t code,
    const RankTables& t
) {
    Rank base = fusion_A_total(steps, outer_ones, t);
    const std::uint32_t limit = std::uint32_t(1) << steps;
    if (code > limit) code = limit;
    for (std::uint32_t c = 0; c < code; ++c)
        base += primitive_count_for_occupied(outer_ones + 1 + popcount32(c), t);
    return base;
}

// Rank a state inside the union component of K_start ... K_{start+steps-1}.
// The caller may supply an already-computed primitive rank. Source Q_start has
// C support[start]=1, which is the fixed distinguished local C bit.
ONEESAN_TC_FUSE_HD Rank fusion_local_rank_with_primitive(
    PackedKey key,
    int start,
    int steps,
    int outer_ones,
    Rank primitive,
    const RankTables& t
) {
    if (key.type == 0) {
        const std::uint32_t code =
            (key.support >> start) & low_mask(steps + 2);
        return fusion_A_prefix(steps, outer_ones, code, t) + primitive;
    }
    const std::uint32_t code =
        (key.support >> (start + 1)) & low_mask(steps);
    return fusion_C_prefix(steps, outer_ones, code, t) + primitive;
}

ONEESAN_TC_FUSE_HD Rank fusion_local_rank(
    PackedKey key,
    int W,
    int start,
    int steps,
    int outer_ones,
    const RankTables& t
) {
    const int len = key.type ? W - 2 : W - 1;
    return fusion_local_rank_with_primitive(
        key, start, steps, outer_ones,
        primitive_rank(key.support, key.left, len, t), t);
}

// Construct the support mask for one block-local A sector.
ONEESAN_TC_FUSE_HD std::uint32_t fusion_A_support(
    std::uint32_t outer_mask,
    int start,
    int steps,
    std::uint32_t code
) {
    return insert_support_window(outer_mask, start, steps + 2, code);
}

// Construct the support mask for one block-local C sector in source Q_start.
// The active first bit is fixed occupied; `code` supplies the following k bits.
ONEESAN_TC_FUSE_HD std::uint32_t fusion_C_support(
    std::uint32_t outer_mask,
    int start,
    int steps,
    std::uint32_t code
) {
    const std::uint32_t local = 1u | ((code & low_mask(steps)) << 1);
    return insert_support_window(outer_mask, start, steps + 1, local);
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_FUSE_HD
