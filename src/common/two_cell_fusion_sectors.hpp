#pragma once

#include "two_cell_fusion_rank.hpp"

#if defined(__CUDACC__)
#define ONEESAN_TC_FSECT_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_FSECT_HD inline
#endif

namespace oneesan::twocell {

struct FusionSector {
    Rank local_base = 0;
    Rank global_base = 0;
    Rank count = 0;
    std::uint32_t support = 0;
    std::uint8_t type = 0; // 0=A, 1=C
    std::uint16_t code = 0;
    bool valid = false;
};

ONEESAN_TC_FSECT_HD int fusion_sector_count(int steps) {
    return (1 << (steps + 2)) + (1 << steps);
}

// A fused outer-support block is a concatenation of primitive-rank intervals.
// For fixed A/C support, stationary rank is exactly support_base + primitive.
// C uses the source-Q_start representation below; stationary canonicalization
// makes the resulting global base independent of the current active position.
ONEESAN_TC_FSECT_HD FusionSector fusion_sector(
    int index,
    std::uint32_t outer_mask,
    int W,
    int start,
    int steps,
    const RankTables& t,
    const StationaryRankTables& s
) {
    FusionSector out{};
    const int outer_ones = popcount32(outer_mask);
    const int a_count = 1 << (steps + 2);
    const int c_count = 1 << steps;
    if (index < 0 || index >= a_count + c_count) return out;

    if (index < a_count) {
        const std::uint32_t code = static_cast<std::uint32_t>(index);
        const std::uint32_t support = fusion_A_support(
            outer_mask, start, steps, code);
        const int occupied = outer_ones + popcount32(code);
        const Rank count = primitive_count_for_occupied(occupied, t);
        out.local_base = fusion_A_prefix(steps, outer_ones, code, t);
        out.global_base = stationary_base_A(support, W, t, s);
        out.count = count;
        out.support = support;
        out.type = 0;
        out.code = static_cast<std::uint16_t>(code);
        out.valid = count != 0;
        return out;
    }

    const std::uint32_t code = static_cast<std::uint32_t>(index - a_count);
    const std::uint32_t support = fusion_C_support(
        outer_mask, start, steps, code);
    const int occupied = outer_ones + 1 + popcount32(code);
    const Rank count = primitive_count_for_occupied(occupied, t);
    out.local_base = fusion_C_prefix(steps, outer_ones, code, t);
    out.global_base = stationary_base_C(support, W, start, t, s);
    out.count = count;
    out.support = support;
    out.type = 1;
    out.code = static_cast<std::uint16_t>(code);
    out.valid = count != 0;
    return out;
}

ONEESAN_TC_FSECT_HD Rank fusion_sector_nonempty_count(
    std::uint32_t outer_mask,
    int W,
    int start,
    int steps,
    const RankTables& t,
    const StationaryRankTables& s
) {
    Rank n = 0;
    const int sectors = fusion_sector_count(steps);
    for (int q = 0; q < sectors; ++q)
        n += fusion_sector(q, outer_mask, W, start, steps, t, s).valid ? 1 : 0;
    return n;
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_FSECT_HD
