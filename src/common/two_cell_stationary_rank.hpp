#pragma once

#include "two_cell_recoupling_rank.hpp"

namespace oneesan::twocell {

struct StationaryRankTables {
    Rank a_sector[kMaxWidth + 1][kMaxWidth + 1]{};
    Rank c_sector[kMaxWidth + 1][kMaxWidth + 1]{};
    Rank a_total[kMaxWidth + 1]{};
    Rank total[kMaxWidth + 1]{};
};

ONEESAN_TC_HD std::uint32_t stationary_c_support(
    std::uint32_t support,
    int active
) {
    // C source coordinates always have support[active]=1. Canonicalize the
    // distinguished active strand to slot zero while preserving the compact
    // occupied-symbol order. This is a right rotation of support[0..active].
    const std::uint32_t prefix = low_mask(active);
    const std::uint32_t suffix = ~low_mask(active + 1);
    return (support & suffix) | 1u | ((support & prefix) << 1);
}

ONEESAN_TC_HD Rank support_rank_fixed(
    std::uint32_t support,
    int len,
    int ones,
    const RankTables& t
) {
    Rank rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        const std::uint32_t bit = std::uint32_t(1) << pos;
        if (!(support & bit)) continue;
        rank += choose_count(len - pos - 1, left, t);
        --left;
    }
    return rank;
}

ONEESAN_TC_HD Rank stationary_rank_A_with_primitive(
    std::uint32_t support,
    int W,
    Rank primitive,
    const RankTables& t,
    const StationaryRankTables& s
) {
    const int occupied = popcount32(support & low_mask(W - 1));
    const Rank sr = support_rank_fixed(support, W - 1, occupied, t);
    return s.a_sector[W][occupied] + sr * t.primitive[occupied][1] + primitive;
}

ONEESAN_TC_HD Rank stationary_rank_C_with_primitive(
    std::uint32_t support,
    int W,
    int active,
    Rank primitive,
    const RankTables& t,
    const StationaryRankTables& s
) {
    const std::uint32_t canonical = stationary_c_support(support, active);
    const int occupied = popcount32(canonical & low_mask(W - 2));
    // canonical bit zero is always occupied, so only rank the remaining support.
    const Rank sr = support_rank_fixed(
        canonical >> 1, W - 3, occupied - 1, t);
    return s.c_sector[W][occupied] + sr * t.primitive[occupied][1] + primitive;
}

ONEESAN_TC_HD Rank stationary_rank_with_primitive(
    PackedKey key,
    int W,
    int active,
    Rank primitive,
    const RankTables& t,
    const StationaryRankTables& s
) {
    return key.type == 0
        ? stationary_rank_A_with_primitive(key.support, W, primitive, t, s)
        : stationary_rank_C_with_primitive(key.support, W, active, primitive, t, s);
}

ONEESAN_TC_HD Rank stationary_rank(
    PackedKey key,
    int W,
    int active,
    const RankTables& t,
    const StationaryRankTables& s
) {
    const int len = key.type ? W - 2 : W - 1;
    const Rank primitive = primitive_rank(key.support, key.left, len, t);
    return stationary_rank_with_primitive(key, W, active, primitive, t, s);
}

#ifndef __CUDA_ARCH__
inline StationaryRankTables make_stationary_rank_tables(const RankTables& t) {
    StationaryRankTables s{};
    for (int W = 3; W <= kMaxWidth; ++W) {
        Rank a = 0;
        for (int occupied = 1; occupied <= W - 1; occupied += 2) {
            s.a_sector[W][occupied] = a;
            a += t.choose[W - 1][occupied] * t.primitive[occupied][1];
        }
        s.a_total[W] = a;

        Rank c = 0;
        for (int occupied = 1; occupied <= W - 2; occupied += 2) {
            s.c_sector[W][occupied] = a + c;
            c += t.choose[W - 3][occupied - 1] * t.primitive[occupied][1];
        }
        s.total[W] = a + c;
    }
    return s;
}
#endif

} // namespace oneesan::twocell
