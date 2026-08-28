#pragma once

#include "two_cell_recoupling_rank.hpp"

#if defined(__CUDACC__)
#define ONEESAN_TC_ST_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_ST_HD inline
#endif

namespace oneesan::twocell {

struct StationaryRankTables {
    Rank a_sector[kMaxWidth + 1][kMaxWidth + 1]{};
    Rank c_sector[kMaxWidth + 1][kMaxWidth + 1]{};
    Rank a_total[kMaxWidth + 1]{};
    Rank total[kMaxWidth + 1]{};
};

ONEESAN_TC_ST_HD std::uint32_t stationary_c_support(
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

ONEESAN_TC_ST_HD std::uint32_t support_insert_bit(
    std::uint32_t support,
    int pos,
    bool bit
) {
    const std::uint32_t lo = support & low_mask(pos);
    const std::uint32_t hi = support >> pos;
    return lo | (std::uint32_t(bit) << pos) | (hi << (pos + 1));
}

ONEESAN_TC_ST_HD std::uint32_t support_remove_bit(
    std::uint32_t support,
    int pos
) {
    const std::uint32_t lo = support & low_mask(pos);
    return lo | ((support >> (pos + 1)) << pos);
}

ONEESAN_TC_ST_HD Rank support_rank_fixed(
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

ONEESAN_TC_ST_HD Rank stationary_base_A(
    std::uint32_t support,
    int W,
    const RankTables& t,
    const StationaryRankTables& s
) {
    const int occupied = popcount32(support & low_mask(W - 1));
    const Rank sr = support_rank_fixed(support, W - 1, occupied, t);
    return s.a_sector[W][occupied] + sr * t.primitive[occupied][1];
}

ONEESAN_TC_ST_HD Rank stationary_base_C(
    std::uint32_t support,
    int W,
    int active,
    const RankTables& t,
    const StationaryRankTables& s
) {
    const std::uint32_t canonical = stationary_c_support(support, active);
    const int occupied = popcount32(canonical & low_mask(W - 2));
    // canonical bit zero is always occupied, so only rank the remaining support.
    const Rank sr = support_rank_fixed(
        canonical >> 1, W - 3, occupied - 1, t);
    return s.c_sector[W][occupied] + sr * t.primitive[occupied][1];
}

ONEESAN_TC_ST_HD Rank stationary_rank_A_with_primitive(
    std::uint32_t support,
    int W,
    Rank primitive,
    const RankTables& t,
    const StationaryRankTables& s
) {
    return stationary_base_A(support, W, t, s) + primitive;
}

ONEESAN_TC_ST_HD Rank stationary_rank_C_with_primitive(
    std::uint32_t support,
    int W,
    int active,
    Rank primitive,
    const RankTables& t,
    const StationaryRankTables& s
) {
    return stationary_base_C(support, W, active, t, s) + primitive;
}

ONEESAN_TC_ST_HD Rank stationary_rank_with_primitive(
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

ONEESAN_TC_ST_HD Rank stationary_rank(
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

// A component has only one, three, or five distinct stationary support bases:
// singleton -> one A support; retained non-deep -> C+A+A; deep -> those three
// plus one sparse A base and one common strand-cut A support. These bases depend
// only on label support and active position, never on primitive connectivity, so
// a support tile can compute them once and reuse them for every primitive rank.
struct StationaryComponentBases {
    Rank slot[5]{};
    int count = 0;
    bool retained = false;
    bool xN_deep = false;
};

ONEESAN_TC_ST_HD StationaryComponentBases stationary_component_bases(
    std::uint32_t label_support,
    int W,
    int i,
    const RankTables& t,
    const StationaryRankTables& s
) {
    StationaryComponentBases out{};
    const bool at_i = ((label_support >> i) & 1u) != 0;
    const bool at_next = ((label_support >> (i + 1)) & 1u) != 0;
    out.retained = at_i;
    out.xN_deep = at_i && !at_next;

    if (!at_i) {
        // C_i(Nq) is represented by A_i(LRq): remove the vacant label slot and
        // replace it by an occupied LR pair.
        std::uint32_t z = support_remove_bit(label_support, i);
        z = support_insert_bit(z, i, true);
        z = support_insert_bit(z, i + 1, true);
        out.slot[0] = stationary_base_A(z, W, t, s);
        out.count = 1;
        return out;
    }

    // Direct retained coordinates C(u), A(insertN_i u), A(insertN_{i+1} u).
    out.slot[0] = stationary_base_C(label_support, W, i, t, s);
    out.slot[1] = stationary_base_A(
        support_insert_bit(label_support, i, false), W, t, s);
    out.slot[2] = stationary_base_A(
        support_insert_bit(label_support, i + 1, false), W, t, s);
    out.count = 3;

    // Precompute the two supports used only by a deep component. For xN this
    // removes the vacant second slot. For LR, deep_collapse removes the R and
    // turns the remaining L slot into a vacancy.
    std::uint32_t collapsed = support_remove_bit(label_support, i + 1);
    if (at_next) collapsed &= ~(std::uint32_t(1) << i);

    std::uint32_t sparse = support_insert_bit(collapsed, i, false);
    sparse = support_insert_bit(sparse, i, false);
    out.slot[3] = stationary_base_A(sparse, W, t, s);

    std::uint32_t cut = support_insert_bit(collapsed, i, true);
    cut = support_insert_bit(cut, i + 1, true);
    out.slot[4] = stationary_base_A(cut, W, t, s);
    out.count = 5;
    return out;
}

ONEESAN_TC_ST_HD Rank stationary_component_source_base(
    const StationaryComponentBases& b,
    int source_index
) {
    if (!b.retained) return b.slot[0];
    if (source_index < 3) return b.slot[source_index];
    return source_index == 3 ? b.slot[3] : b.slot[4];
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

#undef ONEESAN_TC_ST_HD
