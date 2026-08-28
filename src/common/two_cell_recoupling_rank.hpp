#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define ONEESAN_TC_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_HD inline
#endif

namespace oneesan::twocell {

using Rank = std::uint64_t;

constexpr int kMaxWidth = 28;
constexpr int kMaxOuterBits = kMaxWidth - 5;

struct PackedKey {
    std::uint32_t support = 0; // occupied positions
    std::uint32_t left = 0;    // L among occupied positions; occupied non-L is R
    std::uint8_t type = 0;     // 0=A, 1=C
};

// Small immutable tables for the W<=28 sliding recoupling rank and component
// label codec. The complete object is about 22 KiB and is intended for CUDA
// constant memory or a cached read-only parameter block.
struct RankTables {
    Rank choose[kMaxWidth + 1][kMaxWidth + 1]{};
    Rank primitive[kMaxWidth + 1][kMaxWidth + 2]{};
    Rank state_block[kMaxOuterBits + 1]{};
    Rank a_block[kMaxOuterBits + 1]{};
    Rank suffix[kMaxOuterBits + 1][kMaxOuterBits + 1]{};
    Rank a_prefix[kMaxOuterBits + 1][16]{};
    Rank c_prefix[kMaxOuterBits + 1][4]{};
};

ONEESAN_TC_HD int popcount32(std::uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#elif defined(__GNUG__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int n = 0;
    while (x) { x &= x - 1; ++n; }
    return n;
#endif
}

ONEESAN_TC_HD int ctz32(std::uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __ffs(static_cast<int>(x)) - 1;
#elif defined(__GNUG__) || defined(__clang__)
    return __builtin_ctz(x);
#else
    int n = 0;
    while (!(x & 1u)) { x >>= 1; ++n; }
    return n;
#endif
}

ONEESAN_TC_HD std::uint32_t low_mask(int bits) {
    return bits <= 0 ? 0u : ((std::uint32_t(1) << bits) - 1u);
}

ONEESAN_TC_HD Rank choose_count(int n, int k, const RankTables& t) {
    if (n < 0 || k < 0 || k > n || n > kMaxWidth) return 0;
    return t.choose[n][k];
}

// Delete a contiguous support window while preserving the relative order of
// all remaining bits. W<=28 keeps every shift below 32.
ONEESAN_TC_HD std::uint32_t remove_support_window(
    std::uint32_t support, int start, int count
) {
    const std::uint32_t lo = support & low_mask(start);
    const std::uint32_t hi = support >> (start + count);
    return lo | (hi << start);
}

ONEESAN_TC_HD std::uint32_t component_outer_mask(
    std::uint32_t label_support, int window
) {
    return remove_support_window(label_support, window, 3);
}

ONEESAN_TC_HD std::uint32_t state_outer_mask_A(
    std::uint32_t support, int window
) {
    return remove_support_window(support, window, 4);
}

ONEESAN_TC_HD std::uint32_t state_outer_mask_C(
    std::uint32_t support, int window
) {
    return remove_support_window(support, window, 3);
}

// Reference scan retained for probes. It walks every physical slot and is
// intentionally not used by the production rank helpers below.
ONEESAN_TC_HD Rank primitive_rank_scan(
    std::uint32_t support,
    std::uint32_t left,
    int len,
    const RankTables& t
) {
    const int occupied = popcount32(support & low_mask(len));
    Rank rank = 0;
    int h = 1;
    int seen = 0;
    for (int p = 0; p < len; ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (!(support & bit)) continue;
        const int rem = occupied - (++seen);
        if (left & bit) {
            if (h > 0) rank += t.primitive[rem][h - 1];
            ++h;
        } else {
            --h;
        }
    }
    return rank;
}

// R-first primitive lexicographic rank using only L endpoints. If the m-th L
// occurs at occupied ordinal j_m, the height before it is 1+2m-j_m. Choosing L
// skips exactly primitive[occupied-j_m-1][2m-j_m] valid R-first suffixes when
// that height is positive. A valid width<=28 one-defect word has at most 13 L
// endpoints, so this halves the worst production scan from 27 slots to 13
// ffs+popc+table-add iterations.
ONEESAN_TC_HD Rank primitive_rank(
    std::uint32_t support,
    std::uint32_t left,
    int len,
    const RankTables& t
) {
    const std::uint32_t active = support & low_mask(len);
    std::uint32_t lbits = left & active;
    const int occupied = popcount32(active);
    Rank rank = 0;
    int m = 0;
    while (lbits) {
        const int pos = ctz32(lbits);
        const int j = popcount32(active & low_mask(pos));
        const int h_minus_one = 2 * m - j;
        if (h_minus_one >= 0)
            rank += t.primitive[occupied - j - 1][h_minus_one];
        lbits &= lbits - 1;
        ++m;
    }
    return rank;
}

ONEESAN_TC_HD std::uint32_t support_unrank(
    int len,
    int ones,
    Rank rank,
    const RankTables& t
) {
    std::uint32_t support = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        const int rem = len - pos - 1;
        const Rank zero_count = choose_count(rem, left, t);
        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
    }
    return support;
}

// Materialize the R-first primitive rank on a fixed occupied support. This is
// the inverse of primitive_rank and needs no mate table.
ONEESAN_TC_HD std::uint32_t primitive_left_unrank(
    std::uint32_t support,
    int len,
    int occupied,
    Rank rank,
    const RankTables& t
) {
    std::uint32_t left_bits = 0;
    int h = 1;
    int seen = 0;
    for (int pos = 0; pos < len; ++pos) {
        const std::uint32_t bit = std::uint32_t(1) << pos;
        if (!(support & bit)) continue;
        const int rem = occupied - (++seen);
        const Rank r_count = h > 0 ? t.primitive[rem][h - 1] : 0;
        if (rank < r_count) {
            --h; // R
        } else {
            rank -= r_count;
            left_bits |= bit;
            ++h; // L
        }
    }
    return left_bits;
}

ONEESAN_TC_HD Rank component_label_count(int W, const RankTables& t) {
    const int len = W - 2;
    Rank total = 0;
    for (int occupied = 1; occupied <= len; occupied += 2)
        total += choose_count(len, occupied, t) * t.primitive[occupied][1];
    return total;
}

// Dense component id -> unrestricted width-(W-2) one-defect Motzkin label.
// Sectors are ordered by occupied count, then lexicographic support rank, then
// primitive R/L rank. W=28 has M_26=47,337,954,326 such labels.
ONEESAN_TC_HD PackedKey component_label_unrank(
    int W,
    Rank rank,
    const RankTables& t
) {
    const int len = W - 2;
    int occupied = 1;
    for (; occupied <= len; occupied += 2) {
        const Rank pc = t.primitive[occupied][1];
        const Rank sector = choose_count(len, occupied, t) * pc;
        if (rank < sector) {
            const Rank support_rank = rank / pc;
            const Rank primitive_rank_value = rank % pc;
            const std::uint32_t support = support_unrank(
                len, occupied, support_rank, t);
            const std::uint32_t left = primitive_left_unrank(
                support, len, occupied, primitive_rank_value, t);
            return PackedKey{support, left, 1}; // type is unused for a label
        }
        rank -= sector;
    }
    return PackedKey{};
}

ONEESAN_TC_HD Rank block_base(
    std::uint32_t outer_mask,
    int outer_bits,
    const RankTables& t
) {
    Rank base = 0;
    int ones = 0;
    for (int bit = outer_bits - 1; bit >= 0; --bit) {
        if ((outer_mask >> bit) & 1u) {
            base += t.suffix[bit][ones];
            ++ones;
        }
    }
    return base;
}

ONEESAN_TC_HD Rank rank_A_with_block(
    std::uint32_t support,
    std::uint32_t left,
    int W,
    int window,
    int outer_ones,
    Rank base,
    const RankTables& t
) {
    const int code = int((support >> window) & 0x0fu);
    return base + t.a_prefix[outer_ones][code] +
           primitive_rank(support, left, W - 1, t);
}

ONEESAN_TC_HD Rank rank_C_with_block(
    std::uint32_t support,
    std::uint32_t left,
    int W,
    int window,
    int outer_ones,
    Rank base,
    const RankTables& t
) {
    const int code = int(((support >> window) & 1u) |
                         (((support >> (window + 2)) & 1u) << 1));
    return base + t.a_block[outer_ones] + t.c_prefix[outer_ones][code] +
           primitive_rank(support, left, W - 2, t);
}

ONEESAN_TC_HD Rank rank_state(
    PackedKey key,
    int W,
    int window,
    const RankTables& t
) {
    const std::uint32_t mask = key.type == 0
        ? state_outer_mask_A(key.support, window)
        : state_outer_mask_C(key.support, window);
    const int k = popcount32(mask);
    const Rank base = block_base(mask, W - 5, t);
    return key.type == 0
        ? rank_A_with_block(key.support, key.left, W, window, k, base, t)
        : rank_C_with_block(key.support, key.left, W, window, k, base, t);
}

// For a K_i component labelled by u in M_{W-2}, every source shares the
// previous-window outer mask and every destination shares the current-window
// outer mask. Compute these two block descriptors once per component/warp.
struct ComponentBlocks {
    std::uint32_t input_mask = 0;
    std::uint32_t output_mask = 0;
    int input_ones = 0;
    int output_ones = 0;
    Rank input_base = 0;
    Rank output_base = 0;
};

ONEESAN_TC_HD ComponentBlocks component_blocks(
    std::uint32_t label_support,
    int W,
    int i,
    const RankTables& t
) {
    ComponentBlocks b{};
    b.input_mask = component_outer_mask(label_support, i - 1);
    b.output_mask = component_outer_mask(label_support, i);
    b.input_ones = popcount32(b.input_mask);
    b.output_ones = popcount32(b.output_mask);
    b.input_base = block_base(b.input_mask, W - 5, t);
    b.output_base = block_base(b.output_mask, W - 5, t);
    return b;
}

#ifndef __CUDA_ARCH__
inline RankTables make_rank_tables() {
    RankTables t{};
    for (int n = 0; n <= kMaxWidth; ++n) {
        t.choose[n][0] = t.choose[n][n] = 1;
        for (int k = 1; k < n; ++k)
            t.choose[n][k] = t.choose[n - 1][k - 1] + t.choose[n - 1][k];
    }

    t.primitive[0][0] = 1;
    for (int rem = 1; rem <= kMaxWidth; ++rem) {
        for (int h = 0; h <= kMaxWidth; ++h) {
            Rank z = t.primitive[rem - 1][h + 1];
            if (h > 0) z += t.primitive[rem - 1][h - 1];
            t.primitive[rem][h] = z;
        }
    }

    auto pc = [&](int occupied) -> Rank {
        if (occupied <= 0 || occupied > kMaxWidth || !(occupied & 1)) return 0;
        return t.primitive[occupied][1];
    };

    for (int k = 0; k <= kMaxOuterBits; ++k) {
        Rank ap = 0;
        for (int code = 0; code < 16; ++code) {
            t.a_prefix[k][code] = ap;
            ap += pc(k + popcount32(static_cast<std::uint32_t>(code)));
        }
        t.a_block[k] = ap;

        Rank cp = 0;
        for (int code = 0; code < 4; ++code) {
            t.c_prefix[k][code] = cp;
            cp += pc(k + 1 + popcount32(static_cast<std::uint32_t>(code)));
        }
        t.state_block[k] = ap + cp;
    }

    for (int rem = 0; rem <= kMaxOuterBits; ++rem) {
        for (int ones = 0; ones <= kMaxOuterBits; ++ones) {
            Rank z = 0;
            for (int r = 0; r <= rem && ones + r <= kMaxOuterBits; ++r)
                z += t.choose[rem][r] * t.state_block[ones + r];
            t.suffix[rem][ones] = z;
        }
    }
    return t;
}
#endif

} // namespace oneesan::twocell

#undef ONEESAN_TC_HD
