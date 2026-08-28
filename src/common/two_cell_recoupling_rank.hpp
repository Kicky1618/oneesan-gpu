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

// Small immutable tables for the W<=28 sliding recoupling rank. The complete
// object is only about 13 KiB and is intended for CUDA constant memory or a
// cached read-only parameter block.
struct RankTables {
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

ONEESAN_TC_HD std::uint32_t low_mask(int bits) {
    return bits <= 0 ? 0u : ((std::uint32_t(1) << bits) - 1u);
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

ONEESAN_TC_HD Rank primitive_rank(
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
    Rank choose[kMaxWidth + 1][kMaxWidth + 1]{};
    for (int n = 0; n <= kMaxWidth; ++n) {
        choose[n][0] = choose[n][n] = 1;
        for (int k = 1; k < n; ++k)
            choose[n][k] = choose[n - 1][k - 1] + choose[n - 1][k];
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
                z += choose[rem][r] * t.state_block[ones + r];
            t.suffix[rem][ones] = z;
        }
    }
    return t;
}
#endif

} // namespace oneesan::twocell

#undef ONEESAN_TC_HD
