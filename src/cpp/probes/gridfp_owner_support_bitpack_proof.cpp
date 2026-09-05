#include <cstdint>
#include <iostream>
#include <random>

namespace {
std::uint32_t reverse32(std::uint32_t x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

std::uint32_t outer_ref(std::uint32_t outer, int W, int lo, int hi) {
    std::uint32_t full = 0;
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if ((outer >> q) & 1u) full |= 1u << bit;
        ++q;
    }
    return full;
}
std::uint32_t outer_fast(std::uint32_t outer, int W, int lo, int hi) {
    const int O = W - (hi - lo + 1);
    if (O < 32) outer &= O ? ((1u << O) - 1u) : 0u;
    const std::uint32_t low_mask = lo ? ((1u << lo) - 1u) : 0u;
    return (outer & low_mask) | ((outer >> lo) << (hi + 1));
}

std::uint32_t local_ref(std::uint32_t local, int L, int lo, int missing) {
    std::uint32_t full = 0;
    int q = 0;
    for (int bit = lo; bit < lo + L; ++bit) {
        if (bit == missing) continue;
        if ((local >> q) & 1u) full |= 1u << bit;
        ++q;
    }
    return full;
}
std::uint32_t local_fast(std::uint32_t local, int L, int lo, int missing) {
    const int pos = missing - lo;
    const int len = L - 1;
    if (len < 32) local &= len ? ((1u << len) - 1u) : 0u;
    const std::uint32_t low_mask = pos ? ((1u << pos) - 1u) : 0u;
    const std::uint32_t expanded =
        (local & low_mask) | ((local & ~low_mask) << 1);
    return expanded << lo;
}

std::uint32_t label_ref(std::uint32_t full, int W, int missing) {
    std::uint32_t out = 0;
    int q = 0;
    for (int pos = 0; pos < W; ++pos) {
        const int bit = W - 1 - pos;
        if (bit == missing) continue;
        if ((full >> bit) & 1u) out |= 1u << q;
        ++q;
    }
    return out;
}
std::uint32_t label_fast(std::uint32_t full, int W, int missing) {
    const std::uint32_t low_mask = missing ? ((1u << missing) - 1u) : 0u;
    const std::uint32_t compact =
        (full & low_mask) | ((full >> (missing + 1)) << missing);
    return reverse32(compact) >> (32 - (W - 1));
}
}

int main() {
    std::uint64_t exhaustive_outer = 0, exhaustive_local = 0, exhaustive_label = 0;
    for (int W = 2; W <= 10; ++W) {
        for (int lo = 0; lo < W; ++lo) {
            for (int hi = lo; hi < W; ++hi) {
                const int O = W - (hi - lo + 1);
                const std::uint32_t lim = 1u << O;
                for (std::uint32_t x = 0; x < lim; ++x) {
                    ++exhaustive_outer;
                    if (outer_ref(x, W, lo, hi) != outer_fast(x, W, lo, hi)) return 2;
                }
                const int L = hi - lo + 1;
                if (L < 1) continue;
                for (int missing = lo; missing <= hi; ++missing) {
                    const std::uint32_t llim = 1u << (L - 1);
                    for (std::uint32_t x = 0; x < llim; ++x) {
                        ++exhaustive_local;
                        if (local_ref(x, L, lo, missing) != local_fast(x, L, lo, missing)) return 3;
                    }
                }
            }
        }
        const std::uint32_t flim = 1u << W;
        for (int missing = 0; missing < W; ++missing) {
            for (std::uint32_t full = 0; full < flim; ++full) {
                ++exhaustive_label;
                if (label_ref(full, W, missing) != label_fast(full, W, missing)) return 4;
            }
        }
    }

    std::mt19937_64 rng(0x737570706f727462ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int L = W / 2 + 1;
        const int lo = int(rng() % (W - L + 1));
        const int hi = lo + L - 1;
        const int O = W - L;
        const std::uint32_t outer = std::uint32_t(rng()) & ((1u << O) - 1u);
        const int missing = lo + int(rng() % L);
        const std::uint32_t local = std::uint32_t(rng()) & ((1u << (L - 1)) - 1u);
        if (outer_ref(outer, W, lo, hi) != outer_fast(outer, W, lo, hi)) return 5;
        if (local_ref(local, L, lo, missing) != local_fast(local, L, lo, missing)) return 6;
        const std::uint32_t full = std::uint32_t(rng()) & ((1u << W) - 1u);
        if (label_ref(full, W, missing) != label_fast(full, W, missing)) return 7;
    }

    std::cout << "gridfp-owner-support-bitpack-proof OK"
              << " exhaustive_W_max=10"
              << " exhaustive_outer=" << exhaustive_outer
              << " exhaustive_local=" << exhaustive_local
              << " exhaustive_label=" << exhaustive_label
              << " random_cases=" << RANDOM
              << " production_W_max=28"
              << " outer_expand_exact=1 local_expand_exact=1 label_reverse_exact=1\n";
    return 0;
}
