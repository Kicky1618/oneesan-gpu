#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;
Rank64 choose(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    Rank64 z = 1;
    for (int i = 1; i <= k; ++i) z = z * Rank64(n - k + i) / Rank64(i);
    return z;
}
std::uint32_t suffix_mask(int pos, int len) {
    const int width = len - pos;
    const std::uint32_t bits = width == 32 ? ~0u : ((std::uint32_t(1) << width) - 1u);
    return bits << pos;
}
std::uint32_t ordinary_ref(int len, int ones, Rank64 rank) {
    std::uint32_t mask = 0; int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        const int rem = len - pos - 1;
        const Rank64 z = choose(rem, left);
        if (rank < z) continue;
        rank -= z; mask |= std::uint32_t(1) << pos; --left;
    }
    return mask;
}
std::uint32_t ordinary_fast(int len, int ones, Rank64 rank) {
    std::uint32_t mask = 0; int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (!left) break;
        const int remaining = len - pos;
        if (left == remaining) { mask |= suffix_mask(pos, len); break; }
        const int rem = remaining - 1;
        const Rank64 z = choose(rem, left);
        if (rank < z) continue;
        rank -= z; mask |= std::uint32_t(1) << pos; --left;
    }
    return mask;
}
std::uint32_t constrained_ref(int len, int ones, int mark0, int mark1, Rank64 rank) {
    std::uint32_t mask = 0; int left = ones; bool seen = false;
    for (int pos = 0; pos < len; ++pos) {
        const int rem = len - pos - 1;
        const int future = (mark0 > pos) + (mark1 > pos);
        Rank64 z = choose(rem, left);
        if (!seen) z -= choose(rem - future, left);
        if (rank < z) continue;
        rank -= z; mask |= std::uint32_t(1) << pos; --left;
        if (pos == mark0 || pos == mark1) seen = true;
    }
    return mask;
}
std::uint32_t constrained_fast(int len, int ones, int mark0, int mark1, Rank64 rank) {
    std::uint32_t mask = 0; int left = ones; bool seen = false;
    for (int pos = 0; pos < len; ++pos) {
        if (!left) break;
        const int remaining = len - pos;
        if (left == remaining) { mask |= suffix_mask(pos, len); break; }
        const int rem = remaining - 1;
        const int future = (mark0 > pos) + (mark1 > pos);
        Rank64 z = choose(rem, left);
        if (!seen) z -= choose(rem - future, left);
        if (rank < z) continue;
        rank -= z; mask |= std::uint32_t(1) << pos; --left;
        if (pos == mark0 || pos == mark1) seen = true;
    }
    return mask;
}
Rank64 constrained_count(int len, int ones) { return choose(len, ones) - choose(len - 2, ones); }
}
int main() {
    std::uint64_t ordinary_cases = 0, constrained_cases = 0;
    for (int len = 1; len <= 14; ++len) {
        for (int ones = 0; ones <= len; ++ones) {
            const Rank64 n = choose(len, ones);
            for (Rank64 r = 0; r < n; ++r) {
                ++ordinary_cases;
                if (ordinary_ref(len, ones, r) != ordinary_fast(len, ones, r)) return 2;
            }
        }
    }
    for (int len = 2; len <= 10; ++len) {
        for (int a = 0; a < len; ++a) for (int b = 0; b < len; ++b) {
            if (a == b) continue;
            for (int ones = 1; ones <= len; ++ones) {
                const Rank64 n = constrained_count(len, ones);
                for (Rank64 r = 0; r < n; ++r) {
                    ++constrained_cases;
                    if (constrained_ref(len, ones, a, b, r) !=
                        constrained_fast(len, ones, a, b, r)) return 3;
                }
            }
        }
    }
    std::mt19937_64 rng(0x756e72616e6b6578ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int len = 2 + int(rng() % 27);
        const int ones = int(rng() % (len + 1));
        const Rank64 n = choose(len, ones);
        const Rank64 r = n ? rng() % n : 0;
        if (ordinary_ref(len, ones, r) != ordinary_fast(len, ones, r)) return 4;
        const int clen = 2 + int(rng() % 13);
        const int a = int(rng() % clen);
        int b = int(rng() % (clen - 1)); if (b >= a) ++b;
        const int cones = 1 + int(rng() % clen);
        const Rank64 cn = constrained_count(clen, cones);
        if (!cn) continue;
        const Rank64 cr = rng() % cn;
        if (constrained_ref(clen, cones, a, b, cr) !=
            constrained_fast(clen, cones, a, b, cr)) return 5;
    }
    std::cout << "gridfp-runtime-support-unrank-early-exit-proof OK"
              << " ordinary_exhaustive_len_max=14 ordinary_cases=" << ordinary_cases
              << " constrained_exhaustive_len_max=10 constrained_cases=" << constrained_cases
              << " random_cases=" << RANDOM
              << " production_len_max=28 exact=1 zero_tail_break=1 forced_one_suffix=1\n";
    return 0;
}
