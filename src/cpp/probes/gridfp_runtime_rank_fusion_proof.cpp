#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;

Rank64 choose(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    Rank64 z = 1;
    for (int i = 1; i <= k; ++i)
        z = z * Rank64(n - k + i) / Rank64(i);
    return z;
}

Rank64 compact_rank_ref(std::uint32_t mask, int len) {
    const int ones = __builtin_popcount(mask & (len == 32 ? ~0u : ((1u << len) - 1u)));
    Rank64 rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        rank += choose(len - pos - 1, left);
        --left;
    }
    return rank;
}

std::uint32_t erase_two_ref(std::uint32_t local, int L, int a, int b) {
    std::uint32_t out = 0;
    int q = 0;
    for (int pos = 0; pos < L; ++pos) {
        if (pos == a || pos == b) continue;
        if ((local >> pos) & 1u) out |= 1u << q;
        ++q;
    }
    return out;
}

Rank64 fused_main_rank(std::uint32_t full, int W, int lo, int L) {
    Rank64 rank = 0;
    int seen_local = 0;
    std::uint32_t mask = full & (W == 32 ? ~0u : ((1u << W) - 1u));
    while (mask) {
        const int bit = 31 - __builtin_clz(mask);
        if (bit >= lo && bit < lo + L) {
            const int pos = bit - lo;
            ++seen_local;
            rank += choose(L - pos - 1, seen_local);
        }
        mask ^= 1u << bit;
    }
    return rank;
}

Rank64 fused_blocked_rank(
    std::uint32_t full, int W, int lo, int L, int a, int b
) {
    Rank64 rank = 0;
    int seen_local = 0;
    std::uint32_t mask = full & (W == 32 ? ~0u : ((1u << W) - 1u));
    while (mask) {
        const int bit = 31 - __builtin_clz(mask);
        if (bit >= lo && bit < lo + L) {
            const int pos = bit - lo;
            if (pos != a && pos != b) {
                ++seen_local;
                const int compact_pos = pos - int(a < pos) - int(b < pos);
                rank += choose((L - 2) - compact_pos - 1, seen_local);
            }
        }
        mask ^= 1u << bit;
    }
    return rank;
}
}

int main() {
    std::uint64_t exhaustive_main = 0;
    std::uint64_t exhaustive_blocked = 0;
    for (int L = 1; L <= 12; ++L) {
        const std::uint32_t limit = 1u << L;
        for (std::uint32_t local = 0; local < limit; ++local) {
            ++exhaustive_main;
            if (compact_rank_ref(local, L) != fused_main_rank(local, L, 0, L)) {
                std::cerr << "main mismatch L=" << L << " local=" << local << '\n';
                return 2;
            }
            if (L < 2) continue;
            for (int a = 0; a < L; ++a) {
                for (int b = 0; b < L; ++b) {
                    if (a == b) continue;
                    ++exhaustive_blocked;
                    const std::uint32_t compact = erase_two_ref(local, L, a, b);
                    if (compact_rank_ref(compact, L - 2) !=
                        fused_blocked_rank(local, L, 0, L, a, b)) {
                        std::cerr << "blocked mismatch L=" << L << " local=" << local
                                  << " a=" << a << " b=" << b << '\n';
                        return 3;
                    }
                }
            }
        }
    }

    std::mt19937_64 rng(0x72616e6b66757365ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int L = W / 2 + 1;
        const int lo = int(rng() % (W - L + 1));
        const std::uint32_t width_mask = (1u << W) - 1u;
        const std::uint32_t full = std::uint32_t(rng()) & width_mask;
        const std::uint32_t local = (full >> lo) & ((1u << L) - 1u);
        if (compact_rank_ref(local, L) != fused_main_rank(full, W, lo, L)) {
            std::cerr << "random main mismatch W=" << W << " lo=" << lo << '\n';
            return 4;
        }
        const int a = int(rng() % L);
        int b = int(rng() % (L - 1));
        if (b >= a) ++b;
        const std::uint32_t compact = erase_two_ref(local, L, a, b);
        if (compact_rank_ref(compact, L - 2) !=
            fused_blocked_rank(full, W, lo, L, a, b)) {
            std::cerr << "random blocked mismatch W=" << W << " lo=" << lo
                      << " a=" << a << " b=" << b << '\n';
            return 5;
        }
    }

    std::cout << "gridfp-runtime-rank-fusion-proof OK"
              << " exhaustive_L_max=12"
              << " exhaustive_main=" << exhaustive_main
              << " exhaustive_blocked=" << exhaustive_blocked
              << " random_cases=" << RANDOM
              << " production_W_max=28"
              << " main_support_rank_exact=1"
              << " blocked_support_rank_exact=1"
              << " one_high_to_low_support_pass=1\n";
    return 0;
}
