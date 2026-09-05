#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
using MateID = std::uint64_t;
using Rank64 = std::uint64_t;

enum MateValue : std::uint8_t { N=0, R=1, L=2, X=3 };
MateValue mget(MateID m, int k) { return MateValue((m >> (2 * k)) & 3ULL); }
MateID minsert_n(MateID m, int k) {
    const MateID lowmask = k ? ((MateID(1) << (2 * k)) - 1ULL) : 0ULL;
    return (m & lowmask) | ((m & ~lowmask) << 2);
}

std::uint32_t support(MateID m, int width) {
    std::uint32_t out = 0;
    for (int i = 0; i < width; ++i)
        if (mget(m, i) != N) out |= std::uint32_t(1) << i;
    return out;
}

std::uint32_t insert_zero_support(std::uint32_t m, int width, int k) {
    if (width < 32) m &= (std::uint32_t(1) << width) - 1u;
    const std::uint32_t lowmask = k ? ((std::uint32_t(1) << k) - 1u) : 0u;
    return (m & lowmask) | ((m & ~lowmask) << 1);
}

std::vector<int> occupied_sequence(MateID m, int width) {
    std::vector<int> out;
    for (int bit = width - 1; bit >= 0; --bit) {
        const auto v = mget(m, bit);
        if (v != N) out.push_back(int(v));
    }
    return out;
}

Rank64 choose(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    Rank64 z = 1;
    for (int i = 1; i <= k; ++i)
        z = z * Rank64(n - k + i) / Rank64(i);
    return z;
}

Rank64 fused_full_block_rank(
    std::uint32_t full, int W, int lo, int Lw, int erase_a, int erase_b
) {
    Rank64 rank = 0;
    int seen = 0;
    for (int bit = W - 1; bit >= 0; --bit) {
        if (((full >> bit) & 1u) == 0) continue;
        if (bit < lo || bit >= lo + Lw) continue;
        const int pos = bit - lo;
        if (pos == erase_a || pos == erase_b) continue;
        ++seen;
        const int compact_pos = pos - int(erase_a < pos) - int(erase_b < pos);
        rank += choose((Lw - 2) - compact_pos - 1, seen);
    }
    return rank;
}

Rank64 fused_compressed_block_rank(
    std::uint32_t compressed, int W, int missing_bit,
    int lo, int Lw, int erase_a, int erase_b
) {
    Rank64 rank = 0;
    int seen = 0;
    for (int cbit = W - 2; cbit >= 0; --cbit) {
        if (((compressed >> cbit) & 1u) == 0) continue;
        const int bit = cbit + int(cbit >= missing_bit);
        if (bit < lo || bit >= lo + Lw) continue;
        const int pos = bit - lo;
        if (pos == erase_a || pos == erase_b) continue;
        ++seen;
        const int compact_pos = pos - int(erase_a < pos) - int(erase_b < pos);
        rank += choose((Lw - 2) - compact_pos - 1, seen);
    }
    return rank;
}

void enumerate_ternary(int width, int bit, MateID m, std::vector<MateID>& out) {
    if (bit == width) { out.push_back(m); return; }
    enumerate_ternary(width, bit + 1, m, out);
    enumerate_ternary(width, bit + 1, m | (MateID(R) << (2 * bit)), out);
    enumerate_ternary(width, bit + 1, m | (MateID(L) << (2 * bit)), out);
}
}

int main() {
    std::uint64_t exhaustive = 0;
    for (int W = 2; W <= 9; ++W) {
        std::vector<MateID> states;
        enumerate_ternary(W - 1, 0, 0, states);
        for (MateID compressed : states) {
            for (int missing = 0; missing < W; ++missing) {
                ++exhaustive;
                const MateID full = minsert_n(compressed, missing);
                const std::uint32_t cs = support(compressed, W - 1);
                const std::uint32_t fs = support(full, W);
                if (insert_zero_support(cs, W - 1, missing) != fs ||
                    occupied_sequence(compressed, W - 1) != occupied_sequence(full, W)) {
                    std::cerr << "mapping mismatch W=" << W << " missing=" << missing << '\n';
                    return 2;
                }
            }
        }
    }

    std::mt19937_64 rng(0x626c6f636b72616eULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int p = 1 + int(rng() % (W - 1));
        const bool reverse = (rng() & 1) != 0;
        const int missing = reverse ? p - 1 : p;
        const int fixed = reverse ? p : p - 1;
        MateID compressed = 0;
        for (int bit = 0; bit < W - 1; ++bit)
            compressed |= MateID(rng() % 3) << (2 * bit);
        const MateID full = minsert_n(compressed, missing);
        const std::uint32_t cs = support(compressed, W - 1);
        const std::uint32_t fs = support(full, W);
        if (insert_zero_support(cs, W - 1, missing) != fs ||
            occupied_sequence(compressed, W - 1) != occupied_sequence(full, W)) {
            std::cerr << "random mapping mismatch W=" << W << " p=" << p << '\n';
            return 3;
        }

        const int Lw = W / 2 + 1;
        const int lo_min = fixed >= Lw ? fixed - Lw + 1 : 0;
        const int lo_max = missing < W - Lw ? missing : W - Lw;
        if (lo_min > lo_max) continue;
        const int lo = lo_min + int(rng() % (lo_max - lo_min + 1));
        const int erase_a = missing - lo;
        const int erase_b = fixed - lo;
        if (erase_a < 0 || erase_a >= Lw || erase_b < 0 || erase_b >= Lw) return 4;
        if (fused_full_block_rank(fs, W, lo, Lw, erase_a, erase_b) !=
            fused_compressed_block_rank(cs, W, missing, lo, Lw, erase_a, erase_b)) {
            std::cerr << "random rank mismatch W=" << W << " p=" << p
                      << " reverse=" << reverse << " lo=" << lo << '\n';
            return 5;
        }
    }

    std::cout << "gridfp-runtime-blocked-rank-direct-proof OK"
              << " exhaustive_W_max=9 exhaustive_cases=" << exhaustive
              << " random_cases=" << RANDOM
              << " production_W_max=28"
              << " support_insert_zero_exact=1"
              << " occupied_sequence_exact=1"
              << " fused_block_rank_exact=1\n";
    return 0;
}
