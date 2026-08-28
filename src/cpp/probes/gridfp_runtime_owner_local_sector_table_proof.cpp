#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX_W = 28;
static constexpr std::array<std::uint32_t, 1100> EMBEDDED = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_owner_local_sector_end_values.inc"
};

Rank64 choose(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    Rank64 z = 1;
    for (int i = 1; i <= k; ++i) z = z * Rank64(n - k + i) / Rank64(i);
    return z;
}

std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1> primitive_table() {
    std::array<std::array<Rank64, MAX_W + 2>, MAX_W + 1> p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= MAX_W; ++rem)
        for (int h = 0; h <= MAX_W; ++h)
            p[rem][h] = p[rem - 1][h + 1] + (h > 0 ? p[rem - 1][h - 1] : 0);
    return p;
}

int row_base(int W) {
    switch (W) {
    case 8: return 0; case 10: return 20; case 12: return 50;
    case 14: return 92; case 16: return 148; case 18: return 220;
    case 20: return 310; case 22: return 420; case 24: return 552;
    case 26: return 708; case 28: return 890; default: return -1;
    }
}

int binary_sector(int base, int L, int r, Rank64 within, int* comparisons = nullptr) {
    int lo = 0, hi = L;
    int c = 0;
    while (lo < hi) {
        ++c;
        const int mid = lo + ((hi - lo) >> 1);
        if (within < EMBEDDED[base + r * L + mid]) hi = mid;
        else lo = mid + 1;
    }
    if (comparisons) *comparisons = c;
    return lo < L ? lo : -1;
}

int linear_sector(int base, int L, int r, Rank64 within) {
    for (int l = 0; l < L; ++l)
        if (within < EMBEDDED[base + r * L + l]) return l;
    return -1;
}
}

int main() {
    const auto p = primitive_table();
    std::size_t index = 0;
    std::uint32_t max_end = 0;
    std::uint64_t boundary_cases = 0;
    int max_binary_comparisons = 0;
    for (int W = 8; W <= 28; W += 2) {
        const int L = W / 2 + 1;
        const int O = W - L;
        const int base = row_base(W);
        if (base != int(index)) return 2;
        for (int r = 0; r <= O; ++r) {
            Rank64 previous = 0;
            for (int l = 0; l < L; ++l) {
                Rank64 end = previous;
                const int occupied = r + l;
                if (occupied & 1) {
                    const Rank64 supports = choose(L - 1, l) - choose(L - 3, l);
                    end += supports * p[occupied][1];
                }
                if (end > 0xffffffffULL) return 3;
                if (EMBEDDED[index] != end) return 4;
                max_end = std::max(max_end, EMBEDDED[index]);
                if (end > previous) {
                    const Rank64 probes[] = {
                        previous, previous + (end - previous) / 2, end - 1};
                    for (Rank64 x : probes) {
                        ++boundary_cases;
                        int c = 0;
                        const int got = binary_sector(base, L, r, x, &c);
                        max_binary_comparisons = std::max(max_binary_comparisons, c);
                        if (got != l || linear_sector(base, L, r, x) != l) return 5;
                        const Rank64 begin = got
                            ? EMBEDDED[base + r * L + got - 1] : 0;
                        if (x - begin >= end - previous) return 6;
                    }
                }
                previous = end;
                ++index;
            }
            int c = 0;
            if (binary_sector(base, L, r, previous, &c) != -1 ||
                linear_sector(base, L, r, previous) != -1) return 7;
            max_binary_comparisons = std::max(max_binary_comparisons, c);
        }
    }
    if (index != EMBEDDED.size()) return 8;

    std::mt19937_64 rng(0x6c6f63616c736563ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int L = W / 2 + 1;
        const int O = W - L;
        const int r = int(rng() % (O + 1));
        const int base = row_base(W);
        const Rank64 group = EMBEDDED[base + r * L + L - 1];
        if (!group) return 9;
        const Rank64 x = rng() % group;
        if (binary_sector(base, L, r, x) != linear_sector(base, L, r, x)) return 10;
    }

    std::cout << "gridfp-runtime-owner-local-sector-table-proof OK"
              << " W_configs=11 entries=" << EMBEDDED.size()
              << " bytes=" << EMBEDDED.size() * sizeof(std::uint32_t)
              << " max_end=" << max_end
              << " boundary_cases=" << boundary_cases
              << " random_cases=" << RANDOM
              << " production_W_max=28 embedded_exact=1 binary_exact=1"
              << " max_binary_comparisons=" << max_binary_comparisons << '\n';
    return 0;
}
