#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
static constexpr std::array<std::uint32_t, 1100> FULL = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_owner_local_sector_end_values.inc"
};
static constexpr std::array<std::uint32_t, 503> COMPACT = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_owner_local_sector_positive_end_values.inc"
};

Rank64 choose(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    Rank64 z = 1;
    for (int i = 1; i <= k; ++i)
        z = z * Rank64(n - k + i) / Rank64(i);
    return z;
}

std::array<std::array<Rank64, 31>, 29> primitive_table() {
    std::array<std::array<Rank64, 31>, 29> p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= 28; ++rem)
        for (int h = 0; h <= 28; ++h)
            p[rem][h] = p[rem - 1][h + 1] + (h ? p[rem - 1][h - 1] : 0);
    return p;
}

int full_width_base(int W) {
    switch (W) {
    case 8: return 0; case 10: return 20; case 12: return 50;
    case 14: return 92; case 16: return 148; case 18: return 220;
    case 20: return 310; case 22: return 420; case 24: return 552;
    case 26: return 708; case 28: return 890; default: return -1;
    }
}
int compact_width_base(int W) {
    switch (W) {
    case 8: return 0; case 10: return 8; case 12: return 21;
    case 14: return 39; case 16: return 64; case 18: return 96;
    case 20: return 137; case 22: return 187; case 24: return 248;
    case 26: return 320; case 28: return 405; default: return -1;
    }
}
int compact_row_base(int W, int outer) {
    const int L = W / 2 + 1;
    const int even_count = L >> 1;       // outer even: local 1,3,...
    const int odd_count = (L - 1) >> 1; // outer odd: local 2,4,...
    const int prior_even = (outer + 1) >> 1;
    const int prior_odd = outer >> 1;
    return compact_width_base(W) + prior_even * even_count + prior_odd * odd_count;
}
} // namespace

int main() {
    const auto primitive = primitive_table();
    std::size_t compact_index = 0;
    std::uint32_t max_end = 0;
    std::uint64_t compared = 0;
    for (int W = 8; W <= 28; W += 2) {
        const int L = W / 2 + 1;
        const int O = W - L;
        if (compact_width_base(W) != int(compact_index)) return 2;
        for (int outer = 0; outer <= O; ++outer) {
            if (compact_row_base(W, outer) != int(compact_index)) return 3;
            const int full_row = full_width_base(W) + outer * L;
            const int first = (outer & 1) ? 2 : 1;
            Rank64 cumulative = 0;
            for (int local = first; local < L; local += 2) {
                const Rank64 supports =
                    choose(L - 1, local) - choose(L - 3, local);
                cumulative += supports * primitive[outer + local][1];
                if (cumulative > 0xffffffffULL) return 4;
                if (FULL[std::size_t(full_row + local)] != cumulative) return 5;
                if (COMPACT[compact_index] != cumulative) return 6;
                max_end = std::max(max_end, COMPACT[compact_index]);
                ++compact_index;
                ++compared;
            }
        }
    }
    if (compact_index != COMPACT.size()) return 7;
    if (max_end != 448876754u) return 8;
    std::cout << "gridfp-runtime-owner-local-sector-compact-table-proof OK"
              << " W_configs=11 full_entries=" << FULL.size()
              << " compact_entries=" << COMPACT.size()
              << " full_bytes=" << FULL.size() * sizeof(std::uint32_t)
              << " compact_bytes=" << COMPACT.size() * sizeof(std::uint32_t)
              << " saved_bytes=" << (FULL.size() - COMPACT.size()) * sizeof(std::uint32_t)
              << " compared=" << compared
              << " max_end=" << max_end
              << " W28_base=" << compact_width_base(28)
              << " positive_exact=1 formula_exact=1\n";
    return 0;
}
