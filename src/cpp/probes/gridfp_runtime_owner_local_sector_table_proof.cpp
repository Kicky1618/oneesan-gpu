#include <array>
#include <cstdint>
#include <iostream>

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

int binary_sector(int base, int L, int r, Rank64 within) {
    int lo = 0, hi = L;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (within < EMBEDDED[base + r * L + mid]) hi = mid;
        else lo = mid + 1;
    }
    return lo < L ? lo : -1;
}
}

int main() {
    const auto p = primitive_table();
    std::size_t index = 0;
    std::uint32_t max_end = 0;
    std::uint64_t rank_cases = 0;
    int max_binary_comparisons = 0;
    for (int W = 8; W <= 28; W += 2) {
        const int L = W / 2 + 1;
        const int O = W - L;
        const int base = row_base(W);
        if (base != int(index)) return 2;
        for (int r = 0; r <= O; ++r) {
            Rank64 end = 0;
            for (int l = 0; l < L; ++l) {
                const int occupied = r + l;
                if (occupied & 1) {
                    const Rank64 supports = choose(L - 1, l) - choose(L - 3, l);
                    end += supports * p[occupied][1];
                }
                if (end > 0xffffffffULL) return 3;
                if (EMBEDDED[index] != end) return 4;
                max_end = std::max(max_end, EMBEDDED[index]);
                ++index;
            }
            const Rank64 group = end;
            for (Rank64 x = 0; x < group; ++x) {
                ++rank_cases;
                int want = -1;
                Rank64 begin = 0;
                for (int l = 0; l < L; ++l) {
                    const Rank64 e = EMBEDDED[base + r * L + l];
                    if (x < e) { want = l; break; }
                    begin = e;
                }
                const int got = binary_sector(base, L, r, x);
                if (got != want) return 5;
                const Rank64 got_begin = got ? EMBEDDED[base + r * L + got - 1] : 0;
                if (x - got_begin != x - begin) return 6;
            }
            int comps = 0;
            for (int n = L; n > 1; n = (n + 1) >> 1) ++comps;
            max_binary_comparisons = std::max(max_binary_comparisons, comps);
        }
    }
    if (index != EMBEDDED.size()) return 7;
    std::cout << "gridfp-runtime-owner-local-sector-table-proof OK"
              << " W_configs=11 entries=" << EMBEDDED.size()
              << " bytes=" << EMBEDDED.size() * sizeof(std::uint32_t)
              << " max_end=" << max_end
              << " exhaustive_rank_cases=" << rank_cases
              << " production_W_max=28 embedded_exact=1 binary_exact=1"
              << " max_binary_comparisons=" << max_binary_comparisons << '\n';
    return 0;
}
