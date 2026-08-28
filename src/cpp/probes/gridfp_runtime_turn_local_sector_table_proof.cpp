#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX_W = 28;
static constexpr std::array<std::uint32_t, 550> EMBEDDED = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_turn_local_sector_end_values.inc"
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

int width_base(int W) {
    switch (W) {
    case 8: return 0; case 10: return 10; case 12: return 25;
    case 14: return 46; case 16: return 74; case 18: return 110;
    case 20: return 155; case 22: return 210; case 24: return 276;
    case 26: return 354; case 28: return 445; default: return -1;
    }
}

int row_base(int W, int r) {
    const int L = W / 2 + 1;
    const int odd_l = L / 2;
    const int even_l = (L + 1) / 2;
    const int prior_even_r = (r + 1) / 2;
    const int prior_odd_r = r / 2;
    return width_base(W) + prior_even_r * odd_l + prior_odd_r * even_l;
}

int active_count(int L, int r) {
    return (r & 1) ? (L + 1) / 2 : L / 2;
}
int first_local(int r) { return (r & 1) ? 0 : 1; }

int binary_sector(int W, int r, Rank64 within, int* comparisons = nullptr) {
    const int L = W / 2 + 1;
    const int row = row_base(W, r);
    const int count = active_count(L, r);
    int lo = 0, hi = count;
    int c = 0;
    while (lo < hi) {
        ++c;
        const int mid = lo + ((hi - lo) >> 1);
        if (within < EMBEDDED[row + mid]) hi = mid;
        else lo = mid + 1;
    }
    if (comparisons) *comparisons = c;
    return lo < count ? first_local(r) + (lo << 1) : -1;
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
        if (width_base(W) != int(index)) return 2;
        for (int r = 0; r <= O; ++r) {
            if (row_base(W, r) != int(index)) return 3;
            Rank64 previous = 0;
            const int first = first_local(r);
            for (int l = first; l < L; l += 2) {
                const int occupied = r + l;
                if (!(occupied & 1)) return 4;
                const Rank64 end = previous + choose(L - 1, l) * p[occupied][1];
                if (end > 0xffffffffULL) return 5;
                if (EMBEDDED[index] != end) return 6;
                max_end = std::max(max_end, EMBEDDED[index]);
                const Rank64 probes[] = {previous, previous + (end - previous) / 2, end - 1};
                for (Rank64 x : probes) {
                    ++boundary_cases;
                    int c = 0;
                    if (binary_sector(W, r, x, &c) != l) return 7;
                    max_binary_comparisons = std::max(max_binary_comparisons, c);
                }
                previous = end;
                ++index;
            }
            int c = 0;
            if (binary_sector(W, r, previous, &c) != -1) return 8;
            max_binary_comparisons = std::max(max_binary_comparisons, c);
        }
    }
    if (index != EMBEDDED.size()) return 9;

    std::mt19937_64 rng(0x7475726e73656374ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int L = W / 2 + 1;
        const int O = W - L;
        const int r = int(rng() % (O + 1));
        const int row = row_base(W, r);
        const int count = active_count(L, r);
        const Rank64 group = EMBEDDED[row + count - 1];
        const Rank64 x = rng() % group;
        const int got = binary_sector(W, r, x);
        if (got < 0 || got >= L || !((r + got) & 1)) return 10;
        const int slot = (got - first_local(r)) >> 1;
        const Rank64 begin = slot ? EMBEDDED[row + slot - 1] : 0;
        const Rank64 end = EMBEDDED[row + slot];
        if (!(begin <= x && x < end)) return 11;
    }

    std::cout << "gridfp-runtime-turn-local-sector-table-proof OK"
              << " W_configs=11 entries=" << EMBEDDED.size()
              << " bytes=" << EMBEDDED.size() * sizeof(std::uint32_t)
              << " max_end=" << max_end
              << " boundary_cases=" << boundary_cases
              << " random_cases=" << RANDOM
              << " production_W_max=28 embedded_exact=1 parity_compact=1 binary_exact=1"
              << " max_binary_comparisons=" << max_binary_comparisons << '\n';
    return 0;
}
