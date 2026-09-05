#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;

static constexpr std::array<std::uint32_t, 1100> EMBEDDED = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_owner_local_sector_end_values.inc"
};

int row_base(int W) {
    switch (W) {
    case 8: return 0; case 10: return 20; case 12: return 50;
    case 14: return 92; case 16: return 148; case 18: return 220;
    case 20: return 310; case 22: return 420; case 24: return 552;
    case 26: return 708; case 28: return 890; default: return -1;
    }
}

int full_binary(int row, int L, Rank64 within, int* comparisons = nullptr) {
    int lo = 0;
    int hi = L;
    int c = 0;
    while (lo < hi) {
        ++c;
        const int mid = lo + ((hi - lo) >> 1);
        if (within < EMBEDDED[row + mid]) hi = mid;
        else lo = mid + 1;
    }
    if (comparisons) *comparisons = c;
    return lo < L ? lo : -1;
}

int parity_binary(
    int row,
    int L,
    int outer_ones,
    Rank64 within,
    Rank64& begin,
    int* comparisons = nullptr
) {
    const int first = (outer_ones & 1) ? 2 : 1;
    const int count = (L - first + 1) >> 1;
    int lo = 0;
    int hi = count - 1;
    int c = 0;
    while (lo < hi) {
        ++c;
        const int mid = lo + ((hi - lo) >> 1);
        const int l = first + (mid << 1);
        if (within < EMBEDDED[row + l]) hi = mid;
        else lo = mid + 1;
    }
    if (comparisons) *comparisons = c;
    const int l = first + (lo << 1);
    begin = lo ? EMBEDDED[row + first + ((lo - 1) << 1)] : 0;
    return l;
}
} // namespace

int main() {
    std::uint64_t boundary_cases = 0;
    int max_full_comparisons = 0;
    int max_parity_comparisons = 0;

    for (int W = 8; W <= 28; W += 2) {
        const int L = W / 2 + 1;
        const int O = W - L;
        const int base = row_base(W);
        if (base < 0) return 2;
        for (int r = 0; r <= O; ++r) {
            const int row = base + r * L;
            const int first = (r & 1) ? 2 : 1;
            Rank64 previous = 0;
            for (int l = 0; l < L; ++l) {
                const Rank64 end = EMBEDDED[row + l];
                const bool rises = end > previous;
                const bool positive_sector = l >= first && ((l - first) & 1) == 0;
                if (rises != positive_sector) return 3;
                if (rises) {
                    const Rank64 probes[] = {
                        previous,
                        previous + (end - previous) / 2,
                        end - 1,
                    };
                    for (Rank64 x : probes) {
                        ++boundary_cases;
                        Rank64 begin = 0;
                        int full_c = 0;
                        int parity_c = 0;
                        const int full = full_binary(row, L, x, &full_c);
                        const int compact = parity_binary(
                            row, L, r, x, begin, &parity_c);
                        max_full_comparisons = std::max(max_full_comparisons, full_c);
                        max_parity_comparisons = std::max(max_parity_comparisons, parity_c);
                        if (full != l || compact != l || begin != previous) return 4;
                    }
                }
                previous = end;
            }
            if (!previous) return 5;
        }
    }

    std::mt19937_64 rng(0x7061726974797365ULL);
    constexpr std::uint64_t RANDOM_CASES = 1000000;
    for (std::uint64_t i = 0; i < RANDOM_CASES; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int L = W / 2 + 1;
        const int O = W - L;
        const int r = int(rng() % (O + 1));
        const int row = row_base(W) + r * L;
        const Rank64 group = EMBEDDED[row + L - 1];
        const Rank64 x = rng() % group;
        Rank64 begin = 0;
        int full_c = 0;
        int parity_c = 0;
        const int full = full_binary(row, L, x, &full_c);
        const int compact = parity_binary(row, L, r, x, begin, &parity_c);
        max_full_comparisons = std::max(max_full_comparisons, full_c);
        max_parity_comparisons = std::max(max_parity_comparisons, parity_c);
        if (full != compact) return 6;
        const Rank64 want_begin = compact ? EMBEDDED[row + compact - 1] : 0;
        if (begin != want_begin) return 7;
    }

    if (max_full_comparisons != 4 || max_parity_comparisons != 3) return 8;
    std::cout << "gridfp-runtime-owner-local-sector-parity-proof OK"
              << " boundary_cases=" << boundary_cases
              << " random_cases=" << RANDOM_CASES
              << " production_W_max=28 positive_sector_exact=1"
              << " W28_candidates=7"
              << " max_full_comparisons=" << max_full_comparisons
              << " max_parity_comparisons=" << max_parity_comparisons << '\n';
    return 0;
}
