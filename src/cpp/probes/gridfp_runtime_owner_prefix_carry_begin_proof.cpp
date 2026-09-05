#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
using Rank64 = std::uint64_t;

int old_search(const std::vector<Rank64>& prefix, int O, Rank64 rank,
               Rank64& begin, int& loads) {
    loads = 0;
    int lo = 0;
    int hi = O + 1;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const Rank64 end = prefix[std::size_t(mid + 1)];
        ++loads;
        if (rank < end) hi = mid;
        else lo = mid + 1;
    }
    if (lo > O) {
        begin = 0;
        return -1;
    }
    begin = prefix[std::size_t(lo)];
    ++loads;
    return lo;
}

int carry_search(const std::vector<Rank64>& prefix, int O, Rank64 rank,
                 Rank64& begin, int& loads) {
    loads = 0;
    int lo = 0;
    int hi = O + 1;
    begin = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const Rank64 end = prefix[std::size_t(mid + 1)];
        ++loads;
        if (rank < end) {
            hi = mid;
        } else {
            lo = mid + 1;
            begin = end;
        }
        if (begin != prefix[std::size_t(lo)]) return -2;
    }
    return lo <= O ? lo : -1;
}

bool check_case(const std::vector<Rank64>& prefix, int O, Rank64 rank,
                int& max_old_loads, int& max_new_loads) {
    Rank64 a = 0, b = 0;
    int la = 0, lb = 0;
    const int oa = old_search(prefix, O, rank, a, la);
    const int ob = carry_search(prefix, O, rank, b, lb);
    if (oa != ob || oa == -2) return false;
    if (oa >= 0 && a != b) return false;
    if (oa >= 0) {
        if (!(prefix[std::size_t(oa)] <= rank && rank < prefix[std::size_t(oa + 1)]))
            return false;
        max_old_loads = std::max(max_old_loads, la);
        max_new_loads = std::max(max_new_loads, lb);
        if (lb > la || la != lb + 1) return false;
    }
    return true;
}
} // namespace

int main() {
    std::mt19937_64 rng(0x7072656669786265ULL);
    std::uint64_t boundary_cases = 0;
    std::uint64_t repeated_boundaries = 0;
    int max_old_loads = 0;
    int max_new_loads = 0;

    // Deterministic shapes include empty sectors at the front, middle and end.
    for (int O = 0; O <= 13; ++O) {
        std::vector<Rank64> prefix(std::size_t(O + 2), 0);
        for (int r = 0; r <= O; ++r) {
            const Rank64 width = Rank64((r * 17 + O * 11) % 5); // includes zero
            prefix[std::size_t(r + 1)] = prefix[std::size_t(r)] + width;
            if (!width) ++repeated_boundaries;
        }
        if (!prefix.back()) prefix.back() = 1;
        for (int r = 0; r <= O; ++r) {
            const Rank64 lo = prefix[std::size_t(r)];
            const Rank64 hi = prefix[std::size_t(r + 1)];
            if (lo == hi) continue;
            const std::array<Rank64, 3> q{lo, lo + (hi - lo) / 2, hi - 1};
            for (Rank64 rank : q) {
                ++boundary_cases;
                if (!check_case(prefix, O, rank, max_old_loads, max_new_loads)) return 2;
            }
        }
        Rank64 a = 0, b = 0;
        int la = 0, lb = 0;
        if (old_search(prefix, O, prefix.back(), a, la) != -1) return 3;
        if (carry_search(prefix, O, prefix.back(), b, lb) != -1) return 4;
    }

    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t it = 0; it < RANDOM; ++it) {
        const int O = int(rng() % 14);
        std::vector<Rank64> prefix(std::size_t(O + 2), 0);
        for (int r = 0; r <= O; ++r) {
            // Deliberately high zero probability to stress repeated boundaries.
            const Rank64 width = (rng() & 3ULL) ? (1 + rng() % 100000ULL) : 0ULL;
            prefix[std::size_t(r + 1)] = prefix[std::size_t(r)] + width;
            if (!width) ++repeated_boundaries;
        }
        if (!prefix.back()) prefix.back() = 1;
        const Rank64 rank = rng() % prefix.back();
        if (!check_case(prefix, O, rank, max_old_loads, max_new_loads)) return 5;
    }

    // Production widths W=8..28 have O=W/2-1. W=28 therefore has 14 sectors.
    if (max_old_loads != 5 || max_new_loads != 4) return 6;
    std::cout << "gridfp-runtime-owner-prefix-carry-begin-proof OK"
              << " random_cases=" << RANDOM
              << " boundary_cases=" << boundary_cases
              << " repeated_boundaries=" << repeated_boundaries
              << " production_W_max=28 sectors=14"
              << " max_old_global_loads=" << max_old_loads
              << " max_new_global_loads=" << max_new_loads
              << " begin_reload_eliminated=1 exact=1\n";
    return 0;
}
