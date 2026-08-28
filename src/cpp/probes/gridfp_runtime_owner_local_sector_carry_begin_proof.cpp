#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;
static constexpr std::array<std::uint32_t, 1100> END = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_owner_local_sector_end_values.inc"
};

int width_base(int W) {
    switch (W) {
    case 8: return 0; case 10: return 20; case 12: return 50;
    case 14: return 92; case 16: return 148; case 18: return 220;
    case 20: return 310; case 22: return 420; case 24: return 552;
    case 26: return 708; case 28: return 890; default: return -1;
    }
}

int old_search(int W, int outer, Rank64 within, Rank64& begin, int& loads) {
    const int L = W / 2 + 1;
    const int row = width_base(W) + outer * L;
    const int first = (outer & 1) ? 2 : 1;
    const int count = (L - first + 1) >> 1;
    int lo = 0, hi = count - 1;
    loads = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const int l = first + (mid << 1);
        ++loads;
        if (within < END[std::size_t(row + l)]) hi = mid;
        else lo = mid + 1;
    }
    begin = lo ? END[std::size_t(row + first + ((lo - 1) << 1))] : 0;
    if (lo) ++loads;
    return first + (lo << 1);
}

int carry_search(int W, int outer, Rank64 within, Rank64& begin, int& loads) {
    const int L = W / 2 + 1;
    const int row = width_base(W) + outer * L;
    const int first = (outer & 1) ? 2 : 1;
    const int count = (L - first + 1) >> 1;
    int lo = 0, hi = count - 1;
    begin = 0;
    loads = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const int l = first + (mid << 1);
        const Rank64 end = END[std::size_t(row + l)];
        ++loads;
        if (within < end) hi = mid;
        else {
            lo = mid + 1;
            begin = end;
        }
    }
    return first + (lo << 1);
}

bool check(int W, int outer, Rank64 within, int& old_max, int& new_max) {
    Rank64 a = 0, b = 0;
    int la = 0, lb = 0;
    const int oa = old_search(W, outer, within, a, la);
    const int ob = carry_search(W, outer, within, b, lb);
    if (oa != ob || a != b) return false;
    old_max = std::max(old_max, la);
    new_max = std::max(new_max, lb);
    if (lb > la) return false;
    return true;
}
}

int main() {
    int max_old_loads = 0;
    int max_new_loads = 0;
    std::uint64_t boundary_cases = 0;
    for (int W = 8; W <= 28; W += 2) {
        const int L = W / 2 + 1;
        const int O = W - L;
        for (int outer = 0; outer <= O; ++outer) {
            const int row = width_base(W) + outer * L;
            const int first = (outer & 1) ? 2 : 1;
            const int count = (L - first + 1) >> 1;
            Rank64 previous = 0;
            for (int slot = 0; slot < count; ++slot) {
                const int l = first + (slot << 1);
                const Rank64 end = END[std::size_t(row + l)];
                if (!(previous < end)) return 2;
                const std::array<Rank64, 3> q{
                    previous, previous + (end - previous) / 2, end - 1};
                for (Rank64 x : q) {
                    ++boundary_cases;
                    if (!check(W, outer, x, max_old_loads, max_new_loads)) return 3;
                }
                previous = end;
            }
        }
    }

    std::mt19937_64 rng(0x6f776e6572636172ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int L = W / 2 + 1;
        const int O = W - L;
        const int outer = int(rng() % (O + 1));
        const int row = width_base(W) + outer * L;
        const int first = (outer & 1) ? 2 : 1;
        const int count = (L - first + 1) >> 1;
        const Rank64 group = END[std::size_t(row + first + ((count - 1) << 1))];
        const Rank64 within = rng() % group;
        if (!check(W, outer, within, max_old_loads, max_new_loads)) return 4;
    }

    if (max_old_loads != 4 || max_new_loads != 3) return 5;
    std::cout << "gridfp-runtime-owner-local-sector-carry-begin-proof OK"
              << " W_configs=11 random_cases=" << RANDOM
              << " boundary_cases=" << boundary_cases
              << " production_W_max=28"
              << " max_old_table_loads=" << max_old_loads
              << " max_new_table_loads=" << max_new_loads
              << " begin_reload_eliminated=1 exact=1\n";
    return 0;
}
