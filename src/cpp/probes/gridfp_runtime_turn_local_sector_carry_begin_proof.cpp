#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;
static constexpr std::array<std::uint32_t, 550> END = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_turn_local_sector_end_values.inc"
};

int width_base(int W) {
    switch (W) {
    case 8: return 0; case 10: return 10; case 12: return 25;
    case 14: return 46; case 16: return 74; case 18: return 110;
    case 20: return 155; case 22: return 210; case 24: return 276;
    case 26: return 354; case 28: return 445; default: return -1;
    }
}
int row_base(int W, int outer) {
    const int L = W / 2 + 1;
    const int odd_l = L >> 1;
    const int even_l = (L + 1) >> 1;
    const int prior_even_r = (outer + 1) >> 1;
    const int prior_odd_r = outer >> 1;
    return width_base(W) + prior_even_r * odd_l + prior_odd_r * even_l;
}
int count_for(int L, int outer) { return (outer & 1) ? ((L + 1) >> 1) : (L >> 1); }
int first_for(int outer) { return (outer & 1) ? 0 : 1; }

int old_search(int W, int outer, Rank64 within, Rank64& begin, int& loads) {
    const int L = W / 2 + 1;
    const int row = row_base(W, outer);
    const int count = count_for(L, outer);
    int lo = 0, hi = count;
    loads = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        ++loads;
        if (within < END[std::size_t(row + mid)]) hi = mid;
        else lo = mid + 1;
    }
    if (lo >= count) {
        begin = 0;
        return -1;
    }
    begin = lo ? END[std::size_t(row + lo - 1)] : 0;
    if (lo) ++loads;
    return first_for(outer) + (lo << 1);
}

int carry_search(int W, int outer, Rank64 within, Rank64& begin, int& loads) {
    const int L = W / 2 + 1;
    const int row = row_base(W, outer);
    const int count = count_for(L, outer);
    int lo = 0, hi = count;
    begin = 0;
    loads = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const Rank64 end = END[std::size_t(row + mid)];
        ++loads;
        if (within < end) hi = mid;
        else {
            lo = mid + 1;
            begin = end;
        }
    }
    if (lo >= count) return -1;
    return first_for(outer) + (lo << 1);
}

bool check(int W, int outer, Rank64 within, int& old_max, int& new_max) {
    Rank64 a = 0, b = 0;
    int la = 0, lb = 0;
    const int oa = old_search(W, outer, within, a, la);
    const int ob = carry_search(W, outer, within, b, lb);
    if (oa != ob || a != b) return false;
    if (oa >= 0) {
        old_max = std::max(old_max, la);
        new_max = std::max(new_max, lb);
        if (lb > la) return false;
    }
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
            const int row = row_base(W, outer);
            const int count = count_for(L, outer);
            Rank64 previous = 0;
            for (int slot = 0; slot < count; ++slot) {
                const Rank64 end = END[std::size_t(row + slot)];
                if (!(previous < end)) return 2;
                const std::array<Rank64, 3> q{
                    previous, previous + (end - previous) / 2, end - 1};
                for (Rank64 x : q) {
                    ++boundary_cases;
                    if (!check(W, outer, x, max_old_loads, max_new_loads)) return 3;
                }
                previous = end;
            }
            Rank64 a = 0, b = 0;
            int la = 0, lb = 0;
            if (old_search(W, outer, previous, a, la) != -1) return 4;
            if (carry_search(W, outer, previous, b, lb) != -1) return 5;
        }
    }

    std::mt19937_64 rng(0x7475726e63617272ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const int L = W / 2 + 1;
        const int O = W - L;
        const int outer = int(rng() % (O + 1));
        const int row = row_base(W, outer);
        const int count = count_for(L, outer);
        const Rank64 group = END[std::size_t(row + count - 1)];
        const Rank64 within = rng() % group;
        if (!check(W, outer, within, max_old_loads, max_new_loads)) return 6;
    }

    if (max_old_loads != 5 || max_new_loads != 4) return 7;
    std::cout << "gridfp-runtime-turn-local-sector-carry-begin-proof OK"
              << " W_configs=11 random_cases=" << RANDOM
              << " boundary_cases=" << boundary_cases
              << " production_W_max=28"
              << " max_old_table_loads=" << max_old_loads
              << " max_new_table_loads=" << max_new_loads
              << " begin_reload_eliminated=1 exact=1\n";
    return 0;
}
