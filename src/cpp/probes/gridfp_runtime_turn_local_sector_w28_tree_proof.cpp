#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX_W = 28;
static constexpr std::array<std::uint32_t, 550> END = {
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

struct Result {
    int local_ones = -1;
    Rank64 local_within = 0;
    int comparisons = 0;
    int loads = 0;
};

Result generic_search(int row, int first, int count, Rank64 within) {
    int lo = 0;
    int hi = count;
    int comparisons = 0;
    int loads = 0;
    while (lo < hi) {
        ++comparisons;
        ++loads;
        const int mid = lo + ((hi - lo) >> 1);
        if (within < END[row + mid]) hi = mid;
        else lo = mid + 1;
    }
    if (lo >= count) return {-1, 0, comparisons, loads};
    Rank64 begin = 0;
    if (lo) {
        ++loads;
        begin = END[row + lo - 1];
    }
    return {first + (lo << 1), within - begin, comparisons, loads};
}

Result fixed_tree(int row, int first, bool eight, Rank64 within) {
    int loads = 1;
    const Rank64 e3 = END[row + 3];
    int index = 0;
    Rank64 begin = 0;
    if (within < e3) {
        ++loads;
        const Rank64 e1 = END[row + 1];
        if (within < e1) {
            ++loads;
            const Rank64 e0 = END[row];
            if (within < e0) {
                index = 0;
            } else {
                index = 1;
                begin = e0;
            }
        } else {
            ++loads;
            const Rank64 e2 = END[row + 2];
            if (within < e2) {
                index = 2;
                begin = e1;
            } else {
                index = 3;
                begin = e2;
            }
        }
    } else {
        ++loads;
        const Rank64 e5 = END[row + 5];
        if (within < e5) {
            ++loads;
            const Rank64 e4 = END[row + 4];
            if (within < e4) {
                index = 4;
                begin = e3;
            } else {
                index = 5;
                begin = e4;
            }
        } else if (eight) {
            ++loads;
            const Rank64 e6 = END[row + 6];
            if (within < e6) {
                index = 6;
                begin = e5;
            } else {
                index = 7;
                begin = e6;
            }
        } else {
            index = 6;
            begin = e5;
        }
    }
    return {first + (index << 1), within - begin, loads, loads};
}

bool same_value(Result a, Result b) {
    return a.local_ones == b.local_ones && a.local_within == b.local_within;
}
} // namespace

int main() {
    constexpr int W = 28;
    constexpr int L = 15;
    constexpr int O = 13;
    constexpr int base = 445;
    const auto primitive = primitive_table();

    std::uint64_t boundary_cases = 0;
    std::uint64_t weighted_cases = 0;
    std::uint64_t weighted_generic_loads = 0;
    std::uint64_t weighted_tree_loads = 0;
    int max_generic_comparisons = 0;
    int max_generic_loads = 0;
    int max_tree_loads = 0;
    int row = base;

    for (int outer = 0; outer <= O; ++outer) {
        const int first = (outer & 1) ? 0 : 1;
        const int count = (outer & 1) ? 8 : 7;
        Rank64 group = 0;
        for (int slot = 0; slot < count; ++slot) {
            const int local = first + (slot << 1);
            const int occupied = outer + local;
            const Rank64 width = choose(L - 1, local) * primitive[occupied][1];
            if (!width) return 2;
            const Rank64 begin = group;
            group += width;
            if (END[row + slot] != group) return 3;

            const Rank64 probes[] = {
                begin,
                begin + (width >> 1),
                group - 1,
            };
            for (Rank64 within : probes) {
                ++boundary_cases;
                const Result generic = generic_search(row, first, count, within);
                const Result tree = fixed_tree(row, first, count == 8, within);
                if (!same_value(generic, tree) || tree.local_ones != local ||
                    tree.local_within != within - begin) return 4;
                max_generic_comparisons = std::max(max_generic_comparisons, generic.comparisons);
                max_generic_loads = std::max(max_generic_loads, generic.loads);
                max_tree_loads = std::max(max_tree_loads, tree.loads);
            }

            const Result generic = generic_search(row, first, count, begin);
            const Result tree = fixed_tree(row, first, count == 8, begin);
            weighted_cases += width;
            weighted_generic_loads += width * std::uint64_t(generic.loads);
            weighted_tree_loads += width * std::uint64_t(tree.loads);
        }
        if (END[row + count - 1] != group) return 5;
        if (generic_search(row, first, count, group).local_ones != -1) return 6;
        row += count;
    }
    if (row != 550) return 7;

    std::mt19937_64 rng(0x7475726e773238ULL);
    constexpr std::uint64_t RANDOM_CASES = 1000000;
    for (std::uint64_t i = 0; i < RANDOM_CASES; ++i) {
        const int outer = int(rng() % 14);
        const int first = (outer & 1) ? 0 : 1;
        const int count = (outer & 1) ? 8 : 7;
        int r = base;
        for (int x = 0; x < outer; ++x) r += (x & 1) ? 8 : 7;
        const Rank64 group = END[r + count - 1];
        const Rank64 within = rng() % group;
        const Result generic = generic_search(r, first, count, within);
        const Result tree = fixed_tree(r, first, count == 8, within);
        if (!same_value(generic, tree)) return 8;
        max_generic_comparisons = std::max(max_generic_comparisons, generic.comparisons);
        max_generic_loads = std::max(max_generic_loads, generic.loads);
        max_tree_loads = std::max(max_tree_loads, tree.loads);
    }

    if (max_generic_comparisons != 4) return 9;
    if (max_generic_loads != 5) return 10;
    if (max_tree_loads != 3) return 11;
    if (!(weighted_tree_loads < weighted_generic_loads)) return 12;

    const double generic_avg = double(weighted_generic_loads) / double(weighted_cases);
    const double tree_avg = double(weighted_tree_loads) / double(weighted_cases);
    std::cout << "gridfp-runtime-turn-local-sector-w28-tree-proof OK"
              << " W=28 rows=14 count7_rows=7 count8_rows=7"
              << " entries=105 boundary_cases=" << boundary_cases
              << " random_cases=" << RANDOM_CASES
              << " final_endpoint_is_group=1"
              << " max_generic_comparisons=" << max_generic_comparisons
              << " max_generic_loads=" << max_generic_loads
              << " max_tree_loads=" << max_tree_loads
              << " weighted_generic_loads=" << generic_avg
              << " weighted_tree_loads=" << tree_avg
              << " begin_reload_eliminated=1 exact=1\n";
    return 0;
}
