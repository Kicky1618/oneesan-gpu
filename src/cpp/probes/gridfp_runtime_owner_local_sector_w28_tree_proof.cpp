#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using Rank64 = std::uint64_t;

static constexpr std::array<std::uint32_t, 1100> END = {
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_owner_local_sector_end_values.inc"
};

struct Result {
    int local_ones = -1;
    Rank64 local_within = 0;
    int loads = 0;
};

Result generic_parity_search(int row, int first, Rank64 within) {
    constexpr int count = 7;
    int lo = 0;
    int hi = count - 1;
    int loads = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const int l = first + (mid << 1);
        ++loads;
        if (within < END[row + l]) hi = mid;
        else lo = mid + 1;
    }
    const int l = first + (lo << 1);
    Rank64 begin = 0;
    if (lo) {
        ++loads;
        begin = END[row + first + ((lo - 1) << 1)];
    }
    return {l, within - begin, loads};
}

Result fixed_tree(int row, int first, Rank64 within) {
    int loads = 1;
    const Rank64 e3 = END[row + first + 6];
    int index = 0;
    Rank64 begin = 0;
    if (within < e3) {
        ++loads;
        const Rank64 e1 = END[row + first + 2];
        if (within < e1) {
            ++loads;
            const Rank64 e0 = END[row + first];
            if (within < e0) {
                index = 0;
            } else {
                index = 1;
                begin = e0;
            }
        } else {
            ++loads;
            const Rank64 e2 = END[row + first + 4];
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
        const Rank64 e5 = END[row + first + 10];
        if (within < e5) {
            ++loads;
            const Rank64 e4 = END[row + first + 8];
            if (within < e4) {
                index = 4;
                begin = e3;
            } else {
                index = 5;
                begin = e4;
            }
        } else {
            index = 6;
            begin = e5;
        }
    }
    return {first + (index << 1), within - begin, loads};
}

bool equal(Result a, Result b) {
    return a.local_ones == b.local_ones && a.local_within == b.local_within;
}
} // namespace

int main() {
    constexpr int W = 28;
    constexpr int L = 15;
    constexpr int O = 13;
    constexpr int base = 890;

    std::uint64_t interval_probes = 0;
    std::uint64_t weighted_generic_loads = 0;
    std::uint64_t weighted_tree_loads = 0;
    std::uint64_t weighted_cases = 0;
    int max_generic_loads = 0;
    int max_tree_loads = 0;

    for (int outer = 0; outer <= O; ++outer) {
        const int row = base + outer * L;
        const int first = (outer & 1) ? 2 : 1;
        Rank64 begin = 0;
        for (int index = 0; index < 7; ++index) {
            const int l = first + (index << 1);
            const Rank64 end = END[row + l];
            if (!(begin < end)) return 2;
            const Rank64 probes[] = {
                begin,
                begin + ((end - begin) >> 1),
                end - 1,
            };
            for (Rank64 within : probes) {
                ++interval_probes;
                const Result generic = generic_parity_search(row, first, within);
                const Result tree = fixed_tree(row, first, within);
                if (!equal(generic, tree) || tree.local_ones != l ||
                    tree.local_within != within - begin) return 3;
            }
            const Result generic = generic_parity_search(row, first, begin);
            const Result tree = fixed_tree(row, first, begin);
            const Rank64 width = end - begin;
            weighted_generic_loads += width * std::uint64_t(generic.loads);
            weighted_tree_loads += width * std::uint64_t(tree.loads);
            weighted_cases += width;
            if (generic.loads > max_generic_loads) max_generic_loads = generic.loads;
            if (tree.loads > max_tree_loads) max_tree_loads = tree.loads;
            begin = end;
        }
        if (begin != END[row + L - 1]) return 4;
    }

    std::mt19937_64 rng(0x7732387472656537ULL);
    constexpr std::uint64_t random_cases = 1000000;
    for (std::uint64_t i = 0; i < random_cases; ++i) {
        const int outer = int(rng() % 14);
        const int row = base + outer * L;
        const int first = (outer & 1) ? 2 : 1;
        const Rank64 group = END[row + L - 1];
        const Rank64 within = rng() % group;
        const Result generic = generic_parity_search(row, first, within);
        const Result tree = fixed_tree(row, first, within);
        if (!equal(generic, tree)) return 5;
        if (generic.loads > max_generic_loads) max_generic_loads = generic.loads;
        if (tree.loads > max_tree_loads) max_tree_loads = tree.loads;
    }

    if (max_generic_loads != 4 || max_tree_loads != 3) return 6;
    if (!(weighted_tree_loads < weighted_generic_loads)) return 7;

    const double generic_avg = double(weighted_generic_loads) / double(weighted_cases);
    const double tree_avg = double(weighted_tree_loads) / double(weighted_cases);
    std::cout << "gridfp-runtime-owner-local-sector-w28-tree-proof OK"
              << " W=28 rows=14 sectors_per_row=7"
              << " interval_probes=" << interval_probes
              << " random_cases=" << random_cases
              << " max_generic_loads=" << max_generic_loads
              << " max_tree_loads=" << max_tree_loads
              << " weighted_generic_loads=" << generic_avg
              << " weighted_tree_loads=" << tree_avg
              << " begin_reload_eliminated=1 exact=1\n";
    return 0;
}
