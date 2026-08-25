#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

static long double as_ld(U128 x) {
    const U64 lo = U64(x), hi = U64(x >> 64);
    return (long double)hi * 18446744073709551616.0L + (long double)lo;
}

static std::string u128_string(U128 x) {
    if (!x) return "0";
    std::string s;
    while (x) {
        s.push_back(char('0' + x % 10));
        x /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

static U64 low_fixed_count_capped(int occupied, int start, int cap) {
    if (start > cap) return 0;
    std::vector<U64> cur(cap + 2), nxt(cap + 2);
    cur[start] = 1;
    for (int s = 0; s < occupied; ++s) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= cap; ++h) if (cur[h]) {
            if (h) nxt[h - 1] += cur[h];
            if (h + 1 <= cap) nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

static int pair_at(std::uint32_t high_code, int center, int q) {
    const std::uint32_t active = (high_code << 2) | std::uint32_t(center);
    return int((active >> (2 * (q - 1))) & 15u);
}

static U128 packed_desc_loads(U64 rows, U64 cols) {
    if (!rows || !cols) return 0;
    const U128 tasks = (U128(rows) * cols + 31) / 32;
    const U64 period = 32 / std::gcd<U64>(cols, 32);
    const U64 split_boundaries = (rows - 1) - (rows - 1) / period;
    return tasks + split_boundaries;
}

struct HighCode {
    std::uint32_t code = 0;
    std::uint8_t peak = 0;
};

struct Entry {
    int q = 0;
    int k = 0;
    U64 groups = 0;
    U64 dense_cols = 0;
    std::array<U64, 15> rows{};
    std::array<U64, 15> cols{};
};

struct Result {
    U128 useful = 0;
    U128 lanes = 0;
    U128 desc = 0;
    U128 tasks = 0;
    U128 blocks = 0;
    U128 capped = 0;
};

static Result evaluate(
    const std::vector<Entry>& entries,
    int cap,
    int threshold,
    U64 warps_per_block
) {
    Result out;
    struct Group { U64 copies = 0; U128 tasks = 0; };
    std::map<std::pair<int, int>, Group> task_groups;

    for (const Entry& e : entries) {
        const U64 rows = e.rows[size_t(cap)];
        const U64 cols = e.cols[size_t(cap)];
        U128 tasks = 0;
        if (rows && cols) {
            out.useful += U128(e.groups) * rows * cols;
            if (e.dense_cols < U64(threshold)) {
                tasks = (U128(rows) * cols + 31) / 32;
                out.lanes += U128(e.groups) * tasks * 32;
                out.desc += U128(e.groups) * packed_desc_loads(rows, cols);
            } else {
                tasks = rows;
                out.lanes += U128(e.groups) * rows * ((cols + 31) / 32) * 32;
                out.desc += U128(e.groups) * rows;
            }
            out.tasks += U128(e.groups) * tasks;
        }
        auto& g = task_groups[{e.q, e.k}];
        g.copies = e.groups;
        g.tasks += tasks;
    }

    for (const auto& [key, g] : task_groups) {
        (void)key;
        if (!g.tasks) continue;
        U128 blocks = (g.tasks + warps_per_block - 1) / warps_per_block;
        if (blocks > 65535) {
            blocks = 65535;
            out.capped += g.copies;
        }
        out.blocks += U128(g.copies) * blocks;
    }
    return out;
}

static std::vector<Entry> build_entries(int W, int low) {
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    std::vector<std::vector<HighCode>> hc(high + 3);
    auto rec = [&](auto&& self, int pos, int h, int peak, std::uint32_t code) -> void {
        if (pos < 0) {
            hc[h].push_back({code, std::uint8_t(peak)});
            return;
        }
        self(self, pos - 1, h, peak, code);
        if (h) self(self, pos - 1, h - 1, peak,
                    code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1, std::max(peak, h + 1),
             code | (2u << (2 * pos)));
    };
    rec(rec, high - 1, 1, 1, 0);

    std::vector<Entry> entries;
    for (int q = 1; q <= high; ++q) {
        for (int he = 0; he < int(hc.size()); ++he) {
            if (hc[he].empty()) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                std::array<U64, 15> selected{};
                for (const HighCode& x : hc[he]) {
                    const int w = pair_at(x.code, cv, q);
                    if (w != 0xa && w != 0x5 && w != 0x6) continue;
                    for (int cap = 1; cap <= full_cap; ++cap)
                        if (int(x.peak) <= cap) ++selected[size_t(cap)];
                }
                if (!selected[size_t(full_cap)]) continue;
                for (int k = 0; k <= low; ++k) {
                    Entry e;
                    e.q = q;
                    e.k = k;
                    e.groups = choose_u64(low, k);
                    e.dense_cols = low_fixed_count_capped(k, hs, full_cap);
                    for (int cap = 1; cap <= full_cap; ++cap) {
                        e.rows[size_t(cap)] = selected[size_t(cap)];
                        e.cols[size_t(cap)] = low_fixed_count_capped(k, hs, cap);
                    }
                    if (e.dense_cols) entries.push_back(e);
                }
            }
        }
    }
    return entries;
}

static Result aggregate_rows(
    const std::vector<Entry>& entries,
    int W,
    int threshold,
    U64 warps_per_block
) {
    const int full_cap = W / 2;
    Result out;
    for (int row = 1; row <= W; ++row) {
        const Result r = evaluate(
            entries, std::min(row, full_cap), threshold, warps_per_block);
        out.useful += r.useful;
        out.lanes += r.lanes;
        out.desc += r.desc;
        out.tasks += r.tasks;
        out.blocks += r.blocks;
        out.capped += r.capped;
    }
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int threshold = argc > 4 ? std::atoi(argv[4]) : 29;
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    if (W < 4 || W > 28 || low < 1 || high < 1 || high >= 16
        || full_cap >= 15 || threads < 32 || threads > 1024 || (threads & 31)
        || threshold < 1) return 1;

    const auto entries = build_entries(W, low);
    const U64 warps_per_block = U64(threads / 32);
    const Result current = aggregate_rows(entries, W, 16, warps_per_block);
    const Result chosen = aggregate_rows(entries, W, threshold, warps_per_block);

    int best_tasks_threshold = 1;
    int best_blocks_threshold = 1;
    U128 best_tasks = ~U128(0);
    U128 best_blocks = ~U128(0);
    for (int cut = 1; cut <= 1002; ++cut) {
        const Result r = aggregate_rows(entries, W, cut, warps_per_block);
        if (r.tasks < best_tasks) {
            best_tasks = r.tasks;
            best_tasks_threshold = cut;
        }
        if (r.blocks < best_blocks) {
            best_blocks = r.blocks;
            best_blocks_threshold = cut;
        }
    }

    if (W == 28 && low == 14 && threads == 256) {
        if (current.useful != U128(36989860194307ULL)
            || current.lanes != U128(44779140427808ULL)
            || current.desc != U128(1948871708005ULL)
            || current.tasks != U128(835870866393ULL)
            || current.blocks != U128(104486127592ULL)
            || current.capped != 0) {
            std::cerr << "n=27 threshold-16 exact row-depth regression mismatch\n";
            return 2;
        }
        if (threshold == 29
            && (chosen.useful != U128(36989860194307ULL)
                || chosen.lanes != U128(42734081059456ULL)
                || chosen.desc != U128(2131117327561ULL)
                || chosen.tasks != U128(771962761132ULL)
                || chosen.blocks != U128(96497498944ULL)
                || chosen.capped != 0)) {
            std::cerr << "n=27 threshold-29 exact row-depth regression mismatch\n";
            return 3;
        }
        if (best_tasks_threshold != 29 || best_blocks_threshold != 29
            || best_tasks != U128(771962761132ULL)
            || best_blocks != U128(96497498944ULL)) {
            std::cerr << "n=27 threshold optimum regression mismatch\n";
            return 4;
        }
    }

    auto ratio = [](U128 a, U128 b) {
        return double(as_ld(a) / as_ld(b));
    };
    std::cout << std::fixed << std::setprecision(12)
              << "highclosure-rowdepth-threshold W=" << W
              << " low=" << low << " high=" << high
              << " threads=" << threads << " threshold=" << threshold << '\n'
              << "useful=" << u128_string(chosen.useful) << '\n'
              << "current_lanes=" << u128_string(current.lanes)
              << " chosen_lanes=" << u128_string(chosen.lanes)
              << " ratio=" << ratio(chosen.lanes, current.lanes) << '\n'
              << "current_desc=" << u128_string(current.desc)
              << " chosen_desc=" << u128_string(chosen.desc)
              << " ratio=" << ratio(chosen.desc, current.desc) << '\n'
              << "current_tasks=" << u128_string(current.tasks)
              << " chosen_tasks=" << u128_string(chosen.tasks)
              << " ratio=" << ratio(chosen.tasks, current.tasks) << '\n'
              << "current_blocks=" << u128_string(current.blocks)
              << " chosen_blocks=" << u128_string(chosen.blocks)
              << " ratio=" << ratio(chosen.blocks, current.blocks) << '\n'
              << "chosen_capped_group_positions=" << u128_string(chosen.capped) << '\n'
              << "best_tasks_threshold=" << best_tasks_threshold
              << " best_tasks=" << u128_string(best_tasks) << '\n'
              << "best_blocks_threshold=" << best_blocks_threshold
              << " best_blocks=" << u128_string(best_blocks) << '\n';

    if (W == 28 && low == 14 && threads == 256 && argc <= 4) {
        const int cuts[] = {6, 8, 10, 16, 21, 29, 36, 43, 49, 64, 1002};
        std::cout << "pareto threshold lane_ratio desc_ratio task_ratio block_ratio capped\n";
        for (int cut : cuts) {
            const Result r = aggregate_rows(entries, W, cut, warps_per_block);
            std::cout << cut << ' '
                      << ratio(r.lanes, current.lanes) << ' '
                      << ratio(r.desc, current.desc) << ' '
                      << ratio(r.tasks, current.tasks) << ' '
                      << ratio(r.blocks, current.blocks) << ' '
                      << u128_string(r.capped) << '\n';
        }
    }
    return 0;
}
