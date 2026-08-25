#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
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

static U64 low_fixed_count(int occupied, int start) {
    std::vector<U64> cur(64), nxt(64);
    cur[start] = 1;
    for (int s = 0; s < occupied; ++s) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h < 63; ++h) if (cur[h]) {
            if (h) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

static int pair_at(std::uint32_t high_code, int center, int q) {
    const std::uint32_t active = (high_code << 2) | std::uint32_t(center);
    return int((active >> (2 * (q - 1))) & 15u);
}

static U64 capped_blocks(U128 tasks, U64 warps_per_block) {
    if (!tasks) return 0;
    const U128 blocks = (tasks + warps_per_block - 1) / warps_per_block;
    return U64(blocks > U128(65535) ? U128(65535) : blocks);
}

static U64 packed_desc_loads(U64 rows, U64 cols) {
    U64 total = 0;
    for (U64 r = 0; r < rows; ++r) {
        const U64 first = r * cols;
        const U64 last = first + cols - 1;
        total += last / 32 - first / 32 + 1;
    }
    return total;
}

struct Entry {
    U64 rows;
    U64 cols;
    U64 groups;
    int q;
    int k;
};

struct Result {
    U128 useful = 0;
    U128 lanes = 0;
    U128 desc = 0;
    U128 blocks = 0;
    U64 capped = 0;
};

static Result evaluate(
    const std::vector<Entry>& entries, int threshold, U64 warps_per_block
) {
    Result out;
    std::map<std::pair<int, int>, std::pair<U64, U128>> task_groups;
    for (const Entry& e : entries) {
        out.useful += U128(e.groups) * e.rows * e.cols;
        U128 tasks = 0;
        if (e.cols < U64(threshold)) {
            tasks = (U128(e.rows) * e.cols + 31) / 32;
            out.lanes += U128(e.groups) * tasks * 32;
            out.desc += U128(e.groups) * packed_desc_loads(e.rows, e.cols);
        } else {
            tasks = e.rows;
            out.lanes += U128(e.groups) * e.rows * ((e.cols + 31) / 32) * 32;
            out.desc += U128(e.groups) * e.rows;
        }
        auto& g = task_groups[{e.q, e.k}];
        g.first = e.groups;
        g.second += tasks;
    }
    for (const auto& [key, val] : task_groups) {
        (void)key;
        const U64 groups = val.first;
        const U128 tasks = val.second;
        const U128 raw = (tasks + warps_per_block - 1) / warps_per_block;
        if (raw > 65535) out.capped += groups;
        out.blocks += U128(groups) * capped_blocks(tasks, warps_per_block);
    }
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int threshold = argc > 4 ? std::atoi(argv[4]) : 16;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || high < 1 || high >= 16
        || threads < 32 || threads > 1024 || (threads & 31)
        || threshold < 1) return 1;
    const U64 warps_per_block = U64(threads / 32);

    std::vector<std::vector<std::uint32_t>> hc(high + 3);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) { hc[h].push_back(code); return; }
        self(self, pos - 1, h, code);
        if (h) self(self, pos - 1, h - 1, code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1, code | (2u << (2 * pos)));
    };
    rec(rec, high - 1, 1, 0);

    std::vector<std::vector<U64>> lc(low + 1, std::vector<U64>(low + 2));
    for (int k = 0; k <= low; ++k)
        for (int h = 0; h <= low; ++h)
            lc[k][h] = low_fixed_count(k, h);

    std::vector<Entry> entries;
    for (int q = 1; q <= high; ++q) {
        for (int he = 0; he < int(hc.size()); ++he) {
            if (hc[he].empty()) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                U64 rows = 0;
                for (std::uint32_t code : hc[he]) {
                    const int w = pair_at(code, cv, q);
                    if (w == 0xa || w == 0x5 || w == 0x6) ++rows;
                }
                if (!rows) continue;
                for (int k = 0; k <= low; ++k) {
                    const U64 cols = lc[k][hs];
                    if (!cols) continue;
                    entries.push_back({rows, cols, choose_u64(low, k), q, k});
                }
            }
        }
    }

    const Result base = evaluate(entries, 1, warps_per_block);
    const Result chosen = evaluate(entries, threshold, warps_per_block);

    if (W == 28 && low == 14 && threads == 256 && threshold == 16) {
        if (base.useful != U128(1503950445478ULL)
            || base.lanes != U128(3021117696896ULL)
            || base.desc != U128(71386429790ULL)
            || base.blocks != U128(8923348057ULL)
            || chosen.lanes != U128(1814814872992ULL)
            || chosen.desc != U128(79173964114ULL)
            || chosen.blocks != U128(4211269295ULL)
            || chosen.capped != 0) {
            std::cerr << "n=27 hybrid HIGH row-pack regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "highclosure-rowpack-hybrid W=" << W << " low=" << low
              << " high=" << high << " threads=" << threads
              << " threshold=" << threshold << '\n'
              << "useful_states=" << u128_string(chosen.useful) << '\n'
              << "base_lane_slots=" << u128_string(base.lanes)
              << " hybrid_lane_slots=" << u128_string(chosen.lanes)
              << " lane_ratio=" << double(as_ld(chosen.lanes) / as_ld(base.lanes))
              << " useful_lane_fraction="
              << double(as_ld(chosen.useful) / as_ld(chosen.lanes)) << '\n'
              << "base_desc_loads=" << u128_string(base.desc)
              << " hybrid_desc_loads=" << u128_string(chosen.desc)
              << " desc_ratio=" << double(as_ld(chosen.desc) / as_ld(base.desc)) << '\n'
              << "base_launch_blocks=" << u128_string(base.blocks)
              << " hybrid_launch_blocks=" << u128_string(chosen.blocks)
              << " launch_ratio=" << double(as_ld(chosen.blocks) / as_ld(base.blocks))
              << " capped_group_positions=" << chosen.capped << '\n';

    if (W == 28 && low == 14 && threads == 256 && argc <= 4) {
        const int cuts[] = {2, 4, 6, 8, 10, 12, 16, 21, 28, 29, 36, 43, 49, 76, 91, 111, 166, 1002};
        std::cout << "pareto threshold lane_ratio desc_ratio launch_ratio capped\n";
        for (int cut : cuts) {
            const Result r = evaluate(entries, cut, warps_per_block);
            std::cout << cut << ' '
                      << double(as_ld(r.lanes) / as_ld(base.lanes)) << ' '
                      << double(as_ld(r.desc) / as_ld(base.desc)) << ' '
                      << double(as_ld(r.blocks) / as_ld(base.blocks)) << ' '
                      << r.capped << '\n';
        }
    }
    return 0;
}
