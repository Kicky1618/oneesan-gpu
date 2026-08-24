#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
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

struct HighCode {
    std::uint32_t code = 0;
    std::uint8_t peak = 0;
};

struct Entry {
    int q = 0;
    int k = 0;
    U64 groups = 0;
    std::array<U64, 15> rows{};
    std::array<U64, 15> cols{};
};

struct Result {
    U128 useful = 0;
    U128 lanes = 0;
    U128 blocks = 0;
};

static Result evaluate(
    const std::vector<Entry>& entries,
    int cap,
    int threshold,
    U64 warps_per_block
) {
    Result out;
    std::map<std::pair<int, int>, std::pair<U64, U128>> task_groups;
    for (const Entry& e : entries) {
        const U64 rows = e.rows[size_t(cap)];
        const U64 cols = e.cols[size_t(cap)];
        U128 tasks = 0;
        if (rows && cols) {
            out.useful += U128(e.groups) * rows * cols;
            if (cols < U64(threshold)) {
                tasks = (U128(rows) * cols + 31) / 32;
                out.lanes += U128(e.groups) * tasks * 32;
            } else {
                tasks = rows;
                out.lanes += U128(e.groups) * rows * ((cols + 31) / 32) * 32;
            }
        }
        auto& g = task_groups[{e.q, e.k}];
        g.first = e.groups;
        g.second += tasks;
    }
    for (const auto& [key, val] : task_groups) {
        (void)key;
        const U64 groups = val.first;
        const U128 tasks = val.second;
        if (!tasks) continue;
        U128 blocks = (tasks + warps_per_block - 1) / warps_per_block;
        if (blocks > 65535) blocks = 65535;
        out.blocks += U128(groups) * blocks;
    }
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int threshold = argc > 4 ? std::atoi(argv[4]) : 16;
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    if (W < 4 || W > 28 || low < 1 || high < 1 || high >= 16
        || full_cap >= 15 || threads < 32 || threads > 1024 || (threads & 31)
        || threshold < 1) return 1;
    const U64 warps_per_block = U64(threads / 32);

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
                    for (int cap = 1; cap <= full_cap; ++cap) {
                        e.rows[size_t(cap)] = selected[size_t(cap)];
                        e.cols[size_t(cap)] = low_fixed_count_capped(k, hs, cap);
                    }
                    if (e.cols[size_t(full_cap)]) entries.push_back(e);
                }
            }
        }
    }

    const Result dense = evaluate(entries, full_cap, threshold, warps_per_block);
    U128 active_useful = 0;
    U128 compact_lanes = 0;
    U128 compact_blocks = 0;
    for (int row = 1; row <= W; ++row) {
        const int cap = std::min(row, full_cap);
        const Result r = evaluate(entries, cap, threshold, warps_per_block);
        active_useful += r.useful;
        compact_lanes += r.lanes;
        compact_blocks += r.blocks;
    }
    const U128 dense_useful = dense.useful * U128(W);
    const U128 dense_lanes = dense.lanes * U128(W);
    const U128 dense_blocks = dense.blocks * U128(W);

    if (W == 28 && low == 14 && threads == 256 && threshold == 16) {
        static constexpr U64 EXPECT_USEFUL[15] = {
            0ULL,
            234881024ULL,
            50784985927ULL,
            369363350086ULL,
            857704715417ULL,
            1231387068804ULL,
            1417752205726ULL,
            1483630545931ULL,
            1500453105892ULL,
            1503526663444ULL,
            1503916347571ULL,
            1503948792062ULL,
            1503950405099ULL,
            1503950445154ULL,
            1503950445478ULL,
        };
        for (int cap = 1; cap <= full_cap; ++cap) {
            if (evaluate(entries, cap, threshold, warps_per_block).useful
                != U128(EXPECT_USEFUL[cap])) {
                std::cerr << "n=27 HIGH closure row-depth cap regression mismatch cap="
                          << cap << '\n';
                return 2;
            }
        }
        if (dense.useful != U128(1503950445478ULL)
            || dense.lanes != U128(1814814872992ULL)
            || dense.blocks != U128(4211269295ULL)
            || dense_useful != U128(42110612473384ULL)
            || active_useful != U128(36989860194307ULL)
            || dense_lanes != U128(50814816443776ULL)
            || compact_lanes != U128(44720278768384ULL)
            || dense_blocks != U128(117915540260ULL)
            || compact_blocks != U128(104256200361ULL)) {
            std::cerr << "n=27 HIGH closure row-depth aggregate regression mismatch\n";
            return 3;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "highclosure-rowdepth W=" << W
              << " low=" << low << " high=" << high
              << " threads=" << threads << " threshold=" << threshold << '\n'
              << "dense_useful=" << u128_string(dense_useful)
              << " active_useful=" << u128_string(active_useful)
              << " ratio=" << double(as_ld(active_useful) / as_ld(dense_useful))
              << " reduction=" << double(1.0L - as_ld(active_useful) / as_ld(dense_useful))
              << '\n'
              << "dense_hybrid_lanes=" << u128_string(dense_lanes)
              << " exact_compact_lanes=" << u128_string(compact_lanes)
              << " ratio=" << double(as_ld(compact_lanes) / as_ld(dense_lanes))
              << " reduction=" << double(1.0L - as_ld(compact_lanes) / as_ld(dense_lanes))
              << '\n'
              << "dense_launch_blocks=" << u128_string(dense_blocks)
              << " exact_compact_blocks=" << u128_string(compact_blocks)
              << " ratio=" << double(as_ld(compact_blocks) / as_ld(dense_blocks))
              << " reduction=" << double(1.0L - as_ld(compact_blocks) / as_ld(dense_blocks))
              << '\n';
    return 0;
}
