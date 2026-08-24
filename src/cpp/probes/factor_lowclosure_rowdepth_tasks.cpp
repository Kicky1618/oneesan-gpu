#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

static long double as_ld(U128 x) {
    const U64 lo = U64(x), hi = U64(x >> 64);
    return (long double)hi * 18446744073709551616.0L + (long double)lo;
}

struct LowCode {
    std::uint32_t code = 0;
    std::uint8_t peak = 0;
};

static bool closure_pair(std::uint32_t lc, int cv, int low, int p) {
    const std::uint32_t active = lc | (std::uint32_t(cv) << (2 * low));
    const std::uint32_t w = (active >> (2 * (p - 1))) & 15u;
    return w == 0xau || w == 0x5u || w == 0x6u;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int high = W - 1 - low;
    const int full_cap = (W + 1) / 2;
    if (W < 4 || W > 30 || low < 1 || high < 1 || low >= 16 || high >= 16
        || full_cap >= 16 || threads < 32 || threads > 1024 || (threads & 31))
        return 1;
    const U64 warps_per_block = U64(threads / 32);
    const std::uint32_t NM = 1u << high;

    // Cumulative HIGH-row counts for each fixed occupancy mask/end height/cap.
    std::vector<std::array<U64, 16>> high_cum(std::size_t(NM) * (high + 2));
    auto high_rec = [&](auto&& self, int pos, int h, int peak,
                        std::uint32_t mask) -> void {
        if (pos < 0) {
            auto& a = high_cum[std::size_t(mask) * (high + 2) + h];
            for (int cap = peak; cap <= full_cap; ++cap) ++a[size_t(cap)];
            return;
        }
        self(self, pos - 1, h, peak, mask);
        if (h) self(self, pos - 1, h - 1, peak, mask | (1u << pos));
        self(self, pos - 1, h + 1, std::max(peak, h + 1), mask | (1u << pos));
    };
    high_rec(high_rec, high - 1, 1, 1, 0u);

    std::vector<std::vector<LowCode>> low_codes(low + 2);
    for (int hs = 0; hs <= low + 1; ++hs) {
        auto low_rec = [&](auto&& self, int pos, int h, int peak,
                           std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) low_codes[hs].push_back({code, std::uint8_t(peak)});
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, peak, code);
            if (h) self(self, pos - 1, h - 1, peak,
                        code | (1u << (2 * pos)));
            self(self, pos - 1, h + 1, std::max(peak, h + 1),
                 code | (2u << (2 * pos)));
        };
        low_rec(low_rec, low - 1, hs, hs, 0u);
    }

    // selected[pi][he][cv][cap] flattened.
    const int PSTRIDE = (high + 2) * 3 * 16;
    std::vector<U64> selected(std::size_t(low) * PSTRIDE, 0);
    auto six = [&](int pi, int he, int cv, int cap) -> std::size_t {
        return std::size_t(pi) * PSTRIDE
             + (std::size_t(he) * 3 + cv) * 16 + cap;
    };
    for (int p = low; p >= 1; --p) {
        const int pi = low - p;
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low + 1) continue;
                for (const LowCode& x : low_codes[hs]) {
                    if (!closure_pair(x.code, cv, low, p)) continue;
                    for (int cap = int(x.peak); cap <= full_cap; ++cap)
                        ++selected[six(pi, he, cv, cap)];
                }
            }
        }
    }

    U128 dense_tasks = 0, dense_blocks = 0;
    std::array<U128, 16> cap_tasks{}, cap_blocks{};
    for (int p = low; p >= 1; --p) {
        const int pi = low - p;
        for (std::uint32_t mask = 0; mask < NM; ++mask) {
            U64 group_dense = 0;
            std::array<U64, 16> group_cap{};
            for (int he = 0; he <= high + 1; ++he) {
                const auto& hc = high_cum[std::size_t(mask) * (high + 2) + he];
                const U64 dense_rows = hc[size_t(full_cap)];
                if (!dense_rows) continue;
                for (int cv = 0; cv < 3; ++cv) {
                    const U64 dense_sel = selected[six(pi, he, cv, full_cap)];
                    if (dense_sel)
                        group_dense += dense_rows * ((dense_sel + 31) / 32);
                    for (int cap = 1; cap <= full_cap; ++cap) {
                        const U64 rows = hc[size_t(cap)];
                        const U64 sel = selected[six(pi, he, cv, cap)];
                        if (rows && sel)
                            group_cap[size_t(cap)] += rows * ((sel + 31) / 32);
                    }
                }
            }
            dense_tasks += group_dense;
            if (group_dense) {
                const U64 b = (group_dense + warps_per_block - 1) / warps_per_block;
                dense_blocks += std::min<U64>(65535, b);
            }
            for (int cap = 1; cap <= full_cap; ++cap) {
                const U64 t = group_cap[size_t(cap)];
                cap_tasks[size_t(cap)] += t;
                if (t) {
                    const U64 b = (t + warps_per_block - 1) / warps_per_block;
                    cap_blocks[size_t(cap)] += std::min<U64>(65535, b);
                }
            }
        }
    }

    U128 exact_tasks = 0, exact_blocks = 0;
    for (int row = 1; row <= W; ++row) {
        const int cap = std::min(row, full_cap);
        exact_tasks += cap_tasks[size_t(cap)];
        exact_blocks += cap_blocks[size_t(cap)];
    }
    const U128 dense_lanes = dense_tasks * U128(32) * U128(W);
    const U128 exact_lanes = exact_tasks * U128(32);
    const U128 dense_res_blocks = dense_blocks * U128(W);

    if (W == 28 && low == 14 && threads == 256) {
        if (dense_tasks != U128(50626280813ULL)
            || dense_blocks != U128(3964060594ULL)
            || dense_lanes != U128(45361147608448ULL)
            || exact_tasks != U128(1244849733336ULL)
            || exact_lanes != U128(39835191466752ULL)
            || dense_res_blocks != U128(110993696632ULL)
            || exact_blocks != U128(99505629661ULL)) {
            std::cerr << "n=27 LOW closure exact-task regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "lowclosure-rowdepth-tasks W=" << W << " low=" << low
              << " high=" << high << " threads=" << threads << '\n'
              << "dense_lane_slots=" << double(as_ld(dense_lanes))
              << " exact_lane_slots=" << double(as_ld(exact_lanes))
              << " lane_ratio=" << double(as_ld(exact_lanes) / as_ld(dense_lanes))
              << " lane_reduction="
              << double(1.0L - as_ld(exact_lanes) / as_ld(dense_lanes)) << '\n'
              << "dense_launch_blocks=" << double(as_ld(dense_res_blocks))
              << " exact_launch_blocks=" << double(as_ld(exact_blocks))
              << " block_ratio="
              << double(as_ld(exact_blocks) / as_ld(dense_res_blocks))
              << " block_reduction="
              << double(1.0L - as_ld(exact_blocks) / as_ld(dense_res_blocks)) << '\n';
    return 0;
}
