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

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    if (W < 4 || W > 30 || low < 1 || high < 1 || low >= 16 || high >= 16
        || full_cap >= 16 || threads < 32 || (threads & 31)) return 1;
    const int warps_per_block = threads / 32;

    const std::uint32_t nm = 1u << high;
    std::vector<std::array<U64, 16>> high_cum(
        std::size_t(nm) * (high + 2));
    auto hrec = [&](auto&& self, int pos, int h, int peak,
                    std::uint32_t mask) -> void {
        if (pos < 0) {
            auto& a = high_cum[std::size_t(mask) * (high + 2) + h];
            for (int cap = peak; cap <= full_cap; ++cap) ++a[size_t(cap)];
            return;
        }
        self(self, pos - 1, h, peak, mask);
        if (h > 0)
            self(self, pos - 1, h - 1, peak, mask | (1u << pos));
        self(self, pos - 1, h + 1, std::max(peak, h + 1),
             mask | (1u << pos));
    };
    hrec(hrec, high - 1, 1, 1, 0u);

    std::vector<std::array<U64, 16>> low_cum(low + 2);
    for (int h0 = 0; h0 <= low + 1; ++h0) {
        auto lrec = [&](auto&& self, int pos, int h, int peak) -> void {
            if (pos < 0) {
                if (h == 0)
                    for (int cap = peak; cap <= full_cap; ++cap)
                        ++low_cum[size_t(h0)][size_t(cap)];
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, peak);
            if (h > 0) self(self, pos - 1, h - 1, peak);
            self(self, pos - 1, h + 1, std::max(peak, h + 1));
        };
        lrec(lrec, low - 1, h0, h0);
    }

    U128 useful_states = 0;
    U128 scheduled_lanes = 0;
    U128 warp_tasks = 0;
    U128 flat_launch_blocks = 0;
    U128 ideal_warp_launch_blocks = 0;

    for (int row = 1; row <= W; ++row) {
        const int cap = std::min(row, full_cap);
        for (std::uint32_t mask = 0; mask < nm; ++mask) {
            U64 states = 0;
            U64 warps = 0;
            for (int h = 0; h <= high + 1; ++h) {
                const U64 hc = high_cum[
                    std::size_t(mask) * (high + 2) + h][size_t(cap)];
                const U64 lc = low_cum[size_t(h)][size_t(cap)];
                states += hc * lc;
                warps += hc * ((lc + 31u) >> 5);
            }
            useful_states += U128(states) * U128(low);
            scheduled_lanes += U128(warps) * U128(32) * U128(low);
            warp_tasks += U128(warps) * U128(low);
            if (states)
                flat_launch_blocks += U128(std::min<U64>(
                    65535, (states + U64(threads) - 1) / U64(threads))) * U128(low);
            if (warps)
                ideal_warp_launch_blocks += U128(std::min<U64>(
                    65535, (warps + U64(warps_per_block) - 1)
                         / U64(warps_per_block))) * U128(low);
        }
    }

    if (W == 28 && low == 14 && threads == 256) {
        if (useful_states != U128(46983616692250ULL)
            || scheduled_lanes != U128(46989326752256ULL)
            || warp_tasks != U128(1468416461008ULL)
            || flat_launch_blocks != U128(109287433836ULL)
            || ideal_warp_launch_blocks != U128(109293718870ULL)) {
            std::cerr << "n=27 LOW orbit warp-row regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "loworbit-warprow W=" << W << " low=" << low
              << " high=" << high << " threads=" << threads << '\n'
              << "useful_states=" << double(as_ld(useful_states))
              << " scheduled_lanes=" << double(as_ld(scheduled_lanes))
              << " lane_padding_ratio="
              << double(as_ld(scheduled_lanes) / as_ld(useful_states) - 1.0L) << '\n'
              << "flat_divisions=" << double(as_ld(useful_states))
              << " warp_divisions=" << double(as_ld(warp_tasks))
              << " division_speedup="
              << double(as_ld(useful_states) / as_ld(warp_tasks)) << '\n'
              << "flat_launch_blocks=" << double(as_ld(flat_launch_blocks))
              << " ideal_warp_launch_blocks=" << double(as_ld(ideal_warp_launch_blocks))
              << " ideal_grid_delta="
              << double(as_ld(ideal_warp_launch_blocks) / as_ld(flat_launch_blocks) - 1.0L)
              << '\n';
    return 0;
}
