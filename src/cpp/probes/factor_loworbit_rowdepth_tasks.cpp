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
        || full_cap >= 16 || threads < 1) return 1;

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

    U64 dense_group_blocks = 0;
    std::array<U64, 16> exact_group_blocks{};
    U128 dense_states = 0;
    std::array<U128, 16> exact_states{};

    for (std::uint32_t mask = 0; mask < nm; ++mask) {
        U64 dense = 0;
        std::array<U64, 16> active{};
        for (int h = 0; h <= high + 1; ++h) {
            const auto& hc = high_cum[
                std::size_t(mask) * (high + 2) + h];
            dense += hc[size_t(full_cap)]
                   * low_cum[size_t(h)][size_t(full_cap)];
            for (int cap = 1; cap <= full_cap; ++cap)
                active[size_t(cap)] += hc[size_t(cap)]
                    * low_cum[size_t(h)][size_t(cap)];
        }
        dense_states += dense;
        if (dense)
            dense_group_blocks += std::min<U64>(
                65535, (dense + U64(threads) - 1) / U64(threads));
        for (int cap = 1; cap <= full_cap; ++cap) {
            const U64 z = active[size_t(cap)];
            exact_states[size_t(cap)] += z;
            if (z)
                exact_group_blocks[size_t(cap)] += std::min<U64>(
                    65535, (z + U64(threads) - 1) / U64(threads));
        }
    }

    U128 dense_bodies = dense_states * U128(low) * U128(W);
    U128 active_bodies = 0;
    U128 dense_launch_blocks = U128(dense_group_blocks) * low * W;
    U128 exact_launch_blocks = 0;
    for (int row = 1; row <= W; ++row) {
        const int cap = std::min(row, full_cap);
        active_bodies += exact_states[size_t(cap)] * U128(low);
        exact_launch_blocks += U128(exact_group_blocks[size_t(cap)]) * low;
    }

    if (W == 28 && low == 14 && threads == 256) {
        if (dense_states != U128(135015505407ULL)
            || dense_bodies != U128(52926078119544ULL)
            || active_bodies != U128(46983616692250ULL)
            || dense_group_blocks != 307130153ULL
            || dense_launch_blocks != U128(120395019976ULL)
            || exact_launch_blocks != U128(109287433836ULL)) {
            std::cerr << "n=27 LOW orbit exact-task regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "loworbit-rowdepth-tasks W=" << W << " low=" << low
              << " high=" << high << " threads=" << threads << '\n'
              << "dense_bodies=" << double(as_ld(dense_bodies))
              << " active_bodies=" << double(as_ld(active_bodies))
              << " body_ratio="
              << double(as_ld(active_bodies) / as_ld(dense_bodies))
              << " body_reduction="
              << double(1.0L - as_ld(active_bodies) / as_ld(dense_bodies)) << '\n'
              << "dense_launch_blocks=" << double(as_ld(dense_launch_blocks))
              << " exact_launch_blocks=" << double(as_ld(exact_launch_blocks))
              << " block_ratio="
              << double(as_ld(exact_launch_blocks) / as_ld(dense_launch_blocks))
              << " block_reduction="
              << double(1.0L - as_ld(exact_launch_blocks) / as_ld(dense_launch_blocks))
              << '\n';
    return 0;
}
