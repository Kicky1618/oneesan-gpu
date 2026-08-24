#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    if (W < 4 || W > 30 || low < 1 || high < 1 || low >= 16 || high >= 16
        || full_cap >= 16) return 1;

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

    U64 max_group_states = 0;
    U64 max_group_warps = 0;
    U64 max_segment_states = 0;
    U64 max_segment_warps = 0;
    for (int cap = 1; cap <= full_cap; ++cap) {
        for (std::uint32_t mask = 0; mask < nm; ++mask) {
            U64 group_states = 0;
            U64 group_warps = 0;
            for (int h = 0; h <= high + 1; ++h) {
                const U64 hc = high_cum[
                    std::size_t(mask) * (high + 2) + h][size_t(cap)];
                const U64 lc = low_cum[size_t(h)][size_t(cap)];
                const U64 states = hc * lc;
                const U64 warps = hc * ((lc + 31u) >> 5);
                group_states += states;
                group_warps += warps;
                max_segment_states = std::max(max_segment_states, states);
                max_segment_warps = std::max(max_segment_warps, warps);
            }
            max_group_states = std::max(max_group_states, group_states);
            max_group_warps = std::max(max_group_warps, group_warps);
        }
    }

    const bool u32_safe = max_group_warps <= 0xffffffffULL;
    if (!u32_safe) {
        std::cerr << "LOW orbit warp-row u32 overflow max_group_warps="
                  << max_group_warps << '\n';
        return 2;
    }
    if (W == 28 && low == 14) {
        if (max_group_states != 510468519ULL
            || max_group_warps != 15954186ULL
            || max_segment_states != 232555323ULL
            || max_segment_warps != 7268261ULL) {
            std::cerr << "n=27 LOW orbit warp-row u32 regression\n";
            return 3;
        }
    }

    std::cout << "loworbit-warprow-u32 W=" << W
              << " low=" << low
              << " high=" << high
              << " max_group_states=" << max_group_states
              << " max_group_warps=" << max_group_warps
              << " max_segment_states=" << max_segment_states
              << " max_segment_warps=" << max_segment_warps
              << " u32_safe=" << int(u32_safe) << '\n';
    return 0;
}
