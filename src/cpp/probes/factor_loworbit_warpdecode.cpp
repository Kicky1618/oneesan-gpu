#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
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
    const int warp = argc > 3 ? std::atoi(argv[3]) : 32;
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    if (W < 4 || W > 30 || low < 1 || high < 1 || low >= 16 || high >= 16
        || full_cap >= 16 || warp != 32) return 1;

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

    U128 baseline_searches = 0;
    U128 warp_searches = 0;
    U128 total_warps = 0;
    U128 crossing_warps = 0;

    // cap == full_cap uses the physical-order saturated fast path, so only
    // rows 1..full_cap-1 pay compact-prefix decode cost.
    for (int cap = 1; cap < full_cap; ++cap) {
        for (std::uint32_t mask = 0; mask < nm; ++mask) {
            std::vector<U64> seg(high + 2, 0);
            U64 total = 0;
            for (int h = 0; h <= high + 1; ++h) {
                seg[size_t(h)] = high_cum[
                    std::size_t(mask) * (high + 2) + h][size_t(cap)]
                    * low_cum[size_t(h)][size_t(cap)];
                total += seg[size_t(h)];
            }
            if (!total) continue;

            baseline_searches += U128(total) * U128(low);
            const U64 warps = (total + 31u) >> 5;
            std::set<U64> crossing;
            U64 prefix = 0;
            for (int h = 0; h < high + 1; ++h) {
                prefix += seg[size_t(h)];
                if (prefix == 0 || prefix >= total || (prefix & 31u) == 0) continue;
                crossing.insert(prefix >> 5);
            }

            U128 searches = U128(2) * U128(warps);
            for (U64 wid : crossing) {
                const U64 start = wid << 5;
                const U64 lanes = std::min<U64>(32, total - start);
                searches += lanes;
            }
            warp_searches += searches * U128(low);
            total_warps += U128(warps) * U128(low);
            crossing_warps += U128(crossing.size()) * U128(low);
        }
    }

    if (W == 28 && low == 14 && warp == 32) {
        if (baseline_searches != U128(18630360556780ULL)
            || warp_searches != U128(1164504489652ULL)
            || total_warps != U128(582199351034ULL)
            || crossing_warps != U128(3305862ULL)) {
            std::cerr << "n=27 LOW orbit warp-decode regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "loworbit-warpdecode W=" << W << " low=" << low
              << " high=" << high << '\n'
              << "baseline_searches=" << double(as_ld(baseline_searches))
              << " warp_searches=" << double(as_ld(warp_searches))
              << " search_speedup="
              << double(as_ld(baseline_searches) / as_ld(warp_searches)) << '\n'
              << "warps=" << double(as_ld(total_warps))
              << " crossing_warps=" << double(as_ld(crossing_warps))
              << " crossing_ratio="
              << double(as_ld(crossing_warps) / as_ld(total_warps)) << '\n';
    return 0;
}
