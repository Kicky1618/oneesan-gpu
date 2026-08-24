#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

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
    const int high = W - 1 - low;
    const int full_cap = (W + 1) / 2;
    if (W < 4 || W > 30 || low < 1 || high < 1 || low >= 16 || high >= 16
        || full_cap >= 16) return 1;

    std::vector<std::map<int, U64>> high_hist(high + 2);
    auto high_rec = [&](auto&& self, int pos, int h, int peak) -> void {
        if (pos < 0) {
            ++high_hist[h][peak];
            return;
        }
        self(self, pos - 1, h, peak);
        if (h) self(self, pos - 1, h - 1, peak);
        self(self, pos - 1, h + 1, std::max(peak, h + 1));
    };
    high_rec(high - 1, 1, 1);

    std::vector<std::array<U64, 16>> high_cum(high + 2);
    std::vector<U64> high_total(high + 2, 0);
    for (int he = 0; he <= high + 1; ++he) {
        for (const auto& [pk, n] : high_hist[he]) high_total[he] += n;
        for (int cap = 1; cap <= full_cap; ++cap)
            for (const auto& [pk, n] : high_hist[he]) if (pk <= cap)
                high_cum[he][size_t(cap)] += n;
    }

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

    U128 dense_per_row = 0;
    std::array<U128, 16> active_by_cap{};
    for (int p = low; p >= 1; --p) {
        for (int he = 0; he <= high + 1; ++he) {
            if (!high_total[he]) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low + 1) continue;
                U64 selected = 0;
                std::array<U64, 16> selected_cum{};
                for (const LowCode& x : low_codes[hs]) {
                    if (!closure_pair(x.code, cv, low, p)) continue;
                    ++selected;
                    for (int cap = 1; cap <= full_cap; ++cap)
                        if (int(x.peak) <= cap) ++selected_cum[size_t(cap)];
                }
                dense_per_row += U128(high_total[he]) * selected;
                for (int cap = 1; cap <= full_cap; ++cap)
                    active_by_cap[size_t(cap)] +=
                        U128(high_cum[he][size_t(cap)])
                        * selected_cum[size_t(cap)];
            }
        }
    }

    U128 active_total = 0;
    for (int row = 1; row <= W; ++row)
        active_total += active_by_cap[size_t(std::min(row, full_cap))];
    const U128 dense_total = dense_per_row * U128(W);

    U64 low_all_entries = 0;
    for (const auto& v : low_codes) low_all_entries += v.size();

    if (W == 28 && low == 14) {
        static constexpr U64 EXPECT_CAP[15] = {
            0ULL,
            218103808ULL,
            53392006289ULL,
            394628377184ULL,
            920746057120ULL,
            1324466694549ULL,
            1526187837836ULL,
            1597591185761ULL,
            1615842182786ULL,
            1619178764930ULL,
            1619601916817ULL,
            1619637146593ULL,
            1619638897486ULL,
            1619638940933ULL,
            1619638941284ULL,
        };
        for (int cap = 1; cap <= full_cap; ++cap)
            if (active_by_cap[size_t(cap)] != U128(EXPECT_CAP[cap])) return 2;
        if (low_all_entries != 1201917ULL
            || dense_per_row != U128(1619638941284ULL)
            || dense_total != U128(45349890355952ULL)
            || active_total != U128(39825352231352ULL)) {
            std::cerr << "n=27 LOW closure row-depth regression\n";
            return 3;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "lowclosure-rowdepth W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "low_all_peak_entries=" << low_all_entries
              << " metadata_mib=" << double(low_all_entries) / double(1ULL << 20)
              << '\n'
              << "dense_per_row=" << u128_string(dense_per_row)
              << " dense_total=" << u128_string(dense_total) << '\n'
              << "active_total=" << u128_string(active_total)
              << " ratio=" << double(as_ld(active_total) / as_ld(dense_total))
              << " reduction="
              << double(1.0L - as_ld(active_total) / as_ld(dense_total)) << '\n';
    return 0;
}
