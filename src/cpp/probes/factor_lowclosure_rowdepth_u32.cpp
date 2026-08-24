#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

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
    if (W < 4 || W > 28 || low < 1 || high < 1 || low >= 16 || high >= 16
        || full_cap >= 16) return 1;

    const std::uint32_t NM = 1u << high;
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

    U64 max_tasks = 0;
    int max_p = 0, max_cap = 0;
    std::uint32_t max_mask = 0;
    for (int p = low; p >= 1; --p) {
        const int pi = low - p;
        for (std::uint32_t mask = 0; mask < NM; ++mask) {
            std::array<U64, 16> group{};
            for (int he = 0; he <= high + 1; ++he) {
                const auto& hc = high_cum[std::size_t(mask) * (high + 2) + he];
                for (int cv = 0; cv < 3; ++cv) {
                    for (int cap = 1; cap <= full_cap; ++cap) {
                        const U64 rows = hc[size_t(cap)];
                        const U64 sel = selected[six(pi, he, cv, cap)];
                        if (rows && sel)
                            group[size_t(cap)] += rows * ((sel + 31) / 32);
                    }
                }
            }
            for (int cap = 1; cap <= full_cap; ++cap) {
                if (group[size_t(cap)] > max_tasks) {
                    max_tasks = group[size_t(cap)];
                    max_p = p;
                    max_mask = mask;
                    max_cap = cap;
                }
            }
        }
    }

    if (max_tasks > 0xffffffffULL) {
        std::cerr << "LOW closure task ordinal exceeds uint32: " << max_tasks << '\n';
        return 2;
    }
    if (W == 28 && low == 14) {
        if (max_tasks != 14097070ULL || max_p != 13
            || max_mask != 8191u || max_cap != 14) {
            std::cerr << "n=27 LOW closure u32 regression\n";
            return 3;
        }
    }

    std::cout << "lowclosure-u32 W=" << W << " low=" << low
              << " high=" << high
              << " max_tasks=" << max_tasks
              << " max_p=" << max_p
              << " max_mask=" << max_mask
              << " max_cap=" << max_cap
              << " uint32_headroom=" << (0xffffffffULL - max_tasks)
              << '\n';
    return 0;
}
