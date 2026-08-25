#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;
using LD = long double;

static std::vector<U64> high_row_counts(int high) {
    std::vector<U64> cur(high + 3), nxt(high + 3);
    cur[1] = 1;
    for (int step = 0; step < high; ++step) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= high + 1; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur;
}

static std::vector<std::vector<std::uint32_t>> low_codes_by_start(int low) {
    std::vector<std::vector<std::uint32_t>> out(low + 1);
    for (int h0 = 0; h0 <= low; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) out[h0].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h) self(self, pos - 1, h - 1,
                        code | (1u << (2 * pos)));
            self(self, pos - 1, h + 1,
                 code | (2u << (2 * pos)));
        };
        rec(rec, low - 1, h0, 0);
    }
    return out;
}

static bool closure_pair(std::uint32_t low_code, int center, int low, int p) {
    const std::uint32_t active = low_code | (std::uint32_t(center) << (2 * low));
    const std::uint32_t w = (active >> (2 * (p - 1))) & 15u;
    return w == 0xau || w == 0x5u || w == 0x6u;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const LD v08_gib = argc > 3 ? std::strtold(argv[3], nullptr) : 249.116280202L;
    const LD usable_gib = argc > 4 ? std::strtold(argv[4], nullptr) : 268.59L;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || low >= 16 || high < 1 || high >= 16)
        return 1;

    const auto high_rows = high_row_counts(high);
    const auto low_codes = low_codes_by_start(low);
    U64 expected = 0;
    for (int p = low; p >= 1; --p) {
        U64 cols = 0;
        for (int he = 0; he <= high + 1; ++he) {
            if (!high_rows[he]) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                for (std::uint32_t lc : low_codes[hs])
                    if (closure_pair(lc, cv, low, p)) ++cols;
            }
        }
        if (!expected) expected = cols;
        else if (cols != expected) {
            std::cerr << "LOW closure column count depends on p: "
                      << cols << " vs " << expected << '\n';
            return 2;
        }
    }

    const LD col_bytes = LD(4) * LD(low) * LD(expected);
    const LD block_off_bytes = LD(4) * LD(low) * 65.0L;
    const LD delta_bytes = col_bytes + block_off_bytes;
    const LD delta_mib = delta_bytes / LD(U64(1) << 20);
    const LD delta_gib = delta_bytes / LD(U64(1) << 30);
    const LD v09_gib = v08_gib + delta_gib;

    std::cout << std::fixed << std::setprecision(6)
              << "v09-memory-delta W=" << W << " low=" << low << " high=" << high << '\n'
              << "low_closure_cols_per_p=" << expected
              << " low_closure_col_mib=" << double(col_bytes / LD(U64(1) << 20))
              << " low_closure_block_off_kib=" << double(block_off_bytes / 1024.0L) << '\n'
              << "v09_extra_mib=" << double(delta_mib)
              << " v09_peak_gib=" << double(v09_gib)
              << " v09_headroom_gib=" << double(usable_gib - v09_gib) << '\n';

    if (W == 28 && low == 14 && expected != 1088282ULL) {
        std::cerr << "unexpected n=27 LOW closure column count=" << expected << '\n';
        return 3;
    }
    if (v09_gib >= usable_gib) {
        std::cerr << "v0.9 exceeds requested usable HBM\n";
        return 4;
    }
    return 0;
}
