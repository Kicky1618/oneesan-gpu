#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

struct CodePeak {
    std::uint32_t code = 0;
    std::uint8_t peak = 0;
};

static void enum_low_rec(int pos, int h, int start_h, std::uint32_t code,
                         std::uint8_t peak, std::vector<CodePeak>& out) {
    if (pos < 0) {
        if (h == 0) out.push_back({code, peak});
        return;
    }
    if (h < 0 || h > pos + 1) return;
    enum_low_rec(pos - 1, h, start_h, code, peak, out);
    if (h > 0)
        enum_low_rec(pos - 1, h - 1, start_h,
                     code | (1u << (2 * pos)), peak, out);
    enum_low_rec(pos - 1, h + 1, start_h,
                 code | (2u << (2 * pos)),
                 std::uint8_t(std::max<int>(peak, h + 1)), out);
    (void)start_h;
}

static std::vector<std::vector<CodePeak>> enumerate_low(int len) {
    std::vector<std::vector<CodePeak>> by_h(len + 2);
    for (int h = 0; h <= len + 1; ++h)
        enum_low_rec(len - 1, h, h, 0, std::uint8_t(h), by_h[h]);
    return by_h;
}

static void enum_high_rec(int pos, int h, std::uint32_t code, std::uint8_t peak,
                          std::vector<std::vector<CodePeak>>& by_h) {
    if (pos < 0) {
        by_h[h].push_back({code, peak});
        return;
    }
    enum_high_rec(pos - 1, h, code, peak, by_h);
    if (h > 0)
        enum_high_rec(pos - 1, h - 1, code | (1u << (2 * pos)), peak, by_h);
    enum_high_rec(pos - 1, h + 1, code | (2u << (2 * pos)),
                  std::uint8_t(std::max<int>(peak, h + 1)), by_h);
}

static std::vector<std::vector<CodePeak>> enumerate_high(int len) {
    std::vector<std::vector<CodePeak>> by_h(len + 2);
    enum_high_rec(len - 1, 1, 0, 1, by_h);
    return by_h;
}

static U64 capped_state_count(int width, int cap) {
    std::vector<U64> cur(cap + 1), nxt(cap + 1);
    cur[1] = 1;
    for (int p = 0; p < width; ++p) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= cap; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h) nxt[h - 1] += cur[h];
            if (h + 1 <= cap) nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 28 || low < 1 || high < 1 || low > 15 || high > 15) return 1;

    const auto lows = enumerate_low(low);
    const auto highs = enumerate_high(high);
    U64 low_entries = 0, high_entries = 0;
    for (const auto& v : lows) low_entries += v.size();
    for (const auto& v : highs) high_entries += v.size();

    auto main_cap = [&](int cap) -> U64 {
        U64 z = 0;
        for (int he = 0; he < int(highs.size()); ++he) {
            for (const CodePeak& hp : highs[he]) if (hp.peak <= cap) {
                for (int cv = 0; cv < 3; ++cv) {
                    const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                    if (hs < 0 || hs >= int(lows.size())) continue;
                    for (const CodePeak& lp : lows[hs])
                        if (lp.peak <= cap) ++z;
                }
            }
        }
        return z;
    };
    auto block_cap = [&](int cap) -> U64 {
        U64 z = 0;
        for (int h = 0; h < int(highs.size()) && h < int(lows.size()); ++h)
            for (const CodePeak& hp : highs[h]) if (hp.peak <= cap)
                for (const CodePeak& lp : lows[h]) if (lp.peak <= cap) ++z;
        return z;
    };

    const int maxh = std::min(low, high + 1);
    const U64 M = main_cap(maxh);
    const U64 D = block_cap(maxh);
    U64 words = 0;
    for (int row = 1; row <= W; ++row) {
        const int gather_cap = std::min(maxh, std::max(1, row - 1));
        const int scatter_cap = std::min(maxh, row);
        words += main_cap(gather_cap) + main_cap(scatter_cap) + block_cap(scatter_cap);
    }

    if (W == 28 && low == 14) {
        if (low_entries != 1201917ULL || high_entries != 787333ULL
            || M != 385719506620ULL || D != 135015505407ULL) {
            std::cerr << "n=27 exact-factor metadata regression mismatch\n";
            return 2;
        }
        for (int cap = 1; cap <= 14; ++cap) {
            const U64 em = capped_state_count(W, cap);
            const U64 ed = capped_state_count(W - 1, cap);
            if (main_cap(cap) != em || block_cap(cap) != ed) {
                std::cerr << "exact-factor cap mismatch cap=" << cap << '\n';
                return 3;
            }
        }
        // Runtime uses cap=1, rather than the single initial state, for row-1 gather.
        if (words != 22074529070967ULL) {
            std::cerr << "n=27 exact-factor traffic regression mismatch\n";
            return 4;
        }
    }

    const long double tib = (long double)words * 4.0L / (long double)(U64(1) << 40);
    std::cout << std::fixed << std::setprecision(9)
              << "factor-row-depth-factorpeak W=" << W << " low=" << low
              << " high=" << high << '\n'
              << "low_peak_entries=" << low_entries
              << " high_peak_entries=" << high_entries
              << " metadata_mib=" << double(low_entries + high_entries) / double(1ULL << 20)
              << '\n'
              << "main=" << M << " blocked=" << D
              << " exact_cap_words=" << words << '\n'
              << "logical_tib=" << double(tib)
              << " balanced_7of8_peer_tib=" << double(tib * 7.0L / 8.0L) << '\n';
    return 0;
}
