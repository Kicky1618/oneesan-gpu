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

static void enum_low_rec(int pos, int h, std::uint32_t code,
                         std::uint8_t peak, std::vector<CodePeak>& out) {
    if (pos < 0) {
        if (h == 0) out.push_back({code, peak});
        return;
    }
    if (h < 0 || h > pos + 1) return;
    enum_low_rec(pos - 1, h, code, peak, out);
    if (h > 0)
        enum_low_rec(pos - 1, h - 1,
                     code | (1u << (2 * pos)), peak, out);
    enum_low_rec(pos - 1, h + 1,
                 code | (2u << (2 * pos)),
                 std::uint8_t(std::max<int>(peak, h + 1)), out);
}

static std::vector<std::vector<CodePeak>> enumerate_low(int len) {
    std::vector<std::vector<CodePeak>> by_h(len + 2);
    for (int h = 0; h <= len + 1; ++h)
        enum_low_rec(len - 1, h, 0, std::uint8_t(h), by_h[h]);
    return by_h;
}

static void enum_high_rec(int pos, int h, std::uint32_t code,
                          std::uint8_t peak,
                          std::vector<std::vector<CodePeak>>& by_h) {
    if (pos < 0) {
        by_h[h].push_back({code, peak});
        return;
    }
    enum_high_rec(pos - 1, h, code, peak, by_h);
    if (h > 0)
        enum_high_rec(pos - 1, h - 1,
                      code | (1u << (2 * pos)), peak, by_h);
    enum_high_rec(pos - 1, h + 1,
                  code | (2u << (2 * pos)),
                  std::uint8_t(std::max<int>(peak, h + 1)), by_h);
}

static std::vector<std::vector<CodePeak>> enumerate_high(int len) {
    std::vector<std::vector<CodePeak>> by_h(len + 2);
    enum_high_rec(len - 1, 1, 0, 1, by_h);
    return by_h;
}

static bool orbit_dirty_high_row(std::uint32_t high_code, int center,
                                 int high_len) {
    // active=(HIGH code << 2)|center.  For a HIGH position q, mpair() reads
    // the adjacent base-4 symbols as 4*upper + lower.  The block-orbit kernel
    // writes both MAIN members of the three orbit classes:
    //   NN <-> LR, NR <-> RN, NL <-> LN.
    // Closure-source pairs LL/RR/RL read MAIN but only add into BLOCKED.
    static constexpr bool dirty_pair[16] = {
        true,  true,  true,  false,
        true,  false, false, false,
        true,  true,  false, false,
        false, false, false, false,
    };
    const std::uint32_t active = (high_code << 2) | std::uint32_t(center);
    for (int q = 1; q <= high_len; ++q) {
        const std::uint32_t pair = (active >> (2 * (q - 1))) & 15u;
        if (dirty_pair[pair]) return true;
    }
    return false;
}

static std::vector<std::vector<U64>> peak_cumulative(
    const std::vector<std::vector<CodePeak>>& by_h, int max_peak
) {
    std::vector<std::vector<U64>> out(
        by_h.size(), std::vector<U64>(max_peak + 1));
    for (std::size_t h = 0; h < by_h.size(); ++h) {
        for (const CodePeak& x : by_h[h])
            if (x.peak <= max_peak) ++out[h][x.peak];
        for (int p = 1; p <= max_peak; ++p)
            out[h][p] += out[h][p - 1];
    }
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 28 || low < 1 || high < 1 || low > 15 || high > 15)
        return 1;

    const auto lows = enumerate_low(low);
    const auto highs = enumerate_high(high);
    const int maxh = std::min(low, high + 1);
    const auto lc = peak_cumulative(lows, maxh);

    // [ending height][center][cap], with center N=0,R=1,L=2.
    std::vector<std::vector<std::vector<U64>>> all(
        highs.size(), std::vector<std::vector<U64>>(
            3, std::vector<U64>(maxh + 1)));
    auto dirty = all;
    U64 clean_high_row_variants = 0;
    for (std::size_t he = 0; he < highs.size(); ++he) {
        for (const CodePeak& x : highs[he]) {
            if (x.peak > maxh) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = int(he) + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs >= int(lows.size())) continue;
                ++all[he][cv][x.peak];
                if (orbit_dirty_high_row(x.code, cv, high))
                    ++dirty[he][cv][x.peak];
                else
                    ++clean_high_row_variants;
            }
        }
    }
    for (std::size_t he = 0; he < highs.size(); ++he)
        for (int cv = 0; cv < 3; ++cv)
            for (int cap = 1; cap <= maxh; ++cap) {
                all[he][cv][cap] += all[he][cv][cap - 1];
                dirty[he][cv][cap] += dirty[he][cv][cap - 1];
            }

    auto main_cap = [&](int cap, bool dirty_only) -> U64 {
        U64 z = 0;
        const auto& table = dirty_only ? dirty : all;
        for (std::size_t he = 0; he < highs.size(); ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = int(he) + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs >= int(lc.size())) continue;
                z += table[he][cv][cap] * lc[std::size_t(hs)][cap];
            }
        }
        return z;
    };

    U64 skipped_scatter_words = 0;
    for (int row = 1; row <= W; ++row) {
        const int scatter_cap = std::min(maxh, row);
        skipped_scatter_words +=
            main_cap(scatter_cap, false) - main_cap(scatter_cap, true);
    }

    // Exact v0.15 total, independently pinned by factor_row_depth_factorpeak.
    U64 baseline_words = 0;
    if (W == 28 && low == 14) baseline_words = 22074529070967ULL;
    const U64 candidate_words = baseline_words
        ? baseline_words - skipped_scatter_words : 0;

    if (W == 28 && low == 14) {
        if (clean_high_row_variants != 2ULL
            || main_cap(13, false) - main_cap(13, true) != 14ULL
            || main_cap(14, false) - main_cap(14, true) != 14ULL
            || skipped_scatter_words != 224ULL
            || candidate_words != 22074529070743ULL) {
            std::cerr << "n=27 HIGH dirty-scatter regression mismatch\n";
            return 2;
        }
    }

    const long double tib = static_cast<long double>(U64(1) << 40);
    const long double saved_tib =
        static_cast<long double>(skipped_scatter_words) * 4.0L / tib;
    std::cout << std::fixed << std::setprecision(12)
              << "highio-dirty-scatter W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "clean_high_row_variants=" << clean_high_row_variants << '\n'
              << "fullcap_main_words=" << main_cap(maxh, false)
              << " fullcap_dirty_main_words=" << main_cap(maxh, true)
              << " fullcap_skippable_main_words="
              << main_cap(maxh, false) - main_cap(maxh, true) << '\n'
              << "skipped_scatter_words_per_residue=" << skipped_scatter_words
              << " skipped_bytes_per_residue=" << skipped_scatter_words * 4ULL
              << " skipped_tib_per_residue=" << double(saved_tib) << '\n';
    if (baseline_words) {
        std::cout << "baseline_v015_words=" << baseline_words
                  << " candidate_words=" << candidate_words
                  << " logical_tib_saved=" << double(saved_tib)
                  << " balanced_7of8_peer_tib_saved="
                  << double(saved_tib * 7.0L / 8.0L) << '\n';
    }
    std::cout << "decision=reject-negligible\n";
    return 0;
}
