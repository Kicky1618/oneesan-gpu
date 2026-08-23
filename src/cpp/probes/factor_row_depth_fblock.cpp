#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

static std::vector<U64> high_end_counts(int len) {
    std::vector<U64> cur(len + 3), nxt(len + 3);
    cur[1] = 1;
    for (int pos = 0; pos < len; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= len + 1; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur;
}

static U64 low_suffix_count(int len, int start_h) {
    std::vector<U64> cur(len + start_h + 3), nxt(cur.size());
    cur[start_h] = 1;
    for (int pos = 0; pos < len; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h + 1 < int(cur.size()); ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || high < 1) return 1;

    const auto hc = high_end_counts(high);
    std::vector<U64> lc(low + 2);
    for (int h = 0; h <= low + 1; ++h) lc[h] = low_suffix_count(low, h);

    auto main_cap = [&](int cap) -> U64 {
        U64 z = 0;
        for (int he = 0; he < int(hc.size()); ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low + 1) continue;
                if (std::max(he, hs) <= cap) z += hc[he] * lc[hs];
            }
        }
        return z;
    };
    auto block_cap = [&](int cap) -> U64 {
        U64 z = 0;
        for (int h = 0; h < int(hc.size()) && h <= low + 1; ++h)
            if (h <= cap) z += hc[h] * lc[h];
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
    const U64 dense_words = U64(W) * (2 * M + D);

    if (W == 28 && low == 14) {
        static const U64 MC[14] = {
            61163428536ULL,154204141392ULL,252005907636ULL,323991286101ULL,
            363189258486ULL,379303526448ULL,384319614096ULL,385491392496ULL,
            385692748560ULL,385717368078ULL,385719399940ULL,385719503784ULL,
            385719506592ULL,385719506620ULL};
        static const U64 DC[14] = {
            32803431948ULL,67423973085ULL,99047777661ULL,119798216370ULL,
            129956911488ULL,133714131318ULL,134761796118ULL,134979122682ULL,
            135011830266ULL,135015260616ULL,135015495760ULL,135015505224ULL,
            135015505406ULL,135015505407ULL};
        for (int cap = 1; cap <= 14; ++cap)
            if (main_cap(cap) != MC[cap - 1] || block_cap(cap) != DC[cap - 1]) {
                std::cerr << "n=27 FBlock cap regression mismatch cap=" << cap << '\n';
                return 2;
            }
        if (M != 385719506620ULL || D != 135015505407ULL
            || dense_words != 25380726522116ULL
            || words != 23264294823853ULL) {
            std::cerr << "n=27 FBlock traffic regression mismatch\n";
            return 3;
        }
    }

    const long double ratio = (long double)words / (long double)dense_words;
    const long double tib = (long double)words * 4.0L / (long double)(U64(1) << 40);
    std::cout << std::fixed << std::setprecision(9)
              << "factor-row-depth-fblock W=" << W << " low=" << low << " high=" << high << '\n'
              << "main=" << M << " blocked=" << D << '\n'
              << "dense_words=" << dense_words << " fblock_cap_words=" << words
              << " ratio=" << double(ratio)
              << " reduction=" << double(1.0L - ratio) << '\n'
              << "logical_tib=" << double(tib)
              << " balanced_7of8_peer_tib=" << double(tib * 7.0L / 8.0L) << '\n';
    return 0;
}
