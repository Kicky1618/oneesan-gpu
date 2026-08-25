#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

static std::vector<U64> high_counts_capped(int len, int cap, int max_h) {
    std::vector<U64> cur(max_h + 2), nxt(max_h + 2);
    if (cap >= 1) cur[1] = 1;
    for (int pos = 0; pos < len; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= max_h; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h > 0) nxt[h - 1] += cur[h];
            if (h + 1 <= cap) nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur;
}

static std::vector<U64> low_counts_capped(
    int len, std::uint32_t mask, int cap
) {
    std::vector<U64> out(len + 2);
    for (int start = 0; start <= len + 1; ++start) {
        if (start > cap) continue;
        std::vector<U64> cur(cap + 2), nxt(cap + 2);
        cur[start] = 1;
        for (int p = len - 1; p >= 0; --p) {
            std::fill(nxt.begin(), nxt.end(), 0);
            const bool occupied = (mask >> p) & 1u;
            for (int h = 0; h <= cap; ++h) if (cur[h]) {
                if (!occupied) {
                    nxt[h] += cur[h];
                } else {
                    if (h > 0) nxt[h - 1] += cur[h];
                    if (h + 1 <= cap) nxt[h + 1] += cur[h];
                }
            }
            cur.swap(nxt);
        }
        out[start] = cur[0];
    }
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    if (W < 4 || W > 28 || low < 1 || high < 1 || low > 20 || high > 20
        || threads < 1) return 1;

    std::vector<std::vector<U64>> high_cap(full_cap + 1);
    for (int cap = 1; cap <= full_cap; ++cap)
        high_cap[cap] = high_counts_capped(high, cap, high + 1);

    const std::uint32_t nmasks = 1u << low;
    std::vector<U64> exact_bd(W, 0);
    U64 dense_bd_sum = 0;
    U64 low_entries = 0;
    U64 max_low_group = 0;

    for (std::uint32_t mask = 0; mask < nmasks; ++mask) {
        const auto low_full = low_counts_capped(low, mask, full_cap);
        U64 block_n = 0;
        for (int h = 0; h <= high + 1; ++h) {
            block_n += high_cap[full_cap][h] * low_full[h];
            low_entries += low_full[h];
            max_low_group = std::max(max_low_group, low_full[h]);
        }
        dense_bd_sum += block_n
            ? std::min<U64>(65535, (block_n + U64(threads) - 1) / U64(threads))
            : 0;

        for (int row = 1; row <= W; ++row) {
            const int cap = std::min(row, full_cap);
            const auto low_cap = low_counts_capped(low, mask, cap);
            U64 active_n = 0;
            for (int h = 0; h <= high + 1; ++h)
                active_n += high_cap[cap][h] * low_cap[h];
            exact_bd[row - 1] += active_n
                ? std::min<U64>(65535,
                    (active_n + U64(threads) - 1) / U64(threads))
                : 0;
        }
    }

    U64 high_entries = 0;
    U64 max_high_group = 0;
    for (int h = 0; h <= high + 1; ++h) {
        high_entries += high_cap[full_cap][h];
        max_high_group = std::max(max_high_group, high_cap[full_cap][h]);
    }

    U64 exact_sum = 0;
    for (U64 x : exact_bd) exact_sum += x;
    const U64 tight_ctas = dense_bd_sum * U64(high) * U64(W);
    const U64 exact_ctas = exact_sum * U64(high);

    // One compact rank permutation per factor-code entry, plus cap-dependent
    // active counts. LOW mask-ranks fit uint16 at n=27; HIGH all-ranks use
    // uint32. The LOW count table is indexed by (mask,start_height,cap).
    const U64 low_perm_bytes = low_entries * 2;
    const U64 high_perm_bytes = high_entries * 4;
    const U64 low_count_bytes = U64(nmasks) * U64(low + 2)
                              * U64(full_cap + 1) * 2;
    const U64 high_count_bytes = U64(high + 2) * U64(full_cap + 1) * 4;
    const U64 metadata_bytes = low_perm_bytes + high_perm_bytes
                             + low_count_bytes + high_count_bytes;

    if (W == 28 && low == 14 && threads == 256) {
        static constexpr U64 EXPECT_EXACT_BD[10] = {
            262144ULL, 29871782ULL, 166568794ULL, 310992781ULL,
            371900470ULL, 394572120ULL, 400309007ULL, 401266964ULL,
            401363918ULL, 401370925ULL
        };
        const U64 expected_metadata = 1201917ULL * 2ULL
            + 787333ULL * 4ULL
            + (1ULL << 14) * 16ULL * 15ULL * 2ULL
            + 15ULL * 15ULL * 4ULL;
        if (dense_bd_sum != 401370925ULL
            || tight_ctas != 146099016700ULL
            || exact_ctas != 131341022215ULL
            || low_entries != 1201917ULL
            || high_entries != 787333ULL
            || max_low_group != 1001ULL
            || max_high_group != 149019ULL
            || metadata_bytes != expected_metadata) {
            std::cerr << "n=27 exact-task launch aggregate regression mismatch\n";
            return 2;
        }
        for (int r = 0; r < 10; ++r) if (exact_bd[r] != EXPECT_EXACT_BD[r]) {
            std::cerr << "n=27 exact-task launch row regression mismatch row="
                      << r + 1 << '\n';
            return 3;
        }
    }

    const long double ratio = tight_ctas
        ? (long double)exact_ctas / (long double)tight_ctas : 1.0L;
    std::cout << std::fixed << std::setprecision(12)
              << "blockorbit-exacttask-launch W=" << W
              << " low=" << low << " high=" << high
              << " threads=" << threads << '\n'
              << "tight_ctas=" << tight_ctas
              << " exact_ctas=" << exact_ctas
              << " ratio=" << double(ratio)
              << " reduction=" << double(1.0L - ratio) << '\n'
              << "low_entries=" << low_entries
              << " high_entries=" << high_entries
              << " max_low_group=" << max_low_group
              << " max_high_group=" << max_high_group << '\n'
              << "metadata_mib="
              << double(metadata_bytes) / double(1ULL << 20) << '\n';
    for (int r = 0; r < std::min(W, full_cap); ++r)
        std::cout << "row=" << r + 1 << " exact_sum_bd=" << exact_bd[r] << '\n';
    return 0;
}
