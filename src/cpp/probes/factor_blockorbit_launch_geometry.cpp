#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

static std::vector<U64> high_end_counts(int len, int max_h) {
    std::vector<U64> cur(max_h + 2), nxt(max_h + 2);
    cur[1] = 1;
    for (int pos = 0; pos < len; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= max_h; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h > 0) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur;
}

static std::vector<U64> low_mask_counts(int len, std::uint32_t mask) {
    std::vector<U64> out(len + 2);
    for (int start = 0; start <= len + 1; ++start) {
        std::vector<U64> cur(len + 3), nxt(len + 3);
        cur[start] = 1;
        for (int p = len - 1; p >= 0; --p) {
            std::fill(nxt.begin(), nxt.end(), 0);
            const bool occupied = (mask >> p) & 1u;
            for (int h = 0; h <= len + 1; ++h) if (cur[h]) {
                if (!occupied) {
                    nxt[h] += cur[h];
                } else {
                    if (h > 0) nxt[h - 1] += cur[h];
                    if (h + 1 <= len + 1) nxt[h + 1] += cur[h];
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
    if (W < 4 || W > 28 || low < 1 || high < 1 || low > 20 || high > 20
        || threads < 1) return 1;

    const auto hc = high_end_counts(high, high + 1);
    const std::uint32_t nmasks = 1u << low;
    U64 sum_main_blocks = 0;
    U64 sum_block_blocks = 0;
    U64 jobs_main_gt_block = 0;
    U64 max_main_n = 0;
    U64 max_block_n = 0;

    for (std::uint32_t mask = 0; mask < nmasks; ++mask) {
        const auto lc = low_mask_counts(low, mask);
        U64 main_n = 0;
        U64 block_n = 0;
        for (int he = 0; he < int(hc.size()); ++he) {
            if (!hc[he]) continue;
            const int hs0 = he;
            const int hsR = he - 1;
            const int hsL = he + 1;
            if (hs0 >= 0 && hs0 < int(lc.size())) main_n += hc[he] * lc[hs0];
            if (hsR >= 0 && hsR < int(lc.size())) main_n += hc[he] * lc[hsR];
            if (hsL >= 0 && hsL < int(lc.size())) main_n += hc[he] * lc[hsL];
            if (he < int(lc.size())) block_n += hc[he] * lc[he];
        }
        const U64 bm = std::min<U64>(65535, (main_n + U64(threads) - 1) / U64(threads));
        const U64 bd = std::min<U64>(65535, (block_n + U64(threads) - 1) / U64(threads));
        sum_main_blocks += bm;
        sum_block_blocks += bd;
        if (bm > bd) ++jobs_main_gt_block;
        max_main_n = std::max(max_main_n, main_n);
        max_block_n = std::max(max_block_n, block_n);
    }

    const U64 calls_per_job = U64(high) * U64(W);
    const U64 dense_ctas = sum_main_blocks * calls_per_job;
    const U64 tight_ctas = sum_block_blocks * calls_per_job;
    const long double ratio = dense_ctas
        ? (long double)tight_ctas / (long double)dense_ctas : 1.0L;

    if (W == 28 && low == 14 && threads == 256) {
        if (sum_main_blocks != 703136363ULL
            || sum_block_blocks != 401370925ULL
            || jobs_main_gt_block != 14913ULL
            || max_main_n != 961466716ULL
            || max_block_n != 333251570ULL
            || dense_ctas != 255941636132ULL
            || tight_ctas != 146099016700ULL) {
            std::cerr << "n=27 tight-launch regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "blockorbit-launch-geometry W=" << W
              << " low=" << low << " high=" << high
              << " threads=" << threads << '\n'
              << "jobs=" << nmasks
              << " jobs_main_gt_block=" << jobs_main_gt_block << '\n'
              << "sum_bm=" << sum_main_blocks
              << " sum_bd=" << sum_block_blocks
              << " ratio=" << double(ratio)
              << " reduction=" << double(1.0L - ratio) << '\n'
              << "per_residue_dense_ctas=" << dense_ctas
              << " tight_ctas=" << tight_ctas << '\n'
              << "max_main_n=" << max_main_n
              << " max_block_n=" << max_block_n << '\n';
    return 0;
}
