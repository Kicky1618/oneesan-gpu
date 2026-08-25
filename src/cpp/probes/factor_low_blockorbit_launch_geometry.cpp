#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;

static std::vector<U64> low_all_counts(int len) {
    std::vector<U64> out(len + 2);
    for (int h0 = 0; h0 <= len + 1; ++h0) {
        std::vector<U64> cur(len + 3), nxt(len + 3);
        cur[h0] = 1;
        for (int pos = len - 1; pos >= 0; --pos) {
            std::fill(nxt.begin(), nxt.end(), 0);
            for (int h = 0; h <= len + 1; ++h) if (cur[h]) {
                nxt[h] += cur[h];
                if (h > 0) nxt[h - 1] += cur[h];
                if (h + 1 <= len + 1) nxt[h + 1] += cur[h];
            }
            cur.swap(nxt);
        }
        out[h0] = cur[0];
    }
    return out;
}

static std::vector<std::vector<U64>> high_mask_counts(int len) {
    const std::uint32_t nm = 1u << len;
    std::vector<std::vector<U64>> out(nm, std::vector<U64>(len + 2));
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t mask) -> void {
        if (pos < 0) {
            if (h >= 0 && h < int(out[mask].size())) ++out[mask][h];
            return;
        }
        self(self, pos - 1, h, mask); // N
        if (h > 0) self(self, pos - 1, h - 1, mask | (1u << pos)); // R
        self(self, pos - 1, h + 1, mask | (1u << pos)); // L
    };
    rec(rec, len - 1, 1, 0u);
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int high = W - 1 - low;
    if (W < 4 || W > 28 || low < 1 || high < 1 || high >= 20
        || threads < 1) return 1;

    const auto lc = low_all_counts(low);
    const auto hc = high_mask_counts(high);
    const std::uint32_t nmasks = 1u << high;

    U64 sum_main_blocks = 0;
    U64 sum_block_blocks = 0;
    U64 jobs_main_gt_block = 0;
    U64 max_main_n = 0;
    U64 max_block_n = 0;

    for (std::uint32_t mask = 0; mask < nmasks; ++mask) {
        U64 main_n = 0;
        U64 block_n = 0;
        for (int he = 0; he < int(hc[mask].size()); ++he) {
            const U64 rows = hc[mask][he];
            if (!rows) continue;
            const int hs0 = he;
            const int hsR = he - 1;
            const int hsL = he + 1;
            if (hs0 >= 0 && hs0 < int(lc.size())) main_n += rows * lc[hs0];
            if (hsR >= 0 && hsR < int(lc.size())) main_n += rows * lc[hsR];
            if (hsL >= 0 && hsL < int(lc.size())) main_n += rows * lc[hsL];
            if (he >= 0 && he < int(lc.size())) block_n += rows * lc[he];
        }
        const U64 bm = std::min<U64>(
            65535, (main_n + U64(threads) - 1) / U64(threads));
        const U64 bd = std::min<U64>(
            65535, (block_n + U64(threads) - 1) / U64(threads));
        sum_main_blocks += bm;
        sum_block_blocks += bd;
        if (bm > bd) ++jobs_main_gt_block;
        max_main_n = std::max(max_main_n, main_n);
        max_block_n = std::max(max_block_n, block_n);
    }

    const U64 calls_per_job = U64(low) * U64(W);
    const U64 dense_ctas = sum_main_blocks * calls_per_job;
    const U64 tight_ctas = sum_block_blocks * calls_per_job;
    const long double ratio = dense_ctas
        ? (long double)tight_ctas / (long double)dense_ctas : 1.0L;

    if (W == 28 && low == 14 && threads == 256) {
        if (sum_main_blocks != 452054386ULL
            || sum_block_blocks != 307130153ULL
            || jobs_main_gt_block != 5812ULL
            || max_main_n != 1471935235ULL
            || max_block_n != 510468519ULL
            || dense_ctas != 177205319312ULL
            || tight_ctas != 120395019976ULL) {
            std::cerr << "n=27 tight LOW launch regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "low-blockorbit-launch-geometry W=" << W
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
