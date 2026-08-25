#include <algorithm>
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
    U64 sum_dense_bd = 0;
    std::vector<U64> row_bd(W, 0), row_cut_jobs(W, 0);

    for (std::uint32_t mask = 0; mask < nmasks; ++mask) {
        const auto lc = low_mask_counts(low, mask);
        std::vector<U64> prefix(high + 2, 0);
        U64 block_n = 0;
        for (int h = 0; h <= high + 1; ++h) {
            block_n += hc[h] * lc[h];
            prefix[h] = block_n;
        }
        const U64 bd = std::min<U64>(
            65535, (block_n + U64(threads) - 1) / U64(threads));
        sum_dense_bd += bd;

        for (int row = 1; row <= W; ++row) {
            const int cap = std::min(row, high + 1);
            const U64 active_n = prefix[cap];
            const U64 br = active_n
                ? std::min<U64>(65535, (active_n + U64(threads) - 1) / U64(threads))
                : 0;
            row_bd[row - 1] += br;
            if (br < bd) ++row_cut_jobs[row - 1];
        }
    }

    const U64 tight_ctas = sum_dense_bd * U64(high) * U64(W);
    U64 rowcap_sum = 0;
    for (U64 x : row_bd) rowcap_sum += x;
    const U64 rowcap_ctas = rowcap_sum * U64(high);
    const long double ratio = tight_ctas
        ? (long double)rowcap_ctas / (long double)tight_ctas : 1.0L;

    if (W == 28 && low == 14 && threads == 256) {
        static constexpr U64 EXPECT_ROW_BD[9] = {
            126073708ULL, 247591104ULL, 333369706ULL,
            369538839ULL, 394328461ULL, 399112240ULL,
            401183738ULL, 401330885ULL, 401370925ULL
        };
        static constexpr U64 EXPECT_CUT_JOBS[9] = {
            16354ULL, 16172ULL, 15444ULL, 13442ULL, 11440ULL,
            8437ULL, 5005ULL, 2002ULL, 0ULL
        };
        if (sum_dense_bd != 401370925ULL
            || tight_ctas != 146099016700ULL
            || rowcap_ctas != 139099313353ULL) {
            std::cerr << "n=27 row-cap launch aggregate regression mismatch\n";
            return 2;
        }
        for (int r = 0; r < 9; ++r) {
            if (row_bd[r] != EXPECT_ROW_BD[r]
                || row_cut_jobs[r] != EXPECT_CUT_JOBS[r]) {
                std::cerr << "n=27 row-cap launch row regression mismatch row="
                          << r + 1 << '\n';
                return 3;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(12)
              << "blockorbit-rowcap-launch W=" << W
              << " low=" << low << " high=" << high
              << " threads=" << threads << '\n'
              << "tight_ctas=" << tight_ctas
              << " rowcap_ctas=" << rowcap_ctas
              << " ratio=" << double(ratio)
              << " reduction=" << double(1.0L - ratio) << '\n';
    const int shown = std::min(W, high + 1);
    for (int r = 0; r < shown; ++r)
        std::cout << "row=" << r + 1
                  << " sum_bd=" << row_bd[r]
                  << " cut_jobs=" << row_cut_jobs[r] << '\n';
    return 0;
}
