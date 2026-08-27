#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;
static constexpr int H = 13;
static constexpr int L = 14;
static constexpr int S = 30;
static constexpr int NGPU = 8;
using Counts = std::array<std::uint32_t, S>;

static std::vector<Counts> high_counts() {
    std::vector<Counts> out(1u << H);
    for (std::uint32_t mask = 0; mask < (1u << H); ++mask) {
        std::array<U64, S> dp{}, next{};
        dp[1] = 1;
        for (int pos = H - 1; pos >= 0; --pos) {
            next.fill(0);
            bool occ = (mask >> pos) & 1u;
            for (int h = 0; h + 1 < S; ++h) {
                if (!dp[h]) continue;
                if (!occ) next[h] += dp[h];
                else {
                    if (h > 0) next[h - 1] += dp[h];
                    next[h + 1] += dp[h];
                }
            }
            dp = next;
        }
        for (int h = 0; h < S; ++h) out[mask][h] = std::uint32_t(dp[h]);
    }
    return out;
}

static std::vector<Counts> low_counts() {
    std::vector<Counts> out(1u << L);
    for (std::uint32_t mask = 0; mask < (1u << L); ++mask) {
        for (int h0 = 0; h0 < S; ++h0) {
            std::array<U64, S> dp{}, next{};
            dp[h0] = 1;
            for (int pos = L - 1; pos >= 0; --pos) {
                next.fill(0);
                bool occ = (mask >> pos) & 1u;
                for (int h = 0; h + 1 < S; ++h) {
                    if (!dp[h]) continue;
                    if (!occ) next[h] += dp[h];
                    else {
                        if (h > 0) next[h - 1] += dp[h];
                        next[h + 1] += dp[h];
                    }
                }
                dp = next;
            }
            out[mask][h0] = std::uint32_t(dp[0]);
        }
    }
    return out;
}

struct StorageBlock {
    U64 off = 0;
    U64 rows = 0;
    U64 cols = 0;
    int he = 0;
    int hs = 0;
};

static void add_overlap(std::array<U64, NGPU>& dst, U64 start, U64 len,
                        U64 total, U64 chunk) {
    U64 end = start + len;
    int owner = std::min<int>(start / chunk, NGPU - 1);
    while (start < end) {
        U64 shard_end = owner + 1 < NGPU
            ? std::min<U64>((owner + 1) * chunk, total)
            : total;
        U64 take = std::min(end, shard_end) - start;
        dst[owner] += take;
        start += take;
        ++owner;
    }
}

int main() {
    auto hc = high_counts();
    auto lc = low_counts();
    Counts high_all{}, low_all{};
    for (auto const& a : hc) for (int h = 0; h < S; ++h) high_all[h] += a[h];
    for (auto const& a : lc) for (int h = 0; h < S; ++h) low_all[h] += a[h];

    std::vector<StorageBlock> blocks;
    U64 main_states = 0;
    for (int he = 0; he <= H + 1; ++he) {
        for (int delta : {0, -1, +1}) {
            int hs = he + delta;
            StorageBlock b;
            b.off = main_states;
            b.he = he;
            b.hs = hs;
            if (0 <= hs && hs < S) {
                b.rows = high_all[he];
                b.cols = low_all[hs];
                main_states += b.rows * b.cols;
            }
            blocks.push_back(b);
        }
    }

    U64 low_total_runs = 0, low_lane_slots = 0;
    U64 low_max_states = 0, low_max_runs = 0;
    U64 low_sum_states = 0;
    for (std::uint32_t mask = 0; mask < (1u << L); ++mask) {
        U64 size = 0, runs = 0, lane_slots = 0;
        for (auto const& b : blocks) {
            if (!b.rows || b.hs < 0 || b.hs >= S) continue;
            U64 width = lc[mask][b.hs];
            if (!width) continue;
            size += b.rows * width;
            runs += b.rows;
            lane_slots += b.rows * 32 * ((width + 31) / 32);
        }
        low_sum_states += size;
        low_total_runs += runs;
        low_lane_slots += lane_slots;
        low_max_states = std::max(low_max_states, size);
        low_max_runs = std::max(low_max_runs, runs);
    }

    std::vector<Counts> high_begin(1u << H);
    Counts running{};
    for (std::uint32_t mask = 0; mask < (1u << H); ++mask) {
        high_begin[mask] = running;
        for (int h = 0; h < S; ++h) running[h] += hc[mask][h];
    }

    U64 high_sum_states = 0, high_total_runs = 0;
    U64 high_max_states = 0, high_max_runs = 0;
    long double weighted_best_local = 0;
    U64 chunk = (main_states + NGPU - 1) / NGPU;
    for (std::uint32_t mask = 0; mask < (1u << H); ++mask) {
        U64 size = 0, runs = 0;
        std::array<U64, NGPU> shard{};
        for (auto const& b : blocks) {
            if (!b.rows || !b.cols) continue;
            U64 rows = hc[mask][b.he];
            if (!rows) continue;
            U64 row0 = high_begin[mask][b.he];
            U64 len = rows * b.cols;
            U64 start = b.off + row0 * b.cols;
            size += len;
            ++runs;
            add_overlap(shard, start, len, main_states, chunk);
        }
        high_sum_states += size;
        high_total_runs += runs;
        high_max_states = std::max(high_max_states, size);
        high_max_runs = std::max(high_max_runs, runs);
        if (size) {
            weighted_best_local += (long double)*std::max_element(shard.begin(), shard.end());
        }
    }

    std::cout << "W=28 H=13 center=1 L=14\n";
    std::cout << "main_states=" << main_states << "\n";
    U64 hs = 0, ls = 0;
    for (auto x : high_all) hs += x;
    for (auto x : low_all) ls += x;
    std::cout << "high_all_codes=" << hs << " low_all_codes=" << ls << "\n";
    std::cout << "LOW14 groups=" << (1u << L)
              << " max_states=" << low_max_states
              << " max_row_runs=" << low_max_runs
              << " total_row_runs=" << low_total_runs
              << " avg_run=" << std::fixed << std::setprecision(3)
              << double(low_sum_states) / double(low_total_runs)
              << " warp_row_eff="
              << 100.0 * double(low_sum_states) / double(low_lane_slots) << "%\n";
    std::cout << "HIGH13 groups=" << (1u << H)
              << " max_states=" << high_max_states
              << " max_fblock_runs=" << high_max_runs
              << " total_fblock_runs=" << high_total_runs
              << " best_owner_weighted="
              << 100.0 * double(weighted_best_local / (long double)high_sum_states) << "%\n";

    U64 begin_bytes = U64(1u << L) * S * 4 + U64(1u << H) * S * 4;
    U64 old_prefix_bytes = 2 * hs * 8;
    std::int64_t net_metadata_bytes =
        std::int64_t(begin_bytes) - std::int64_t(old_prefix_bytes);
    std::cout << "mask_begin_mib=" << double(begin_bytes) / (1u << 20)
              << " removable_canonical_prefix_mib="
              << double(old_prefix_bytes) / (1u << 20)
              << " net_metadata_delta_mib="
              << double(net_metadata_bytes) / (1u << 20) << "\n";

    if (main_states != 385719506620ULL ||
        low_sum_states != main_states || high_sum_states != main_states) {
        std::cerr << "state partition mismatch\n";
        return 1;
    }
    return 0;
}
