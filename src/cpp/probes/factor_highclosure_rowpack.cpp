#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

static long double as_ld(U128 x) {
    const U64 lo = U64(x), hi = U64(x >> 64);
    return (long double)hi * 18446744073709551616.0L + (long double)lo;
}

static std::string u128_string(U128 x) {
    if (!x) return "0";
    std::string s;
    while (x) {
        s.push_back(char('0' + x % 10));
        x /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

static U64 low_fixed_count(int occupied, int start) {
    std::vector<U64> cur(64), nxt(64);
    cur[start] = 1;
    for (int s = 0; s < occupied; ++s) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h < 63; ++h) if (cur[h]) {
            if (h) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

static int pair_at(std::uint32_t high_code, int center, int q) {
    const std::uint32_t active = (high_code << 2) | std::uint32_t(center);
    return int((active >> (2 * (q - 1))) & 15u);
}

static U64 capped_blocks(U128 items, U64 units_per_block) {
    if (!items) return 0;
    const U128 blocks = (items + units_per_block - 1) / units_per_block;
    return U64(blocks > U128(65535) ? U128(65535) : blocks);
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || high < 1 || high >= 16
        || threads < 32 || threads > 1024 || (threads & 31)) return 1;
    const U64 warps_per_block = U64(threads / 32);

    std::vector<std::vector<std::uint32_t>> hc(high + 3);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) { hc[h].push_back(code); return; }
        self(self, pos - 1, h, code);
        if (h) self(self, pos - 1, h - 1, code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1, code | (2u << (2 * pos)));
    };
    rec(rec, high - 1, 1, 0);

    std::vector<std::vector<U64>> lc(low + 1, std::vector<U64>(low + 2));
    for (int k = 0; k <= low; ++k)
        for (int h = 0; h <= low; ++h)
            lc[k][h] = low_fixed_count(k, h);

    U128 useful_states = 0;
    U128 row_warp_slots = 0;
    U128 packed_warp_slots = 0;
    U128 row_launch_blocks = 0;
    U128 packed_launch_blocks = 0;
    U64 row_capped_group_positions = 0;
    U64 packed_capped_group_positions = 0;
    U64 global_rows_per_p = 0;
    bool first = true;

    for (int q = 1; q <= high; ++q) {
        std::vector<std::vector<U64>> closure_rows(
            high + 2, std::vector<U64>(3));
        U64 rows_this_p = 0;
        for (int he = 0; he < int(hc.size()); ++he) {
            if (hc[he].empty()) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                U64 rows = 0;
                for (std::uint32_t code : hc[he]) {
                    const int w = pair_at(code, cv, q);
                    if (w == 0xa || w == 0x5 || w == 0x6) ++rows;
                }
                closure_rows[he][cv] = rows;
                rows_this_p += rows;
            }
        }
        if (first) {
            global_rows_per_p = rows_this_p;
            first = false;
        } else if (rows_this_p != global_rows_per_p) {
            std::cerr << "HIGH closure row count depends on position\n";
            return 2;
        }

        for (int k = 0; k <= low; ++k) {
            const U64 groups = choose_u64(low, k);
            U128 row_tasks_one_group = 0;
            U128 packed_tasks_one_group = 0;
            for (int he = 0; he <= high + 1; ++he) {
                for (int cv = 0; cv < 3; ++cv) {
                    const U64 rows = closure_rows[he][cv];
                    if (!rows) continue;
                    const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                    if (hs < 0 || hs > low) continue;
                    const U64 cols = lc[k][hs];
                    if (!cols) continue;

                    useful_states += U128(groups) * rows * cols;
                    row_warp_slots += U128(groups) * rows
                        * U128((cols + 31) / 32) * 32;

                    // Proposed v0.10 mapping: flatten all selected closure rows
                    // within one FBlock into contiguous row-major state items.
                    // One warp owns 32 consecutive items. It may span row
                    // boundaries when the fixed LOW-mask width is small.
                    const U128 items = U128(rows) * cols;
                    const U128 tasks = (items + 31) / 32;
                    packed_warp_slots += U128(groups) * tasks * 32;
                    packed_tasks_one_group += tasks;
                    row_tasks_one_group += rows;
                }
            }

            const U64 row_blocks = capped_blocks(row_tasks_one_group, warps_per_block);
            const U64 packed_blocks = capped_blocks(
                packed_tasks_one_group, warps_per_block);
            if (row_tasks_one_group
                && (row_tasks_one_group + warps_per_block - 1) / warps_per_block > 65535)
                row_capped_group_positions += groups;
            if (packed_tasks_one_group
                && (packed_tasks_one_group + warps_per_block - 1) / warps_per_block > 65535)
                packed_capped_group_positions += groups;
            row_launch_blocks += U128(groups) * row_blocks;
            packed_launch_blocks += U128(groups) * packed_blocks;
        }
    }

    if (W == 28 && low == 14 && threads == 256) {
        if (global_rows_per_p != 715533ULL
            || useful_states != U128(1503950445478ULL)
            || row_warp_slots != U128(3021117696896ULL)
            || packed_warp_slots != U128(1503993904960ULL)
            || row_launch_blocks != U128(8923348057ULL)
            || packed_launch_blocks != U128(4616872663ULL)
            || row_capped_group_positions != 0
            || packed_capped_group_positions != 19123ULL) {
            std::cerr << "n=27 HIGH closure row-pack regression mismatch\n";
            return 3;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "highclosure-rowpack W=" << W << " low=" << low
              << " high=" << high << " threads=" << threads << '\n'
              << "closure_rows_per_p=" << global_rows_per_p << '\n'
              << "useful_states=" << u128_string(useful_states) << '\n'
              << "row_warp_lane_slots=" << u128_string(row_warp_slots)
              << " useful_lane_fraction="
              << double(as_ld(useful_states) / as_ld(row_warp_slots)) << '\n'
              << "packed_warp_lane_slots=" << u128_string(packed_warp_slots)
              << " packed_vs_row_lane_ratio="
              << double(as_ld(packed_warp_slots) / as_ld(row_warp_slots))
              << " packed_useful_lane_fraction="
              << double(as_ld(useful_states) / as_ld(packed_warp_slots)) << '\n'
              << "row_launch_blocks=" << u128_string(row_launch_blocks)
              << " packed_launch_blocks=" << u128_string(packed_launch_blocks)
              << " packed_launch_ratio="
              << double(as_ld(packed_launch_blocks) / as_ld(row_launch_blocks)) << '\n'
              << "row_capped_group_positions=" << row_capped_group_positions
              << " packed_capped_group_positions=" << packed_capped_group_positions << '\n'
              << "per_residue_row_lane_slots="
              << u128_string(row_warp_slots * U128(W))
              << " per_residue_packed_lane_slots="
              << u128_string(packed_warp_slots * U128(W)) << '\n'
              << "per_residue_row_launch_blocks="
              << u128_string(row_launch_blocks * U128(W))
              << " per_residue_packed_launch_blocks="
              << u128_string(packed_launch_blocks * U128(W)) << '\n';
    return 0;
}
