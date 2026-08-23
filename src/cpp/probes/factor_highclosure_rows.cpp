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
        || threads < 1 || threads > 1024) return 1;
    const U64 warps_per_block = U64((threads + 31) / 32);

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

    U128 full_state_threads = 0;
    U128 closure_states = 0;
    U128 closure_rows = 0;
    U128 candidate_rows = 0;
    U128 warp_slots = 0;
    U128 old_launch_blocks = 0;
    U128 compact_launch_blocks = 0;
    U64 global_rows_per_p = 0;
    bool first = true;

    for (int q = 1; q <= high; ++q) {
        U128 q_states = 0, q_rows = 0, q_candidates = 0, q_warps = 0, q_full = 0;
        U64 q_global_rows = 0;
        std::vector<U128> group_main(low + 1), group_rows(low + 1);
        for (int he = 0; he < int(hc.size()); ++he) {
            if (hc[he].empty()) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                U64 cr = 0;
                for (std::uint32_t code : hc[he]) {
                    const int w = pair_at(code, cv, q);
                    if (w == 0xa || w == 0x5 || w == 0x6) ++cr; // LL/RR/RL
                }
                q_global_rows += cr;
                for (int k = 0; k <= low; ++k) {
                    const U64 cols = lc[k][hs];
                    if (!cols) continue;
                    const U64 groups = choose_u64(low, k);
                    const U128 block_main = U128(hc[he].size()) * cols;
                    group_main[k] += block_main;
                    group_rows[k] += cr;
                    q_full += U128(groups) * block_main;
                    q_candidates += U128(groups) * hc[he].size();
                    q_rows += U128(groups) * cr;
                    q_states += U128(groups) * cr * cols;
                    q_warps += U128(groups) * cr * ((cols + 31) / 32);
                }
            }
        }
        if (first) { global_rows_per_p = q_global_rows; first = false; }
        else if (q_global_rows != global_rows_per_p) {
            std::cerr << "closure-row count depends on q: " << q_global_rows
                      << " vs " << global_rows_per_p << '\n';
            return 2;
        }
        for (int k = 0; k <= low; ++k) {
            const U64 groups = choose_u64(low, k);
            old_launch_blocks += U128(groups) * capped_blocks(group_main[k], U64(threads));
            compact_launch_blocks += U128(groups)
                * capped_blocks(group_rows[k], warps_per_block);
        }
        full_state_threads += q_full;
        closure_states += q_states;
        closure_rows += q_rows;
        candidate_rows += q_candidates;
        warp_slots += 32 * q_warps;
    }

    const U128 residue_full = full_state_threads * U128(W);
    const U128 residue_closure = closure_states * U128(W);
    const U128 residue_warp_slots = warp_slots * U128(W);
    const U128 residue_old_blocks = old_launch_blocks * U128(W);
    const U128 residue_compact_blocks = compact_launch_blocks * U128(W);

    std::cout << std::fixed << std::setprecision(6)
              << "highclosure-rows W=" << W << " low=" << low << " high=" << high
              << " threads=" << threads << '\n'
              << "global_closure_rows_per_p=" << global_rows_per_p
              << " compact_row_table_mib="
              << double((long double)global_rows_per_p * high * 4 / (1ULL << 20)) << '\n'
              << "full_state_threads=" << u128_string(full_state_threads) << '\n'
              << "exact_closure_states=" << u128_string(closure_states)
              << " exact_state_ratio="
              << double(as_ld(closure_states) / as_ld(full_state_threads)) << '\n'
              << "candidate_rows=" << u128_string(candidate_rows)
              << " closure_rows=" << u128_string(closure_rows) << '\n'
              << "warp_lane_slots=" << u128_string(warp_slots)
              << " warp_lane_ratio=" << double(as_ld(warp_slots) / as_ld(full_state_threads))
              << " useful_lane_fraction=" << double(as_ld(closure_states) / as_ld(warp_slots))
              << '\n'
              << "old_launch_blocks=" << u128_string(old_launch_blocks)
              << " compact_launch_blocks=" << u128_string(compact_launch_blocks)
              << " launch_block_ratio="
              << double(as_ld(compact_launch_blocks) / as_ld(old_launch_blocks)) << '\n'
              << "per_residue_full_state_threads=" << u128_string(residue_full)
              << " per_residue_closure_states=" << u128_string(residue_closure)
              << " per_residue_warp_lane_slots=" << u128_string(residue_warp_slots) << '\n'
              << "per_residue_old_launch_blocks=" << u128_string(residue_old_blocks)
              << " per_residue_compact_launch_blocks=" << u128_string(residue_compact_blocks)
              << '\n';
    return 0;
}
