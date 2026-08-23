#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

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

static std::vector<U64> high_row_counts(int high) {
    std::vector<U64> cur(high + 3), nxt(high + 3);
    cur[1] = 1;
    for (int pos = high - 1; pos >= 0; --pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= high + 1; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h) nxt[h - 1] += cur[h];
            nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur;
}

static std::vector<std::vector<std::uint32_t>> build_low_storage(int low) {
    const std::uint32_t nm = 1u << low;
    std::vector<std::vector<std::uint32_t>> out(low + 1);
    std::vector<std::vector<std::uint32_t>> groups(nm);
    for (int h0 = 0; h0 <= low; ++h0) {
        for (auto& v : groups) v.clear();
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code,
                       std::uint32_t occ) -> void {
            if (pos < 0) {
                if (h == 0) groups[occ].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code, occ); // N
            if (h) self(self, pos - 1, h - 1,
                        code | (1u << (2 * pos)), occ | (1u << pos)); // R
            self(self, pos - 1, h + 1,
                 code | (2u << (2 * pos)), occ | (1u << pos)); // L
        };
        rec(rec, low - 1, h0, 0, 0);
        auto& dst = out[h0];
        for (std::uint32_t mask = 0; mask < nm; ++mask)
            dst.insert(dst.end(), groups[mask].begin(), groups[mask].end());
    }
    return out;
}

static bool is_closure_pair(std::uint32_t code, int center, int low, int p) {
    const std::uint32_t active = code | (std::uint32_t(center) << (2 * low));
    const std::uint32_t w = (active >> (2 * (p - 1))) & 15u;
    return w == 0xau || w == 0x5u || w == 0x6u; // LL/RR/RL
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || low >= 16 || high < 1 || high >= 16)
        return 1;

    const auto high_rows = high_row_counts(high);
    const auto low_codes = build_low_storage(low);

    U128 total_full_words = 0, total_selected_words = 0;
    U128 total_full_warp_slots = 0, total_compact_warp_slots = 0;
    U128 full_sector8 = 0, sparse_sector8 = 0;
    U128 full_sector32 = 0, sparse_sector32 = 0;
    U64 global_cols_per_p = 0;
    bool first = true;

    for (int p = low; p >= 1; --p) {
        U128 p_selected = 0;
        U64 p_global_cols = 0;
        for (int he = 0; he < int(high_rows.size()); ++he) {
            const U64 rows = high_rows[he];
            if (!rows) continue;
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                const auto& cols = low_codes[hs];
                if (cols.empty()) continue;

                U64 selected = 0;
                U64 sec8 = 0, sec32 = 0;
                bool prev8 = false, prev32 = false;
                U64 last8 = ~U64(0), last32 = ~U64(0);
                for (U64 lr = 0; lr < cols.size(); ++lr) {
                    if (!is_closure_pair(cols[size_t(lr)], cv, low, p)) continue;
                    ++selected;
                    const U64 s8 = lr / 8;
                    const U64 s32 = lr / 32;
                    if (!prev8 || s8 != last8) { ++sec8; last8 = s8; prev8 = true; }
                    if (!prev32 || s32 != last32) { ++sec32; last32 = s32; prev32 = true; }
                }

                p_global_cols += selected;
                p_selected += U128(rows) * selected;
                total_full_words += U128(rows) * cols.size();
                total_selected_words += U128(rows) * selected;
                total_full_warp_slots += U128(rows) * ((cols.size() + 31) / 32) * 32;
                total_compact_warp_slots += U128(rows) * ((selected + 31) / 32) * 32;
                full_sector8 += U128(rows) * ((cols.size() + 7) / 8);
                sparse_sector8 += U128(rows) * sec8;
                full_sector32 += U128(rows) * ((cols.size() + 31) / 32);
                sparse_sector32 += U128(rows) * sec32;
            }
        }
        if (first) { global_cols_per_p = p_global_cols; first = false; }
        else if (p_global_cols != global_cols_per_p) {
            std::cerr << "LOW closure column count depends on p: " << p_global_cols
                      << " vs " << global_cols_per_p << '\n';
            return 2;
        }
        std::cout << "p=" << p << " closure_cols=" << p_global_cols
                  << " closure_states=" << u128_string(p_selected) << '\n';
    }

    std::cout << std::fixed << std::setprecision(6)
              << "lowclosure-columns W=" << W << " low=" << low << " high=" << high << '\n'
              << "global_closure_cols_per_p=" << global_cols_per_p
              << " compact_col_table_mib="
              << double((long double)global_cols_per_p * low * 4 / (1ULL << 20)) << '\n'
              << "full_words=" << u128_string(total_full_words)
              << " selected_words=" << u128_string(total_selected_words)
              << " selected_ratio=" << double(as_ld(total_selected_words) / as_ld(total_full_words))
              << '\n'
              << "full_warp_lane_slots=" << u128_string(total_full_warp_slots)
              << " compact_warp_lane_slots=" << u128_string(total_compact_warp_slots)
              << " compact_lane_ratio="
              << double(as_ld(total_compact_warp_slots) / as_ld(total_full_warp_slots))
              << " useful_lane_fraction="
              << double(as_ld(total_selected_words) / as_ld(total_compact_warp_slots)) << '\n'
              << "full_32B_sectors=" << u128_string(full_sector8)
              << " sparse_32B_sectors=" << u128_string(sparse_sector8)
              << " sparse_32B_ratio=" << double(as_ld(sparse_sector8) / as_ld(full_sector8))
              << '\n'
              << "full_128B_lines=" << u128_string(full_sector32)
              << " sparse_128B_lines=" << u128_string(sparse_sector32)
              << " sparse_128B_ratio=" << double(as_ld(sparse_sector32) / as_ld(full_sector32))
              << '\n';
    return 0;
}
