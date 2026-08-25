#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

static long double as_ld(U128 x) {
    const U64 lo = U64(x), hi = U64(x >> 64);
    return (long double)hi * 18446744073709551616.0L + (long double)lo;
}

static std::vector<std::vector<U64>> high_mask_rows(int high) {
    const std::uint32_t nm = 1u << high;
    std::vector<std::vector<U64>> out(nm, std::vector<U64>(high + 2));
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t mask) -> void {
        if (pos < 0) {
            if (h >= 0 && h < int(out[mask].size())) ++out[mask][h];
            return;
        }
        self(self, pos - 1, h, mask); // N
        if (h) self(self, pos - 1, h - 1, mask | (1u << pos)); // R
        self(self, pos - 1, h + 1, mask | (1u << pos)); // L
    };
    rec(rec, high - 1, 1, 0);
    return out;
}

static std::vector<std::vector<std::uint32_t>> low_codes_by_start(int low) {
    std::vector<std::vector<std::uint32_t>> out(low + 1);
    for (int h0 = 0; h0 <= low; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) out[h0].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h) self(self, pos - 1, h - 1,
                        code | (1u << (2 * pos)));
            self(self, pos - 1, h + 1,
                 code | (2u << (2 * pos)));
        };
        rec(rec, low - 1, h0, 0);
    }
    return out;
}

static bool closure_pair(std::uint32_t lc, int cv, int low, int p) {
    const std::uint32_t active = lc | (std::uint32_t(cv) << (2 * low));
    const std::uint32_t w = (active >> (2 * (p - 1))) & 15u;
    return w == 0xau || w == 0x5u || w == 0x6u;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || low >= 16 || high < 1 || high >= 16
        || threads < 32 || threads > 1024 || (threads & 31)) return 1;

    const int warps_per_block = threads / 32;
    const auto rows = high_mask_rows(high);
    const auto low_codes = low_codes_by_start(low);
    const std::uint32_t nm = 1u << high;
    const int nblocks = 3 * (high + 2);

    U128 old_state_threads = 0;
    U128 selected_states = 0;
    U128 compact_lane_slots = 0;
    U128 old_launch_blocks = 0;
    U128 compact_launch_blocks = 0;
    U64 capped_old = 0, capped_compact = 0;

    for (int p = low; p >= 1; --p) {
        std::vector<U64> selected(nblocks);
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int bid = 3 * he + cv;
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low) continue;
                U64 z = 0;
                for (std::uint32_t lc : low_codes[hs])
                    if (closure_pair(lc, cv, low, p)) ++z;
                selected[bid] = z;
            }
        }

        for (std::uint32_t mask = 0; mask < nm; ++mask) {
            U64 main_n = 0;
            U64 tasks = 0;
            U64 useful = 0;
            for (int he = 0; he <= high + 1; ++he) {
                const U64 r = rows[mask][he];
                if (!r) continue;
                for (int cv = 0; cv < 3; ++cv) {
                    const int bid = 3 * he + cv;
                    const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                    if (hs < 0 || hs > low) continue;
                    const U64 cols = low_codes[hs].size();
                    const U64 sel = selected[bid];
                    main_n += r * cols;
                    useful += r * sel;
                    tasks += r * ((sel + 31) / 32);
                }
            }
            if (!main_n) continue;
            old_state_threads += main_n;
            selected_states += useful;
            compact_lane_slots += U128(tasks) * 32;

            const U64 old_raw = (main_n + U64(threads) - 1) / U64(threads);
            const U64 compact_raw = tasks
                ? (tasks + U64(warps_per_block) - 1) / U64(warps_per_block) : 0;
            if (old_raw > 65535) ++capped_old;
            if (compact_raw > 65535) ++capped_compact;
            old_launch_blocks += std::min<U64>(65535, old_raw);
            compact_launch_blocks += std::min<U64>(65535, compact_raw);
        }
    }

    if (W == 28 && low == 14 && threads == 256) {
        if (old_state_threads != U128(5400073092680ULL)
            || selected_states != U128(1619638941284ULL)
            || compact_lane_slots != U128(1620040986016ULL)
            || old_launch_blocks != U128(6328761404ULL)
            || compact_launch_blocks != U128(3964060594ULL)
            || capped_old != 81368ULL || capped_compact != 33320ULL) {
            std::cerr << "n=27 LOW closure launch-model regression\n";
            return 2;
        }
    }

    const U128 residue_old_blocks = old_launch_blocks * U128(W);
    const U128 residue_compact_blocks = compact_launch_blocks * U128(W);
    std::cout << std::fixed << std::setprecision(6)
              << "lowclosure-launch W=" << W << " low=" << low
              << " high=" << high << " threads=" << threads << '\n'
              << "old_state_threads=" << double(as_ld(old_state_threads))
              << " selected_states=" << double(as_ld(selected_states))
              << " selected_ratio="
              << double(as_ld(selected_states) / as_ld(old_state_threads)) << '\n'
              << "compact_lane_slots=" << double(as_ld(compact_lane_slots))
              << " compact_lane_ratio="
              << double(as_ld(compact_lane_slots) / as_ld(old_state_threads))
              << " useful_lane_fraction="
              << double(as_ld(selected_states) / as_ld(compact_lane_slots)) << '\n'
              << "old_launch_blocks=" << double(as_ld(old_launch_blocks))
              << " compact_launch_blocks=" << double(as_ld(compact_launch_blocks))
              << " launch_block_ratio="
              << double(as_ld(compact_launch_blocks) / as_ld(old_launch_blocks)) << '\n'
              << "old_capped_group_positions=" << capped_old
              << " compact_capped_group_positions=" << capped_compact << '\n'
              << "per_residue_old_blocks=" << double(as_ld(residue_old_blocks))
              << " per_residue_compact_blocks=" << double(as_ld(residue_compact_blocks))
              << '\n';
    return 0;
}
