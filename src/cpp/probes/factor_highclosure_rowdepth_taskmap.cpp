#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;

static int peak_code(std::uint32_t code, int len, int start) {
    int h = start, peak = h;
    for (int p = len - 1; p >= 0; --p) {
        const auto v = MateValue((code >> (2 * p)) & 3u);
        if (v == R) --h;
        else if (v == L) { ++h; peak = std::max(peak, h); }
    }
    return peak;
}

static std::vector<std::vector<std::vector<std::uint32_t>>> build_low_groups(int low) {
    const std::uint32_t nm = 1u << low;
    std::vector<std::vector<std::vector<std::uint32_t>>> out(
        low + 1, std::vector<std::vector<std::uint32_t>>(nm));
    for (int h0 = 0; h0 <= low; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code,
                       std::uint32_t mask) -> void {
            if (pos < 0) { if (h == 0) out[h0][mask].push_back(code); return; }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code, mask);
            if (h) self(self, pos - 1, h - 1,
                        code | (1u << (2 * pos)), mask | (1u << pos));
            self(self, pos - 1, h + 1,
                 code | (2u << (2 * pos)), mask | (1u << pos));
        };
        rec(rec, low - 1, h0, 0, 0);
    }
    return out;
}

static std::vector<std::vector<std::uint32_t>> build_high_rows(int high) {
    std::vector<std::vector<std::uint32_t>> out(high + 2);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) { out[h].push_back(code); return; }
        self(self, pos - 1, h, code);
        if (h) self(self, pos - 1, h - 1, code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1, code | (2u << (2 * pos)));
    };
    rec(rec, high - 1, 1, 0);
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : W / 2;
    const int threshold = argc > 3 ? std::atoi(argv[3]) : 16;
    const int high = W - 1 - low;
    const int full_cap = (W + 1) / 2;
    if (W < 4 || W > 13 || low < 1 || high < 1 || low >= 16 || high >= 16
        || threshold < 1) return 1;

    const auto low_groups = build_low_groups(low);
    const auto high_rows = build_high_rows(high);
    std::vector<std::uint32_t> representative(low + 1, 0xffffffffu);
    for (int hs = 0; hs <= low; ++hs)
        for (const auto& g : low_groups[hs]) if (!g.empty()) {
            representative[hs] = g.front();
            break;
        }

    U64 expected_total = 0, compact_total = 0, warp_tasks = 0, lane_slots = 0;
    for (int p = W - 1; p >= low + 1; --p) {
        struct Block { int cv = 0, hs = 0; std::vector<std::uint32_t> high; };
        std::vector<Block> blocks;
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                if (hs < 0 || hs > low || representative[hs] == 0xffffffffu) continue;
                Block b;
                b.cv = cv;
                b.hs = hs;
                for (std::uint32_t hc : high_rows[he]) {
                    const MateID m = MateID(representative[hs])
                        | (MateID(cv) << (2 * low))
                        | (MateID(hc) << (2 * (low + 1)));
                    const auto w = mpair(m, p);
                    if (w != LL && w != RR && w != RL) continue;
                    const auto z = include_horizontal(m, W, p);
                    if (z.valid && z.blocked) b.high.push_back(hc);
                }
                if (!b.high.empty()) blocks.push_back(std::move(b));
            }
        }

        for (std::uint32_t mask = 0; mask < (1u << low); ++mask) {
            for (int cap = 1; cap <= full_cap; ++cap) {
                std::unordered_set<MateID> expected, compact;
                for (const Block& b : blocks) {
                    const auto& dense_cols = low_groups[b.hs][mask];
                    if (dense_cols.empty()) continue;
                    std::vector<std::uint32_t> rows, cols;
                    for (auto hc : b.high)
                        if (peak_code(hc, high, 1) <= cap) rows.push_back(hc);
                    for (auto lc : dense_cols)
                        if (peak_code(lc, low, b.hs) <= cap) cols.push_back(lc);
                    std::stable_sort(rows.begin(), rows.end(), [&](auto x, auto y) {
                        const int px = peak_code(x, high, 1);
                        const int py = peak_code(y, high, 1);
                        return px != py ? px < py : x < y;
                    });
                    std::stable_sort(cols.begin(), cols.end(), [&](auto x, auto y) {
                        const int px = peak_code(x, low, b.hs);
                        const int py = peak_code(y, low, b.hs);
                        return px != py ? px < py : x < y;
                    });

                    for (auto hc : b.high) if (peak_code(hc, high, 1) <= cap)
                        for (auto lc : dense_cols) if (peak_code(lc, low, b.hs) <= cap) {
                            const MateID m = MateID(lc)
                                | (MateID(b.cv) << (2 * low))
                                | (MateID(hc) << (2 * (low + 1)));
                            if (!expected.insert(m).second) return 2;
                        }

                    // Preserve v0.11 attribution: packing is decided from the
                    // dense physical LOW width, not the row-dependent active count.
                    const bool pack = dense_cols.size() < std::size_t(threshold);
                    if (pack) {
                        const U64 items = U64(rows.size()) * cols.size();
                        const U64 tasks = (items + 31) / 32;
                        warp_tasks += tasks;
                        lane_slots += tasks * 32;
                        for (U64 t = 0; t < tasks; ++t)
                            for (U64 lane = 0; lane < 32; ++lane) {
                                const U64 item = t * 32 + lane;
                                if (item >= items) continue;
                                const U64 rr = item / cols.size();
                                const U64 cc = item - rr * cols.size();
                                const MateID m = MateID(cols[cc])
                                    | (MateID(b.cv) << (2 * low))
                                    | (MateID(rows[rr]) << (2 * (low + 1)));
                                if (!compact.insert(m).second) return 3;
                            }
                    } else {
                        warp_tasks += rows.size();
                        lane_slots += U64(rows.size()) * ((cols.size() + 31) / 32) * 32;
                        for (auto hc : rows)
                            for (U64 lane = 0; lane < 32; ++lane)
                                for (U64 q = lane; q < cols.size(); q += 32) {
                                    const MateID m = MateID(cols[q])
                                        | (MateID(b.cv) << (2 * low))
                                        | (MateID(hc) << (2 * (low + 1)));
                                    if (!compact.insert(m).second) return 4;
                                }
                    }
                }
                if (expected != compact) {
                    std::cerr << "HIGH closure row-depth compact task-map mismatch W="
                              << W << " p=" << p << " mask=" << mask
                              << " cap=" << cap << " expected=" << expected.size()
                              << " compact=" << compact.size() << '\n';
                    return 5;
                }
                expected_total += expected.size();
                compact_total += compact.size();
            }
        }
    }

    if (threshold == 16 && W == 10 && low == 5) {
        if (expected_total != 12719 || compact_total != 12719
            || warp_tasks != 2306 || lane_slots != 73792) return 6;
    }
    if (threshold == 16 && W == 12 && low == 6) {
        if (expected_total != 146624 || compact_total != 146624
            || warp_tasks != 9311 || lane_slots != 297952) return 7;
    }

    std::cout << "factor-highclosure-rowdepth-taskmap OK W=" << W
              << " low=" << low << " high=" << high
              << " threshold=" << threshold
              << " expected=" << expected_total
              << " compact=" << compact_total
              << " warp_tasks=" << warp_tasks
              << " lane_slots=" << lane_slots << '\n';
    return 0;
}
