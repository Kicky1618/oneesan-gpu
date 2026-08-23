#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;

static std::vector<std::vector<std::vector<std::uint32_t>>> build_low_groups(int low) {
    const std::uint32_t nm = 1u << low;
    std::vector<std::vector<std::vector<std::uint32_t>>> out(
        low + 1, std::vector<std::vector<std::uint32_t>>(nm));
    for (int h0 = 0; h0 <= low; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code,
                       std::uint32_t mask) -> void {
            if (pos < 0) {
                if (h == 0) out[h0][mask].push_back(code);
                return;
            }
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
        if (pos < 0) {
            out[h].push_back(code);
            return;
        }
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
    if (W < 4 || W > 13 || low < 1 || high < 1 || low >= 16 || high >= 16
        || threshold < 1) return 1;

    const auto low_groups = build_low_groups(low);
    const auto high_rows = build_high_rows(high);
    std::vector<std::uint32_t> representative(low + 1, 0xffffffffu);
    for (int hs = 0; hs <= low; ++hs) {
        for (const auto& g : low_groups[hs]) if (!g.empty()) {
            representative[hs] = g.front();
            break;
        }
    }

    U64 candidates = 0, tasks = 0, lane_slots = 0;
    for (int p = W - 1; p >= low + 1; --p) {
        struct SelectedBlock {
            int cv = 0, hs = 0;
            std::vector<std::uint32_t> high_codes;
        };
        std::vector<SelectedBlock> blocks;
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                if (hs < 0 || hs > low || representative[hs] == 0xffffffffu) continue;
                SelectedBlock b;
                b.cv = cv;
                b.hs = hs;
                for (std::uint32_t hc : high_rows[he]) {
                    const MateID m = MateID(representative[hs])
                        | (MateID(cv) << (2 * low))
                        | (MateID(hc) << (2 * (low + 1)));
                    const MateValuePair w = mpair(m, p);
                    if (w != LL && w != RR && w != RL) continue;
                    const IncludeResult z = include_horizontal(m, W, p);
                    if (z.valid && z.blocked) b.high_codes.push_back(hc);
                }
                if (!b.high_codes.empty()) blocks.push_back(std::move(b));
            }
        }

        for (std::uint32_t mask = 0; mask < (1u << low); ++mask) {
            std::unordered_set<MateID> old_map, hybrid_map;
            for (const SelectedBlock& b : blocks) {
                const auto& cols = low_groups[b.hs][mask];
                if (cols.empty()) continue;

                for (std::uint32_t hc : b.high_codes) {
                    for (std::uint32_t lc : cols) {
                        const MateID m = MateID(lc)
                            | (MateID(b.cv) << (2 * low))
                            | (MateID(hc) << (2 * (low + 1)));
                        if (!old_map.insert(m).second) return 2;
                    }
                }

                if (cols.size() < size_t(threshold)) {
                    const U64 items = U64(b.high_codes.size()) * cols.size();
                    const U64 block_tasks = (items + 31) / 32;
                    tasks += block_tasks;
                    lane_slots += block_tasks * 32;
                    for (U64 task = 0; task < block_tasks; ++task) {
                        for (U64 lane = 0; lane < 32; ++lane) {
                            const U64 item = task * 32 + lane;
                            if (item >= items) continue;
                            const U64 row = item / cols.size();
                            const U64 lr = item - row * cols.size();
                            const MateID m = MateID(cols[size_t(lr)])
                                | (MateID(b.cv) << (2 * low))
                                | (MateID(b.high_codes[size_t(row)]) << (2 * (low + 1)));
                            if (!hybrid_map.insert(m).second) return 3;
                        }
                    }
                } else {
                    tasks += b.high_codes.size();
                    lane_slots += U64(b.high_codes.size()) * ((cols.size() + 31) / 32) * 32;
                    for (std::uint32_t hc : b.high_codes) {
                        for (U64 lane = 0; lane < 32; ++lane) {
                            for (U64 lr = lane; lr < cols.size(); lr += 32) {
                                const MateID m = MateID(cols[size_t(lr)])
                                    | (MateID(b.cv) << (2 * low))
                                    | (MateID(hc) << (2 * (low + 1)));
                                if (!hybrid_map.insert(m).second) return 4;
                            }
                        }
                    }
                }
            }
            if (old_map != hybrid_map) {
                std::cerr << "hybrid HIGH closure task-map mismatch W=" << W
                          << " p=" << p << " mask=" << mask
                          << " old=" << old_map.size()
                          << " hybrid=" << hybrid_map.size() << '\n';
                return 5;
            }
            candidates += old_map.size();
        }
    }

    std::cout << "factor-highclosure-rowpack-hybrid-taskmap OK W=" << W
              << " low=" << low << " high=" << high
              << " threshold=" << threshold
              << " candidates=" << candidates
              << " tasks=" << tasks
              << " lane_slots=" << lane_slots << '\n';
    return 0;
}
