#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;

static std::vector<MateID> enumerate_states(int width) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        if (pos < 0) {
            if (h == 0) out.push_back(m);
            return;
        }
        if (h < 0 || h > pos + 1) return;
        self(self, pos - 1, h, m);
        if (h > 0) self(self, pos - 1, h - 1,
                        m | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             m | (MateID(L) << (2 * pos)));
    };
    rec(rec, width - 1, 1, 0);
    return out;
}

static int seg_end_height(std::uint32_t code, int len) {
    int h = 1;
    for (int p = len - 1; p >= 0; --p) {
        const auto v = MateValue((code >> (2 * p)) & 3u);
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return h;
}

static std::uint32_t occ(std::uint32_t code, int len) {
    std::uint32_t z = 0;
    for (int p = 0; p < len; ++p)
        if (((code >> (2 * p)) & 3u) != N) z |= 1u << p;
    return z;
}

static bool closure_pair(MateValuePair w) {
    return w == LL || w == RR || w == RL;
}

struct LowStorage {
    std::vector<std::vector<std::uint32_t>> codes;
    std::vector<std::map<std::uint32_t, std::uint32_t>> rank;
};

static LowStorage build_low_storage(int low) {
    LowStorage s;
    s.codes.resize(low + 1);
    s.rank.resize(low + 1);
    const std::uint32_t nm = 1u << low;
    std::vector<std::vector<std::uint32_t>> groups(nm);
    for (int h0 = 0; h0 <= low; ++h0) {
        for (auto& v : groups) v.clear();
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code,
                       std::uint32_t mask) -> void {
            if (pos < 0) {
                if (h == 0) groups[mask].push_back(code);
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
        auto& dst = s.codes[h0];
        for (std::uint32_t mask = 0; mask < nm; ++mask)
            dst.insert(dst.end(), groups[mask].begin(), groups[mask].end());
        for (std::uint32_t r = 0; r < dst.size(); ++r)
            s.rank[h0][dst[r]] = r;
    }
    return s;
}

struct HighRows {
    // [mask][height] in storage-mask rank order.
    std::vector<std::vector<std::vector<std::uint32_t>>> codes;
};

static HighRows build_high_rows(int high) {
    const std::uint32_t nm = 1u << high;
    HighRows out;
    out.codes.resize(nm, std::vector<std::vector<std::uint32_t>>(high + 2));
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            out.codes[occ(code, high)][h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h) self(self, pos - 1, h - 1,
                    code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1,
             code | (2u << (2 * pos)));
    };
    rec(rec, high - 1, 1, 0);
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : W / 2;
    const int high = W - 1 - low;
    if (W < 4 || W > 13 || low < 1 || high < 1 || low >= 16 || high >= 16)
        return 1;

    constexpr U64 LOW_MASK64 = ~U64(0);
    const U64 low_code_mask = (U64(1) << (2 * low)) - 1;
    const U64 high_code_mask = (U64(1) << (2 * high)) - 1;
    (void)LOW_MASK64;

    const auto states = enumerate_states(W);
    const auto low_storage = build_low_storage(low);
    const auto high_rows = build_high_rows(high);

    U64 task_count = 0, lane_slots = 0, sources = 0;
    for (int p = low; p >= 1; --p) {
        // Production compact list is [FBlock][LOW all-rank], independent of
        // the fixed HIGH occupancy mask.
        std::vector<std::vector<std::uint32_t>> closure_lr(3 * (high + 2));
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                const int bid = 3 * he + cv;
                if (hs < 0 || hs > low || bid >= int(closure_lr.size())) continue;
                const auto& lc = low_storage.codes[hs];
                for (std::uint32_t lr = 0; lr < lc.size(); ++lr) {
                    const U64 active = U64(lc[lr]) | (U64(cv) << (2 * low));
                    const auto w = MateValuePair((active >> (2 * (p - 1))) & 15u);
                    if (!closure_pair(w)) continue;
                    // Pick any compatible HIGH row; validity of LOW-window
                    // normal/cross closure is checked below for actual states.
                    closure_lr[bid].push_back(lr);
                }
            }
        }

        for (std::uint32_t mask = 0; mask < (1u << high); ++mask) {
            std::unordered_set<MateID> expected, got;
            expected.reserve(states.size() / 4 + 1);
            got.reserve(states.size() / 4 + 1);

            for (MateID m : states) {
                const std::uint32_t hc = std::uint32_t(
                    (m >> (2 * (low + 1))) & high_code_mask);
                if (occ(hc, high) != mask) continue;
                const auto w = mpair(m, p);
                if (!closure_pair(w)) continue;
                const IncludeResult z = include_horizontal(m, W, p);
                if (z.valid) expected.insert(m);
            }

            for (int he = 0; he <= high + 1; ++he) {
                const auto& hrs = high_rows.codes[mask][he];
                if (hrs.empty()) continue;
                for (int cv = 0; cv < 3; ++cv) {
                    const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                    const int bid = 3 * he + cv;
                    if (hs < 0 || hs > low || bid >= int(closure_lr.size())) continue;
                    const auto& cols = closure_lr[bid];
                    const U64 chunks = (cols.size() + 31) / 32;
                    task_count += U64(hrs.size()) * chunks;
                    lane_slots += U64(hrs.size()) * chunks * 32;
                    for (std::uint32_t hc : hrs) {
                        for (U64 chunk = 0; chunk < chunks; ++chunk) {
                            for (U64 lane = 0; lane < 32; ++lane) {
                                const U64 qi = chunk * 32 + lane;
                                if (qi >= cols.size()) continue;
                                const std::uint32_t lr = cols[size_t(qi)];
                                const std::uint32_t lc = low_storage.codes[hs][lr];
                                MateID m = MateID(lc)
                                    | (MateID(cv) << (2 * low))
                                    | (MateID(hc) << (2 * (low + 1)));
                                const IncludeResult z = include_horizontal(m, W, p);
                                if (!z.valid || !closure_pair(mpair(m, p))) continue;
                                if (!got.insert(m).second) {
                                    std::cerr << "duplicate compact LOW closure source W=" << W
                                              << " p=" << p << " mask=" << mask
                                              << " bid=" << bid << '\n';
                                    return 10;
                                }
                            }
                        }
                    }
                }
            }

            if (got != expected) {
                std::cerr << "compact LOW closure task-map mismatch W=" << W
                          << " p=" << p << " mask=" << mask
                          << " got=" << got.size()
                          << " expected=" << expected.size() << '\n';
                return 11;
            }
            sources += got.size();
        }
    }

    std::cout << "factor-lowclosure-taskmap OK W=" << W
              << " low=" << low << " high=" << high
              << " main_states=" << states.size()
              << " sources=" << sources
              << " tasks=" << task_count
              << " lane_slots=" << lane_slots << '\n';
    return 0;
}
