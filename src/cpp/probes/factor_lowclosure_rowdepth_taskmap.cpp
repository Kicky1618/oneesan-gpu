#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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

static int peak_code(std::uint32_t code, int len, int start_h) {
    int h = start_h, peak = h;
    for (int pos = len - 1; pos >= 0; --pos) {
        const std::uint32_t v = (code >> (2 * pos)) & 3u;
        if (v == std::uint32_t(R)) --h;
        else if (v == std::uint32_t(L)) {
            ++h;
            peak = std::max(peak, h);
        }
    }
    return peak;
}

static int peak_state(MateID m, int width) {
    int h = 1, peak = 1;
    for (int pos = width - 1; pos >= 0; --pos) {
        const MateValue v = mget(m, pos);
        if (v == R) --h;
        else if (v == L) {
            ++h;
            peak = std::max(peak, h);
        }
    }
    return peak;
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

struct LowCode {
    std::uint32_t code = 0;
    std::uint8_t peak = 0;
};

static std::vector<std::vector<LowCode>> build_low_storage(int low) {
    std::vector<std::vector<LowCode>> out(low + 1);
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
        for (std::uint32_t mask = 0; mask < nm; ++mask) {
            for (std::uint32_t code : groups[mask])
                out[h0].push_back({code, std::uint8_t(peak_code(code, low, h0))});
        }
    }
    return out;
}

struct HighCode {
    std::uint32_t code = 0;
    std::uint8_t peak = 0;
};

struct HighRows {
    std::vector<std::vector<std::vector<HighCode>>> codes;
};

static HighRows build_high_rows(int high) {
    const std::uint32_t nm = 1u << high;
    HighRows out;
    out.codes.resize(nm, std::vector<std::vector<HighCode>>(high + 2));
    auto rec = [&](auto&& self, int pos, int h, int peak,
                   std::uint32_t code) -> void {
        if (pos < 0) {
            out.codes[occ(code, high)][h].push_back(
                {code, std::uint8_t(peak)});
            return;
        }
        self(self, pos - 1, h, peak, code);
        if (h) self(self, pos - 1, h - 1, peak,
                    code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1, std::max(peak, h + 1),
             code | (2u << (2 * pos)));
    };
    rec(rec, high - 1, 1, 1, 0);
    return out;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : W / 2;
    const int high = W - 1 - low;
    const int full_cap = (W + 1) / 2;
    if (W < 4 || W > 13 || low < 1 || high < 1 || low >= 16 || high >= 16)
        return 1;

    const U64 high_code_mask = (U64(1) << (2 * high)) - 1;
    const auto states = enumerate_states(W);
    const auto low_storage = build_low_storage(low);
    const auto high_rows = build_high_rows(high);

    U64 expected_total = 0, compact_total = 0, task_total = 0, lane_slots = 0;
    for (int p = low; p >= 1; --p) {
        struct SelectedCol {
            std::uint32_t lr = 0;
            std::uint8_t peak = 0;
        };
        std::vector<std::vector<SelectedCol>> closure_lr(3 * (high + 2));
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                const int bid = 3 * he + cv;
                if (hs < 0 || hs > low || bid >= int(closure_lr.size())) continue;
                const auto& lc = low_storage[hs];
                for (std::uint32_t lr = 0; lr < lc.size(); ++lr) {
                    const U64 active = U64(lc[lr].code) | (U64(cv) << (2 * low));
                    const auto w = MateValuePair((active >> (2 * (p - 1))) & 15u);
                    if (closure_pair(w))
                        closure_lr[bid].push_back({lr, lc[lr].peak});
                }
                std::stable_sort(closure_lr[bid].begin(), closure_lr[bid].end(),
                    [](const SelectedCol& a, const SelectedCol& b) {
                        return a.peak != b.peak ? a.peak < b.peak : a.lr < b.lr;
                    });
            }
        }

        for (std::uint32_t mask = 0; mask < (1u << high); ++mask) {
            for (int cap = 1; cap <= full_cap; ++cap) {
                std::unordered_set<MateID> expected, got;
                expected.reserve(states.size() / 4 + 1);
                got.reserve(states.size() / 4 + 1);

                for (MateID m : states) {
                    const std::uint32_t hc = std::uint32_t(
                        (m >> (2 * (low + 1))) & high_code_mask);
                    if (occ(hc, high) != mask || !closure_pair(mpair(m, p)))
                        continue;
                    if (peak_state(m, W) <= cap) expected.insert(m);
                }

                for (int he = 0; he <= high + 1; ++he) {
                    const auto& dense_hrs = high_rows.codes[mask][he];
                    if (dense_hrs.empty()) continue;
                    std::vector<HighCode> hrs;
                    for (const HighCode& x : dense_hrs)
                        if (int(x.peak) <= cap) hrs.push_back(x);
                    std::stable_sort(hrs.begin(), hrs.end(),
                        [](const HighCode& a, const HighCode& b) {
                            return a.peak != b.peak ? a.peak < b.peak : a.code < b.code;
                        });
                    if (hrs.empty()) continue;

                    for (int cv = 0; cv < 3; ++cv) {
                        const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                        const int bid = 3 * he + cv;
                        if (hs < 0 || hs > low || bid >= int(closure_lr.size())) continue;
                        const auto& dense_cols = closure_lr[bid];
                        std::size_t ncols = 0;
                        while (ncols < dense_cols.size()
                               && int(dense_cols[ncols].peak) <= cap) ++ncols;
                        if (!ncols) continue;

                        const U64 chunks = (U64(ncols) + 31) / 32;
                        task_total += U64(hrs.size()) * chunks;
                        lane_slots += U64(hrs.size()) * chunks * 32;
                        for (const HighCode& hx : hrs) {
                            for (U64 chunk = 0; chunk < chunks; ++chunk) {
                                for (U64 lane = 0; lane < 32; ++lane) {
                                    const U64 qi = chunk * 32 + lane;
                                    if (qi >= ncols) continue;
                                    const std::uint32_t lr = dense_cols[size_t(qi)].lr;
                                    const std::uint32_t lc = low_storage[hs][lr].code;
                                    const MateID m = MateID(lc)
                                        | (MateID(cv) << (2 * low))
                                        | (MateID(hx.code) << (2 * (low + 1)));
                                    if (!got.insert(m).second) {
                                        std::cerr << "duplicate row-depth LOW closure source W="
                                                  << W << " p=" << p << " mask="
                                                  << mask << " cap=" << cap
                                                  << " bid=" << bid << '\n';
                                        return 10;
                                    }
                                }
                            }
                        }
                    }
                }

                if (got != expected) {
                    std::cerr << "row-depth LOW closure task-map mismatch W=" << W
                              << " p=" << p << " mask=" << mask
                              << " cap=" << cap << " got=" << got.size()
                              << " expected=" << expected.size() << '\n';
                    return 11;
                }
                expected_total += expected.size();
                compact_total += got.size();
            }
        }
    }

    std::cout << "factor-lowclosure-rowdepth-taskmap OK W=" << W
              << " low=" << low << " high=" << high
              << " main_states=" << states.size()
              << " expected=" << expected_total
              << " compact=" << compact_total
              << " tasks=" << task_total
              << " lane_slots=" << lane_slots << '\n';
    return 0;
}
