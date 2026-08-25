#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;

static int max_height(MateID m, int width) {
    int h = 1, peak = 1;
    for (int p = width - 1; p >= 0; --p) {
        const MateValue v = mget(m, p);
        if (v == R) --h;
        else if (v == L) {
            ++h;
            peak = std::max(peak, h);
        }
    }
    return peak;
}

static int segment_peak(std::uint32_t code, int len, int start_h) {
    int h = start_h, peak = h;
    for (int p = len - 1; p >= 0; --p) {
        const std::uint32_t v = (code >> (2 * p)) & 3u;
        if (v == std::uint32_t(R)) --h;
        else if (v == std::uint32_t(L)) {
            ++h;
            peak = std::max(peak, h);
        }
    }
    return peak;
}

static std::uint32_t occupancy(std::uint32_t code, int len) {
    std::uint32_t m = 0;
    for (int p = 0; p < len; ++p)
        if ((code >> (2 * p)) & 3u) m |= 1u << p;
    return m;
}

static std::vector<MateID> enumerate_states(int width) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        if (pos < 0) {
            if (h == 0) out.push_back(m);
            return;
        }
        if (h < 0 || h > pos + 1) return;
        self(self, pos - 1, h, m);
        if (h > 0)
            self(self, pos - 1, h - 1, m | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1, m | (MateID(L) << (2 * pos)));
    };
    rec(rec, width - 1, 1, 0);
    return out;
}

static std::vector<std::vector<std::uint32_t>> low_codes(int low) {
    std::vector<std::vector<std::uint32_t>> out(low + 2);
    for (int h0 = 0; h0 <= low + 1; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) out[h0].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h > 0)
                self(self, pos - 1, h - 1,
                     code | (std::uint32_t(R) << (2 * pos)));
            self(self, pos - 1, h + 1,
                 code | (std::uint32_t(L) << (2 * pos)));
        };
        rec(rec, low - 1, h0, 0u);
    }
    return out;
}

struct HighCode {
    std::uint32_t code = 0;
    int end_h = 0;
};

static std::vector<HighCode> high_codes(int high) {
    std::vector<HighCode> out;
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            out.push_back({code, h});
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0)
            self(self, pos - 1, h - 1,
                 code | (std::uint32_t(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             code | (std::uint32_t(L) << (2 * pos)));
    };
    rec(rec, high - 1, 1, 0u);
    return out;
}

static void verify(int W, int low) {
    const int high = W - 1 - low;
    const int full_cap = W / 2;
    const auto all = enumerate_states(W - 1);
    const auto lows = low_codes(low);
    const auto highs = high_codes(high);
    const std::uint32_t high_mask_bits = (1u << (2 * high)) - 1u;
    const std::uint32_t nm = 1u << high;
    U64 expected_total = 0, compact_total = 0;

    std::vector<std::vector<std::vector<std::uint32_t>>> hg(
        nm, std::vector<std::vector<std::uint32_t>>(high + 2));
    for (const HighCode& x : highs)
        hg[occupancy(x.code, high)][x.end_h].push_back(x.code);

    for (std::uint32_t mask = 0; mask < nm; ++mask) {
        for (int cap = 1; cap <= full_cap; ++cap) {
            std::unordered_set<MateID> expected, compact;
            for (MateID m : all) {
                const std::uint32_t hc = std::uint32_t(
                    (m >> (2 * low)) & high_mask_bits);
                if (occupancy(hc, high) != mask) continue;
                if (max_height(m, W - 1) <= cap) expected.insert(m);
            }

            for (int h = 0; h <= high + 1; ++h) {
                for (std::uint32_t hc : hg[mask][h]) {
                    if (segment_peak(hc, high, 1) > cap) continue;
                    for (std::uint32_t lc : lows[h]) {
                        if (segment_peak(lc, low, h) > cap) continue;
                        const MateID m = MateID(lc) | (MateID(hc) << (2 * low));
                        if (!compact.insert(m).second) {
                            std::cerr << "duplicate compact LOW orbit task W=" << W
                                      << " mask=" << mask << " cap=" << cap << '\n';
                            std::exit(30);
                        }
                    }
                }
            }
            if (expected != compact) {
                std::cerr << "LOW orbit compact task-map mismatch W=" << W
                          << " mask=" << mask << " cap=" << cap
                          << " expected=" << expected.size()
                          << " compact=" << compact.size() << '\n';
                std::exit(31);
            }
            expected_total += expected.size();
            compact_total += compact.size();
        }
    }

    std::cout << "loworbit-rowdepth-taskmap OK W=" << W
              << " low=" << low << " high=" << high
              << " expected=" << expected_total
              << " compact=" << compact_total << '\n';
}

int main() {
    verify(8, 4);
    verify(10, 5);
    verify(12, 6);
    return 0;
}
