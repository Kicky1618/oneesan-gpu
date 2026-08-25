#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;

struct FactorCode {
    std::uint32_t code = 0;
    int peak = 0;
};

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
    int h = start_h, peak = start_h;
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

static int segment_end(std::uint32_t code, int len, int start_h) {
    int h = start_h;
    for (int p = len - 1; p >= 0; --p) {
        const std::uint32_t v = (code >> (2 * p)) & 3u;
        if (v == std::uint32_t(R)) --h;
        else if (v == std::uint32_t(L)) ++h;
    }
    return h;
}

static std::uint32_t occupancy(std::uint32_t code, int len) {
    std::uint32_t mask = 0;
    for (int p = 0; p < len; ++p)
        if ((code >> (2 * p)) & 3u) mask |= 1u << p;
    return mask;
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

static void verify(int W, int low) {
    const int high = W - 1 - low;
    const auto blocked = enumerate_states(W - 1);
    const std::uint32_t low_code_mask = (1u << (2 * low)) - 1u;
    const std::uint32_t nmasks = 1u << low;

    std::vector<std::unordered_map<std::uint32_t, int>> high_by_h(high + 2);
    std::vector<std::unordered_map<std::uint32_t, int>> low_by_mask_h(
        std::size_t(nmasks) * std::size_t(low + 2));

    for (MateID m : blocked) {
        const std::uint32_t lc = std::uint32_t(m) & low_code_mask;
        const std::uint32_t hc = std::uint32_t(m >> (2 * low));
        const int h = segment_end(hc, high, 1);
        if (h < 0 || h > high + 1 || h > low + 1) std::exit(20);
        const std::uint32_t mask = occupancy(lc, low);
        const int hp = segment_peak(hc, high, 1);
        const int lp = segment_peak(lc, low, h);
        auto [hi, hins] = high_by_h[h].emplace(hc, hp);
        if (!hins && hi->second != hp) std::exit(21);
        auto& lm = low_by_mask_h[std::size_t(mask) * (low + 2) + h];
        auto [li, lins] = lm.emplace(lc, lp);
        if (!lins && li->second != lp) std::exit(22);
    }

    std::uint64_t total_checks = 0;
    for (int cap = 1; cap <= W / 2; ++cap) {
        std::unordered_set<MateID> expect;
        expect.reserve(blocked.size() * 2 + 1);
        for (MateID m : blocked)
            if (max_height(m, W - 1) <= cap) expect.insert(m);

        std::unordered_set<MateID> got;
        got.reserve(expect.size() * 2 + 1);
        for (std::uint32_t mask = 0; mask < nmasks; ++mask) {
            for (int h = 0; h <= high + 1 && h <= low + 1; ++h) {
                const auto& hs = high_by_h[h];
                const auto& ls = low_by_mask_h[
                    std::size_t(mask) * (low + 2) + h];
                for (const auto& [hc, hp] : hs) {
                    if (hp > cap) continue;
                    for (const auto& [lc, lp] : ls) {
                        if (lp > cap) continue;
                        const MateID m = MateID(lc) | (MateID(hc) << (2 * low));
                        if (!got.insert(m).second) {
                            std::cerr << "duplicate compact task W=" << W
                                      << " cap=" << cap << '\n';
                            std::exit(23);
                        }
                    }
                }
            }
        }
        if (got != expect) {
            std::cerr << "exact compact task-set mismatch W=" << W
                      << " cap=" << cap
                      << " got=" << got.size()
                      << " expect=" << expect.size() << '\n';
            std::exit(24);
        }
        total_checks += got.size();
    }

    std::cout << "blockorbit-exacttask-taskmap OK W=" << W
              << " low=" << low << " high=" << high
              << " blocked=" << blocked.size()
              << " accumulated_active=" << total_checks << '\n';
}

int main() {
    verify(6, 3);
    verify(8, 4);
    verify(10, 5);
    verify(12, 6);
    return 0;
}
