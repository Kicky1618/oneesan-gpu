#include "../../common/gridfp_closure_inverse.hpp"
#include "../../common/gridfp_closure_pattern10.hpp"
#include "../../common/gridfp_transition_reverse.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#ifndef LOW_LUT_K
#define LOW_LUT_K 14
#endif
#ifndef HIGH_LUT_K
#define HIGH_LUT_K 13
#endif
#ifndef TARGET_W
#define TARGET_W (LOW_LUT_K + HIGH_LUT_K + 1)
#endif

using oneesan::gridfp::CLOSURE_PATTERN10_NONE;
using oneesan::gridfp::L;
using oneesan::gridfp::MateID;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::blocked_exclude_reverse;
using oneesan::gridfp::closure_pattern10_encode;
using oneesan::gridfp::high_cross_preimage_partial;
using oneesan::gridfp::low_cross_preimage_partial;
using oneesan::gridfp::minsert;

static std::vector<std::vector<MateID>> enumerate_low_codes() {
    constexpr int K = LOW_LUT_K;
    std::vector<std::vector<MateID>> out(K + 2);
    auto rec = [&](auto&& self, int pos, int h, MateID code, int h0) -> void {
        if (pos < 0) {
            if (h == 0) out[h0].push_back(code);
            return;
        }
        if (h < 0 || h > pos + 1) return;
        self(self, pos - 1, h, code, h0);
        if (h > 0) self(self, pos - 1, h - 1, code | (MateID(R) << (2 * pos)), h0);
        self(self, pos - 1, h + 1, code | (MateID(L) << (2 * pos)), h0);
    };
    for (int h0 = 0; h0 <= K + 1; ++h0) rec(rec, K - 1, h0, 0, h0);
    return out;
}

static std::vector<std::vector<MateID>> enumerate_high_codes() {
    constexpr int K = HIGH_LUT_K;
    std::vector<std::vector<MateID>> out(K + 2);
    auto rec = [&](auto&& self, int pos, int h, MateID code) -> void {
        if (pos < 0) {
            if (h >= 0 && h < int(out.size())) out[h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0) self(self, pos - 1, h - 1, code | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1, code | (MateID(L) << (2 * pos)));
    };
    rec(rec, K - 1, 1, 0);
    return out;
}

static uint16_t low_pair(MateID d, int p) {
    const uint16_t id = closure_pattern10_encode(d, LOW_LUT_K + 1, p);
    MateID source = 0;
    int depth = low_cross_preimage_partial(d, LOW_LUT_K + 1, p, source);
    if (depth < 0 || depth > 15) {
        std::cerr << "LOW depth overflow p=" << p << " depth=" << depth << '\n';
        std::exit(2);
    }
    if (id == CLOSURE_PATTERN10_NONE) depth = 0;
    return uint16_t((uint16_t(id) << 4) | uint16_t(depth));
}

static uint16_t high_pair(MateID d, int rel) {
    const uint16_t id = closure_pattern10_encode(d, HIGH_LUT_K + 1, rel);
    MateID source = 0;
    int depth = high_cross_preimage_partial(d, HIGH_LUT_K + 1, rel, source);
    if (depth < 0 || depth > 15) {
        std::cerr << "HIGH depth overflow rel=" << rel << " depth=" << depth << '\n';
        std::exit(3);
    }
    if (id == CLOSURE_PATTERN10_NONE) depth = 0;
    return uint16_t((uint16_t(id) << 4) | uint16_t(depth));
}

int main() {
    static_assert(LOW_LUT_K > 0 && HIGH_LUT_K > 0);
    static_assert(LOW_LUT_K <= 14 && HIGH_LUT_K <= 14,
                  "pattern10 depthcode proof targets the 10-bit factor regime");
    static_assert(TARGET_W == LOW_LUT_K + HIGH_LUT_K + 1);

    auto low = enumerate_low_codes();
    auto high = enumerate_high_codes();
    constexpr int LOW_H = HIGH_LUT_K + 2;
    constexpr int HIGH_H = LOW_LUT_K + 2;

    using Contexts = std::vector<std::vector<std::set<uint16_t>>>;
    Contexts f_low(LOW_LUT_K + 1, std::vector<std::set<uint16_t>>(LOW_H));
    Contexts r_low(LOW_LUT_K + 1, std::vector<std::set<uint16_t>>(LOW_H));
    Contexts f_high(HIGH_LUT_K + 1, std::vector<std::set<uint16_t>>(HIGH_H));
    Contexts r_high(HIGH_LUT_K + 1, std::vector<std::set<uint16_t>>(HIGH_H));

    // Forward LOW p=1 is a main-state transition: the center value remains in
    // the local word. Context is source main-block he, while the LOW factor is
    // selected by hs=he+delta(center).
    for (int he = 0; he <= HIGH_LUT_K + 1; ++he) {
        for (int c = int(N); c <= int(L); ++c) {
            int hs = he + (c == int(L) ? 1 : c == int(R) ? -1 : 0);
            if (hs < 0 || hs > LOW_LUT_K + 1) continue;
            for (MateID code : low[hs]) {
                MateID d = code | (MateID(c) << (2 * LOW_LUT_K));
                f_low[1][he].insert(low_pair(d, 1));
            }
        }
    }
    for (int p = 2; p <= LOW_LUT_K; ++p) {
        for (int h = 0; h < LOW_H && h < int(low.size()); ++h) {
            for (MateID code : low[h]) f_low[p][h].insert(low_pair(minsert(code, p, N), p));
        }
    }
    for (int p = 1; p <= LOW_LUT_K; ++p) {
        for (int h = 0; h < LOW_H && h < int(low.size()); ++h) {
            for (MateID code : low[h]) {
                r_low[p][h].insert(low_pair(blocked_exclude_reverse(code, LOW_LUT_K + 1, p), p));
            }
        }
    }

    for (int rel = 1; rel <= HIGH_LUT_K; ++rel) {
        for (int h = 0; h < HIGH_H && h < int(high.size()); ++h) {
            for (MateID code : high[h]) f_high[rel][h].insert(high_pair(minsert(code, rel, N), rel));
        }
    }
    for (int rel = 1; rel < HIGH_LUT_K; ++rel) {
        for (int h = 0; h < HIGH_H && h < int(high.size()); ++h) {
            for (MateID code : high[h]) {
                r_high[rel][h].insert(high_pair(blocked_exclude_reverse(code, HIGH_LUT_K + 1, rel), rel));
            }
        }
    }
    // Reverse HIGH edge mirrors the forward p=1 special case. The high factor
    // is grouped by he, but runtime context is hs.
    const int edge = HIGH_LUT_K;
    for (int he = 0; he <= HIGH_LUT_K + 1 && he < int(high.size()); ++he) {
        for (int c = int(N); c <= int(L); ++c) {
            int hs = he + (c == int(L) ? 1 : c == int(R) ? -1 : 0);
            if (hs < 0 || hs >= HIGH_H) continue;
            for (MateID code : high[he]) {
                MateID d = MateID(c) | (code << 2);
                r_high[edge][hs].insert(high_pair(d, edge));
            }
        }
    }

    size_t max_pairs = 0, contexts = 0, total_pairs = 0;
    std::string worst = "none";
    int worst_p = 0, worst_h = 0;
    auto scan = [&](const Contexts& x, const char* tag) {
        for (int p = 1; p < int(x.size()); ++p) {
            for (int h = 0; h < int(x[p].size()); ++h) {
                if (x[p][h].empty()) continue;
                ++contexts;
                total_pairs += x[p][h].size();
                if (x[p][h].size() > max_pairs) {
                    max_pairs = x[p][h].size();
                    worst = tag;
                    worst_p = p;
                    worst_h = h;
                }
            }
        }
    };
    scan(f_low, "forward-low");
    scan(r_low, "reverse-low");
    scan(f_high, "forward-high");
    scan(r_high, "reverse-high");

    size_t low_codes = 0, high_codes = 0;
    for (const auto& v : low) low_codes += v.size();
    for (const auto& v : high) high_codes += v.size();

    const bool fits = max_pairs <= 1024;
    std::cout << "gridfp-pattern10-depthcode-bound OK W=" << TARGET_W
              << " low_codes=" << low_codes
              << " high_codes=" << high_codes
              << " contexts=" << contexts
              << " total_unique_pairs=" << total_pairs
              << " max_pairs_per_phase_context=" << max_pairs
              << " worst=" << worst
              << " worst_p=" << worst_p
              << " worst_height=" << worst_h
              << " fits10=" << (fits ? 1 : 0)
              << " production_subset_proved=" << (fits ? 1 : 0)
              << '\n';
    return fits ? 0 : 1;
}
