#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

static U64 capped_state_count(int width, int cap) {
    std::vector<U64> cur(cap + 1), nxt(cap + 1);
    cur[1] = 1;
    for (int pos = 0; pos < width; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= cap; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h > 0) nxt[h - 1] += cur[h];
            if (h + 1 <= cap) nxt[h + 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur[0];
}

static bool orbit_rep(MateValuePair w) {
    return w == NN || w == NR || w == NL;
}

static void verify_width(int W, int low) {
    const auto blocked = enumerate_states(W - 1);
    const int max_cap = (W + 1) / 2;
    U64 checked = 0;

    for (int p = low; p >= 1; --p) {
        for (MateID d : blocked) {
            const MateID source = blocked_exclude(d, p);
            const MateValuePair w = mpair(source, p);
            if (!orbit_rep(w)) {
                std::cerr << "LOW blocked exclusion did not create orbit representative W="
                          << W << " p=" << p << '\n';
                std::exit(20);
            }

            MateValuePair cw = LR;
            if (w == NR) cw = RN;
            else if (w == NL) cw = LN;
            const MateID companion = msetpair(source, p, cw);

            const int dd = max_height(d, W - 1);
            const int ds = max_height(source, W);
            const int dc = max_height(companion, W);
            for (int cap = 1; cap <= max_cap; ++cap) {
                if (dd > cap && (ds <= cap || dc <= cap)) {
                    std::cerr << "unsafe LOW BLOCKED row-depth prune W=" << W
                              << " p=" << p << " cap=" << cap
                              << " blocked_depth=" << dd
                              << " source_depth=" << ds
                              << " companion_depth=" << dc << '\n';
                    std::exit(21);
                }
                ++checked;
            }
        }
    }

    std::cout << "loworbit-rowdepth-semantics OK W=" << W
              << " low=" << low << " checks=" << checked << '\n';
}

int main(int argc, char** argv) {
    const int max_w = argc > 1 ? std::atoi(argv[1]) : 12;
    if (max_w < 6 || max_w > 14) return 1;
    for (int W = 6; W <= max_w; W += 2) verify_width(W, W / 2);

    constexpr int W = 28;
    constexpr int LOW = 14;
    constexpr U64 D = 135015505407ULL;
    U64 active_per_low_position = 0;
    for (int row = 1; row <= W; ++row)
        active_per_low_position += capped_state_count(W - 1, std::min(row, W / 2));

    const U64 dense_bodies = D * U64(LOW) * U64(W);
    const U64 active_bodies = active_per_low_position * U64(LOW);
    if (dense_bodies != 52926078119544ULL
        || active_bodies != 46983616692250ULL) {
        std::cerr << "n=27 LOW orbit body-count regression mismatch\n";
        return 2;
    }

    const long double ratio =
        (long double)active_bodies / (long double)dense_bodies;
    std::cout << std::fixed << std::setprecision(12)
              << "n27_loworbit_dense_bodies=" << dense_bodies
              << " active_bodies=" << active_bodies
              << " ratio=" << double(ratio)
              << " removable=" << double(1.0L - ratio) << '\n';
    return 0;
}
