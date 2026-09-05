#include "../../common/gridfp_transition.hpp"
#include "../../common/gridfp_transition_reverse.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <vector>

namespace {
using namespace oneesan::gridfp;

struct Key {
    MateID mate = 0;
    bool blocked = false;
    bool operator<(const Key& o) const {
        return blocked != o.blocked ? blocked < o.blocked : mate < o.mate;
    }
};
using Terms = std::map<Key, int>;

void add(Terms& z, Key k, int c = 1) {
    z[k] += c;
    if (!z[k]) z.erase(k);
}

void project_reverse(Terms& z, MateID mate, bool blocked, int W) {
    if (!blocked || mget(mate, 1) != N) {
        add(z, Key{mate, blocked});
        return;
    }
    const MateID nn = blocked_exclude_reverse(mate, W, 2);
    add(z, Key{nn, false});
    add(z, Key{msetpair(nn, 2, LR), false}, -1);
}

Terms generic_step(MateID src, int W) {
    Terms z;
    add(z, Key{src, false});
    const IncludeResult x = include_horizontal_reverse(src, W, 1);
    if (x.valid) project_reverse(z, x.mate, x.blocked, W);
    return z;
}

Terms direct_step(MateID src, int W) {
    Terms z;
    add(z, Key{src, false});
    const MateValuePair pair = mpair(src, 1);
    switch (pair) {
    case NN:
        add(z, Key{msetpair(src, 1, LR), false});
        break;
    case NL:
        add(z, Key{msetpair(src, 1, LN), false});
        break;
    case NR:
        add(z, Key{msetpair(src, 1, RN), false});
        break;
    case LN: case RN: {
        const MateID b = mshrink(src, 0);
        project_reverse(z, b, true, W);
        break;
    }
    case RR: {
        MateID t = msetpair(src, 1, NN);
        const int q = closure_match_right(t, W, 1);
        if (q >= 0) {
            t = mset(t, q, R);
            project_reverse(z, mshrink(t, 1), true, W);
        }
        break;
    }
    case RL:
        project_reverse(z, mshrink(msetpair(src, 1, NN), 1), true, W);
        break;
    default:
        break;
    }
    return z;
}

std::vector<MateID> generate_valid(int W) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        const int rem = W - pos;
        if (h < 0 || h > rem) return;
        if (pos == W) { if (h == 0) out.push_back(m); return; }
        const int bit = W - 1 - pos;
        self(self, pos + 1, h, m);
        if (h > 0) self(self, pos + 1, h - 1, m | (MateID(R) << (2 * bit)));
        self(self, pos + 1, h + 1, m | (MateID(L) << (2 * bit)));
    };
    rec(rec, 0, 1, 0);
    return out;
}

using CountTable = std::array<std::array<std::uint64_t, 31>, 29>;
CountTable make_counts() {
    CountTable f{}; f[0][0] = 1;
    for (int rem = 1; rem <= 28; ++rem) {
        for (int h = 0; h <= 29; ++h) {
            std::uint64_t z = f[rem - 1][h];
            if (h > 0) z += f[rem - 1][h - 1];
            if (h < 30) z += f[rem - 1][h + 1];
            f[rem][h] = z;
        }
    }
    return f;
}

MateID unrank_valid(int W, std::uint64_t rank, const CountTable& f) {
    MateID m = 0; int h = 1;
    for (int pos = 0; pos < W; ++pos) {
        const int rem = W - pos - 1, bit = W - 1 - pos;
        const std::uint64_t n = f[rem][h];
        if (rank < n) continue;
        rank -= n;
        const std::uint64_t r = h > 0 ? f[rem][h - 1] : 0;
        if (rank < r) { m |= MateID(R) << (2 * bit); --h; continue; }
        rank -= r; m |= MateID(L) << (2 * bit); ++h;
    }
    return m;
}

void check(MateID forward, int W,
           std::uint64_t& three_term,
           std::uint64_t& rr_cases,
           std::uint64_t& rl_cases) {
    const MateID src = mirror_mate(forward, W);
    const Terms a = generic_step(src, W);
    const Terms b = direct_step(src, W);
    if (a != b) {
        std::cerr << "mismatch W=" << W << " src=" << src
                  << " pair=" << int(mpair(src, 1)) << '\n';
        std::exit(2);
    }
    if (a.size() == 3) ++three_term;
    if (mpair(src, 1) == RR) ++rr_cases;
    if (mpair(src, 1) == RL) ++rl_cases;
}
} // namespace

int main() {
    std::uint64_t exhaustive = 0, random = 0;
    std::uint64_t three_term = 0, rr_cases = 0, rl_cases = 0;
    for (int W = 2; W <= 12; ++W) {
        for (MateID m : generate_valid(W)) {
            check(m, W, three_term, rr_cases, rl_cases);
            ++exhaustive;
        }
    }

    const CountTable f = make_counts();
    if (f[28][1] != 385719506620ULL) return 3;
    std::mt19937_64 rng(0x6c6f776578707374ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        check(unrank_valid(28, rng() % f[28][1], f), 28,
              three_term, rr_cases, rl_cases);
        ++random;
    }
    if (!three_term || !rr_cases || !rl_cases) return 4;
    std::cout << "gridfp-runtime-turn-direct-low-expand-step-proof OK"
              << " exhaustive_width_max=12 exhaustive_cases=" << exhaustive
              << " random_width=28 random_cases=" << random
              << " three_term_projection_cases=" << three_term
              << " rr_cases=" << rr_cases << " rl_cases=" << rl_cases
              << " source_scope=main_only reverse_p=1"
              << " include_result_dispatch=0"
              << " generic_project_reverse_calls=0"
              << " full_mirror_passes=0 step_exact=1\n";
    return 0;
}
