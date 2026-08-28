#include "../../common/gridfp_transition.hpp"

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

void project_forward(Terms& z, MateID mate, bool blocked, int W) {
    const int q = W - 2;
    if (!blocked || mget(mate, q - 1) != N) {
        add(z, Key{mate, blocked});
        return;
    }
    const MateID nn = blocked_exclude(mate, q);
    add(z, Key{nn, false});
    add(z, Key{msetpair(nn, q, LR), false}, -1);
}

Terms generic_step(MateID src, int W) {
    Terms z;
    add(z, Key{src, false});
    const int p = W - 1;
    const IncludeResult x = include_horizontal(src, W, p);
    if (x.valid) project_forward(z, x.mate, x.blocked, W);
    return z;
}

Terms direct_step(MateID src, int W) {
    Terms z;
    add(z, Key{src, false});
    const int p = W - 1;
    const MateValuePair pair = mpair(src, p);
    switch (pair) {
    case NN:
        add(z, Key{msetpair(src, p, LR), false});
        break;
    case NR: case NL:
        project_forward(z, mshrink(src, p), true, W);
        break;
    case RN:
        add(z, Key{msetpair(src, p, NR), false});
        break;
    case LN:
        add(z, Key{msetpair(src, p, NL), false});
        break;
    case LL: {
        MateID t = msetpair(src, p, NN);
        const int q = closure_match_left(t, p);
        if (q >= 0) {
            t = mset(t, q, L);
            project_forward(z, mshrink(t, p - 1), true, W);
        }
        break;
    }
    case RL:
        project_forward(z, mshrink(msetpair(src, p, NN), p - 1), true, W);
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

void check(MateID src, int W,
           std::uint64_t& three_term,
           std::uint64_t& ll_cases,
           std::uint64_t& rl_cases) {
    const Terms a = generic_step(src, W);
    const Terms b = direct_step(src, W);
    if (a != b) {
        std::cerr << "mismatch W=" << W << " src=" << src
                  << " pair=" << int(mpair(src, W - 1)) << '\n';
        std::exit(2);
    }
    if (a.size() == 3) ++three_term;
    if (mpair(src, W - 1) == LL) ++ll_cases;
    if (mpair(src, W - 1) == RL) ++rl_cases;
}
} // namespace

int main() {
    std::uint64_t exhaustive = 0, random = 0;
    std::uint64_t three_term = 0, ll_cases = 0, rl_cases = 0;
    for (int W = 3; W <= 12; ++W) {
        for (MateID m : generate_valid(W)) {
            check(m, W, three_term, ll_cases, rl_cases);
            ++exhaustive;
        }
    }

    const CountTable f = make_counts();
    if (f[28][1] != 385719506620ULL) return 3;
    std::mt19937_64 rng(0x6869676865787073ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        check(unrank_valid(28, rng() % f[28][1], f), 28,
              three_term, ll_cases, rl_cases);
        ++random;
    }
    if (!three_term || !ll_cases || !rl_cases) return 4;
    std::cout << "gridfp-runtime-turn-direct-high-expand-step-proof OK"
              << " exhaustive_width_max=12 exhaustive_cases=" << exhaustive
              << " random_width=28 random_cases=" << random
              << " three_term_projection_cases=" << three_term
              << " ll_cases=" << ll_cases << " rl_cases=" << rl_cases
              << " source_scope=main_only forward_p=Wm1"
              << " include_result_dispatch=0"
              << " generic_project_forward_calls=0"
              << " full_mirror_passes=0 step_exact=1\n";
    return 0;
}
