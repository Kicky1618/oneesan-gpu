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
    bool operator==(const Key& o) const {
        return mate == o.mate && blocked == o.blocked;
    }
};
using Terms = std::map<Key, int>;

void add(Terms& z, Key k) { ++z[k]; }

Terms generic_low(Key src, int W) {
    Terms z;
    if (!src.blocked) {
        add(z, src);
        const IncludeResult x = include_horizontal(src.mate, W, 1);
        if (x.valid) {
            if (x.blocked) std::exit(20);
            add(z, Key{x.mate, false});
        }
        return z;
    }
    add(z, Key{blocked_exclude(src.mate, 1), false});
    return z;
}

Terms direct_low(Key src, int W) {
    Terms z;
    if (src.blocked) {
        add(z, Key{blocked_exclude(src.mate, 1), false});
        return z;
    }
    add(z, src);
    const MateValuePair pair = mpair(src.mate, 1);
    switch (pair) {
    case NN:
        add(z, Key{msetpair(src.mate, 1, LR), false});
        break;
    case NR:
        add(z, Key{msetpair(src.mate, 1, RN), false});
        break;
    case RN:
        add(z, Key{msetpair(src.mate, 1, NR), false});
        break;
    case RR: {
        MateID t = msetpair(src.mate, 1, NN);
        const int q = closure_match_right(t, W, 1);
        if (q >= 0) add(z, Key{mset(t, q, R), false});
        break;
    }
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
        const int rem = W - pos - 1;
        const int bit = W - 1 - pos;
        const std::uint64_t n = f[rem][h];
        if (rank < n) continue;
        rank -= n;
        const std::uint64_t r = h > 0 ? f[rem][h - 1] : 0;
        if (rank < r) { m |= MateID(R) << (2 * bit); --h; continue; }
        rank -= r; m |= MateID(L) << (2 * bit); ++h;
    }
    return m;
}

void check(Key k, int W, std::uint64_t& terms, std::uint64_t& rr) {
    const Terms a = generic_low(k, W);
    const Terms b = direct_low(k, W);
    if (a != b) {
        std::cerr << "mismatch W=" << W << " blocked=" << k.blocked
                  << " mate=" << k.mate << '\n';
        std::exit(2);
    }
    terms += a.size();
    if (!k.blocked && mpair(k.mate, 1) == RR) ++rr;
}

} // namespace

int main() {
    std::uint64_t exhaustive_main = 0, exhaustive_blocked = 0;
    std::uint64_t random_main = 0, random_blocked = 0, emitted_terms = 0, rr_cases = 0;
    for (int W = 2; W <= 12; ++W) {
        for (MateID m : generate_valid(W)) { check(Key{m, false}, W, emitted_terms, rr_cases); ++exhaustive_main; }
        for (MateID b : generate_valid(W - 1)) { check(Key{b, true}, W, emitted_terms, rr_cases); ++exhaustive_blocked; }
    }
    const CountTable f = make_counts();
    if (f[28][1] != 385719506620ULL || f[27][1] != 135015505407ULL) return 3;
    std::mt19937_64 rng(0x6c6f777475726e31ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        check(Key{unrank_valid(28, rng() % f[28][1], f), false}, 28, emitted_terms, rr_cases);
        check(Key{unrank_valid(27, rng() % f[27][1], f), true}, 28, emitted_terms, rr_cases);
        ++random_main; ++random_blocked;
    }
    if (!rr_cases) return 4;
    std::cout << "gridfp-runtime-turn-direct-low-compress-step-proof OK"
              << " exhaustive_width_max=12 exhaustive_main=" << exhaustive_main
              << " exhaustive_blocked=" << exhaustive_blocked
              << " random_width=28 random_main=" << random_main
              << " random_blocked=" << random_blocked
              << " rr_cases=" << rr_cases
              << " emitted_terms=" << emitted_terms
              << " local_cases=NN,NR,RN"
              << " rr_closure=closure_match_right"
              << " generic_include_calls=0"
              << " step_exact=1\n";
    return 0;
}
