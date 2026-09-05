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
    bool operator==(const Key& o) const {
        return mate == o.mate && blocked == o.blocked;
    }
};
using Terms = std::map<Key, int>;

void add(Terms& z, Key k, int c = 1) {
    z[k] += c;
    if (!z[k]) z.erase(k);
}

Key mirror_key(Key k, int W) {
    return Key{mirror_mate(k.mate, k.blocked ? W - 1 : W), k.blocked};
}

Terms low_compress(Key src, int W) {
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

Terms old_high_compress(Key src, int W) {
    const Terms tmp = low_compress(mirror_key(src, W), W);
    Terms z;
    for (const auto& [k, c] : tmp) add(z, mirror_key(k, W), c);
    return z;
}

Terms direct_high_compress(Key src, int W) {
    Terms z;
    if (!src.blocked) {
        add(z, src);
        const IncludeResult x = include_horizontal_reverse(src.mate, W, W - 1);
        if (x.valid) {
            if (x.blocked) std::exit(21);
            add(z, Key{x.mate, false});
        }
        return z;
    }
    add(z, Key{blocked_exclude_reverse(src.mate, W, W - 1), false});
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

void check(Key k, int W, std::uint64_t& terms) {
    const Terms a = old_high_compress(k, W);
    const Terms b = direct_high_compress(k, W);
    if (a != b) {
        std::cerr << "mismatch W=" << W << " blocked=" << k.blocked
                  << " mate=" << k.mate << '\n';
        std::exit(2);
    }
    terms += a.size();
}

} // namespace

int main() {
    std::uint64_t exhaustive_main = 0, exhaustive_blocked = 0;
    std::uint64_t random_main = 0, random_blocked = 0, emitted_terms = 0;
    for (int W = 2; W <= 12; ++W) {
        for (MateID m : generate_valid(W)) {
            check(Key{m, false}, W, emitted_terms); ++exhaustive_main;
        }
        for (MateID b : generate_valid(W - 1)) {
            check(Key{b, true}, W, emitted_terms); ++exhaustive_blocked;
        }
    }

    const CountTable f = make_counts();
    if (f[28][1] != 385719506620ULL || f[27][1] != 135015505407ULL) return 3;
    std::mt19937_64 rng(0x686967687475726eULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        check(Key{unrank_valid(28, rng() % f[28][1], f), false}, 28, emitted_terms);
        check(Key{unrank_valid(27, rng() % f[27][1], f), true}, 28, emitted_terms);
        ++random_main; ++random_blocked;
    }

    std::cout << "gridfp-runtime-turn-direct-high-compress-step-proof OK"
              << " exhaustive_width_max=12"
              << " exhaustive_main=" << exhaustive_main
              << " exhaustive_blocked=" << exhaustive_blocked
              << " random_width=28 random_main=" << random_main
              << " random_blocked=" << random_blocked
              << " emitted_terms=" << emitted_terms
              << " direct_main=include_horizontal_reverse"
              << " direct_blocked=blocked_exclude_reverse"
              << " source_mirror_passes=0 result_mirror_passes=0"
              << " step_exact=1\n";
    return 0;
}
