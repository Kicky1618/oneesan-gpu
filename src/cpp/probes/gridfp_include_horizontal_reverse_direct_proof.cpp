#include "../../common/gridfp_transition_reverse.hpp"

#include <cstdint>
#include <iostream>
#include <random>

namespace {
using namespace oneesan::gridfp;

IncludeResult legacy_reverse(MateID m, int width, int p) {
    IncludeResult z = include_horizontal(mirror_mate(m, width), width, width - p);
    if (!z.valid) return z;
    z.mate = mirror_mate(z.mate, z.blocked ? width - 1 : width);
    return z;
}

IncludeResult direct_reverse(MateID m, int width, int p) {
    IncludeResult z{};
    MateID t = m;
    switch (mpair(m, p)) {
    case NN:
        z.mate = msetpair(m, p, LR); z.valid = true; return z;
    case LN: case RN:
        if (p == width - 1) {
            z.mate = msetpair(m, p, mpair(m, p) == LN ? NL : NR);
            z.valid = true;
            return z;
        }
        z.mate = mshrink(m, p - 1); z.valid = true; z.blocked = true; return z;
    case NL:
        z.mate = msetpair(m, p, LN); z.valid = true; return z;
    case NR:
        z.mate = msetpair(m, p, RN); z.valid = true; return z;
    case LL: {
        t = msetpair(m, p, NN); int q = p - 1, s = 1;
        while (s) {
            --q; if (q < 0) return z;
            const auto v = mget(t, q);
            if (v == L) ++s; else if (v == R) --s;
        }
        t = mset(t, q, L);
        if (p == width - 1) { z.mate = t; z.valid = true; return z; }
        z.mate = mshrink(t, p); z.valid = true; z.blocked = true; return z;
    }
    case RR: {
        t = msetpair(m, p, NN); int q = p, s = 1;
        while (s) {
            ++q; if (q >= width) return z;
            const auto v = mget(t, q);
            if (v == L) --s; else if (v == R) ++s;
        }
        t = mset(t, q, R);
        if (p == width - 1) { z.mate = t; z.valid = true; return z; }
        z.mate = mshrink(t, p); z.valid = true; z.blocked = true; return z;
    }
    case RL:
        t = msetpair(m, p, NN);
        if (p == width - 1) { z.mate = t; z.valid = true; return z; }
        z.mate = mshrink(t, p); z.valid = true; z.blocked = true; return z;
    default:
        return z;
    }
}

bool equal_result(const IncludeResult& a, const IncludeResult& b) {
    return a.mate == b.mate && a.valid == b.valid && a.blocked == b.blocked;
}
}

int main() {
    std::uint64_t exhaustive = 0;
    for (int width = 2; width <= 7; ++width) {
        const int bits = 2 * width;
        const MateID limit = MateID(1) << bits;
        for (MateID m = 0; m < limit; ++m) {
            for (int p = 1; p < width; ++p) {
                ++exhaustive;
                const auto a = legacy_reverse(m, width, p);
                const auto b = direct_reverse(m, width, p);
                if (!equal_result(a, b)) {
                    std::cerr << "mismatch width=" << width << " p=" << p
                              << " mate=" << m
                              << " legacy=" << a.mate << ',' << a.valid << ',' << a.blocked
                              << " direct=" << b.mate << ',' << b.valid << ',' << b.blocked
                              << '\n';
                    return 2;
                }
            }
        }
    }

    std::mt19937_64 rng(0x726576696e636c75ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int width = 2 + int(rng() % 27);
        const int p = 1 + int(rng() % (width - 1));
        const int bits = 2 * width;
        const MateID mask = bits == 64 ? ~MateID(0) : ((MateID(1) << bits) - 1);
        const MateID m = rng() & mask;
        if (!equal_result(legacy_reverse(m, width, p), direct_reverse(m, width, p))) {
            std::cerr << "random mismatch width=" << width << " p=" << p << '\n';
            return 3;
        }
    }

    std::cout << "gridfp-include-horizontal-reverse-direct-proof OK"
              << " exhaustive_width_max=7 exhaustive_cases=" << exhaustive
              << " random_cases=" << RANDOM
              << " production_width_max=28"
              << " direct_coordinate_exact=1\n";
    return 0;
}
