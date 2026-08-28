#include "../../common/gridfp_transition_reverse.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <utility>

namespace {
using namespace oneesan::gridfp;
using Key = std::pair<MateID,int>;
using Terms = std::map<Key,int>;

Key mirror_key(Key k, int W) {
    return {mirror_mate(k.first, k.second ? W - 1 : W), k.second};
}
void add(Terms& z, Key k, int c) {
    if (!c) return;
    const int v = (z[k] += c);
    if (!v) z.erase(k);
}

void project_forward(Key k, int W, int q, Terms& z) {
    if (!k.second || mget(k.first, q - 1) != N) {
        add(z, k, 1);
        return;
    }
    const MateID nn = blocked_exclude(k.first, q);
    add(z, {nn,0}, 1);
    add(z, {msetpair(nn, q, LR),0}, -1);
}

Terms step_forward(Key src, int W, int p) {
    Terms z;
    if (!src.second) {
        add(z, src, 1);
        const IncludeResult x = include_horizontal(src.first, W, p);
        if (x.valid) project_forward({x.mate, int(x.blocked)}, W, p - 1, z);
    } else {
        add(z, {blocked_exclude(src.first, p),0}, 1);
    }
    return z;
}

Terms legacy_reverse(Key src, int W, int p) {
    const int fp = W - p;
    const Terms tmp = step_forward(mirror_key(src, W), W, fp);
    Terms z;
    for (const auto& [k,c] : tmp) add(z, mirror_key(k, W), c);
    return z;
}

void project_reverse(Key k, int W, int q, Terms& z) {
    if (!k.second || mget(k.first, q - 1) != N) {
        add(z, k, 1);
        return;
    }
    const MateID nn = blocked_exclude_reverse(k.first, W, q);
    add(z, {nn,0}, 1);
    add(z, {msetpair(nn, q, LR),0}, -1);
}

Terms direct_reverse(Key src, int W, int p) {
    Terms z;
    if (!src.second) {
        add(z, src, 1);
        const IncludeResult x = include_horizontal_reverse(src.first, W, p);
        if (x.valid) project_reverse({x.mate, int(x.blocked)}, W, p + 1, z);
    } else {
        add(z, {blocked_exclude_reverse(src.first, W, p),0}, 1);
    }
    return z;
}
}

int main() {
    std::uint64_t exhaustive = 0;
    for (int W = 2; W <= 6; ++W) {
        for (int blocked = 0; blocked <= 1; ++blocked) {
            const int len = blocked ? W - 1 : W;
            const MateID limit = MateID(1) << (2 * len);
            for (MateID m = 0; m < limit; ++m) {
                for (int p = 1; p < W; ++p) {
                    ++exhaustive;
                    if (legacy_reverse({m,blocked}, W, p) !=
                        direct_reverse({m,blocked}, W, p)) {
                        std::cerr << "mismatch W=" << W << " p=" << p
                                  << " blocked=" << blocked << " mate=" << m << '\n';
                        return 2;
                    }
                }
            }
        }
    }

    std::mt19937_64 rng(0x72756e74696d6572ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int W = 2 + int(rng() % 27);
        const int p = 1 + int(rng() % (W - 1));
        const int blocked = int(rng() & 1);
        const int len = blocked ? W - 1 : W;
        const MateID mask = (MateID(1) << (2 * len)) - 1;
        const MateID m = rng() & mask;
        if (legacy_reverse({m,blocked}, W, p) !=
            direct_reverse({m,blocked}, W, p)) {
            std::cerr << "random mismatch W=" << W << " p=" << p
                      << " blocked=" << blocked << '\n';
            return 3;
        }
    }

    std::cout << "gridfp-runtime-reverse-small-step-direct-proof OK"
              << " exhaustive_W_max=6 exhaustive_cases=" << exhaustive
              << " random_cases=" << RANDOM
              << " production_W_max=28"
              << " direct_projection_exact=1 direct_step_exact=1\n";
    return 0;
}
