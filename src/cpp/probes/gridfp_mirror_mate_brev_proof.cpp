#include "../../common/gridfp_transition.hpp"

#include <cstdint>
#include <iostream>
#include <random>

namespace {
using namespace oneesan::gridfp;

MateValue mirror_value_ref(MateValue v) {
    if (v == R) return L;
    if (v == L) return R;
    return v;
}
MateID legacy_mirror(MateID m, int width) {
    MateID out = 0;
    for (int i = 0; i < width; ++i)
        out |= MateID(mirror_value_ref(mget(m, i))) << (2 * (width - 1 - i));
    return out;
}
MateID reverse_bits64(MateID x) {
    x = ((x >> 1) & 0x5555555555555555ULL) |
        ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) |
        ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL) |
        ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
    x = ((x >> 8) & 0x00ff00ff00ff00ffULL) |
        ((x & 0x00ff00ff00ff00ffULL) << 8);
    x = ((x >> 16) & 0x0000ffff0000ffffULL) |
        ((x & 0x0000ffff0000ffffULL) << 16);
    x = (x >> 32) | (x << 32);
    return x;
}
MateID fast_mirror(MateID m, int width) {
    if (width <= 0) return 0;
    const MateID rev = reverse_bits64(m);
    const int shift = 64 - 2 * width;
    return shift ? (rev >> shift) : rev;
}
}

int main() {
    std::uint64_t exhaustive = 0;
    for (int width = 0; width <= 8; ++width) {
        const int bits = 2 * width;
        const MateID limit = bits ? (MateID(1) << bits) : 1;
        for (MateID m = 0; m < limit; ++m) {
            ++exhaustive;
            if (legacy_mirror(m, width) != fast_mirror(m, width)) {
                std::cerr << "mismatch width=" << width << " mate=" << m << '\n';
                return 2;
            }
        }
    }

    std::mt19937_64 rng(0x627265766d617465ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int width = int(rng() % 33);
        const int bits = 2 * width;
        const MateID mask = bits == 64 ? ~MateID(0) : (bits ? (MateID(1) << bits) - 1 : 0);
        const MateID m = rng() & mask;
        if (legacy_mirror(m, width) != fast_mirror(m, width)) return 3;
    }

    std::cout << "gridfp-mirror-mate-brev-proof OK"
              << " exhaustive_width_max=8 exhaustive_cases=" << exhaustive
              << " random_cases=" << RANDOM
              << " width_max=32"
              << " bit_reverse_exact=1 rl_swap_implicit=1\n";
    return 0;
}
