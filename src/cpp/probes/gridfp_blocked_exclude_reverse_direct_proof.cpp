#include "../../common/gridfp_transition_reverse.hpp"

#include <cstdint>
#include <iostream>
#include <random>

namespace {
using namespace oneesan::gridfp;

MateID legacy_blocked_exclude_reverse(MateID compressed, int width, int p) {
    MateID mirrored = mirror_mate(compressed, width - 1);
    MateID expanded = blocked_exclude(mirrored, width - p);
    return mirror_mate(expanded, width);
}

MateID direct_blocked_exclude_reverse(MateID compressed, int width, int p) {
    (void)width;
    return minsert(compressed, p - 1, N);
}
}

int main() {
    std::uint64_t exhaustive = 0;
    for (int width = 2; width <= 8; ++width) {
        const int bits = 2 * (width - 1);
        const MateID limit = MateID(1) << bits;
        for (MateID compressed = 0; compressed < limit; ++compressed) {
            for (int p = 1; p < width; ++p) {
                ++exhaustive;
                const MateID a = legacy_blocked_exclude_reverse(compressed, width, p);
                const MateID b = direct_blocked_exclude_reverse(compressed, width, p);
                if (a != b) {
                    std::cerr << "mismatch width=" << width << " p=" << p
                              << " compressed=" << compressed << '\n';
                    return 2;
                }
            }
        }
    }

    std::mt19937_64 rng(0x726576657273654eULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int width = 2 + int(rng() % 27);
        const int p = 1 + int(rng() % (width - 1));
        const int bits = 2 * (width - 1);
        const MateID mask = bits == 64 ? ~MateID(0) : ((MateID(1) << bits) - 1);
        const MateID compressed = rng() & mask;
        if (legacy_blocked_exclude_reverse(compressed, width, p) !=
            direct_blocked_exclude_reverse(compressed, width, p))
            return 3;
    }

    std::cout << "gridfp-blocked-exclude-reverse-direct-proof OK"
              << " exhaustive_width_max=8 exhaustive_cases=" << exhaustive
              << " random_cases=" << RANDOM
              << " production_width_max=28"
              << " identity=minsert_p_minus_1 exact=1\n";
    return 0;
}
