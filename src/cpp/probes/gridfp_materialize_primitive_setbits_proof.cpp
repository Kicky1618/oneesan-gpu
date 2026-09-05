#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using MateID = std::uint64_t;
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
std::array<std::array<Rank64, MAX + 2>, MAX + 1> P{};

void build() {
    P[0][0] = 1;
    for (int rem = 1; rem <= MAX; ++rem) {
        for (int h = 0; h <= MAX; ++h) {
            Rank64 z = P[rem - 1][h + 1];
            if (h > 0) z += P[rem - 1][h - 1];
            P[rem][h] = z;
        }
    }
}

MateID ref(std::uint32_t support, int len, int occupied, Rank64 rank) {
    MateID m = 0;
    int h = 1;
    int seen = 0;
    for (int pos = 0; pos < len; ++pos) {
        if (((support >> pos) & 1u) == 0) continue;
        const int rem = occupied - (++seen);
        const Rank64 r_count = h > 0 ? P[rem][h - 1] : 0;
        int v = 1; // R
        if (rank < r_count) {
            --h;
        } else {
            rank -= r_count;
            v = 2; // L
            ++h;
        }
        m |= MateID(v) << (2 * (len - 1 - pos));
    }
    return m;
}

MateID fast(std::uint32_t support, int len, int occupied, Rank64 rank) {
    MateID m = 0;
    int h = 1;
    int seen = 0;
    if (len < 32) support &= (1u << len) - 1u;
    while (support) {
        const int pos = __builtin_ffs(support) - 1;
        const int rem = occupied - (++seen);
        const Rank64 r_count = h > 0 ? P[rem][h - 1] : 0;
        int v = 1;
        if (rank < r_count) {
            --h;
        } else {
            rank -= r_count;
            v = 2;
            ++h;
        }
        m |= MateID(v) << (2 * (len - 1 - pos));
        support &= support - 1u;
    }
    return m;
}
}

int main() {
    build();
    std::uint64_t exhaustive = 0;
    for (int len = 1; len <= 10; ++len) {
        const std::uint32_t limit = 1u << len;
        for (std::uint32_t support = 0; support < limit; ++support) {
            const int occupied = __builtin_popcount(support);
            if (!(occupied & 1)) continue;
            const Rank64 count = P[occupied][1];
            for (Rank64 rank = 0; rank < count; ++rank) {
                ++exhaustive;
                if (ref(support, len, occupied, rank) != fast(support, len, occupied, rank)) {
                    std::cerr << "exhaustive mismatch len=" << len << " support=" << support
                              << " rank=" << rank << '\n';
                    return 2;
                }
            }
        }
    }

    std::mt19937_64 rng(0x6d6174657269616cULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int len = 1 + int(rng() % 28);
        std::uint32_t support = std::uint32_t(rng()) & ((1u << len) - 1u);
        int occupied = __builtin_popcount(support);
        if (!(occupied & 1)) {
            const int bit = int(rng() % len);
            support ^= 1u << bit;
            occupied = __builtin_popcount(support);
        }
        if (!occupied) { support = 1u; occupied = 1; }
        const Rank64 count = P[occupied][1];
        if (!count) return 3;
        const Rank64 rank = rng() % count;
        if (ref(support, len, occupied, rank) != fast(support, len, occupied, rank)) {
            std::cerr << "random mismatch len=" << len << " support=" << support
                      << " rank=" << rank << '\n';
            return 4;
        }
    }

    std::cout << "gridfp-materialize-primitive-setbits-proof OK"
              << " exhaustive_len_max=10 exhaustive_cases=" << exhaustive
              << " random_cases=" << RANDOM
              << " production_len_max=28"
              << " output_exact=1 traversal=support_setbits\n";
    return 0;
}
