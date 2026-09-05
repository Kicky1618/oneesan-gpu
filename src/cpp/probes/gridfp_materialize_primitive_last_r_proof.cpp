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
    if (len < 32) support &= (std::uint32_t(1) << len) - 1u;
    while (support) {
        const int pos = __builtin_ffs(support) - 1;
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
        support &= support - 1u;
    }
    return m;
}

MateID last_r(std::uint32_t support, int len, int occupied, Rank64 rank) {
    MateID m = 0;
    int h = 1;
    int seen = 0;
    if (len < 32) support &= (std::uint32_t(1) << len) - 1u;
    while (support) {
        const int pos = __builtin_ffs(support) - 1;
        if ((occupied & 1) && seen + 1 == occupied) {
            // Every valid primitive starts at height one, stays non-negative,
            // and has odd length. Immediately before its final symbol, the
            // only valid state is height one with residual rank zero; the last
            // symbol is therefore forced R.
            m |= MateID(1) << (2 * (len - 1 - pos));
            break;
        }
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

bool verify_final_state(int occupied, Rank64 initial_rank) {
    int h = 1;
    Rank64 rank = initial_rank;
    for (int seen = 0; seen < occupied - 1; ++seen) {
        const int rem = occupied - seen - 1;
        const Rank64 r_count = h > 0 ? P[rem][h - 1] : 0;
        if (rank < r_count) {
            --h;
        } else {
            rank -= r_count;
            ++h;
        }
    }
    return h == 1 && rank == 0;
}

} // namespace

int main() {
    build();
    std::uint64_t all_ranks = 0;
    for (int occupied = 1; occupied <= 27; occupied += 2) {
        const Rank64 count = P[occupied][1];
        for (Rank64 rank = 0; rank < count; ++rank) {
            ++all_ranks;
            if (!verify_final_state(occupied, rank)) {
                std::cerr << "final-state mismatch occupied=" << occupied
                          << " rank=" << rank << '\n';
                return 2;
            }
            const std::uint32_t support = occupied == 32
                ? ~0u : ((std::uint32_t(1) << occupied) - 1u);
            if (ref(support, 28, occupied, rank) !=
                last_r(support, 28, occupied, rank)) {
                std::cerr << "canonical-support mismatch occupied=" << occupied
                          << " rank=" << rank << '\n';
                return 3;
            }
        }
    }

    std::mt19937_64 rng(0x6c6173745f725f31ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int len = 1 + int(rng() % 28);
        std::uint32_t support = std::uint32_t(rng()) &
            ((std::uint32_t(1) << len) - 1u);
        int occupied = __builtin_popcount(support);
        if (!(occupied & 1)) {
            support ^= std::uint32_t(1) << int(rng() % len);
            occupied = __builtin_popcount(support);
        }
        if (!occupied) {
            support = 1u;
            occupied = 1;
        }
        const Rank64 count = P[occupied][1];
        if (!count) return 4;
        const Rank64 rank = rng() % count;
        if (ref(support, len, occupied, rank) !=
            last_r(support, len, occupied, rank)) {
            std::cerr << "random-support mismatch len=" << len
                      << " occupied=" << occupied << " support=" << support
                      << " rank=" << rank << '\n';
            return 5;
        }
    }

    if (all_ranks != 3707851ULL) {
        std::cerr << "unexpected primitive rank total=" << all_ranks << '\n';
        return 6;
    }

    std::cout << "gridfp-materialize-primitive-last-r-proof OK"
              << " occupied_max=27 all_primitive_ranks=" << all_ranks
              << " random_support_cases=" << RANDOM
              << " final_pre_state_h=1 final_pre_state_rank=0"
              << " final_symbol=R output_exact=1"
              << " saved_primitive_table_loads_per_call=1"
              << " saved_rank_branch_per_call=1\n";
    return 0;
}
