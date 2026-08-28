#include <array>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
std::array<std::array<Rank64, MAX + 2>, MAX + 1> P{};
static constexpr std::uint32_t PACKED[] = {
#include "../../cuda/gridfp/gridfp_reduced_production_materialize_primitive_packed_values.inc"
};
static_assert(sizeof(PACKED) / sizeof(PACKED[0]) == 104);

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

int row_base(int rem) {
    if (rem <= 13) {
        const int m = rem >> 1;
        return (rem & 1) ? m * (m + 2) : m * m + m - 1;
    }
    const int s = 26 - rem;
    const int m = s >> 1;
    const int tail = (s & 1) ? (m + 1) * (m + 2) : (m + 1) * (m + 1);
    return 104 - tail;
}

int packed_index(int rem, int h_minus_1) {
    return row_base(rem) + (h_minus_1 >> 1);
}

bool state_represented(int rem, int h_minus_1) {
    if (rem < 1 || rem > 26 || h_minus_1 < 0) return false;
    if ((h_minus_1 & 1) != (rem & 1)) return false;
    const int max_hm1 = rem < 26 - rem ? rem : 26 - rem;
    return h_minus_1 <= max_hm1;
}

} // namespace

int main() {
    build();
    std::array<bool, 104> touched{};
    std::uint64_t all_ranks = 0;
    std::uint64_t packed_loads = 0;
    std::uint32_t max_value = 0;

    for (std::uint32_t v : PACKED) max_value = v > max_value ? v : max_value;

    for (int occupied = 1; occupied <= 27; occupied += 2) {
        const Rank64 count = P[occupied][1];
        for (Rank64 initial_rank = 0; initial_rank < count; ++initial_rank) {
            ++all_ranks;
            int h = 1;
            Rank64 rank = initial_rank;
            for (int seen = 0; seen < occupied - 1; ++seen) {
                const int rem = occupied - seen - 1;
                Rank64 r_count = 0;
                if (h > 0) {
                    const int hm1 = h - 1;
                    if (!state_represented(rem, hm1)) {
                        std::cerr << "unrepresented reachable state occupied=" << occupied
                                  << " rank=" << initial_rank << " rem=" << rem
                                  << " h=" << h << '\n';
                        return 2;
                    }
                    const int ix = packed_index(rem, hm1);
                    if (ix < 0 || ix >= int(touched.size())) return 3;
                    if (Rank64(PACKED[ix]) != P[rem][hm1]) {
                        std::cerr << "packed threshold mismatch rem=" << rem
                                  << " hm1=" << hm1 << " ix=" << ix
                                  << " packed=" << PACKED[ix]
                                  << " expected=" << P[rem][hm1] << '\n';
                        return 4;
                    }
                    touched[std::size_t(ix)] = true;
                    ++packed_loads;
                    r_count = PACKED[ix];
                }
                if (rank < r_count) {
                    --h;
                } else {
                    rank -= r_count;
                    ++h;
                }
            }
            if (h != 1 || rank != 0) {
                std::cerr << "bad final pre-state occupied=" << occupied
                          << " rank=" << initial_rank << " h=" << h
                          << " residual=" << rank << '\n';
                return 5;
            }
        }
    }

    std::size_t touched_count = 0;
    for (bool x : touched) touched_count += x ? 1u : 0u;
    if (touched_count != touched.size()) {
        std::cerr << "not all packed cells reachable touched=" << touched_count << '\n';
        return 6;
    }
    if (all_ranks != 3707851ULL) return 7;
    if (max_value != 742900u) return 8;

    std::cout << "gridfp-materialize-primitive-packed-proof OK"
              << " all_primitive_ranks=" << all_ranks
              << " reachable_threshold_cells=" << touched_count
              << " packed_entries=" << touched.size()
              << " packed_bytes=" << touched.size() * sizeof(std::uint32_t)
              << " full_u64_entries=" << (MAX + 1) * (MAX + 2)
              << " full_u64_bytes=" << (MAX + 1) * (MAX + 2) * sizeof(Rank64)
              << " packed_max_value=" << max_value
              << " packed_load_observations=" << packed_loads
              << " threshold_exact=1 final_R_assumed=1\n";
    return 0;
}
