#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Rank = std::uint64_t;
static constexpr int MAX_W = 28;

Rank choose_u64(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    Rank x = 1;
    for (int i = 1; i <= k; ++i)
        x = x * Rank(n - k + i) / Rank(i);
    return x;
}

std::array<std::array<Rank, MAX_W + 2>, MAX_W + 1> primitive_table() {
    std::array<std::array<Rank, MAX_W + 2>, MAX_W + 1> p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= MAX_W; ++rem) {
        for (int h = 0; h <= MAX_W; ++h) {
            Rank z = p[rem - 1][h + 1];
            if (h) z += p[rem - 1][h - 1];
            p[rem][h] = z;
        }
    }
    return p;
}

Rank state_group_size(
    const std::array<std::array<Rank, MAX_W + 2>, MAX_W + 1>& p,
    int L,
    int outer_ones
) {
    Rank total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        const Rank supports =
            choose_u64(L, local) + choose_u64(L - 2, local - 1);
        total += supports * p[occupied][1];
    }
    return total;
}

Rank support_slab_group_size(int L, int outer_ones) {
    Rank total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        total += choose_u64(L, local) + choose_u64(L - 2, local - 1);
    }
    return total;
}

struct OwnerMemory {
    Rank states = 0;
    Rank groups = 0;
    Rank slabs = 0;
};

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int K = argc > 2 ? std::atoi(argv[2]) : 13;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    if (W < 8 || W > MAX_W || K < 1 || K + 2 > W ||
        ngpu < 2 || ngpu > 64) return 2;

    const int L = K + 2;
    const int O = W - L;
    const auto primitive = primitive_table();
    std::vector<Rank> state_group(static_cast<std::size_t>(O + 1));
    std::vector<Rank> slab_group(static_cast<std::size_t>(O + 1));
    Rank total_states = 0;
    Rank total_slabs = 0;
    for (int r = 0; r <= O; ++r) {
        state_group[static_cast<std::size_t>(r)] =
            state_group_size(primitive, L, r);
        slab_group[static_cast<std::size_t>(r)] =
            support_slab_group_size(L, r);
        total_states += choose_u64(O, r) * state_group[static_cast<std::size_t>(r)];
        total_slabs += choose_u64(O, r) * slab_group[static_cast<std::size_t>(r)];
    }

    std::vector<OwnerMemory> owner(static_cast<std::size_t>(ngpu));
    Rank prefix = 0;
    for (int r = 0; r <= O; ++r) {
        const Rank group = state_group[static_cast<std::size_t>(r)];
        const Rank slabs = slab_group[static_cast<std::size_t>(r)];
        const Rank count = choose_u64(O, r);
        for (Rank sr = 0; sr < count; ++sr) {
            const __uint128_t midpoint =
                __uint128_t(prefix) + __uint128_t(sr) * group + group / 2;
            int g = static_cast<int>(midpoint * Rank(ngpu) / total_states);
            if (g >= ngpu) g = ngpu - 1;
            owner[static_cast<std::size_t>(g)].states += group;
            owner[static_cast<std::size_t>(g)].groups += 1;
            owner[static_cast<std::size_t>(g)].slabs += slabs;
        }
        prefix += count * group;
    }

    Rank check_states = 0, check_groups = 0, check_slabs = 0;
    for (const auto& x : owner) {
        check_states += x.states;
        check_groups += x.groups;
        check_slabs += x.slabs;
    }
    if (check_states != total_states || check_groups != (Rank(1) << O) ||
        check_slabs != total_slabs) return 3;

    if (W == 28 && K == 13) {
        if (total_states != 473397057701ULL) return 4;
        if (total_slabs != 5ULL * (1ULL << 25)) return 5;
    }

    const double GiB = double(1ULL << 30);
    const double MiB = double(1ULL << 20);
    const double b300_gib = 288e9 / GiB;
    double worst_combined = 0.0;
    int worst_gpu = -1;

    std::cout << std::fixed << std::setprecision(6);
    for (int g = 0; g < ngpu; ++g) {
        const auto& x = owner[static_cast<std::size_t>(g)];
        const double state_gib = double(x.states) * 4.0 / GiB;
        // One packed u32 entry is sufficient for every support slab.  The
        // actual persistent cross-segment + all-local-leader lists are strict
        // subsets, so this is a deterministic allocation upper bound.
        const double list_upper_mib = double(x.slabs) * 4.0 / MiB;
        const double combined_gib = state_gib + double(x.slabs) * 4.0 / GiB;
        const double headroom_gib = b300_gib - combined_gib;
        if (combined_gib > worst_combined) {
            worst_combined = combined_gib;
            worst_gpu = g;
        }
        std::cout << "persistent-memory-owner"
                  << " W=" << W
                  << " K=" << K
                  << " ngpu=" << ngpu
                  << " gpu=" << g
                  << " groups=" << x.groups
                  << " support_slabs=" << x.slabs
                  << " state_GiB=" << state_gib
                  << " persistent_list_upper_MiB=" << list_upper_mib
                  << " state_plus_list_upper_GiB=" << combined_gib
                  << " B300_headroom_before_scratch_GiB=" << headroom_gib
                  << '\n';
    }

    std::cout << "gridfp-persistent-memory-bound"
              << " W=" << W
              << " K=" << K
              << " local_window=" << L
              << " outer_bits=" << O
              << " ngpu=" << ngpu
              << " states=" << total_states
              << " support_slabs=" << total_slabs
              << " persistent_entry_bytes=4"
              << " persistent_list_total_upper_MiB="
              << double(total_slabs) * 4.0 / MiB
              << " worst_gpu=" << worst_gpu
              << " worst_state_plus_list_upper_GiB=" << worst_combined
              << " B300_GiB=" << b300_gib
              << " worst_B300_headroom_before_scratch_GiB="
              << (b300_gib - worst_combined)
              << " list_upper_bound_exact=1 table_bytes=0\n";
    return 0;
}
