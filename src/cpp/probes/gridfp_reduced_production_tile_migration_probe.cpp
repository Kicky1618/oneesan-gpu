#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_multigpu_owner_probe_main_unused
#include "gridfp_reduced_production_multigpu_owner_probe.cpp"
#pragma pop_macro("main")

namespace {

Rank overlap_state_weight(int base_occupied) {
    if (base_occupied & 1) {
        // overlap 00 and 11; both are main-only at Q_q
        return catalan((base_occupied + 1) / 2) +
               catalan((base_occupied + 3) / 2);
    }
    // Two one-bit overlap supports are main states, and the canonical 01
    // support additionally has a blocked coordinate at the tile boundary.
    return 3 * catalan((base_occupied + 2) / 2);
}

struct OwnerHist {
    std::vector<std::vector<Rank>> by_owner_ones;
};

OwnerHist owner_hist_for_common(
    std::uint32_t common_mask,
    int common_bits,
    int exclusive_bits,
    int L,
    int ngpu
) {
    const int O = common_bits + exclusive_bits;
    OwnerHist h;
    h.by_owner_ones.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<Rank>(static_cast<std::size_t>(exclusive_bits + 1), 0));
    const Rank count = Rank(1) << exclusive_bits;
    for (Rank x = 0; x < count; ++x) {
        const std::uint32_t compact = common_mask | (std::uint32_t(x) << common_bits);
        const int owner = weighted_owner(compact, L, O, ngpu);
        ++h.by_owner_ones[static_cast<std::size_t>(owner)]
                         [static_cast<std::size_t>(__builtin_popcountll(x))];
    }
    return h;
}

struct MigrationReport {
    Rank total = 0;
    Rank moved = 0;
};

MigrationReport mixed_tile_migration(int Kold, int Knew, int ngpu) {
    constexpr int W = 28;
    const int common_bits = W - (Kold + Knew + 2);
    if (common_bits < 0 || common_bits > 20) fail("migration common width");
    const int Lold = Kold + 2;
    const int Lnew = Knew + 2;

    MigrationReport out;
    const Rank common_count = Rank(1) << common_bits;
    for (Rank c = 0; c < common_count; ++c) {
        const int common_ones = __builtin_popcountll(c);
        const OwnerHist old_hist = owner_hist_for_common(
            static_cast<std::uint32_t>(c), common_bits, Knew, Lold, ngpu);
        const OwnerHist new_hist = owner_hist_for_common(
            static_cast<std::uint32_t>(c), common_bits, Kold, Lnew, ngpu);

        for (int old_owner = 0; old_owner < ngpu; ++old_owner) {
            for (int bo = 0; bo <= Knew; ++bo) {
                const Rank cb = old_hist.by_owner_ones[static_cast<std::size_t>(old_owner)]
                                                      [static_cast<std::size_t>(bo)];
                if (!cb) continue;
                for (int new_owner = 0; new_owner < ngpu; ++new_owner) {
                    for (int ao = 0; ao <= Kold; ++ao) {
                        const Rank ca = new_hist.by_owner_ones[static_cast<std::size_t>(new_owner)]
                                                          [static_cast<std::size_t>(ao)];
                        if (!ca) continue;
                        const Rank weight = overlap_state_weight(common_ones + bo + ao);
                        const __uint128_t z = __uint128_t(cb) * ca * weight;
                        if (z > std::numeric_limits<Rank>::max()) fail("migration accumulation term");
                        const Rank term = static_cast<Rank>(z);
                        out.total += term;
                        if (old_owner != new_owner) out.moved += term;
                    }
                }
            }
        }
    }
    if (out.total != 473397057701ULL) fail("migration total dimension");
    return out;
}

std::pair<Rank, Rank> owner_load_range(int K, int ngpu) {
    const LoadReport r = exact_w28_owner_load(K, ngpu);
    const auto [lo, hi] = std::minmax_element(r.states.begin(), r.states.end());
    return {*lo, *hi};
}

Rank max_group_for_k(int K) {
    constexpr int W = 28;
    const int L = K + 2;
    const int O = W - L;
    Rank best = 0;
    for (int r = 0; r <= O; ++r) best = std::max(best, fixed_outer_group_size(L, r));
    return best;
}

} // namespace

int main(int argc, char** argv) {
    const int ngpu = argc > 1 ? std::atoi(argv[1]) : 8;
    if (ngpu < 2 || ngpu > 64) return 2;

    Rank best_moved = std::numeric_limits<Rank>::max();
    int best_a = -1, best_b = -1;
    for (int a = 7; a <= 18; ++a) {
        const int b = 25 - a;
        if (b < 7 || b > 18) continue;
        const MigrationReport m = mixed_tile_migration(a, b, ngpu);
        const auto [alo, ahi] = owner_load_range(a, ngpu);
        const auto [blo, bhi] = owner_load_range(b, ngpu);
        const Rank max_group = std::max(max_group_for_k(a), max_group_for_k(b));
        const double moved_tib = double(m.moved) * 4.0 / double(1ULL << 40);
        const double moved_frac = double(m.moved) / double(m.total);
        const double min_gib = double(std::min(alo, blo)) * 4.0 / double(1ULL << 30);
        const double max_gib = double(std::max(ahi, bhi)) * 4.0 / double(1ULL << 30);
        const double max_group_gib = double(max_group) * 4.0 / double(1ULL << 30);
        std::cout << "W=28 tile_split=" << a << '+' << b
                  << " ngpu=" << ngpu
                  << " moved_states=" << m.moved
                  << " moved_fraction=" << moved_frac
                  << " redistribution_TiB=" << moved_tib
                  << " owner_min_GiB=" << min_gib
                  << " owner_max_GiB=" << max_gib
                  << " max_whole_group_GiB=" << max_group_gib
                  << " redistributions_per_interior_row=1\n";
        if (m.moved < best_moved) {
            best_moved = m.moved;
            best_a = a;
            best_b = b;
        }
    }

    std::cout << "W=28 best_tested_split=" << best_a << '+' << best_b
              << " moved_states=" << best_moved
              << " redistribution_TiB=" << double(best_moved) * 4.0 / double(1ULL << 40)
              << " criterion=min_bytes among splits 7..18"
              << "\n";
    std::cout << "ALL_OK production_tile_migration_model=1\n";
    return 0;
}
