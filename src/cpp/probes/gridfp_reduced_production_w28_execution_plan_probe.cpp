#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_two_window_schedule_probe_main_unused
#include "gridfp_reduced_production_two_window_schedule_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <iomanip>

namespace {

Rank execution_component_group(int L, int outer_ones) {
    __uint128_t z = 0;
    for (int local = 0; local <= L - 1; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        z += __uint128_t(choose_u64(L - 1, local) - choose_u64(L - 3, local)) *
             catalan((occupied + 1) / 2);
    }
    return static_cast<Rank>(z);
}

Rank execution_turn_compress_group(int L, int outer_ones) {
    __uint128_t z = 0;
    for (int local = 0; local <= L - 1; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        z += __uint128_t(choose_u64(L - 1, local)) * catalan((occupied + 1) / 2);
    }
    return static_cast<Rank>(z);
}

Rank fixed_blocked_shift_states_k13() {
    __uint128_t z = 0;
    for (int r = 0; r <= 13; ++r)
        z += __uint128_t(choose_u64(13, r)) * catalan(r + 1);
    return static_cast<Rank>(z);
}

} // namespace

int main() {
    constexpr int W = 28;
    constexpr int K = 13;
    constexpr int L = 15;
    constexpr int O = 13;
    constexpr int NGPU = 8;
    constexpr Rank MAIN = 385719506620ULL;
    constexpr Rank DIM = 473397057701ULL;
    constexpr Rank COMPONENTS = 118389089432ULL;
    constexpr Rank M27 = 135015505407ULL;
    constexpr int REDUCED_STEPS_PER_ROW = 25;

    const OwnerPlan owner = make_owner_plan(W, K, NGPU);
    const auto h = owner_hist_k13(NGPU);

    std::array<Rank,NGPU> component_owner{};
    std::array<Rank,NGPU> compress_owner{};
    for (int g = 0; g < NGPU; ++g) {
        for (int r = 0; r <= O; ++r) {
            component_owner[static_cast<std::size_t>(g)] +=
                h[static_cast<std::size_t>(g)][static_cast<std::size_t>(r)] *
                execution_component_group(L, r);
            compress_owner[static_cast<std::size_t>(g)] +=
                h[static_cast<std::size_t>(g)][static_cast<std::size_t>(r)] *
                execution_turn_compress_group(L, r);
        }
    }

    Rank component_sum = 0, compress_sum = 0;
    for (Rank z : component_owner) component_sum += z;
    for (Rank z : compress_owner) compress_sum += z;
    if (component_sum != COMPONENTS) fail("execution component count");
    if (compress_sum != M27) fail("execution turn compress count");

    Rank redistribution_total = 0, redistribution_moved = 0;
    for (int so = 0; so < NGPU; ++so) for (int a = 0; a <= K; ++a) {
        const Rank ca = h[static_cast<std::size_t>(so)][static_cast<std::size_t>(a)];
        for (int d = 0; d < NGPU; ++d) for (int b = 0; b <= K; ++b) {
            const Rank cb = h[static_cast<std::size_t>(d)][static_cast<std::size_t>(b)];
            const Rank term = ca * cb * overlap_weight(a + b);
            redistribution_total += term;
            if (so != d) redistribution_moved += term;
        }
    }
    if (redistribution_total != DIM) fail("execution redistribution dimension");

    const Rank fixed_blocked = fixed_blocked_shift_states_k13();
    if (fixed_blocked >= DIM) fail("execution fixed blocked range");
    const Rank rotated_states = DIM - fixed_blocked;

    const __uint128_t interior_bytes = __uint128_t(REDUCED_STEPS_PER_ROW) * 8 * DIM;
    const __uint128_t turn_bytes = __uint128_t(8) * (DIM + MAIN);
    const __uint128_t redistribution_hbm_bytes = __uint128_t(8) * rotated_states;
    const __uint128_t total_hbm_bytes = interior_bytes + turn_bytes + redistribution_hbm_bytes;
    const __uint128_t p2p_payload_bytes = __uint128_t(4) * redistribution_moved;

    auto to_double = [](__uint128_t x) { return static_cast<double>(x); };
    const double tib = double(1ULL << 40);
    const double gib = double(1ULL << 30);

    std::cout << std::setprecision(12);
    std::cout << "W=28 execution_plan=two_window_k13_single_stream"
              << " reduced_states=" << DIM
              << " main_states=" << MAIN
              << " components_per_reduced_step=" << COMPONENTS
              << " reduced_steps_per_row=" << REDUCED_STEPS_PER_ROW
              << " redistributions_per_row=1"
              << " turn_compress_components=" << M27
              << " turn_expand_components=" << COMPONENTS
              << "\n";

    for (int g = 0; g < NGPU; ++g) {
        std::cout << "gpu=" << g
                  << " state_GiB=" << double(owner.size[static_cast<std::size_t>(g)]) * 4.0 / gib
                  << " interior_components_per_step=" << component_owner[static_cast<std::size_t>(g)]
                  << " interior_components_per_row="
                  << component_owner[static_cast<std::size_t>(g)] * Rank(REDUCED_STEPS_PER_ROW)
                  << " turn_compress_components=" << compress_owner[static_cast<std::size_t>(g)]
                  << " turn_expand_components=" << component_owner[static_cast<std::size_t>(g)]
                  << "\n";
    }

    std::cout << "aggregate_interior_HBM_TiB=" << to_double(interior_bytes) / tib
              << " aggregate_turn_HBM_TiB=" << to_double(turn_bytes) / tib
              << " aggregate_redistribution_HBM_TiB=" << to_double(redistribution_hbm_bytes) / tib
              << " aggregate_row_HBM_TiB=" << to_double(total_hbm_bytes) / tib
              << " per_gpu_row_HBM_TiB=" << to_double(total_hbm_bytes) / (NGPU * tib)
              << " ideal_row_seconds_at_64TBps=" << to_double(total_hbm_bytes) / 64.0e12
              << "\n";

    std::cout << "redistribution_fixed_blocked_states=" << fixed_blocked
              << " redistribution_rotated_states=" << rotated_states
              << " cross_owner_states=" << redistribution_moved
              << " cross_owner_fraction=" << double(redistribution_moved) / double(DIM)
              << " P2P_payload_lower_bound_TiB=" << to_double(p2p_payload_bytes) / tib
              << " direct_cycle_remote_traffic_may_exceed_payload_lower_bound=1"
              << "\n";

    std::cout << "memory_plan state_streams_per_gpu=1"
              << " second_full_state_buffer_bytes=0"
              << " component_table_bytes=0 inverse_table_bytes=0"
              << " run_table_bytes=0 visited_bytes=0"
              << " owner_component_plan_bytes_per_gpu=" << (3 * (O + 2) * sizeof(Rank))
              << " subgroup_width=8 components_per_warp=4"
              << "\n";
    std::cout << "ALL_OK production_w28_execution_plan=1\n";
    return 0;
}
