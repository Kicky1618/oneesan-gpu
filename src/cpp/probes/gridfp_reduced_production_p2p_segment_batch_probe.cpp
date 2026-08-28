#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_segmented_rotation_probe_main_unused
#include "gridfp_reduced_production_p2p_segmented_rotation_probe.cpp"
#pragma pop_macro("main")

#include <array>

namespace {

Rank ceil_div_rank(Rank a, Rank b) {
    return (a + b - 1) / b;
}

void print_w28_segment_batch_plan(double scratch_cap_gib) {
    constexpr int W = 28, K = 13, ngpu = 8;
    const RunTraffic t = equal_run_traffic_model(W, K, ngpu);
    if (t.total_states != 473397057701ULL)
        fail("segment batch total states");

    std::array<Rank, ngpu> inbound{};
    std::array<Rank, ngpu> outbound{};
    Rank moved = 0;
    for (int s = 0; s < ngpu; ++s) {
        for (int d = 0; d < ngpu; ++d) {
            if (s == d) continue;
            inbound[static_cast<std::size_t>(d)] += t.pair_states[s][d];
            outbound[static_cast<std::size_t>(s)] += t.pair_states[s][d];
            moved += t.pair_states[s][d];
        }
    }
    if (moved != 409769189454ULL)
        fail("segment batch W28 moved states");

    // One boundary item is one complete primitive run.  W=28 has at most 27
    // occupied sites, hence primitive multiplicity Catalan(14).
    const Rank max_item_states = catalan(14);
    if (max_item_states != 2674440ULL)
        fail("segment batch max primitive multiplicity");

    Rank max_inbound = 0;
    for (int g = 0; g < ngpu; ++g) {
        max_inbound = std::max(max_inbound, inbound[static_cast<std::size_t>(g)]);
        if (inbound[static_cast<std::size_t>(g)] != outbound[static_cast<std::size_t>(g)])
            fail("segment batch symmetric traffic");
        std::cout << "W=28 gpu=" << g
                  << " inbound_states=" << inbound[static_cast<std::size_t>(g)]
                  << " inbound_GiB="
                  << double(inbound[static_cast<std::size_t>(g)]) * 4.0 /
                     double(1ULL << 30)
                  << " outbound_GiB="
                  << double(outbound[static_cast<std::size_t>(g)]) * 4.0 /
                     double(1ULL << 30)
                  << "\n";
    }

    // List scheduling independently for each destination GPU: assigning the
    // next boundary run to that GPU's least-loaded batch has makespan at most
    // average + largest item.  Destinations are independent, so the same global
    // batch index set can be used on all GPUs.
    int chosen_batches = -1;
    double chosen_bound_gib = 0.0;
    for (int batches = 2; batches <= 16; ++batches) {
        Rank worst_states = 0;
        for (int g = 0; g < ngpu; ++g) {
            const Rank avg_ceil = ceil_div_rank(
                inbound[static_cast<std::size_t>(g)], Rank(batches));
            const Rank bound = avg_ceil + max_item_states;
            worst_states = std::max(worst_states, bound);
        }
        const double bound_gib = double(worst_states) * 4.0 / double(1ULL << 30);
        std::cout << "W=28 scratch_batches=" << batches
                  << " greedy_guaranteed_max_GiB_per_gpu=" << bound_gib
                  << " max_boundary_item_MiB="
                  << double(max_item_states) * 4.0 / double(1ULL << 20)
                  << "\n";
        if (chosen_batches < 0 && bound_gib <= scratch_cap_gib) {
            chosen_batches = batches;
            chosen_bound_gib = bound_gib;
        }
    }
    if (chosen_batches < 0) fail("segment batch scratch cap too small");

    const LoadReport owner_load = exact_w28_owner_load(K, ngpu);
    const Rank max_state = *std::max_element(owner_load.states.begin(), owner_load.states.end());
    const double max_state_gib = double(max_state) * 4.0 / double(1ULL << 30);
    const double b300_288gb_decimal_gib = 288e9 / double(1ULL << 30);
    const Rank packed_one_direction_bytes = 1011352736ULL;
    const double both_schedule_gib =
        2.0 * double(packed_one_direction_bytes) / double(1ULL << 30);
    // Deliberately pessimistic: charge both complete cluster-wide schedules to
    // the single GPU that also owns the largest state shard.
    const double pessimistic_used_gib =
        max_state_gib + scratch_cap_gib + both_schedule_gib;

    std::cout << "W=28 K=13"
              << " exact_network_payload_TiB="
              << double(moved) * 4.0 / double(1ULL << 40)
              << " max_inbound_GiB="
              << double(max_inbound) * 4.0 / double(1ULL << 30)
              << " scratch_cap_GiB=" << scratch_cap_gib
              << " chosen_batches=" << chosen_batches
              << " greedy_bound_GiB=" << chosen_bound_gib
              << " kernels_per_redistribution=" << (2 * chosen_batches)
              << " global_phase_barriers=" << chosen_batches
              << " max_state_GiB=" << max_state_gib
              << " B300_288GB_decimal_GiB=" << b300_288gb_decimal_gib
              << " both_packed_schedules_cluster_GiB=" << both_schedule_gib
              << " pessimistic_state_plus_scratch_plus_all_schedule_GiB="
              << pessimistic_used_gib
              << " pessimistic_headroom_GiB="
              << (b300_288gb_decimal_gib - pessimistic_used_gib)
              << " boundary_minimal_network_candidate=1\n";

    if (scratch_cap_gib == 39.0) {
        if (chosen_batches != 5 || chosen_bound_gib >= 39.0)
            fail("W28 39GiB five-batch guarantee");
    }
}

} // namespace

int main(int argc, char** argv) {
    const double scratch_cap_gib = argc > 1 ? std::atof(argv[1]) : 39.0;
    if (!(scratch_cap_gib > 0.0)) return 2;
    print_w28_segment_batch_plan(scratch_cap_gib);
    std::cout << "ALL_OK production_p2p_segment_batch_plan=1\n";
    return 0;
}
