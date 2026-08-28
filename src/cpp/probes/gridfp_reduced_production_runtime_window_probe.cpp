#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_probe_main_unused
#include "gridfp_reduced_production_shift_cycle_probe.cpp"
#pragma pop_macro("main")

namespace {

void verify_w28_runtime_windows() {
    constexpr int W = 28;
    constexpr int ngpu = 8;
    Rank best_moved = std::numeric_limits<Rank>::max();
    int best_k = -1;

    for (int K = 13; K <= 20; ++K) {
        const int L = K + 2;
        const int shift = W - L;
        if (shift < 1 || shift > K) fail("runtime window shift range");
        const RunTraffic t = shifted_run_traffic_model(W, K, shift, ngpu);
        if (t.total_states != 473397057701ULL) fail("runtime window dimension");
        const LoadReport load = exact_w28_owner_load(K, ngpu);
        const auto [lo, hi] = std::minmax_element(load.states.begin(), load.states.end());
        const int high_min_p = W - K;
        const int low_max_p = K + 1;
        if (high_min_p > low_max_p + 1) fail("runtime window leaves interior gap");

        std::cout << "W=28 K=" << K
                  << " local_window=" << L
                  << " window_shift=" << shift
                  << " physical_overlap=" << (L - shift)
                  << " moved_states=" << t.moved_states
                  << " moved_fraction=" << double(t.moved_states) / double(t.total_states)
                  << " peer_TiB=" << double(t.moved_states) * 4.0 / double(1ULL << 40)
                  << " owner_min_GiB=" << double(*lo) * 4.0 / double(1ULL << 30)
                  << " owner_max_GiB=" << double(*hi) * 4.0 / double(1ULL << 30)
                  << " high_min_p=" << high_min_p
                  << " low_max_p=" << low_max_p
                  << " one_redistribution_covers_interior=1\n";
        if (t.moved_states < best_moved) {
            best_moved = t.moved_states;
            best_k = K;
        }
    }
    if (best_k != 13) fail("W28 runtime K=13 no longer traffic-optimal in tested range");
    std::cout << "W=28 best_K=" << best_k
              << " best_peer_TiB=" << double(best_moved) * 4.0 / double(1ULL << 40)
              << " two_redistributions_per_two_rows_TiB="
              << double(best_moved) * 8.0 / double(1ULL << 40)
              << " criterion=min_exact_peer_bytes_K13_to20\n";
}

} // namespace

int main() {
    verify_w28_runtime_windows();
    std::cout << "ALL_OK production_runtime_window_plan=1\n";
    return 0;
}
