#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_persistent_pipeline_plan_probe_main_unused
#include "gridfp_reduced_production_persistent_pipeline_plan_probe.cpp"
#pragma pop_macro("main")

#include <algorithm>
#include <iomanip>
#include <iostream>

int main() {
    initialize_tables();
    PipelineStats forward{};
    PipelineStats reverse{};
    necklace_bits[0] = 0;
    pipeline_generate_necklaces(1, 1, forward, reverse);
    pipeline_blocked(forward, reverse);

    constexpr double GiB = double(1ULL << 30);
    constexpr double B300_GiB = 288e9 / GiB;
    double worst_lower = 0.0;
    int worst_gpu = -1;

    std::cout << std::fixed << std::setprecision(9);
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        Rank outgoing_words_f = 0;
        Rank outgoing_words_r = 0;
        Rank cross_entries_f = 0;
        Rank cross_entries_r = 0;
        for (int b = 0; b < PIPE_BATCHES; ++b) {
            outgoing_words_f += forward.scratch_words[gpu][b];
            outgoing_words_r += reverse.scratch_words[gpu][b];
            cross_entries_f += forward.compressed_entries[gpu][b];
            cross_entries_r += reverse.compressed_entries[gpu][b];
        }
        if (outgoing_words_f != outgoing_words_r ||
            cross_entries_f != cross_entries_r ||
            forward.local_entries[gpu] != reverse.local_entries[gpu])
            return 2;

        const Rank list_entries =
            forward.local_entries[gpu] + cross_entries_f;
        const double state_gib = double(owner_states[gpu]) * 4.0 / GiB;
        const double dual_list_gib = double(list_entries) * 8.0 / GiB;

        // With B=8 and two scratch planes, each plane owns four batches.
        // max(even) >= sum(even)/4 and max(odd) >= sum(odd)/4, hence
        // max(even)+max(odd) >= total_outgoing/4.  Each word is u32, so the
        // byte lower bound is total_outgoing_words itself.
        const double b8_scratch_lower_gib = double(outgoing_words_f) / GiB;
        const double peak_lower =
            state_gib + dual_list_gib + b8_scratch_lower_gib;
        if (peak_lower > worst_lower) {
            worst_lower = peak_lower;
            worst_gpu = gpu;
        }

        std::cout << "persistent-b8-lower-bound"
                  << " gpu=" << gpu
                  << " outgoing_words=" << outgoing_words_f
                  << " state_GiB=" << state_gib
                  << " dual_list_GiB=" << dual_list_gib
                  << " scratch_lower_GiB=" << b8_scratch_lower_gib
                  << " peak_lower_GiB=" << peak_lower
                  << " B300_headroom_upper_GiB=" << (B300_GiB - peak_lower)
                  << '\n';
    }

    if (!(worst_lower > B300_GiB)) return 3;
    std::cout << "persistent-b8-lower-bound-summary"
              << " worst_gpu=" << worst_gpu
              << " peak_lower_GiB=" << worst_lower
              << " B300_GiB=" << B300_GiB
              << " impossible_on_B300=1"
              << " independent_of_batch_hash=1"
              << " metadata_ignored=1\n";
    std::cout << "ALL_OK production_persistent_b8_lower_bound=1"
              << " B8_double_scratch_impossible=1\n";
    return 0;
}
