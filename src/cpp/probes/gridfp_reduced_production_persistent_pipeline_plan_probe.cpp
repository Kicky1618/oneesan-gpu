#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_cycle_owner_compression_probe_main_unused
#include "gridfp_reduced_production_cycle_owner_compression_probe.cpp"
#pragma pop_macro("main")

namespace {

constexpr int PIPE_BATCHES = 16;

struct PipelineStats {
    Rank scratch_words[NGPU][PIPE_BATCHES]{};
    Rank compressed_entries[NGPU][PIPE_BATCHES]{};
    Rank cycle_values[PIPE_BATCHES]{};
    Rank local_entries[NGPU]{};
    Rank local_values = 0;
};

int main_batch16(Mask support) {
    Mask h = Mask(__builtin_popcount(support)) * 0x9e3779b1u;
    constexpr std::pair<int, Mask> terms[] = {
        {1, 0x85ebca6bu},
        {3, 0xc2b2ae35u},
        {5, 0x27d4eb2fu},
        {7, 0x165667b1u},
    };
    for (const auto [distance, coefficient] : terms) {
        h ^= Mask(__builtin_popcount(
                 support & rotate_bits(support, W, distance))) *
             coefficient;
    }
    return int(mix32(h) & (PIPE_BATCHES - 1));
}

int blocked_batch16(Mask a, Mask b, int occupied) {
    const Mask lo = std::min(a, b);
    const Mask hi = std::max(a, b);
    Mask h = lo * 0x9e3779b1u;
    h ^= hi * 0x85ebca6bu;
    h ^= Mask(occupied) * 0xc2b2ae35u;
    return int(mix32(h) & (PIPE_BATCHES - 1));
}

void pipeline_process_main(
    PipelineStats& stats,
    int period,
    bool reverse
) {
    int occupied = 0;
    Mask support = 0;
    for (int i = 1; i <= W; ++i) {
        occupied += necklace_bits[i];
        if (necklace_bits[i]) support |= Mask(1) << (i - 1);
    }
    if (!(occupied & 1) || period <= 1) return;

    const Rank pc = primitive[occupied][1];
    const int step = reverse ? 15 : 13;
    int route_owner[W]{};
    Mask cur = support;
    for (int h = 0; h < period; ++h) {
        route_owner[h] = owner_of_support(cur, reverse);
        cur = rotate_bits(cur, W, step);
    }

    bool all_local = true;
    for (int h = 1; h < period; ++h)
        all_local = all_local && route_owner[h] == route_owner[0];
    if (all_local) {
        ++stats.local_entries[route_owner[0]];
        stats.local_values += Rank(period) * pc;
        return;
    }

    int segments_per_owner[NGPU]{};
    int h = 0;
    while (h < period) {
        const int owner = route_owner[h];
        int len = 1;
        while (h + len < period && route_owner[h + len] == owner) ++len;
        ++segments_per_owner[owner];
        h += len;
    }
    if (route_owner[0] == route_owner[period - 1])
        --segments_per_owner[route_owner[0]];

    const int batch = main_batch16(support);
    stats.cycle_values[batch] += Rank(period) * pc;
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        const int segments = segments_per_owner[gpu];
        if (!segments) continue;
        ++stats.compressed_entries[gpu][batch];
        stats.scratch_words[gpu][batch] += Rank(segments) * pc;
    }
}

void pipeline_generate_necklaces(
    int t,
    int period,
    PipelineStats& forward,
    PipelineStats& reverse
) {
    if (t > W) {
        if (W % period == 0) {
            pipeline_process_main(forward, period, false);
            pipeline_process_main(reverse, period, true);
        }
        return;
    }
    necklace_bits[t] = necklace_bits[t - period];
    pipeline_generate_necklaces(t + 1, period, forward, reverse);
    for (int bit = necklace_bits[t - period] + 1; bit <= 1; ++bit) {
        necklace_bits[t] = bit;
        pipeline_generate_necklaces(t + 1, t, forward, reverse);
    }
}

void pipeline_blocked(
    PipelineStats& forward,
    PipelineStats& reverse
) {
    for (Mask a = 0; a < (Mask(1) << K); ++a) {
        for (Mask b = 0; b <= a; ++b) {
            const int free_occupied =
                __builtin_popcount(a) + __builtin_popcount(b);
            if (free_occupied & 1) continue;
            if (a == b) continue;

            const int owner_a = owner_lut[a];
            const int owner_b = owner_lut[b];
            const int occupied = free_occupied + 1;
            const Rank pc = primitive[occupied][1];
            const int batch = blocked_batch16(a, b, occupied);
            for (PipelineStats* stats : {&forward, &reverse}) {
                if (owner_a == owner_b) {
                    ++stats->local_entries[owner_a];
                    stats->local_values += 2 * pc;
                } else {
                    stats->cycle_values[batch] += 2 * pc;
                    for (const int owner : {owner_a, owner_b}) {
                        ++stats->compressed_entries[owner][batch];
                        stats->scratch_words[owner][batch] += pc;
                    }
                }
            }
        }
    }
}

void report_pipeline(
    const char* direction,
    const PipelineStats& stats,
    double node_hbm_tbps,
    double node_nvlink_tbps
) {
    Rank peer_by_batch[PIPE_BATCHES]{};
    Rank total_peer = 0;
    Rank total_cycle_values = 0;
    Rank total_local_entries = 0;
    for (int batch = 0; batch < PIPE_BATCHES; ++batch) {
        total_cycle_values += stats.cycle_values[batch];
        for (int gpu = 0; gpu < NGPU; ++gpu)
            peer_by_batch[batch] += stats.scratch_words[gpu][batch];
        total_peer += peer_by_batch[batch];
    }
    for (int gpu = 0; gpu < NGPU; ++gpu)
        total_local_entries += stats.local_entries[gpu];

    if (total_peer != 409769189454ULL ||
        stats.local_values != 18956500538ULL ||
        total_cycle_values + stats.local_values != 473330026916ULL ||
        total_local_entries != 5910700ULL)
        std::exit(11);

    constexpr double GiB = double(1ULL << 30);
    constexpr double B300_GiB = 288e9 / GiB;
    double worst_peak = 0.0;
    int worst_gpu = -1;
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        Rank even_scratch = 0;
        Rank odd_scratch = 0;
        Rank list_entries = stats.local_entries[gpu];
        for (int batch = 0; batch < PIPE_BATCHES; ++batch) {
            if (batch & 1)
                odd_scratch = std::max(odd_scratch, stats.scratch_words[gpu][batch]);
            else
                even_scratch = std::max(even_scratch, stats.scratch_words[gpu][batch]);
            list_entries += stats.compressed_entries[gpu][batch];
        }

        const double state_gib = double(owner_states[gpu]) * 4.0 / GiB;
        const double dual_list_gib = double(list_entries) * 8.0 / GiB;
        const double scratch0_gib = double(even_scratch) * 4.0 / GiB;
        const double scratch1_gib = double(odd_scratch) * 4.0 / GiB;
        const double metadata_gib =
            double(PIPE_BATCHES * (W + 1) *
                   (MAX_SEGMENTS_PER_OWNER + 1) * 2 * sizeof(Rank)) /
            GiB;
        const double peak = state_gib + dual_list_gib +
            scratch0_gib + scratch1_gib + metadata_gib;
        if (peak > worst_peak) {
            worst_peak = peak;
            worst_gpu = gpu;
        }
        std::cout << "persistent-pipeline-owner"
                  << " direction=" << direction
                  << " gpu=" << gpu
                  << " state_GiB=" << state_gib
                  << " dual_list_MiB=" << dual_list_gib * 1024.0
                  << " scratch_even_GiB=" << scratch0_gib
                  << " scratch_odd_GiB=" << scratch1_gib
                  << " peak_GiB=" << peak
                  << " B300_headroom_GiB=" << (B300_GiB - peak)
                  << '\n';
    }

    const double hbm = node_hbm_tbps * 1.0e12;
    const double nvlink = node_nvlink_tbps * 1.0e12;
    const double local_seconds =
        double(stats.local_values) * 2.0 * 4.0 / hbm;
    double sequential = local_seconds;
    for (int batch = 0; batch < PIPE_BATCHES; ++batch) {
        const double a_hbm =
            double(stats.cycle_values[batch]) * 2.0 * 4.0;
        const double b_hbm = double(peer_by_batch[batch]) * 2.0 * 4.0;
        const double b_nvlink = double(peer_by_batch[batch]) * 4.0;
        sequential += a_hbm / hbm + std::max(b_hbm / hbm, b_nvlink / nvlink);
    }

    double pipelined = local_seconds +
        double(stats.cycle_values[0]) * 2.0 * 4.0 / hbm;
    for (int batch = 0; batch + 1 < PIPE_BATCHES; ++batch) {
        const double b_hbm = double(peer_by_batch[batch]) * 2.0 * 4.0;
        const double b_nvlink = double(peer_by_batch[batch]) * 4.0;
        const double next_a_hbm =
            double(stats.cycle_values[batch + 1]) * 2.0 * 4.0;
        pipelined += std::max(
            (b_hbm + next_a_hbm) / hbm,
            b_nvlink / nvlink);
    }
    const int last = PIPE_BATCHES - 1;
    pipelined += std::max(
        double(peer_by_batch[last]) * 2.0 * 4.0 / hbm,
        double(peer_by_batch[last]) * 4.0 / nvlink);

    std::cout << "persistent-pipeline-plan"
              << " direction=" << direction
              << " batches=" << PIPE_BATCHES
              << " logical_peer_values=" << total_peer
              << " cross_cycle_values=" << total_cycle_values
              << " local_cycle_values=" << stats.local_values
              << " worst_gpu=" << worst_gpu
              << " peak_GiB=" << worst_peak
              << " B300_headroom_GiB=" << (B300_GiB - worst_peak)
              << " node_HBM_TBps=" << node_hbm_tbps
              << " node_NVLink_TBps=" << node_nvlink_tbps
              << " sequential_floor_ms=" << sequential * 1000.0
              << " pipelined_floor_ms=" << pipelined * 1000.0
              << " ideal_pipeline_speedup=" << sequential / pipelined
              << " double_scratch=1 descriptor_bytes=0"
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const double hbm = argc > 1 ? std::atof(argv[1]) : 64.0;
    const double nvlink = argc > 2 ? std::atof(argv[2]) : 14.4;
    if (!(hbm > 0.0) || !(nvlink > 0.0)) return 2;

    initialize_tables();
    PipelineStats forward{};
    PipelineStats reverse{};
    necklace_bits[0] = 0;
    pipeline_generate_necklaces(1, 1, forward, reverse);
    pipeline_blocked(forward, reverse);

    std::cout << std::fixed << std::setprecision(9);
    report_pipeline("forward", forward, hbm, nvlink);
    report_pipeline("reverse", reverse, hbm, nvlink);
    std::cout << "ALL_OK production_persistent_pipeline_plan=1\n";
    return 0;
}
