#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_persistent_pipeline_plan_probe_main_unused
#include "gridfp_reduced_production_persistent_pipeline_plan_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {

struct EagerLoadStats {
    Rank a_values[NGPU][PIPE_BATCHES]{};
    Rank local_values[NGPU]{};
    Rank cross_total = 0;
    Rank local_total = 0;
};

void eager_process_main(EagerLoadStats& stats, int period, bool reverse) {
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
    for (int hop = 0; hop < period; ++hop) {
        route_owner[hop] = owner_of_support(cur, reverse);
        cur = rotate_bits(cur, W, step);
    }

    bool all_local = true;
    for (int hop = 1; hop < period; ++hop)
        all_local = all_local && route_owner[hop] == route_owner[0];
    if (all_local) {
        const Rank values = Rank(period) * pc;
        stats.local_values[route_owner[0]] += values;
        stats.local_total += values;
        return;
    }

    const int batch = main_batch16(support);
    for (int hop = 0; hop < period; ++hop)
        stats.a_values[route_owner[hop]][batch] += pc;
    stats.cross_total += Rank(period) * pc;
}

void eager_generate_necklaces(
    int t,
    int period,
    EagerLoadStats& forward,
    EagerLoadStats& reverse
) {
    if (t > W) {
        if (W % period == 0) {
            eager_process_main(forward, period, false);
            eager_process_main(reverse, period, true);
        }
        return;
    }
    necklace_bits[t] = necklace_bits[t - period];
    eager_generate_necklaces(t + 1, period, forward, reverse);
    for (int bit = necklace_bits[t - period] + 1; bit <= 1; ++bit) {
        necklace_bits[t] = bit;
        eager_generate_necklaces(t + 1, t, forward, reverse);
    }
}

void eager_blocked(EagerLoadStats& forward, EagerLoadStats& reverse) {
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
            for (EagerLoadStats* stats : {&forward, &reverse}) {
                if (owner_a == owner_b) {
                    stats->local_values[owner_a] += 2 * pc;
                    stats->local_total += 2 * pc;
                } else {
                    stats->a_values[owner_a][batch] += pc;
                    stats->a_values[owner_b][batch] += pc;
                    stats->cross_total += 2 * pc;
                }
            }
        }
    }
}

struct LoadSummary {
    Rank staged_metric = 0;
    Rank eager_bound = 0;
    double worst_batch_imbalance = 0.0;
    int worst_batch = -1;
};

LoadSummary summarize_load(const EagerLoadStats& stats) {
    LoadSummary out;
    std::array<Rank, NGPU> gpu_total{};
    Rank check_cross = 0;
    for (int batch = 0; batch < PIPE_BATCHES; ++batch) {
        Rank batch_total = 0;
        Rank batch_max = 0;
        for (int gpu = 0; gpu < NGPU; ++gpu) {
            const Rank v = stats.a_values[gpu][batch];
            batch_total += v;
            batch_max = std::max(batch_max, v);
            gpu_total[gpu] += v;
        }
        check_cross += batch_total;
        out.staged_metric += batch_max;
        const double avg = double(batch_total) / NGPU;
        const double imbalance = avg ? double(batch_max) / avg : 0.0;
        if (imbalance > out.worst_batch_imbalance) {
            out.worst_batch_imbalance = imbalance;
            out.worst_batch = batch;
        }
    }
    for (Rank v : gpu_total) out.eager_bound = std::max(out.eager_bound, v);
    if (check_cross != stats.cross_total) std::exit(21);
    return out;
}

void report_load(
    const char* direction,
    const EagerLoadStats& stats,
    double node_hbm_tbps
) {
    if (stats.cross_total != 454373526378ULL ||
        stats.local_total != 18956500538ULL)
        std::exit(22);

    const LoadSummary s = summarize_load(stats);
    if (s.staged_metric != 62070698845ULL ||
        s.eager_bound != 57545699830ULL)
        std::exit(23);

    const double local_hbm = node_hbm_tbps / NGPU * 1.0e12;
    const double staged_ms = double(s.staged_metric) * 8.0 / local_hbm * 1000.0;
    const double eager_ms = double(s.eager_bound) * 8.0 / local_hbm * 1000.0;

    std::cout << "persistent-eager-load"
              << " direction=" << direction
              << " batches=" << PIPE_BATCHES
              << " cross_A_values=" << stats.cross_total
              << " local_A_values=" << stats.local_total
              << " staged_metric=" << s.staged_metric
              << " eager_bound=" << s.eager_bound
              << " eager_A_upper_speedup="
              << double(s.staged_metric) / double(s.eager_bound)
              << " staged_A_floor_ms=" << staged_ms
              << " eager_A_bound_ms=" << eager_ms
              << " eager_A_savings_ms=" << (staged_ms - eager_ms)
              << " worst_batch=" << s.worst_batch
              << " worst_batch_imbalance=" << s.worst_batch_imbalance
              << " per_gpu_HBM_TBps=" << (node_hbm_tbps / NGPU)
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const double hbm = argc > 1 ? std::atof(argv[1]) : 64.0;
    if (!(hbm > 0.0)) return 2;

    initialize_tables();
    EagerLoadStats forward{};
    EagerLoadStats reverse{};
    necklace_bits[0] = 0;
    eager_generate_necklaces(1, 1, forward, reverse);
    eager_blocked(forward, reverse);

    for (int gpu = 0; gpu < NGPU; ++gpu) {
        if (forward.local_values[gpu] != reverse.local_values[gpu]) return 3;
        for (int batch = 0; batch < PIPE_BATCHES; ++batch)
            if (forward.a_values[gpu][batch] != reverse.a_values[gpu][batch]) return 4;
    }

    std::cout << std::fixed << std::setprecision(9);
    report_load("forward", forward, hbm);
    report_load("reverse", reverse, hbm);
    std::cout << "ALL_OK production_persistent_eager_load=1"
              << " direction_load_identical=1"
              << " exact_W28=1\n";
    return 0;
}
