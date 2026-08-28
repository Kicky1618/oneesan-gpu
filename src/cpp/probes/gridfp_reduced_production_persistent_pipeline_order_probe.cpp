#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_persistent_pipeline_plan_probe_main_unused
#include "gridfp_reduced_production_persistent_pipeline_plan_probe.cpp"
#pragma pop_macro("main")

#include <cmath>
#include <limits>
#include <vector>

namespace {

struct OrderResult {
    double seconds = 0.0;
    std::array<int, PIPE_BATCHES> order{};
};

OrderResult optimize_order(
    const PipelineStats& stats,
    double hbm_tbps,
    double nvlink_tbps
) {
    const double hbm = hbm_tbps * 1.0e12;
    const double nvlink = nvlink_tbps * 1.0e12;
    std::array<Rank, PIPE_BATCHES> peer{};
    std::array<double, PIPE_BATCHES> a{};
    std::array<double, PIPE_BATCHES> b_tail{};
    for (int b = 0; b < PIPE_BATCHES; ++b) {
        for (int g = 0; g < NGPU; ++g)
            peer[b] += stats.scratch_words[g][b];
        a[b] = double(stats.cycle_values[b]) * 8.0 / hbm;
        b_tail[b] = std::max(
            double(peer[b]) * 8.0 / hbm,
            double(peer[b]) * 4.0 / nvlink);
    }

    auto edge = [&](int i, int j) {
        return std::max(
            (double(peer[i]) * 8.0 +
             double(stats.cycle_values[j]) * 8.0) / hbm,
            double(peer[i]) * 4.0 / nvlink);
    };

    const int states = 1 << PIPE_BATCHES;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dp(static_cast<std::size_t>(states) * PIPE_BATCHES, INF);
    std::vector<std::int8_t> parent(
        static_cast<std::size_t>(states) * PIPE_BATCHES, std::int8_t(-1));
    auto at = [&](int mask, int last) -> double& {
        return dp[static_cast<std::size_t>(mask) * PIPE_BATCHES + last];
    };
    auto par = [&](int mask, int last) -> std::int8_t& {
        return parent[static_cast<std::size_t>(mask) * PIPE_BATCHES + last];
    };

    for (int first = 0; first < PIPE_BATCHES; ++first)
        at(1 << first, first) = a[first];

    for (int mask = 1; mask < states; ++mask) {
        for (int last = 0; last < PIPE_BATCHES; ++last) {
            const double cur = at(mask, last);
            if (!std::isfinite(cur)) continue;
            for (int next = 0; next < PIPE_BATCHES; ++next) {
                if (mask & (1 << next)) continue;
                const int nm = mask | (1 << next);
                const double cand = cur + edge(last, next);
                if (cand < at(nm, next)) {
                    at(nm, next) = cand;
                    par(nm, next) = static_cast<std::int8_t>(last);
                }
            }
        }
    }

    const int full = states - 1;
    double best = INF;
    int last = -1;
    for (int x = 0; x < PIPE_BATCHES; ++x) {
        const double cand = at(full, x) + b_tail[x];
        if (cand < best) {
            best = cand;
            last = x;
        }
    }
    if (last < 0) std::exit(31);

    OrderResult out;
    out.seconds = best +
        double(stats.local_values) * 8.0 / hbm;
    int mask = full;
    for (int pos = PIPE_BATCHES - 1; pos >= 0; --pos) {
        out.order[static_cast<std::size_t>(pos)] = last;
        const int prev = par(mask, last);
        mask ^= 1 << last;
        last = prev;
    }
    if (mask != 0) std::exit(32);
    return out;
}

double natural_floor(
    const PipelineStats& stats,
    double hbm_tbps,
    double nvlink_tbps
) {
    const double hbm = hbm_tbps * 1.0e12;
    const double nvlink = nvlink_tbps * 1.0e12;
    std::array<Rank, PIPE_BATCHES> peer{};
    for (int b = 0; b < PIPE_BATCHES; ++b)
        for (int g = 0; g < NGPU; ++g)
            peer[b] += stats.scratch_words[g][b];

    double z = double(stats.local_values) * 8.0 / hbm;
    z += double(stats.cycle_values[0]) * 8.0 / hbm;
    for (int b = 0; b + 1 < PIPE_BATCHES; ++b) {
        z += std::max(
            (double(peer[b]) * 8.0 +
             double(stats.cycle_values[b + 1]) * 8.0) / hbm,
            double(peer[b]) * 4.0 / nvlink);
    }
    z += std::max(
        double(peer[PIPE_BATCHES - 1]) * 8.0 / hbm,
        double(peer[PIPE_BATCHES - 1]) * 4.0 / nvlink);
    return z;
}

double order_peak_gib(
    const PipelineStats& stats,
    const std::array<int, PIPE_BATCHES>& order
) {
    constexpr double GiB = double(1ULL << 30);
    double worst = 0.0;
    for (int g = 0; g < NGPU; ++g) {
        Rank plane[2]{};
        Rank list_entries = stats.local_entries[g];
        for (int b = 0; b < PIPE_BATCHES; ++b)
            list_entries += stats.compressed_entries[g][b];
        for (int pos = 0; pos < PIPE_BATCHES; ++pos) {
            const int b = order[static_cast<std::size_t>(pos)];
            plane[pos & 1] = std::max(plane[pos & 1], stats.scratch_words[g][b]);
        }
        const double state = double(owner_states[g]) * 4.0 / GiB;
        const double dual_list = double(list_entries) * 8.0 / GiB;
        const double scratch = double(plane[0] + plane[1]) * 4.0 / GiB;
        const double metadata =
            double(PIPE_BATCHES * (W + 1) *
                   (MAX_SEGMENTS_PER_OWNER + 1) * 2 * sizeof(Rank)) / GiB;
        worst = std::max(worst, state + dual_list + scratch + metadata);
    }
    return worst;
}

void report_order(
    const char* direction,
    const PipelineStats& stats,
    double hbm,
    double nvlink
) {
    const double natural = natural_floor(stats, hbm, nvlink);
    const OrderResult best = optimize_order(stats, hbm, nvlink);
    const double peak = order_peak_gib(stats, best.order);
    constexpr double B300_GiB = 288e9 / double(1ULL << 30);

    std::cout << "persistent-pipeline-order"
              << " direction=" << direction
              << " batches=" << PIPE_BATCHES
              << " natural_floor_ms=" << natural * 1000.0
              << " optimized_floor_ms=" << best.seconds * 1000.0
              << " order_speedup=" << natural / best.seconds
              << " peak_GiB=" << peak
              << " B300_headroom_GiB=" << (B300_GiB - peak)
              << " order=";
    for (int i = 0; i < PIPE_BATCHES; ++i) {
        if (i) std::cout << ',';
        std::cout << best.order[static_cast<std::size_t>(i)];
    }
    std::cout << '\n';
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
    report_order("forward", forward, hbm, nvlink);
    report_order("reverse", reverse, hbm, nvlink);
    std::cout << "ALL_OK production_persistent_pipeline_order=1 exact_dp=1\n";
    return 0;
}
