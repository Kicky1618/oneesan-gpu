#include "../../common/two_cell_fusion_rank.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

using oneesan::twocell::Rank;
using oneesan::twocell::RankTables;

struct DepthPlan {
    Rank total_states = 0;
    Rank fitted_states = 0;
    Rank fitted_blocks = 0;
    Rank total_blocks = 0;
    int max_outer_ones = -1;
};

DepthPlan make_plan(
    int W,
    int steps,
    Rank shared_bytes,
    Rank workspace_bytes,
    const RankTables& t
) {
    DepthPlan p{};
    const int outer_bits = W - steps - 3;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = t.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(steps, o, t);
        p.total_blocks += blocks;
        p.total_states += blocks * n;
        if (workspace_bytes + n * sizeof(std::uint32_t) <= shared_bytes) {
            p.fitted_blocks += blocks;
            p.fitted_states += blocks * n;
            p.max_outer_ones = o;
        }
    }
    return p;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const Rank workspace = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 4096ULL;
    if (W < 8 || W > oneesan::twocell::kMaxWidth) return 2;

    const auto t = oneesan::twocell::make_rank_tables();
    const Rank shared = shared_kib * 1024ULL;
    double best_reduction = -1.0;
    int best_steps = -1;

    std::cout << std::setprecision(12);
    for (int steps = 2; steps <= oneesan::twocell::kMaxFusionSteps; ++steps) {
        if (steps > W - 3) break;
        const auto p = make_plan(W, steps, shared, workspace, t);
        const double fitted = p.total_states
            ? double(p.fitted_states) / double(p.total_states) : 0.0;
        // A k-step unfused stationary execution performs k loads+k stores per
        // state.  A fitted fused block performs one load+one store, hence the
        // fractional traffic saving on fitted states is 1-1/k.
        const double reduction = fitted * (1.0 - 1.0 / double(steps));
        if (reduction > best_reduction) {
            best_reduction = reduction;
            best_steps = steps;
        }
        std::cout << "W=" << W
                  << " steps=" << steps
                  << " outer_bits=" << (W - steps - 3)
                  << " blocks=" << p.total_blocks
                  << " states=" << p.total_states
                  << " max_fused_outer_ones=" << p.max_outer_ones
                  << " fused_state_fraction=" << fitted
                  << " HBM_reduction=" << reduction
                  << " HBM_ratio=" << (1.0 - reduction)
                  << "\n";
    }

    if (W == 28 && shared_kib == 228 && best_steps != 2) {
        std::cerr << "unexpected W28/228KiB optimum steps=" << best_steps << '\n';
        return 3;
    }
    std::cout << "BEST steps=" << best_steps
              << " HBM_reduction=" << best_reduction
              << " shared_KiB=" << shared_kib
              << " workspace_bytes=" << workspace
              << "\n";
    return 0;
}
