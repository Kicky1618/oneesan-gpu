#include "../../common/two_cell_fusion_rank.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

using oneesan::twocell::Rank;
using oneesan::twocell::RankTables;

Rank ceil_div(Rank a, Rank b) {
    return (a + b - 1) / b;
}

int min_cluster_for_block(
    Rank states,
    Rank per_cta_limit,
    Rank per_cta_reserve,
    int max_cluster
) {
    for (int c : {1, 2, 4, 8}) {
        if (c > max_cluster) break;
        const Rank local_bytes = ceil_div(states * sizeof(std::uint32_t), Rank(c));
        if (local_bytes + per_cta_reserve <= per_cta_limit) return c;
    }
    return 0;
}

void print_plan(
    int W,
    Rank shared_limit,
    Rank reserve,
    int max_cluster,
    const RankTables& rt
) {
    const int outer_bits = W - 5;
    Rank total_states = 0;
    Rank fused_states = 0;
    Rank unfused_states = 0;
    Rank cluster_states[9]{};
    Rank cluster_blocks[9]{};
    int max_fused_outer = -1;

    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        const Rank states = blocks * n;
        total_states += states;
        const int c = min_cluster_for_block(
            n, shared_limit, reserve, max_cluster);
        if (c) {
            fused_states += states;
            cluster_states[c] += states;
            cluster_blocks[c] += blocks;
            max_fused_outer = o;
        } else {
            unfused_states += states;
        }
    }

    const double f = total_states ? double(fused_states) / double(total_states) : 0.0;
    std::cout << "fusion2_cluster_plan"
              << " W=" << W
              << " per_cta_shared_limit=" << shared_limit
              << " per_cta_reserve=" << reserve
              << " max_cluster=" << max_cluster
              << " max_fused_outer_ones=" << max_fused_outer
              << " fused_state_fraction=" << std::setprecision(12) << f
              << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
              << " fallback_state_fraction=" << double(unfused_states) / double(total_states)
              << "\n";

    for (int c : {1, 2, 4, 8}) {
        if (c > max_cluster || !cluster_states[c]) continue;
        std::cout << "  cluster=" << c
                  << " states=" << cluster_states[c]
                  << " state_fraction=" << double(cluster_states[c]) / double(total_states)
                  << " outer_blocks=" << cluster_blocks[c]
                  << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const Rank reserve = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 2048ULL;
    if (W < 6 || W > oneesan::twocell::kMaxWidth) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    for (int max_cluster : {1, 2, 4, 8})
        print_plan(W, shared_kib * 1024ULL, reserve, max_cluster, rt);

    if (W == 28 && shared_kib == 228) {
        std::cout << "W=28_reference outer15_bytes="
                  << oneesan::twocell::fusion_block_size(2, 15, rt) * 4ULL
                  << " outer16_bytes="
                  << oneesan::twocell::fusion_block_size(2, 16, rt) * 4ULL
                  << " outer17_bytes="
                  << oneesan::twocell::fusion_block_size(2, 17, rt) * 4ULL
                  << " outer18_bytes="
                  << oneesan::twocell::fusion_block_size(2, 18, rt) * 4ULL
                  << "\n";
    }
    std::cout << "NOTE cluster plan is capacity-only; actual CUDA cluster/DSM availability and bandwidth must be queried on target hardware.\n";
    return 0;
}
