#include "../../common/two_cell_fusion_slices.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

using oneesan::twocell::Rank;
using oneesan::twocell::RankTables;

struct FusionCoverage {
    Rank total_states = 0;
    Rank fused_states = 0;
    Rank blocks[4]{};       // chosen cluster sizes 1,2,4,8
    Rank states[4]{};
    Rank fallback_blocks = 0;
    Rank fallback_states = 0;
    int max_fused_outer_ones = -1;

    double fraction() const {
        return total_states ? double(fused_states) / double(total_states) : 0.0;
    }
};

int cluster_slot(int c) {
    if (c == 1) return 0;
    if (c == 2) return 1;
    if (c == 4) return 2;
    if (c == 8) return 3;
    return -1;
}

Rank max_owner_states(
    int steps,
    int outer_ones,
    int owners,
    const RankTables& rt
) {
    Rank worst = 0;
    for (int owner = 0; owner < owners; ++owner) {
        Rank local = 0;
        const std::uint32_t a_codes = std::uint32_t(1) << (steps + 2);
        for (std::uint32_t code = 0; code < a_codes; ++code) {
            const Rank count = oneesan::twocell::primitive_count_for_occupied(
                outer_ones + oneesan::twocell::popcount32(code), rt);
            local += oneesan::twocell::primitive_slice_count(
                count, owner, owners);
        }
        const std::uint32_t c_codes = std::uint32_t(1) << steps;
        for (std::uint32_t code = 0; code < c_codes; ++code) {
            const Rank count = oneesan::twocell::primitive_count_for_occupied(
                outer_ones + 1 + oneesan::twocell::popcount32(code), rt);
            local += oneesan::twocell::primitive_slice_count(
                count, owner, owners);
        }
        worst = std::max(worst, local);
    }
    return worst;
}

int choose_cluster(
    int steps,
    int outer_ones,
    Rank per_cta_shared,
    Rank reserve,
    int max_cluster,
    const RankTables& rt
) {
    for (int c : {1, 2, 4, 8}) {
        if (c > max_cluster) break;
        const Rank states = max_owner_states(steps, outer_ones, c, rt);
        const Rank bytes = states * sizeof(std::uint32_t) + reserve;
        if (bytes <= per_cta_shared) return c;
    }
    return 0;
}

FusionCoverage coverage(
    int W,
    int steps,
    Rank per_cta_shared,
    Rank reserve,
    int max_cluster,
    const RankTables& rt
) {
    FusionCoverage out{};
    const int outer_bits = W - steps - 3;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank block_count = rt.choose[outer_bits][o];
        const Rank block_states = oneesan::twocell::fusion_block_size(steps, o, rt);
        const Rank weighted = block_count * block_states;
        out.total_states += weighted;

        const int c = choose_cluster(
            steps, o, per_cta_shared, reserve, max_cluster, rt);
        if (!c) {
            out.fallback_blocks += block_count;
            out.fallback_states += weighted;
            continue;
        }
        const int slot = cluster_slot(c);
        out.blocks[slot] += block_count;
        out.states[slot] += weighted;
        out.fused_states += weighted;
        out.max_fused_outer_ones = o;
    }
    return out;
}

void print_coverage(const char* name, const FusionCoverage& c) {
    static constexpr std::array<int, 4> clusters{1, 2, 4, 8};
    std::cout << name
              << " total_states=" << c.total_states
              << " fused_states=" << c.fused_states
              << " fused_fraction=" << c.fraction()
              << " pair_traffic_reduction=" << 0.5 * c.fraction()
              << " max_fused_outer_ones=" << c.max_fused_outer_ones
              << " fallback_states=" << c.fallback_states
              << '\n';
    for (int q = 0; q < 4; ++q) {
        if (!c.blocks[q]) continue;
        std::cout << "  cluster=" << clusters[q]
                  << " blocks=" << c.blocks[q]
                  << " states=" << c.states[q]
                  << " state_fraction="
                  << double(c.states[q]) / double(c.total_states)
                  << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const Rank reserve = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 4096ULL;
    const int max_cluster = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 6 || W > oneesan::twocell::kMaxWidth ||
        shared_kib == 0 || reserve >= shared_kib * 1024ULL ||
        (max_cluster != 1 && max_cluster != 2 &&
         max_cluster != 4 && max_cluster != 8)) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const Rank shared = shared_kib * 1024ULL;
    const auto interior = coverage(W, 2, shared, reserve, max_cluster, rt);
    const auto boundary = coverage(W, 1, shared, reserve, max_cluster, rt);

    // One width-28 snake row is organized as 12 two-interior-transfer pairs
    // plus one (last interior transfer + physical turn) boundary pair. More
    // generally, the number of interior pairs is (W-4)/2 when W is even.
    const int interior_pairs = (W - 4) / 2;
    const int boundary_pairs = 1;
    const int pair_count = interior_pairs + boundary_pairs;
    const double row_reduction = pair_count
        ? (double(interior_pairs) * interior.fraction() + boundary.fraction()) /
              (2.0 * double(pair_count))
        : 0.0;

    std::cout << std::setprecision(12)
              << "two-cell-row-cluster-traffic-plan"
              << " W=" << W
              << " per_cta_shared_KiB=" << shared_kib
              << " reserve_bytes=" << reserve
              << " max_cluster=" << max_cluster
              << " interior_pairs=" << interior_pairs
              << " boundary_pairs=" << boundary_pairs
              << '\n';
    print_coverage("interior_fusion2", interior);
    print_coverage("boundary_K_plus_turn", boundary);
    std::cout << "row_value_traffic_reduction=" << row_reduction
              << " row_value_traffic_ratio=" << (1.0 - row_reduction)
              << " model_only=1 measured=0\n";

    if (W == 28 && shared_kib == 228 && reserve == 4096 && max_cluster == 8) {
        // Keep a broad sanity range rather than hard-coding floating-point
        // decimals from one compiler. Current exact combinatorial planning is
        // about 47.3% reduction for the 26-operation row core.
        if (!(row_reduction > 0.47 && row_reduction < 0.48)) return 3;
    }
    return 0;
}
