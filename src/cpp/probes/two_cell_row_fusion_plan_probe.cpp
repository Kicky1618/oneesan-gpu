#include "../../common/two_cell_fusion_rank.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

using oneesan::twocell::Rank;
using oneesan::twocell::RankTables;

struct Fraction {
    Rank fit = 0;
    Rank total = 0;
    int max_o = -1;
    double value() const { return total ? double(fit) / double(total) : 0.0; }
};

Fraction fused_fraction(
    int W,
    int coordinate_steps,
    Rank shared_bytes,
    Rank workspace,
    const RankTables& t
) {
    Fraction f{};
    const int outer_bits = W - coordinate_steps - 3;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = t.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(coordinate_steps, o, t);
        f.total += blocks * n;
        if (workspace + n * sizeof(std::uint32_t) <= shared_bytes) {
            f.fit += blocks * n;
            f.max_o = o;
        }
    }
    return f;
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
    const auto interior = fused_fraction(W, 2, shared, workspace, t);
    const auto boundary = fused_fraction(W, 1, shared, workspace, t);

    const int interior_steps = W - 3;
    const int before_turn = interior_steps - 1; // final K is paired with the turn
    const int interior_pairs = before_turn / 2;
    const int leftover = before_turn % 2;
    const int boundary_pairs = 1;
    const int row_ops = interior_steps + 1; // all interior K plus physical turn

    // Every 2-op fused pair saves half of the baseline traffic on the states
    // whose union block fits.  An odd leftover (only for odd W) remains a
    // stationary one-step operation with no fusion saving.
    const double saved_equivalent_ops =
        double(interior_pairs) * interior.value() +
        double(boundary_pairs) * boundary.value();
    const double row_reduction = saved_equivalent_ops / double(row_ops);

    std::cout << std::setprecision(12)
              << "W=" << W
              << " row_ops=" << row_ops
              << " interior_pairs=" << interior_pairs
              << " boundary_pairs=" << boundary_pairs
              << " leftover_single=" << leftover
              << " interior_fused_state_fraction=" << interior.value()
              << " boundary_fused_state_fraction=" << boundary.value()
              << " row_HBM_reduction=" << row_reduction
              << " row_HBM_ratio=" << (1.0 - row_reduction)
              << "\n";

    if (W == 28 && shared_kib == 228) {
        if (interior_pairs != 12 || leftover != 0)
            return 3;
        const Rank R = 165727043758ULL;
        const long double baseline_bytes =
            static_cast<long double>(row_ops) * 8.0L * static_cast<long double>(R);
        const long double hybrid_bytes = baseline_bytes * (1.0L - row_reduction);
        std::cout << "W=28_theory row_baseline_TB="
                  << static_cast<double>(baseline_bytes / 1.0e12L)
                  << " row_hybrid_TB="
                  << static_cast<double>(hybrid_bytes / 1.0e12L)
                  << " roofline64TBps_baseline_s="
                  << static_cast<double>(baseline_bytes / 64.0e12L)
                  << " roofline64TBps_hybrid_s="
                  << static_cast<double>(hybrid_bytes / 64.0e12L)
                  << "\n";
    }
    std::cout << "ALL_OK paired_row_fusion_plan=1\n";
    return 0;
}
