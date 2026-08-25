#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || high >= 31) return 1;

    // For the production balanced split (W=28, LOW=14, HIGH=13), every HIGH
    // occupancy mask has at least one valid prefix and LOW continuation, so all
    // 2^HIGH groups are processed on every row.
    const std::uint64_t groups_per_row = std::uint64_t(1) << high;
    const std::uint64_t groups_per_residue = groups_per_row * std::uint64_t(W);
    const std::uint64_t baseline_waits =
        groups_per_residue * std::uint64_t(2 * low);
    const std::uint64_t pair_waits =
        groups_per_residue * std::uint64_t(low);
    const std::uint64_t group_waits = groups_per_residue;
    const std::uint64_t group_skips = baseline_waits - group_waits;

    if (W == 28 && low == 14) {
        if (groups_per_row != 8192ULL
            || groups_per_residue != 229376ULL
            || baseline_waits != 6422528ULL
            || pair_waits != 3211264ULL
            || group_waits != 229376ULL
            || group_skips != 6193152ULL) {
            std::cerr << "n=27 LOW group sync regression\n";
            return 2;
        }
    }

    std::cout << "low-group-sync W=" << W
              << " low=" << low
              << " high=" << high
              << " groups_per_residue=" << groups_per_residue
              << " baseline_waits=" << baseline_waits
              << " pair_waits=" << pair_waits
              << " group_waits=" << group_waits
              << " group_skips=" << group_skips
              << " baseline_reduction="
              << (baseline_waits
                    ? 1.0 - double(group_waits) / double(baseline_waits)
                    : 0.0)
              << " pair_reduction="
              << (pair_waits
                    ? 1.0 - double(group_waits) / double(pair_waits)
                    : 0.0)
              << '\n';
    return 0;
}
