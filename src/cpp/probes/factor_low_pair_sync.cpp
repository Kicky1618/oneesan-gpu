#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || high < 1 || high >= 31) return 1;

    const std::uint32_t nm = 1u << high;
    std::vector<std::uint8_t> high_mask_seen(nm, 0);
    auto high_rec = [&](auto&& self, int pos, int h, std::uint32_t mask) -> void {
        if (pos < 0) {
            high_mask_seen[mask] = 1;
            return;
        }
        self(self, pos - 1, h, mask);
        if (h > 0) self(self, pos - 1, h - 1, mask | (1u << pos));
        self(self, pos - 1, h + 1, mask | (1u << pos));
    };
    high_rec(high - 1, 1, 0u);

    std::uint64_t active_masks = 0;
    for (std::uint8_t x : high_mask_seen) active_masks += x != 0;

    // Every nonempty fixed-HIGH group is processed at every row and every LOW
    // position.  Baseline waits once after orbit and once after closure; pair
    // sync removes only the former.
    const std::uint64_t steps = active_masks * std::uint64_t(W) * std::uint64_t(low);
    const std::uint64_t baseline_syncs = steps * 2;
    const std::uint64_t paired_syncs = steps;
    const std::uint64_t skipped_syncs = steps;

    if (W == 28 && low == 14) {
        if (active_masks != 8192ULL
            || baseline_syncs != 6422528ULL
            || paired_syncs != 3211264ULL
            || skipped_syncs != 3211264ULL) {
            std::cerr << "n=27 LOW pair sync regression\n";
            return 2;
        }
    }

    std::cout << "low-pair-sync W=" << W
              << " low=" << low
              << " high=" << high
              << " active_masks=" << active_masks
              << " baseline_syncs=" << baseline_syncs
              << " paired_syncs=" << paired_syncs
              << " skipped_syncs=" << skipped_syncs
              << " reduction="
              << (baseline_syncs ? 1.0 - double(paired_syncs) / double(baseline_syncs) : 0.0)
              << '\n';
    return 0;
}
