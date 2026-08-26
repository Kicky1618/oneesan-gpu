#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

static U64 align8(U64 x) { return (x + 7ULL) & ~7ULL; }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || low >= 31) return 1;

    const U64 masks = U64(1) << low;
    const U64 main_blocks = U64(3) * U64(high + 2);
    const U64 blocked_blocks = U64(high + 2);
    const U64 fblock_bytes = masks * (main_blocks + blocked_blocks) * 24ULL;

    const U64 prefix_bytes = U64(high + 3) * 8ULL;
    const U64 low_count_bytes = U64(high + 2) * 2ULL;
    const U64 plan_bytes = align8(prefix_bytes + low_count_bytes) + 8ULL;
    const U64 plan_entries = masks * U64(W / 2);
    const U64 plan_bytes_total = plan_entries * plan_bytes;
    const U64 pinned_bytes = fblock_bytes + plan_bytes_total;

    if (W == 28 && low == 14) {
        if (main_blocks != 45ULL
            || blocked_blocks != 15ULL
            || fblock_bytes != 23592960ULL
            || plan_bytes != 168ULL
            || plan_entries != 229376ULL
            || plan_bytes_total != 38535168ULL
            || pinned_bytes != 62128128ULL) {
            std::cerr << "n=27 HIGH pinned-config regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-pinned-config W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "masks=" << masks
              << " main_blocks_per_mask=" << main_blocks
              << " blocked_blocks_per_mask=" << blocked_blocks << '\n'
              << "pinned_fblock_bytes=" << fblock_bytes
              << " pinned_fblock_mib="
              << double(fblock_bytes) / double(1ULL << 20) << '\n'
              << "pinned_plan_entries=" << plan_entries
              << " plan_bytes=" << plan_bytes
              << " pinned_plan_bytes=" << plan_bytes_total
              << " pinned_plan_mib="
              << double(plan_bytes_total) / double(1ULL << 20) << '\n'
              << "pinned_total_bytes=" << pinned_bytes
              << " pinned_total_mib="
              << double(pinned_bytes) / double(1ULL << 20) << '\n';
    return 0;
}
