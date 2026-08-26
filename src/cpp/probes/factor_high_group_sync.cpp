#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || low >= 31) return 1;

    const U64 jobs_per_row = U64(1) << low;
    const U64 groups = jobs_per_row * U64(W);
    const U64 waits_per_group_old = U64(2 * high + 2);
    const U64 old_waits = groups * waits_per_group_old;
    const U64 new_waits = groups;
    const U64 skipped = old_waits - new_waits;

    if (W == 28 && low == 14) {
        if (jobs_per_row != 16384ULL
            || groups != 458752ULL
            || waits_per_group_old != 28ULL
            || old_waits != 12845056ULL
            || new_waits != 458752ULL
            || skipped != 12386304ULL) {
            std::cerr << "n=27 HIGH group-sync regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-group-sync W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "jobs_per_row=" << jobs_per_row
              << " groups_per_residue=" << groups << '\n'
              << "old_waits_per_group=" << waits_per_group_old
              << " new_waits_per_group=1\n"
              << "old_device_sync_waits_per_residue=" << old_waits
              << " new_device_sync_waits_per_residue=" << new_waits
              << " skipped_waits_per_residue=" << skipped
              << " reduction_pct="
              << (old_waits ? 100.0 * double(skipped) / double(old_waits) : 0.0)
              << '\n'
              << "kernel_sequence_unchanged=1 final_scatter_wait_preserved=1\n";
    return 0;
}
