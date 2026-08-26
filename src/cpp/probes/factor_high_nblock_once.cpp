#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || low >= 31 || ngpu < 1) return 1;

    const U64 masks = U64(1) << low;
    if (masks < U64(ngpu)) return 2;
    const U64 jobs = masks * U64(W);
    const U64 row_workers = U64(W) * U64(ngpu);

    constexpr U64 retained_v062_config_per_job = 5;
    constexpr U64 invariant_nblock_per_job = 2;
    constexpr U64 variable_config_per_job =
        retained_v062_config_per_job - invariant_nblock_per_job;

    const U64 old_nblock_calls = jobs * invariant_nblock_per_job;
    const U64 new_nblock_calls = row_workers * invariant_nblock_per_job;
    const U64 removed_nblock_calls = old_nblock_calls - new_nblock_calls;
    const U64 old_config_calls = jobs * retained_v062_config_per_job;
    const U64 new_config_calls = jobs * variable_config_per_job + new_nblock_calls;
    const U64 removed_config_calls = old_config_calls - new_config_calls;

    if (W == 28 && low == 14 && ngpu == 8) {
        if (jobs != 458752ULL
            || row_workers != 224ULL
            || old_nblock_calls != 917504ULL
            || new_nblock_calls != 448ULL
            || removed_nblock_calls != 917056ULL
            || old_config_calls != 2293760ULL
            || new_config_calls != 1376704ULL
            || removed_config_calls != 917056ULL) {
            std::cerr << "n=27 HIGH nblock-once regression mismatch\n";
            return 3;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-nblock-once W=" << W
              << " low=" << low << " high=" << high
              << " ngpu=" << ngpu << '\n'
              << "high_jobs_per_residue=" << jobs
              << " row_workers_per_residue=" << row_workers << '\n'
              << "old_nblock_calls_per_residue=" << old_nblock_calls
              << " new_nblock_calls_per_residue=" << new_nblock_calls
              << " removed_nblock_calls_per_residue=" << removed_nblock_calls
              << " nblock_reduction_pct="
              << 100.0 * double(removed_nblock_calls) / double(old_nblock_calls)
              << '\n'
              << "old_retained_config_calls_per_residue=" << old_config_calls
              << " new_retained_config_calls_per_residue=" << new_config_calls
              << " removed_config_calls_per_residue=" << removed_config_calls
              << " total_retained_config_reduction_pct="
              << 100.0 * double(removed_config_calls) / double(old_config_calls)
              << '\n';
    return 0;
}
