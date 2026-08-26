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

    const U64 jobs = (U64(1) << low) * U64(W);
    constexpr U64 old_copies_per_job = 12;
    constexpr U64 retained_copies_per_job = 5;
    constexpr U64 removed_copies_per_job =
        old_copies_per_job - retained_copies_per_job;
    const U64 old_calls = jobs * old_copies_per_job;
    const U64 retained_calls = jobs * retained_copies_per_job;
    const U64 removed_calls = jobs * removed_copies_per_job;

    if (W == 28 && low == 14) {
        if (jobs != 458752ULL
            || old_calls != 5505024ULL
            || retained_calls != 2293760ULL
            || removed_calls != 3211264ULL) {
            std::cerr << "n=27 HIGH dead-symbol-copy regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-dead-symbol-copies W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "high_jobs_per_residue=" << jobs << '\n'
              << "old_symbol_copies_per_job=" << old_copies_per_job
              << " retained_symbol_copies_per_job=" << retained_copies_per_job
              << " removed_symbol_copies_per_job=" << removed_copies_per_job << '\n'
              << "old_symbol_copy_calls_per_residue=" << old_calls
              << " retained_symbol_copy_calls_per_residue=" << retained_calls
              << " removed_symbol_copy_calls_per_residue=" << removed_calls
              << " reduction_pct="
              << 100.0 * double(removed_calls) / double(old_calls) << '\n'
              << "removed_symbols=D_F_FIX_LOW,D_MAIN_FIXED,D_MAIN_OCC,D_BLOCK_FIXED,D_BLOCK_OCC,D_MAIN_DP,D_BLOCK_DP\n";
    return 0;
}
