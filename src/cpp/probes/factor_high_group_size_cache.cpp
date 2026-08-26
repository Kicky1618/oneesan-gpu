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

    const U64 masks = U64(1) << low;
    const U64 jobs = masks * U64(W);
    const U64 baseline_runtime_make_spec = jobs * 2;
    const U64 cached_runtime_make_spec = 0;
    const U64 setup_make_spec = masks * 2;
    const U64 cache_bytes = masks * 2 * sizeof(U64);

    if (W == 28 && low == 14) {
        if (masks != 16384ULL
            || jobs != 458752ULL
            || baseline_runtime_make_spec != 917504ULL
            || setup_make_spec != 32768ULL
            || cache_bytes != 262144ULL) {
            std::cerr << "n=27 HIGH group-size cache regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-group-size-cache W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "masks=" << masks
              << " high_jobs_per_residue=" << jobs << '\n'
              << "baseline_runtime_make_spec_per_residue="
              << baseline_runtime_make_spec
              << " cached_runtime_make_spec_per_residue="
              << cached_runtime_make_spec
              << " setup_make_spec=" << setup_make_spec << '\n'
              << "cache_bytes=" << cache_bytes
              << " cache_mib="
              << double(cache_bytes) / double(1ULL << 20) << '\n'
              << "runtime_make_spec_reduction_pct=100.000000000\n";
    return 0;
}
