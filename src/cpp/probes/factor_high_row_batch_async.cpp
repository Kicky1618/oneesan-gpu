#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

static U64 align8(U64 x) { return (x + 7ULL) & ~7ULL; }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || low >= 31 || ngpu < 1) return 1;

    const U64 masks = U64(1) << low;
    const U64 jobs = masks * U64(W);
    const U64 active_workers = std::min<U64>(masks, U64(ngpu));
    const U64 v060_waits = jobs;
    const U64 v061_waits = U64(W) * active_workers;
    const U64 legacy_waits = jobs * U64(2 * high + 2);

    const int full_cap = W / 2;
    const U64 unsaturated_rows = U64(std::max(0, full_cap - 1));
    const U64 saturated_rows = U64(W) - unsaturated_rows;
    const U64 old_plan_copy_calls = jobs * 2;
    const U64 async_plan_copy_calls = masks * unsaturated_rows * 2;
    const U64 removed_plan_copy_calls = old_plan_copy_calls - async_plan_copy_calls;

    const U64 prefix_bytes = U64(high + 3) * sizeof(U64);
    const U64 low_count_bytes = U64(high + 2) * sizeof(std::uint16_t);
    const U64 plan_bytes = align8(prefix_bytes + low_count_bytes) + sizeof(U64);
    const U64 plan_entries = masks * U64(full_cap);
    const U64 plan_cache_bytes = plan_entries * plan_bytes;

    if (W == 28 && low == 14 && ngpu == 8) {
        if (jobs != 458752ULL
            || legacy_waits != 12845056ULL
            || v060_waits != 458752ULL
            || v061_waits != 224ULL
            || unsaturated_rows != 13ULL
            || saturated_rows != 15ULL
            || old_plan_copy_calls != 917504ULL
            || async_plan_copy_calls != 425984ULL
            || removed_plan_copy_calls != 491520ULL
            || plan_bytes != 168ULL
            || plan_entries != 229376ULL
            || plan_cache_bytes != 38535168ULL) {
            std::cerr << "n=27 HIGH row-batch async regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-row-batch-async W=" << W
              << " low=" << low << " high=" << high
              << " gpus=" << ngpu << '\n'
              << "high_jobs_per_residue=" << jobs
              << " active_workers_per_row=" << active_workers << '\n'
              << "legacy_waits_per_residue=" << legacy_waits
              << " v060_waits_per_residue=" << v060_waits
              << " v061_waits_per_residue=" << v061_waits << '\n'
              << "wait_reduction_vs_v060_pct="
              << 100.0 * (1.0 - double(v061_waits) / double(v060_waits))
              << " wait_reduction_vs_legacy_pct="
              << 100.0 * (1.0 - double(v061_waits) / double(legacy_waits)) << '\n'
              << "unsaturated_rows=" << unsaturated_rows
              << " saturated_rows=" << saturated_rows << '\n'
              << "old_plan_copy_calls=" << old_plan_copy_calls
              << " async_plan_copy_calls=" << async_plan_copy_calls
              << " removed_plan_copy_calls=" << removed_plan_copy_calls << '\n'
              << "plan_cache_entries=" << plan_entries
              << " plan_bytes=" << plan_bytes
              << " plan_cache_mib="
              << double(plan_cache_bytes) / double(1ULL << 20) << '\n';
    return 0;
}
