#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || low >= 31 || high < 1 || ngpu < 1) return 1;

    const U64 masks = U64(1) << low;
    const U64 unsaturated_rows = U64(W / 2 - 1);
    const U64 prefix_bytes = U64(high + 3) * 8;
    const U64 low_count_bytes = U64(high + 2) * 2;
    const U64 payload_per_job = prefix_bytes + low_count_bytes;

    U64 class_worker_visits_per_row = 0;
    for (int k = 0; k <= low; ++k)
        class_worker_visits_per_row +=
            std::min<U64>(U64(ngpu), choose_u64(low, k));

    const U64 old_jobs = masks * unsaturated_rows;
    const U64 old_calls = old_jobs * 2;
    const U64 new_calls_upper = unsaturated_rows * class_worker_visits_per_row * 2;
    const U64 old_payload = old_jobs * payload_per_job;
    const U64 new_payload_upper =
        unsaturated_rows * class_worker_visits_per_row * payload_per_job;

    if (W == 28 && low == 14 && ngpu == 8) {
        if (unsaturated_rows != 13ULL
            || class_worker_visits_per_row != 106ULL
            || payload_per_job != 158ULL
            || old_calls != 425984ULL
            || new_calls_upper != 2756ULL
            || old_payload != 33652736ULL
            || new_payload_upper != 217724ULL) {
            std::cerr << "n=27 HIGH row-plan copy-dedup regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-row-plan-copy-dedup W=" << W
              << " low=" << low << " high=" << high
              << " ngpu=" << ngpu << '\n'
              << "unsaturated_rows=" << unsaturated_rows
              << " class_worker_visits_upper_per_row="
              << class_worker_visits_per_row << '\n'
              << "payload_per_job_bytes=" << payload_per_job << '\n'
              << "old_copy_calls_per_residue=" << old_calls
              << " dedup_copy_calls_upper_per_residue=" << new_calls_upper
              << " call_reduction_lower_pct="
              << 100.0 * (1.0 - double(new_calls_upper) / double(old_calls))
              << '\n'
              << "old_payload_bytes_per_residue=" << old_payload
              << " dedup_payload_upper_bytes_per_residue=" << new_payload_upper
              << " payload_reduction_lower_pct="
              << 100.0 * (1.0 - double(new_payload_upper) / double(old_payload))
              << '\n';
    return 0;
}
