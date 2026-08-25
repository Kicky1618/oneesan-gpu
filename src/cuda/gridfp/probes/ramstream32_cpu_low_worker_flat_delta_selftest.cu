#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"

int main() {
    std::unordered_map<uint64_t,uint32_t> refs{{1,2},{2,1},{4,1}};

    // The same page appears on both affected boundaries with opposite signs.
    // It must cancel before the unique-count query.
    CpuLowWorkerUniqueFlatDelta d;
    cpu_low_worker_unique_flat_delta_append(d, std::vector<uint64_t>{1,2,3}, -1);
    cpu_low_worker_unique_flat_delta_append(d, std::vector<uint64_t>{1,3,5}, +1);
    cpu_low_worker_unique_flat_delta_normalize(d);
    // page1 and page3 cancel; remaining delta is page2:-1, page5:+1.
    if (d.size() != 2) return 1;
    if (d[0].first != 2 || d[0].second != -1) return 2;
    if (d[1].first != 5 || d[1].second != +1) return 3;
    if (cpu_low_worker_unique_after_flat_delta(refs, d) != 3) return 4;

    cpu_low_worker_unique_apply_flat_delta(refs, d);
    if (refs.size() != 3) return 5;
    if (refs.count(2) != 0 || refs.count(5) != 1) return 6;
    if (refs.at(1) != 2 || refs.at(4) != 1 || refs.at(5) != 1) return 7;

    // Two removals of a doubly referenced page are legal and delete it.
    CpuLowWorkerUniqueFlatDelta twice{{1,-1},{1,-1}};
    cpu_low_worker_unique_flat_delta_normalize(twice);
    if (twice.size() != 1 || twice[0].second != -2) return 8;
    if (cpu_low_worker_unique_after_flat_delta(refs, twice) != 2) return 9;
    cpu_low_worker_unique_apply_flat_delta(refs, twice);
    if (refs.count(1) != 0 || refs.size() != 2) return 10;

    // Pure duplicate insertion merges to +2 and still adds one unique page.
    CpuLowWorkerUniqueFlatDelta add{{9,+1},{9,+1}};
    cpu_low_worker_unique_flat_delta_normalize(add);
    if (add.size() != 1 || add[0].second != 2) return 11;
    if (cpu_low_worker_unique_after_flat_delta(refs, add) != 3) return 12;

    std::cout << "cpu-low-worker-flat-delta-selftest OK"
              << " cancellation=1"
              << " double_remove=1"
              << " double_insert=1\n";
    return 0;
}
