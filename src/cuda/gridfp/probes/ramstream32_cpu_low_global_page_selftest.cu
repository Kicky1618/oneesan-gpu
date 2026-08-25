#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page_global.hpp"

static uint64_t unique_count(std::initializer_list<uint64_t> xs) {
    std::vector<uint64_t> v(xs);
    return cpu_low_domain_global_unique_count(v);
}

int main() {
    // Two local boundary signatures share page 2. Summing local cardinalities
    // gives 4, while the exact global union contains only {1,2,3}.
    std::vector<uint64_t> pages = {1, 2, 2, 3};
    if (cpu_low_domain_global_unique_count(pages) != 3) return 1;

    // Main/block arrays must remain distinct even when the numeric page index
    // is equal; bit 63 is the research planner's address-space tag.
    constexpr uint64_t TAG = 1ull << 63;
    if (unique_count({7, TAG | 7}) != 2) return 2;

    // A local-penalty improvement need not improve the global union: replacing
    // A={1,2} by A'={1} against B={2,3} changes local sum 4->3 but global
    // unique remains {1,2,3}=3.
    if (unique_count({1, 2, 3}) != 3) return 3;

    // A different replacement A''={3} against B={2,3} does improve the exact
    // union to {2,3}=2. This is the class of distinction the v5.24 objective
    // can detect while a per-boundary sum cannot represent exactly.
    if (unique_count({3, 2, 3}) != 2) return 4;

    CpuLowDomainGlobalPageScore a{2, 10};
    CpuLowDomainGlobalPageScore b{3, 0};
    CpuLowDomainGlobalPageScore c{2, 11};
    if (!cpu_low_domain_global_page_score_less(a, b)) return 5;
    if (!cpu_low_domain_global_page_score_less(a, c)) return 6;
    if (!cpu_low_domain_global_page_score_equal(a, CpuLowDomainGlobalPageScore{2, 10}))
        return 7;

    std::cout << "cpu-low-global-page-selftest OK"
              << " local_sum_example=4"
              << " global_unique_example=3"
              << " global_improved_example=2\n";
    return 0;
}
