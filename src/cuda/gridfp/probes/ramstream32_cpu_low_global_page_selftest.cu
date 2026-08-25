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
    // Validate the successor index against the original linear mask scan for
    // every relevant endpoint height and every threshold, including nmasks.
    build_full_dp();
    G_FACTOR = build_factor_tables();
    CpuLowDomainPageMaskIndex index = cpu_low_build_domain_page_mask_index();
    constexpr uint32_t S = FactorTablesHost::STRIDE;
    const uint32_t nmasks = uint32_t(1) << HIGH_LUT_K;
    uint64_t mask_index_checked = 0;
    for (uint32_t h = 0; h <= uint32_t(HIGH_LUT_K + 1); ++h) {
        uint32_t first = nmasks;
        for (uint32_t mask = 0; mask < nmasks; ++mask) {
            size_t ix = size_t(mask) * S + h;
            if (G_FACTOR.high_mask_off[ix + 1] != G_FACTOR.high_mask_off[ix]) {
                first = mask;
                break;
            }
        }
        if (index.first_nonempty[h] != first) return 10;
        for (uint32_t threshold = 0; threshold <= nmasks; ++threshold) {
            uint32_t want = nmasks;
            for (uint32_t mask = threshold; mask < nmasks; ++mask) {
                size_t ix = size_t(mask) * S + h;
                if (G_FACTOR.high_mask_off[ix + 1] != G_FACTOR.high_mask_off[ix]) {
                    want = mask;
                    break;
                }
            }
            if (index.next(h, threshold) != want) return 11;
            ++mask_index_checked;
        }
    }

    // Rows outside StorageLayout's HIGH endpoint range remain sentinel-only.
    for (uint32_t h = uint32_t(HIGH_LUT_K + 2); h < index.stride; ++h) {
        if (index.first_nonempty[h] != nmasks) return 12;
    }

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
              << " mask_index_checked=" << mask_index_checked
              << " max_indexed_height=" << index.max_indexed_height
              << " local_sum_example=4"
              << " global_unique_example=3"
              << " global_improved_example=2\n";
    return 0;
}
