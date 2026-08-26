#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_worker_dense_page.hpp"

#include <iostream>
#include <vector>

int main() {
    std::vector<CpuLowDomainGlobalPageSignature> raw(3);
    raw[1].pages_2m = {100, 200};
    raw[1].pages_4k = {10, 20};
    raw[2].pages_2m = {200, 300};
    raw[2].pages_4k = {20, 30};

    CpuLowWorkerDensePageIndex index =
        cpu_low_build_worker_dense_page_index_from_raw(raw);
    if (index.universe_2m != std::vector<uint64_t>({100, 200, 300})) return 1;
    if (index.universe_4k != std::vector<uint64_t>({10, 20, 30})) return 2;
    if (index.boundary[1].pages_2m != std::vector<uint32_t>({0, 1})) return 3;
    if (index.boundary[2].pages_2m != std::vector<uint32_t>({1, 2})) return 4;
    if (index.boundary[1].pages_4k != std::vector<uint32_t>({0, 1})) return 5;
    if (index.boundary[2].pages_4k != std::vector<uint32_t>({1, 2})) return 6;

    std::vector<uint32_t> refs2(index.universe_2m.size(), 0);
    cpu_low_worker_dense_ref_add(refs2, index.boundary[1].pages_2m, +1);
    if (refs2 != std::vector<uint32_t>({1, 1, 0})) return 7;
    if (cpu_low_worker_dense_ref_unique(refs2) != 2) return 8;

    CpuLowWorkerDenseDelta d;
    cpu_low_worker_dense_delta_add(d, index.boundary[1].pages_2m, -1);
    cpu_low_worker_dense_delta_add(d, index.boundary[2].pages_2m, +1);
    cpu_low_worker_dense_delta_normalize(d);
    if (d.entries.size() != 2) return 9;
    if (d.entries[0] != std::make_pair(uint32_t(0), -1)) return 10;
    if (d.entries[1] != std::make_pair(uint32_t(2), +1)) return 11;
    if (cpu_low_worker_dense_unique_after_delta(refs2, 2, d) != 2) return 12;
    uint64_t unique2 = 2;
    cpu_low_worker_dense_apply_delta(refs2, unique2, d);
    if (refs2 != std::vector<uint32_t>({0, 1, 1})) return 13;
    if (unique2 != 2) return 14;

    CpuLowWorkerDenseDelta twice;
    cpu_low_worker_dense_delta_add(twice, index.boundary[2].pages_2m, +1);
    cpu_low_worker_dense_delta_add(twice, index.boundary[2].pages_2m, +1);
    cpu_low_worker_dense_delta_normalize(twice);
    if (twice.entries != std::vector<std::pair<uint32_t,int>>({{1,2},{2,2}})) return 15;
    if (cpu_low_worker_dense_unique_after_delta(refs2, unique2, twice) != 2) return 16;

    if (!index.bytes()) return 17;
    std::cout << "cpu-low-worker-dense-page-selftest OK"
              << " universe_2m=3 universe_4k=3"
              << " cancellation=1 duplicate_ref=1 dense_delta=1\n";
    return 0;
}