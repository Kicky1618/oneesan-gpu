#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_worker_dense_page.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

static bool same_dense_index(
    const CpuLowWorkerDensePageIndex& a,
    const CpuLowWorkerDensePageIndex& b
) {
    if (a.universe_2m != b.universe_2m
        || a.universe_4k != b.universe_4k
        || a.boundary.size() != b.boundary.size()) return false;
    for (size_t i = 0; i < a.boundary.size(); ++i) {
        if (a.boundary[i].pages_2m != b.boundary[i].pages_2m
            || a.boundary[i].pages_4k != b.boundary[i].pages_4k) return false;
    }
    return true;
}

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
    if (!index.bytes() || index.reserved_bytes() < index.bytes()) return 17;

    // Real W10 topology: prove the low-memory two-pass builder is an exact
    // representation replacement for the legacy retain-all-raw builder.
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(TARGET_W, low_wp);

    std::vector<CpuLowStaticJobCost> ordered;
    ordered.reserve(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].main_size && !jobs[i].block_size) continue;
        ordered.push_back({i, jobs[i].mask, cpu_low_sparse_job_cells(jobs[i], sparse)});
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        if (a.mask != b.mask) return a.mask < b.mask;
        return a.index < b.index;
    });
    CpuLowDomainPageMaskIndex mask_index = cpu_low_build_domain_page_mask_index();
    auto legacy = cpu_low_build_worker_dense_page_index(
        ordered, layout, storage, mask_index);
    auto streamed = cpu_low_build_worker_dense_page_index_streaming(
        ordered, layout, storage, mask_index);
    if (!same_dense_index(legacy, streamed)) return 18;
    if (streamed.reserved_bytes() < streamed.bytes()) return 19;

    std::cout << "cpu-low-worker-dense-page-selftest OK"
              << " universe_2m=3 universe_4k=3"
              << " cancellation=1 duplicate_ref=1 dense_delta=1"
              << " streaming_equivalence=1"
              << " real_boundaries=" << streamed.boundary.size()
              << '\n';
    return 0;
}
