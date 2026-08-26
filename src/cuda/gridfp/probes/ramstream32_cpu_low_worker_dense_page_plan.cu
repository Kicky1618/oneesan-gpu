#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_worker_dense_page.hpp"

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    if (n < 2 || n + 1 != TARGET_W || n + 1 > MAXW) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(n + 1, low_wp);

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

    auto mt0 = std::chrono::steady_clock::now();
    CpuLowDomainPageMaskIndex mask_index = cpu_low_build_domain_page_mask_index();
    double mask_index_build_s = ram_seconds_since(mt0);
    CpuLowWorkerDensePageIndex dense = cpu_low_build_worker_dense_page_index(
        ordered, layout, storage, mask_index);

    uint64_t entries2 = 0, entries4 = 0;
    uint64_t max_boundary2 = 0, max_boundary4 = 0;
    for (size_t k = 1; k < dense.boundary.size(); ++k) {
        entries2 += dense.boundary[k].pages_2m.size();
        entries4 += dense.boundary[k].pages_4k.size();
        max_boundary2 = std::max<uint64_t>(
            max_boundary2, dense.boundary[k].pages_2m.size());
        max_boundary4 = std::max<uint64_t>(
            max_boundary4, dense.boundary[k].pages_4k.size());
    }

    uint64_t raw_signature_payload = (entries2 + entries4) * sizeof(uint64_t);
    uint64_t dense_signature_payload = (entries2 + entries4) * sizeof(uint32_t);
    uint64_t dense_universe_bytes =
        (dense.universe_2m.size() + dense.universe_4k.size()) * sizeof(uint64_t);
    uint64_t dense_ref_bytes =
        (dense.universe_2m.size() + dense.universe_4k.size()) * sizeof(uint32_t);
    uint64_t candidate_peak_entries = 2 * (max_boundary2 + max_boundary4);

    if (entries2 && dense.universe_2m.empty()) return 2;
    if (entries4 && dense.universe_4k.empty()) return 3;
    if (dense.boundary.size() != ordered.size() + 1) return 4;

    std::cout << std::setprecision(12)
              << "cpu_low_worker_dense_page_plan OK"
              << " objective=dense-page-id-substrate-v5.30-plan"
              << " n=" << n
              << " ordered_jobs=" << ordered.size()
              << " boundaries=" << (ordered.empty() ? 0 : ordered.size() - 1)
              << " universe_2m=" << dense.universe_2m.size()
              << " universe_4k=" << dense.universe_4k.size()
              << " signature_entries_2m=" << entries2
              << " signature_entries_4k=" << entries4
              << " max_boundary_entries_2m=" << max_boundary2
              << " max_boundary_entries_4k=" << max_boundary4
              << " candidate_peak_delta_entries=" << candidate_peak_entries
              << " raw_signature_payload_mib="
              << double(raw_signature_payload) / double(1 << 20)
              << " dense_signature_payload_mib="
              << double(dense_signature_payload) / double(1 << 20)
              << " dense_universe_mib="
              << double(dense_universe_bytes) / double(1 << 20)
              << " dense_ref_mib="
              << double(dense_ref_bytes) / double(1 << 20)
              << " dense_total_index_mib="
              << double(dense.bytes()) / double(1 << 20)
              << " mask_index_build_s=" << mask_index_build_s
              << " dense_build_s=" << dense.build_s
              << '\n';
    return 0;
}
