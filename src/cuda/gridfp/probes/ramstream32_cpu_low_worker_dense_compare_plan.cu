#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page.hpp"
#include "../ramstream32_cpu_low_domain_worker_locality.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_dense_coalesce.hpp"

static uint64_t dense_compare_max_cells(const CpuLowSparsePersistentPool& pool) {
    return pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    int workers = argc > 2 ? std::max(1, std::atoi(argv[2])) : 64;
    int domain_size = argc > 3 ? std::atoi(argv[3]) : 32;
    if (n < 2 || n + 1 != TARGET_W || n + 1 > MAXW) return 1;
    if (workers <= 0 || domain_size <= 0 || domain_size > workers) return 1;
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

    CpuLowSparsePersistentPool flat(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool dense(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    flat.prepare_static_schedule(jobs, sparse);
    dense.prepare_static_schedule(jobs, sparse);
    cpu_low_apply_domain_page_tiebreak(flat, jobs, sparse, storage, layout);
    cpu_low_apply_domain_page_tiebreak(dense, jobs, sparse, storage, layout);
    cpu_low_apply_domain_worker_locality(flat, jobs, sparse);
    cpu_low_apply_domain_worker_locality(dense, jobs, sparse);
    if (flat.sticky_worker_jobs != dense.sticky_worker_jobs
        || flat.sticky_worker_cells != dense.sticky_worker_cells) {
        std::cerr << "dense compare parent schedule mismatch\n";
        return 2;
    }
    uint64_t parent_max = dense_compare_max_cells(flat);

    CpuLowDomainWorkerUniqueCoalesceStats flat_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            flat, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueDenseStats dense_stats =
        cpu_low_apply_domain_worker_unique_dense_coalesce(
            dense, jobs, sparse, storage, layout);

    if (flat.sticky_worker_jobs != dense.sticky_worker_jobs
        || flat.sticky_worker_cells != dense.sticky_worker_cells) {
        std::cerr << "dense compare final schedule mismatch\n";
        return 3;
    }
    if (flat_stats.unique_pages_2m_before != dense_stats.unique_pages_2m_before
        || flat_stats.unique_pages_2m_after != dense_stats.unique_pages_2m_after
        || flat_stats.unique_pages_4k_before != dense_stats.unique_pages_4k_before
        || flat_stats.unique_pages_4k_after != dense_stats.unique_pages_4k_after
        || flat_stats.owner_transitions_before != dense_stats.owner_transitions_before
        || flat_stats.owner_transitions_after != dense_stats.owner_transitions_after) {
        std::cerr << "dense compare exact objective mismatch\n";
        return 4;
    }
    if (flat_stats.candidate_evaluations != dense_stats.candidate_evaluations
        || flat_stats.cap_rejections != dense_stats.cap_rejections
        || flat_stats.accepted_moves != dense_stats.accepted_moves
        || flat_stats.moved_cells != dense_stats.moved_cells) {
        std::cerr << "dense compare search trace mismatch\n";
        return 5;
    }
    if (flat_stats.flat_delta_normalizations != dense_stats.dense_delta_normalizations) {
        std::cerr << "dense compare delta normalization mismatch\n";
        return 6;
    }
    if (dense_compare_max_cells(flat) > parent_max
        || dense_compare_max_cells(dense) > parent_max) {
        std::cerr << "dense compare max-worker regression\n";
        return 7;
    }

    double speedup = dense_stats.build_s > 0.0
        ? flat_stats.build_s / dense_stats.build_s : 0.0;
    std::cout << std::setprecision(12)
              << "cpu_low_worker_dense_compare_plan OK"
              << " objective=flat-vs-dense-exact-v5.30-plan"
              << " n=" << n
              << " workers=" << workers
              << " domain_size=" << domain_size
              << " exact_pages_2m=" << dense_stats.unique_pages_2m_after
              << " exact_pages_4k=" << dense_stats.unique_pages_4k_after
              << " exact_transitions=" << dense_stats.owner_transitions_after
              << " candidate_evaluations=" << dense_stats.candidate_evaluations
              << " accepted_moves=" << dense_stats.accepted_moves
              << " parent_max_worker_cells=" << parent_max
              << " final_max_worker_cells=" << dense_compare_max_cells(dense)
              << " flat_build_s=" << flat_stats.build_s
              << " dense_build_s=" << dense_stats.build_s
              << " dense_vs_flat_speedup=" << speedup
              << " dense_index_mib="
              << double(dense_stats.dense_index_bytes) / double(1 << 20)
              << " dense_index_build_s=" << dense_stats.dense_index_build_s
              << " flat_delta_peak_entries=" << flat_stats.flat_delta_peak_entries
              << " dense_delta_peak_entries=" << dense_stats.dense_delta_peak_entries
              << " identical_schedule=1 identical_trace=1\n";
    return 0;
}
