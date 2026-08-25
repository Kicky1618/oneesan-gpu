#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page_global.hpp"

static uint64_t max_cells(const CpuLowSparsePersistentPool& pool) {
    return pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
}

static uint64_t total_cells(const CpuLowSparsePersistentPool& pool) {
    return std::accumulate(
        pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end(), uint64_t(0));
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

    auto score_index_t0 = std::chrono::steady_clock::now();
    CpuLowDomainPageMaskIndex score_index = cpu_low_build_domain_page_mask_index();
    double score_index_build_s = ram_seconds_since(score_index_t0);
    double score_index_mib = double(
        score_index.first_nonempty.size() * sizeof(uint32_t)
        + score_index.next_nonempty.size() * sizeof(uint32_t)) / double(1 << 20);

    CpuLowSparsePersistentPool refined(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool local(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool global(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    refined.prepare_static_schedule(jobs, sparse);
    local.prepare_static_schedule(jobs, sparse);
    global.prepare_static_schedule(jobs, sparse);

    uint64_t expected_total = total_cells(refined);
    if (total_cells(local) != expected_total || total_cells(global) != expected_total) {
        std::cerr << "cpu LOW global page plan initial accounting mismatch\n";
        return 2;
    }

    CpuLowDomainGlobalPageScore refined_score =
        cpu_low_domain_global_page_score_for_pool(
            refined, jobs, storage, layout, score_index);
    uint64_t refined_max = max_cells(refined);

    CpuLowDomainPageTieStats local_stats = cpu_low_apply_domain_page_tiebreak(
        local, jobs, sparse, storage, layout);
    local.schedule_build_s += local_stats.build_s;
    CpuLowDomainGlobalPageScore local_score =
        cpu_low_domain_global_page_score_for_pool(
            local, jobs, storage, layout, score_index);
    uint64_t local_max = max_cells(local);

    CpuLowDomainPageTieStats global_local_stats = cpu_low_apply_domain_page_tiebreak(
        global, jobs, sparse, storage, layout);
    global.schedule_build_s += global_local_stats.build_s;
    CpuLowDomainGlobalPageStats global_stats =
        cpu_low_apply_domain_global_page_tiebreak(
            global, jobs, sparse, storage, layout);
    global.schedule_build_s += global_stats.build_s;
    CpuLowDomainGlobalPageScore global_score =
        cpu_low_domain_global_page_score_for_pool(
            global, jobs, storage, layout, score_index);
    uint64_t global_max = max_cells(global);

    if (local_max > refined_max || global_max > local_max) {
        std::cerr << "cpu LOW global page plan max-worker regression"
                  << " refined=" << refined_max
                  << " local=" << local_max
                  << " global=" << global_max << '\n';
        return 3;
    }
    if (cpu_low_domain_global_page_score_less(local_score, global_score)) {
        std::cerr << "cpu LOW global page plan global-score regression"
                  << " local_2m=" << local_score.pages_2m
                  << " local_4k=" << local_score.pages_4k
                  << " global_2m=" << global_score.pages_2m
                  << " global_4k=" << global_score.pages_4k << '\n';
        return 4;
    }
    if (total_cells(local) != expected_total || total_cells(global) != expected_total) {
        std::cerr << "cpu LOW global page plan final accounting mismatch\n";
        return 5;
    }

    double avg = workers ? double(expected_total) / workers : 0.0;
    std::cout << std::setprecision(12)
              << "cpu_low_global_page_plan OK"
              << " objective=global-unique-max-guard-page-sum-v5.24-plan"
              << " n=" << n
              << " workers=" << workers
              << " domain_size=" << domain_size
              << " domains=" << ((workers + domain_size - 1) / domain_size)
              << " total_cells=" << expected_total
              << " refined_max_worker_cells=" << refined_max
              << " local_max_worker_cells=" << local_max
              << " global_max_worker_cells=" << global_max
              << " refined_imbalance=" << (avg ? double(refined_max) / avg : 0.0)
              << " local_imbalance=" << (avg ? double(local_max) / avg : 0.0)
              << " global_imbalance=" << (avg ? double(global_max) / avg : 0.0)
              << " refined_global_pages_2m=" << refined_score.pages_2m
              << " refined_global_pages_4k=" << refined_score.pages_4k
              << " local_global_pages_2m=" << local_score.pages_2m
              << " local_global_pages_4k=" << local_score.pages_4k
              << " global_pages_2m=" << global_score.pages_2m
              << " global_pages_4k=" << global_score.pages_4k
              << " score_mask_index_mib=" << score_index_mib
              << " score_mask_index_build_s=" << score_index_build_s
              << " local_boundary_moves=" << local_stats.boundary_moves
              << " local_candidate_evaluations=" << local_stats.candidate_evaluations
              << " local_max_guard_rejections=" << local_stats.max_guard_rejections
              << " local_page_improve_sum_increase_moves="
              << local_stats.page_improve_sum_increase_moves
              << " local_mask_index_mib=" << local_stats.mask_index_mib
              << " local_mask_index_build_s=" << local_stats.mask_index_build_s
              << " global_local_mask_index_mib=" << global_local_stats.mask_index_mib
              << " global_local_mask_index_build_s=" << global_local_stats.mask_index_build_s
              << " global_boundary_moves=" << global_stats.boundary_moves
              << " global_moved_jobs=" << global_stats.moved_jobs
              << " global_candidate_evaluations=" << global_stats.candidate_evaluations
              << " global_max_guard_rejections=" << global_stats.max_guard_rejections
              << " global_page_improving_moves=" << global_stats.page_improving_moves
              << " global_page_tie_load_moves=" << global_stats.page_tie_load_moves
              << " global_page_improve_sum_increase_moves="
              << global_stats.page_improve_sum_increase_moves
              << " global_mask_index_mib=" << global_stats.mask_index_mib
              << " global_mask_index_build_s=" << global_stats.mask_index_build_s
              << " global_build_s=" << global_stats.build_s
              << " local_total_build_s=" << local.schedule_build_s
              << " global_total_build_s=" << global.schedule_build_s
              << '\n';
    return 0;
}
