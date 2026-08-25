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
#include "../ramstream32_cpu_low_domain_worker_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"

static uint64_t pool_max_cells(const CpuLowSparsePersistentPool& pool) {
    return pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
}

static uint64_t pool_total_cells(const CpuLowSparsePersistentPool& pool) {
    return std::accumulate(
        pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end(), uint64_t(0));
}

static bool same_schedule(
    const CpuLowSparsePersistentPool& a,
    const CpuLowSparsePersistentPool& b
) {
    return a.sticky_worker_jobs == b.sticky_worker_jobs
        && a.sticky_worker_cells == b.sticky_worker_cells;
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

    CpuLowSparsePersistentPool direct(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool hybrid(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool selected(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    direct.prepare_static_schedule(jobs, sparse);
    hybrid.prepare_static_schedule(jobs, sparse);
    selected.prepare_static_schedule(jobs, sparse);

    cpu_low_apply_domain_page_tiebreak(direct, jobs, sparse, storage, layout);
    cpu_low_apply_domain_page_tiebreak(hybrid, jobs, sparse, storage, layout);
    cpu_low_apply_domain_page_tiebreak(selected, jobs, sparse, storage, layout);
    CpuLowDomainWorkerLocalityStats direct_parent =
        cpu_low_apply_domain_worker_locality(direct, jobs, sparse);
    CpuLowDomainWorkerLocalityStats hybrid_parent =
        cpu_low_apply_domain_worker_locality(hybrid, jobs, sparse);
    CpuLowDomainWorkerLocalityStats selected_parent =
        cpu_low_apply_domain_worker_locality(selected, jobs, sparse);

    if (!same_schedule(direct, hybrid) || !same_schedule(direct, selected)) {
        std::cerr << "worker multistart v5.25 branch mismatch\n";
        return 2;
    }
    uint64_t parent_total = pool_total_cells(direct);
    uint64_t parent_max = pool_max_cells(direct);

    CpuLowDomainWorkerUniqueCoalesceStats direct_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            direct, jobs, sparse, storage, layout);

    CpuLowDomainWorkerCoalesceStats hybrid_local_stats =
        cpu_low_apply_domain_worker_coalesce(
            hybrid, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueCoalesceStats hybrid_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            hybrid, jobs, sparse, storage, layout);

    CpuLowDomainWorkerMultiStartDecision decision = cpu_low_choose_worker_multistart(
        direct, direct_stats, hybrid, hybrid_stats);
    const CpuLowSparsePersistentPool& winner =
        decision.source == CPU_LOW_WORKER_MULTISTART_HYBRID ? hybrid : direct;
    cpu_low_copy_worker_static_schedule(selected, winner);

    if (!same_schedule(selected, winner)) {
        std::cerr << "worker multistart selected schedule copy mismatch\n";
        return 3;
    }
    if (pool_total_cells(direct) != parent_total
        || pool_total_cells(hybrid) != parent_total
        || pool_total_cells(selected) != parent_total) {
        std::cerr << "worker multistart cell accounting mismatch\n";
        return 4;
    }

    CpuLowDomainWorkerUniqueScore parent_score{
        direct_stats.unique_pages_2m_before,
        direct_stats.unique_pages_4k_before,
        direct_stats.owner_transitions_before};
    CpuLowDomainWorkerUniqueScore raw_v526_score{
        hybrid_stats.unique_pages_2m_before,
        hybrid_stats.unique_pages_4k_before,
        hybrid_stats.owner_transitions_before};
    CpuLowDomainWorkerUniqueScore selected_score =
        decision.source == CPU_LOW_WORKER_MULTISTART_HYBRID
        ? decision.hybrid_score : decision.direct_score;

    if (cpu_low_worker_unique_score_less(parent_score, decision.direct_score)) {
        std::cerr << "worker multistart direct branch regression\n";
        return 5;
    }
    if (cpu_low_worker_unique_score_less(raw_v526_score, decision.hybrid_score)) {
        std::cerr << "worker multistart hybrid cleanup regression\n";
        return 6;
    }
    if (cpu_low_worker_unique_score_less(decision.direct_score, selected_score)
        || cpu_low_worker_unique_score_less(decision.hybrid_score, selected_score)) {
        std::cerr << "worker multistart selector did not choose exact minimum\n";
        return 7;
    }
    if (cpu_low_worker_unique_score_less(raw_v526_score, selected_score)) {
        std::cerr << "worker multistart failed to dominate raw v5.26\n";
        return 8;
    }

    uint64_t direct_max = pool_max_cells(direct);
    uint64_t hybrid_max = pool_max_cells(hybrid);
    uint64_t selected_max = pool_max_cells(selected);
    if (direct_max > parent_max || hybrid_local_stats.max_worker_cells_after > parent_max
        || hybrid_max > hybrid_local_stats.max_worker_cells_after
        || selected_max > parent_max) {
        std::cerr << "worker multistart max-worker regression\n";
        return 9;
    }

    std::cout << std::setprecision(12)
              << "cpu_low_domain_worker_multistart_plan OK"
              << " objective=multistart-global-unique-worker-v5.28-plan"
              << " n=" << n
              << " workers=" << workers
              << " domain_size=" << domain_size
              << " domains=" << direct_parent.domains
              << " total_cells=" << parent_total
              << " parent_max_worker_cells=" << parent_max
              << " direct_max_worker_cells=" << direct_max
              << " hybrid_raw_v526_max_worker_cells="
              << hybrid_local_stats.max_worker_cells_after
              << " hybrid_max_worker_cells=" << hybrid_max
              << " selected_max_worker_cells=" << selected_max
              << " parent_pages_2m=" << parent_score.pages_2m
              << " parent_pages_4k=" << parent_score.pages_4k
              << " parent_transitions=" << parent_score.transitions
              << " direct_pages_2m=" << decision.direct_score.pages_2m
              << " direct_pages_4k=" << decision.direct_score.pages_4k
              << " direct_transitions=" << decision.direct_score.transitions
              << " raw_v526_pages_2m=" << raw_v526_score.pages_2m
              << " raw_v526_pages_4k=" << raw_v526_score.pages_4k
              << " raw_v526_transitions=" << raw_v526_score.transitions
              << " hybrid_pages_2m=" << decision.hybrid_score.pages_2m
              << " hybrid_pages_4k=" << decision.hybrid_score.pages_4k
              << " hybrid_transitions=" << decision.hybrid_score.transitions
              << " selected_source="
              << cpu_low_worker_multistart_source_name(decision.source)
              << " selected_pages_2m=" << selected_score.pages_2m
              << " selected_pages_4k=" << selected_score.pages_4k
              << " selected_transitions=" << selected_score.transitions
              << " direct_accepted_moves=" << direct_stats.accepted_moves
              << " hybrid_v526_accepted_moves=" << hybrid_local_stats.accepted_moves
              << " hybrid_v527_accepted_moves=" << hybrid_stats.accepted_moves
              << " direct_build_s=" << direct_stats.build_s
              << " hybrid_v526_build_s=" << hybrid_local_stats.build_s
              << " hybrid_v527_build_s=" << hybrid_stats.build_s
              << " parent_direct_converted_domains=" << direct_parent.converted_domains
              << " parent_hybrid_converted_domains=" << hybrid_parent.converted_domains
              << " parent_selected_converted_domains=" << selected_parent.converted_domains
              << '\n';
    return 0;
}
