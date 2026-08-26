#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page.hpp"
#include "../ramstream32_cpu_low_domain_worker_locality.hpp"
#include "../ramstream32_cpu_low_domain_worker_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_run_coalesce.hpp"

static uint64_t run_plan_max_cells(const CpuLowSparsePersistentPool& p) {
    return p.sticky_worker_cells.empty() ? 0
        : *std::max_element(p.sticky_worker_cells.begin(), p.sticky_worker_cells.end());
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    int workers = argc > 2 ? std::max(1, std::atoi(argv[2])) : 64;
    int domain_size = argc > 3 ? std::atoi(argv[3]) : 32;
    int max_run = argc > 4 ? std::atoi(argv[4]) : 4;
    if (n < 2 || n + 1 != TARGET_W || n + 1 > MAXW) return 1;
    if (workers <= 0 || domain_size <= 0 || domain_size > workers
        || max_run <= 0 || max_run > 64) return 1;
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

    CpuLowSparsePersistentPool direct(workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool hybrid(workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool selected(workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    for (CpuLowSparsePersistentPool* p : {&direct, &hybrid, &selected}) {
        p->prepare_static_schedule(jobs, sparse);
        cpu_low_apply_domain_page_tiebreak(*p, jobs, sparse, storage, layout);
        cpu_low_apply_domain_worker_locality(*p, jobs, sparse);
    }
    if (direct.sticky_worker_jobs != hybrid.sticky_worker_jobs
        || direct.sticky_worker_jobs != selected.sticky_worker_jobs
        || direct.sticky_worker_cells != hybrid.sticky_worker_cells
        || direct.sticky_worker_cells != selected.sticky_worker_cells) return 2;

    CpuLowWorkerExactWorkspace ws = cpu_low_build_worker_exact_workspace(
        jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueDenseStats direct_stats =
        cpu_low_apply_domain_worker_unique_shared_coalesce(direct, jobs, sparse, ws);
    CpuLowDomainWorkerCoalesceStats hybrid_local =
        cpu_low_apply_domain_worker_coalesce(hybrid, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueDenseStats hybrid_stats =
        cpu_low_apply_domain_worker_unique_shared_coalesce(hybrid, jobs, sparse, ws);

    CpuLowDomainWorkerUniqueCoalesceStats compat_direct{};
    compat_direct.unique_pages_2m_after = direct_stats.unique_pages_2m_after;
    compat_direct.unique_pages_4k_after = direct_stats.unique_pages_4k_after;
    compat_direct.owner_transitions_after = direct_stats.owner_transitions_after;
    CpuLowDomainWorkerUniqueCoalesceStats compat_hybrid{};
    compat_hybrid.unique_pages_2m_after = hybrid_stats.unique_pages_2m_after;
    compat_hybrid.unique_pages_4k_after = hybrid_stats.unique_pages_4k_after;
    compat_hybrid.owner_transitions_after = hybrid_stats.owner_transitions_after;
    CpuLowDomainWorkerMultiStartDecision decision = cpu_low_choose_worker_multistart(
        direct, compat_direct, hybrid, compat_hybrid);
    const CpuLowSparsePersistentPool& winner =
        decision.source == CPU_LOW_WORKER_MULTISTART_HYBRID ? hybrid : direct;
    cpu_low_copy_worker_static_schedule(selected, winner);

    CpuLowDomainWorkerUniqueScore before =
        decision.source == CPU_LOW_WORKER_MULTISTART_HYBRID
        ? cpu_low_worker_unique_dense_after_score(hybrid_stats)
        : cpu_low_worker_unique_dense_after_score(direct_stats);
    uint64_t before_max = run_plan_max_cells(selected);
    CpuLowDomainWorkerRunCoalesceStats run =
        cpu_low_apply_domain_worker_unique_run_coalesce(
            selected, jobs, sparse, ws, uint32_t(max_run));
    CpuLowDomainWorkerUniqueScore after{
        run.unique_pages_2m_after,
        run.unique_pages_4k_after,
        run.owner_transitions_after};
    if (cpu_low_worker_unique_score_less(before, after)) {
        std::cerr << "run plan exact objective regression\n";
        return 3;
    }
    if (run_plan_max_cells(selected) > before_max) {
        std::cerr << "run plan max-worker regression\n";
        return 4;
    }

    std::cout << std::setprecision(12)
              << "cpu_low_worker_run_plan OK"
              << " objective=bounded-run-global-unique-v5.32-plan"
              << " n=" << n
              << " workers=" << workers
              << " domain_size=" << domain_size
              << " max_run=" << max_run
              << " selected_source="
              << cpu_low_worker_multistart_source_name(decision.source)
              << " before_pages_2m=" << before.pages_2m
              << " before_pages_4k=" << before.pages_4k
              << " before_transitions=" << before.transitions
              << " after_pages_2m=" << after.pages_2m
              << " after_pages_4k=" << after.pages_4k
              << " after_transitions=" << after.transitions
              << " pages_2m_delta=" << int64_t(after.pages_2m) - int64_t(before.pages_2m)
              << " pages_4k_delta=" << int64_t(after.pages_4k) - int64_t(before.pages_4k)
              << " transition_delta=" << int64_t(after.transitions) - int64_t(before.transitions)
              << " max_worker_cells_before=" << before_max
              << " max_worker_cells_after=" << run_plan_max_cells(selected)
              << " candidate_evaluations=" << run.candidate_evaluations
              << " cap_rejections=" << run.cap_rejections
              << " accepted_runs=" << run.accepted_runs
              << " moved_jobs=" << run.moved_jobs
              << " moved_cells=" << run.moved_cells
              << " max_run_used=" << run.max_run_used
              << " page_improving_runs=" << run.page_improving_runs
              << " transition_only_runs=" << run.transition_only_runs
              << " move_limit_hit=" << (run.move_limit_hit ? 1 : 0)
              << " run_build_s=" << run.build_s
              << " workspace_build_s=" << ws.build_s
              << " hybrid_v526_build_s=" << hybrid_local.build_s
              << '\n';
    return 0;
}
