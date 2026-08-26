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

static bool shared_same_schedule(
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

    CpuLowSparsePersistentPool legacy_direct(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool legacy_hybrid(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool shared_direct(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);
    CpuLowSparsePersistentPool shared_hybrid(
        workers, CPU_LOW_SCHEDULE_DOMAIN, domain_size, true);

    for (CpuLowSparsePersistentPool* p : {
            &legacy_direct, &legacy_hybrid, &shared_direct, &shared_hybrid}) {
        p->prepare_static_schedule(jobs, sparse);
        cpu_low_apply_domain_page_tiebreak(*p, jobs, sparse, storage, layout);
        cpu_low_apply_domain_worker_locality(*p, jobs, sparse);
    }
    if (!shared_same_schedule(legacy_direct, legacy_hybrid)
        || !shared_same_schedule(legacy_direct, shared_direct)
        || !shared_same_schedule(legacy_direct, shared_hybrid)) {
        std::cerr << "shared multistart parent schedule mismatch\n";
        return 2;
    }

    CpuLowWorkerExactWorkspace ws = cpu_low_build_worker_exact_workspace(
        jobs, sparse, storage, layout);
    if (!ws.structural_audit_ok) {
        std::cerr << "shared multistart workspace audit failed\n";
        return 8;
    }

    CpuLowDomainWorkerUniqueCoalesceStats legacy_direct_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            legacy_direct, jobs, sparse, storage, layout);
    CpuLowDomainWorkerCoalesceStats legacy_local =
        cpu_low_apply_domain_worker_coalesce(
            legacy_hybrid, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueCoalesceStats legacy_hybrid_stats =
        cpu_low_apply_domain_worker_unique_coalesce(
            legacy_hybrid, jobs, sparse, storage, layout);

    CpuLowDomainWorkerUniqueDenseStats shared_direct_stats =
        cpu_low_apply_domain_worker_unique_shared_coalesce(
            shared_direct, jobs, sparse, ws);
    CpuLowDomainWorkerCoalesceStats shared_local =
        cpu_low_apply_domain_worker_coalesce(
            shared_hybrid, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueDenseStats shared_hybrid_stats =
        cpu_low_apply_domain_worker_unique_shared_coalesce(
            shared_hybrid, jobs, sparse, ws);

    if (!shared_same_schedule(legacy_direct, shared_direct)) {
        std::cerr << "shared direct schedule mismatch\n";
        return 3;
    }
    if (!shared_same_schedule(legacy_hybrid, shared_hybrid)) {
        std::cerr << "shared hybrid schedule mismatch\n";
        return 4;
    }
    if (legacy_direct_stats.candidate_evaluations
            != shared_direct_stats.candidate_evaluations
        || legacy_direct_stats.accepted_moves != shared_direct_stats.accepted_moves
        || legacy_hybrid_stats.candidate_evaluations
            != shared_hybrid_stats.candidate_evaluations
        || legacy_hybrid_stats.accepted_moves != shared_hybrid_stats.accepted_moves
        || legacy_local.accepted_moves != shared_local.accepted_moves) {
        std::cerr << "shared multistart search trace mismatch\n";
        return 5;
    }

    CpuLowDomainWorkerUniqueScore ld = cpu_low_worker_unique_after_score(
        legacy_direct_stats);
    CpuLowDomainWorkerUniqueScore lh = cpu_low_worker_unique_after_score(
        legacy_hybrid_stats);
    CpuLowDomainWorkerUniqueScore sd = cpu_low_worker_unique_dense_after_score(
        shared_direct_stats);
    CpuLowDomainWorkerUniqueScore sh = cpu_low_worker_unique_dense_after_score(
        shared_hybrid_stats);
    if (ld.pages_2m != sd.pages_2m || ld.pages_4k != sd.pages_4k
        || ld.transitions != sd.transitions
        || lh.pages_2m != sh.pages_2m || lh.pages_4k != sh.pages_4k
        || lh.transitions != sh.transitions) {
        std::cerr << "shared multistart exact score mismatch\n";
        return 6;
    }

    CpuLowDomainWorkerMultiStartDecision legacy_decision =
        cpu_low_choose_worker_multistart(
            legacy_direct, legacy_direct_stats,
            legacy_hybrid, legacy_hybrid_stats);
    CpuLowDomainWorkerUniqueCoalesceStats compat_sd{};
    compat_sd.unique_pages_2m_after = sd.pages_2m;
    compat_sd.unique_pages_4k_after = sd.pages_4k;
    compat_sd.owner_transitions_after = sd.transitions;
    CpuLowDomainWorkerUniqueCoalesceStats compat_sh{};
    compat_sh.unique_pages_2m_after = sh.pages_2m;
    compat_sh.unique_pages_4k_after = sh.pages_4k;
    compat_sh.owner_transitions_after = sh.transitions;
    CpuLowDomainWorkerMultiStartDecision shared_decision =
        cpu_low_choose_worker_multistart(
            shared_direct, compat_sd, shared_hybrid, compat_sh);
    if (legacy_decision.source != shared_decision.source) {
        std::cerr << "shared multistart selector mismatch\n";
        return 7;
    }

    double legacy_total = legacy_direct_stats.build_s
        + legacy_local.build_s + legacy_hybrid_stats.build_s;
    double shared_total = ws.build_s
        + shared_direct_stats.build_s
        + shared_local.build_s + shared_hybrid_stats.build_s;
    double speedup = shared_total > 0.0 ? legacy_total / shared_total : 0.0;

    std::cout << std::setprecision(12)
              << "cpu_low_worker_shared_multistart_plan OK"
              << " objective=shared-dense-multistart-v5.31-plan"
              << " n=" << n
              << " workers=" << workers
              << " domain_size=" << domain_size
              << " selected_source="
              << cpu_low_worker_multistart_source_name(shared_decision.source)
              << " direct_pages_2m=" << sd.pages_2m
              << " direct_pages_4k=" << sd.pages_4k
              << " hybrid_pages_2m=" << sh.pages_2m
              << " hybrid_pages_4k=" << sh.pages_4k
              << " legacy_total_build_s=" << legacy_total
              << " workspace_build_s=" << ws.build_s
              << " workspace_mib=" << double(ws.bytes()) / double(1 << 20)
              << " workspace_audit_ok=" << (ws.structural_audit_ok ? 1 : 0)
              << " workspace_audited_jobs=" << ws.audited_jobs
              << " workspace_audited_cells=" << ws.audited_cells
              << " workspace_audit_s=" << ws.audit_s
              << " shared_direct_search_s=" << shared_direct_stats.build_s
              << " shared_v526_build_s=" << shared_local.build_s
              << " shared_hybrid_search_s=" << shared_hybrid_stats.build_s
              << " shared_total_build_s=" << shared_total
              << " shared_vs_legacy_speedup=" << speedup
              << " direct_candidate_evaluations="
              << shared_direct_stats.candidate_evaluations
              << " hybrid_candidate_evaluations="
              << shared_hybrid_stats.candidate_evaluations
              << " identical_direct_schedule=1"
              << " identical_hybrid_schedule=1"
              << " identical_selector=1\n";
    return 0;
}
