#pragma once

#include "ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// Research-only v5.28 selector.
//
// Two exact-objective branches start from the same v5.25 schedule:
//
//   direct: v5.25 -> v5.27
//   hybrid: v5.25 -> v5.26 -> v5.27
//
// The final v5.27 pass cannot worsen the exact unique tuple relative to its
// input. Therefore the hybrid branch is <= the raw v5.26 result, while the
// direct branch is <= the common v5.25 result. Selecting the better exact tuple
// dominates v5.25, raw v5.26, and direct v5.27 structurally. If exact tuples
// tie, prefer the smaller max worker load, then the direct branch for stable
// deterministic provenance.

enum CpuLowDomainWorkerMultiStartSource : uint8_t {
    CPU_LOW_WORKER_MULTISTART_DIRECT = 0,
    CPU_LOW_WORKER_MULTISTART_HYBRID = 1,
};

struct CpuLowDomainWorkerMultiStartDecision {
    CpuLowDomainWorkerMultiStartSource source = CPU_LOW_WORKER_MULTISTART_DIRECT;
    CpuLowDomainWorkerUniqueScore direct_score{};
    CpuLowDomainWorkerUniqueScore hybrid_score{};
    uint64_t direct_max_worker_cells = 0;
    uint64_t hybrid_max_worker_cells = 0;
};

static const char* cpu_low_worker_multistart_source_name(
    CpuLowDomainWorkerMultiStartSource source
) {
    return source == CPU_LOW_WORKER_MULTISTART_HYBRID ? "hybrid" : "direct";
}

static CpuLowDomainWorkerUniqueScore cpu_low_worker_unique_after_score(
    const CpuLowDomainWorkerUniqueCoalesceStats& s
) {
    return {s.unique_pages_2m_after,
            s.unique_pages_4k_after,
            s.owner_transitions_after};
}

static CpuLowDomainWorkerMultiStartDecision cpu_low_choose_worker_multistart(
    const CpuLowSparsePersistentPool& direct_pool,
    const CpuLowDomainWorkerUniqueCoalesceStats& direct_stats,
    const CpuLowSparsePersistentPool& hybrid_pool,
    const CpuLowDomainWorkerUniqueCoalesceStats& hybrid_stats
) {
    if (direct_pool.workers != hybrid_pool.workers
        || direct_pool.domain_size != hybrid_pool.domain_size
        || direct_pool.schedule_mode != hybrid_pool.schedule_mode
        || direct_pool.sticky_source_jobs != hybrid_pool.sticky_source_jobs
        || direct_pool.sticky_source_sparse != hybrid_pool.sticky_source_sparse) {
        std::cerr << "cpu LOW worker multistart incompatible branch provenance\n";
        std::exit(212);
    }

    CpuLowDomainWorkerMultiStartDecision d;
    d.direct_score = cpu_low_worker_unique_after_score(direct_stats);
    d.hybrid_score = cpu_low_worker_unique_after_score(hybrid_stats);
    d.direct_max_worker_cells = direct_pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            direct_pool.sticky_worker_cells.begin(), direct_pool.sticky_worker_cells.end());
    d.hybrid_max_worker_cells = hybrid_pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            hybrid_pool.sticky_worker_cells.begin(), hybrid_pool.sticky_worker_cells.end());

    bool hybrid_better = cpu_low_worker_unique_score_less(
        d.hybrid_score, d.direct_score);
    bool direct_better = cpu_low_worker_unique_score_less(
        d.direct_score, d.hybrid_score);
    if (hybrid_better
        || (!direct_better
            && d.hybrid_max_worker_cells < d.direct_max_worker_cells))
        d.source = CPU_LOW_WORKER_MULTISTART_HYBRID;
    return d;
}

static void cpu_low_copy_worker_static_schedule(
    CpuLowSparsePersistentPool& dst,
    const CpuLowSparsePersistentPool& src
) {
    if (dst.workers != src.workers
        || dst.domain_size != src.domain_size
        || dst.schedule_mode != src.schedule_mode
        || dst.sticky_source_jobs != src.sticky_source_jobs
        || dst.sticky_source_sparse != src.sticky_source_sparse) {
        std::cerr << "cpu LOW worker multistart copy provenance mismatch\n";
        std::exit(213);
    }
    dst.sticky_worker_jobs = src.sticky_worker_jobs;
    dst.sticky_worker_cells = src.sticky_worker_cells;
}
