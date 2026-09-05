#pragma once

#include "ramstream32_cpu_low_domain_worker_unique_run_coalesce.hpp"
#include "ramstream32_cpu_low_domain_worker_unique_swap_coalesce.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// Research-only v5.34 alternating fixed-point search.
// Both component passes strictly improve the same exact tuple. Alternating them
// therefore cannot cycle; a fixed point is reached when one full round accepts
// neither a bounded-run move nor a bounded swap.

enum CpuLowWorkerFixedPointOrder : uint8_t {
    CPU_LOW_WORKER_FIXED_RUN_SWAP = 0,
    CPU_LOW_WORKER_FIXED_SWAP_RUN = 1,
};

struct CpuLowWorkerFixedPointStats {
    CpuLowWorkerFixedPointOrder order = CPU_LOW_WORKER_FIXED_RUN_SWAP;
    uint32_t max_run = 0;
    uint32_t max_swap = 0;
    uint32_t rounds = 0;
    uint64_t run_accepted = 0;
    uint64_t swap_accepted = 0;
    uint64_t run_candidates = 0;
    uint64_t swap_candidates = 0;
    uint64_t run_cap_rejections = 0;
    uint64_t swap_cap_rejections = 0;
    uint64_t moved_jobs = 0;
    uint64_t moved_cells = 0;
    uint32_t max_run_used = 0;
    uint32_t max_left_swap_used = 0;
    uint32_t max_right_swap_used = 0;
    bool component_limit_hit = false;
    bool round_limit_hit = false;
    CpuLowDomainWorkerUniqueScore before{};
    CpuLowDomainWorkerUniqueScore after{};
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double build_s = 0.0;
};

static const char* cpu_low_worker_fixed_order_name(CpuLowWorkerFixedPointOrder x) {
    return x == CPU_LOW_WORKER_FIXED_SWAP_RUN ? "swap-run" : "run-swap";
}

static CpuLowDomainWorkerUniqueScore cpu_low_worker_run_before_score(
    const CpuLowDomainWorkerRunCoalesceStats& s
) {
    return {s.unique_pages_2m_before, s.unique_pages_4k_before,
            s.owner_transitions_before};
}
static CpuLowDomainWorkerUniqueScore cpu_low_worker_run_after_score(
    const CpuLowDomainWorkerRunCoalesceStats& s
) {
    return {s.unique_pages_2m_after, s.unique_pages_4k_after,
            s.owner_transitions_after};
}
static CpuLowDomainWorkerUniqueScore cpu_low_worker_swap_before_score(
    const CpuLowDomainWorkerSwapStats& s
) {
    return {s.unique_pages_2m_before, s.unique_pages_4k_before,
            s.owner_transitions_before};
}
static CpuLowDomainWorkerUniqueScore cpu_low_worker_swap_after_score(
    const CpuLowDomainWorkerSwapStats& s
) {
    return {s.unique_pages_2m_after, s.unique_pages_4k_after,
            s.owner_transitions_after};
}

static CpuLowWorkerFixedPointStats cpu_low_apply_worker_exact_fixedpoint(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const CpuLowWorkerExactWorkspace& ws,
    CpuLowWorkerFixedPointOrder order,
    uint32_t max_run = 4,
    uint32_t max_swap = 4,
    uint32_t max_rounds = 32
) {
    CpuLowWorkerFixedPointStats out;
    auto t0 = std::chrono::steady_clock::now();
    cpu_low_validate_worker_exact_workspace(ws, jobs, sparse);
    if (!max_run || !max_swap || !max_rounds) {
        std::cerr << "cpu LOW fixedpoint limits must be positive\n";
        std::exit(284);
    }
    out.order = order;
    out.max_run = max_run;
    out.max_swap = max_swap;
    out.max_worker_cells_before = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());

    bool have_before = false;
    bool converged = false;
    for (uint32_t round = 0; round < max_rounds; ++round) {
        uint64_t round_run = 0, round_swap = 0;
        CpuLowDomainWorkerRunCoalesceStats rs;
        CpuLowDomainWorkerSwapStats ss;

        if (order == CPU_LOW_WORKER_FIXED_RUN_SWAP) {
            rs = cpu_low_apply_domain_worker_unique_run_coalesce(
                pool, jobs, sparse, ws, max_run);
            if (!have_before) { out.before = cpu_low_worker_run_before_score(rs); have_before = true; }
            ss = cpu_low_apply_domain_worker_unique_swap_coalesce(
                pool, jobs, sparse, ws, max_swap);
            out.after = cpu_low_worker_swap_after_score(ss);
        } else {
            ss = cpu_low_apply_domain_worker_unique_swap_coalesce(
                pool, jobs, sparse, ws, max_swap);
            if (!have_before) { out.before = cpu_low_worker_swap_before_score(ss); have_before = true; }
            rs = cpu_low_apply_domain_worker_unique_run_coalesce(
                pool, jobs, sparse, ws, max_run);
            out.after = cpu_low_worker_run_after_score(rs);
        }

        if (rs.move_limit_hit || ss.move_limit_hit)
            out.component_limit_hit = true;
        round_run = rs.accepted_runs;
        round_swap = ss.accepted_swaps;
        out.run_accepted += round_run;
        out.swap_accepted += round_swap;
        out.run_candidates += rs.candidate_evaluations;
        out.swap_candidates += ss.candidate_evaluations;
        out.run_cap_rejections += rs.cap_rejections;
        out.swap_cap_rejections += ss.cap_rejections;
        out.moved_jobs += rs.moved_jobs + ss.moved_jobs;
        out.moved_cells += rs.moved_cells + ss.moved_cells;
        out.max_run_used = std::max(out.max_run_used, rs.max_run_used);
        out.max_left_swap_used = std::max(out.max_left_swap_used, ss.max_left_used);
        out.max_right_swap_used = std::max(out.max_right_swap_used, ss.max_right_used);
        out.rounds = round + 1;

        if (out.component_limit_hit) break;
        if (!round_run && !round_swap) {
            converged = true;
            break;
        }
    }
    if (!converged && !out.component_limit_hit)
        out.round_limit_hit = true;

    out.max_worker_cells_after = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
    if (out.max_worker_cells_after > out.max_worker_cells_before) {
        std::cerr << "cpu LOW fixedpoint max-worker regression\n";
        std::exit(285);
    }
    if (cpu_low_worker_unique_score_less(out.before, out.after)) {
        std::cerr << "cpu LOW fixedpoint exact objective regression\n";
        std::exit(286);
    }
    out.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_worker_exact_fixedpoint"
              << " objective=alternating-run-swap-v5.34-plan"
              << " order=" << cpu_low_worker_fixed_order_name(order)
              << " max_run=" << max_run
              << " max_swap=" << max_swap
              << " rounds=" << out.rounds
              << " run_accepted=" << out.run_accepted
              << " swap_accepted=" << out.swap_accepted
              << " run_candidates=" << out.run_candidates
              << " swap_candidates=" << out.swap_candidates
              << " moved_jobs=" << out.moved_jobs
              << " max_run_used=" << out.max_run_used
              << " max_left_swap_used=" << out.max_left_swap_used
              << " max_right_swap_used=" << out.max_right_swap_used
              << " component_limit_hit=" << (out.component_limit_hit ? 1 : 0)
              << " round_limit_hit=" << (out.round_limit_hit ? 1 : 0)
              << " before_2m=" << out.before.pages_2m
              << " before_4k=" << out.before.pages_4k
              << " before_transitions=" << out.before.transitions
              << " after_2m=" << out.after.pages_2m
              << " after_4k=" << out.after.pages_4k
              << " after_transitions=" << out.after.transitions
              << " max_worker_cells_before=" << out.max_worker_cells_before
              << " max_worker_cells_after=" << out.max_worker_cells_after
              << " build_s=" << out.build_s << '\n';
    return out;
}
