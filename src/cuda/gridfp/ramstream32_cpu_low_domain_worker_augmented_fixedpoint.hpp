#pragma once

#include "ramstream32_cpu_low_domain_worker_unique_fixedpoint.hpp"
#include "ramstream32_cpu_low_domain_worker_neutral_balance.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Research-only v5.36 augmented fixed point.
// Primary objective is the exact page tuple. On an exact tie, the descending
// worker-load profile is the secondary objective. Exact fixed-point selection
// and neutral balance are both required to improve this combined potential.

struct CpuLowWorkerBestExactStats {
    CpuLowDomainWorkerUniqueScore before{};
    CpuLowDomainWorkerUniqueScore after{};
    bool schedule_changed = false;
    bool exact_improved = false;
    bool profile_improved = false;
    const char* selected_order = "current";
    CpuLowWorkerFixedPointStats run_swap{};
    CpuLowWorkerFixedPointStats swap_run{};
};

static bool cpu_low_worker_score_equal_aug(
    const CpuLowDomainWorkerUniqueScore& a,
    const CpuLowDomainWorkerUniqueScore& b
) {
    return a.pages_2m == b.pages_2m
        && a.pages_4k == b.pages_4k
        && a.transitions == b.transitions;
}

static CpuLowWorkerBestExactStats cpu_low_apply_best_exact_fixedpoint(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const CpuLowWorkerExactWorkspace& ws,
    uint32_t max_run = 4,
    uint32_t max_swap = 4
) {
    CpuLowWorkerBestExactStats out;
    CpuLowSparsePersistentPool rs(
        pool.workers, CPU_LOW_SCHEDULE_DOMAIN, pool.domain_size, true);
    CpuLowSparsePersistentPool sr(
        pool.workers, CPU_LOW_SCHEDULE_DOMAIN, pool.domain_size, true);
    rs.prepare_static_schedule(jobs, sparse);
    sr.prepare_static_schedule(jobs, sparse);
    cpu_low_copy_worker_static_schedule(rs, pool);
    cpu_low_copy_worker_static_schedule(sr, pool);

    out.run_swap = cpu_low_apply_worker_exact_fixedpoint(
        rs, jobs, sparse, ws, CPU_LOW_WORKER_FIXED_RUN_SWAP, max_run, max_swap);
    out.swap_run = cpu_low_apply_worker_exact_fixedpoint(
        sr, jobs, sparse, ws, CPU_LOW_WORKER_FIXED_SWAP_RUN, max_run, max_swap);
    if (out.run_swap.component_limit_hit || out.run_swap.round_limit_hit
        || out.swap_run.component_limit_hit || out.swap_run.round_limit_hit) {
        std::cerr << "cpu LOW augmented exact component limit hit\n";
        std::exit(301);
    }
    if (!cpu_low_worker_score_equal_aug(out.run_swap.before, out.swap_run.before)) {
        std::cerr << "cpu LOW augmented exact parent mismatch\n";
        std::exit(302);
    }
    out.before = out.run_swap.before;
    out.after = out.before;

    auto current_profile = cpu_low_worker_sorted_load_profile(pool.sticky_worker_cells);
    auto rs_profile = cpu_low_worker_sorted_load_profile(rs.sticky_worker_cells);
    auto sr_profile = cpu_low_worker_sorted_load_profile(sr.sticky_worker_cells);

    const CpuLowSparsePersistentPool* best_pool = &pool;
    std::vector<uint64_t> best_profile = current_profile;
    CpuLowDomainWorkerUniqueScore best_score = out.before;
    const char* best_name = "current";

    auto consider = [&](const CpuLowSparsePersistentPool& cand,
                        const CpuLowDomainWorkerUniqueScore& score,
                        const std::vector<uint64_t>& profile,
                        const char* name) {
        bool exact_better = cpu_low_worker_unique_score_less(score, best_score);
        bool exact_worse = cpu_low_worker_unique_score_less(best_score, score);
        if (exact_better || (!exact_worse && profile < best_profile)) {
            best_pool = &cand;
            best_score = score;
            best_profile = profile;
            best_name = name;
        }
    };
    consider(rs, out.run_swap.after, rs_profile, "run-swap");
    consider(sr, out.swap_run.after, sr_profile, "swap-run");

    if (best_pool != &pool) {
        cpu_low_copy_worker_static_schedule(pool, *best_pool);
        out.schedule_changed = true;
    }
    out.after = best_score;
    out.selected_order = best_name;
    out.exact_improved = cpu_low_worker_unique_score_less(out.after, out.before);
    out.profile_improved = cpu_low_worker_score_equal_aug(out.after, out.before)
        && best_profile < current_profile;
    if (cpu_low_worker_unique_score_less(out.before, out.after)) {
        std::cerr << "cpu LOW augmented exact selector regression\n";
        std::exit(303);
    }
    if (cpu_low_worker_score_equal_aug(out.after, out.before)
        && current_profile < best_profile) {
        std::cerr << "cpu LOW augmented exact load-profile regression\n";
        std::exit(304);
    }
    rs.shutdown();
    sr.shutdown();
    return out;
}

struct CpuLowWorkerAugmentedFixedPointStats {
    uint32_t rounds = 0;
    uint64_t exact_schedule_changes = 0;
    uint64_t exact_primary_improvements = 0;
    uint64_t exact_profile_improvements = 0;
    uint64_t neutral_moves = 0;
    uint64_t neutral_candidates = 0;
    bool neutral_limit_hit = false;
    bool round_limit_hit = false;
    CpuLowDomainWorkerUniqueScore before{};
    CpuLowDomainWorkerUniqueScore after{};
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double build_s = 0.0;
};

static CpuLowWorkerAugmentedFixedPointStats cpu_low_apply_worker_augmented_fixedpoint(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const CpuLowWorkerExactWorkspace& ws,
    uint32_t max_run = 4,
    uint32_t max_swap = 4,
    uint32_t max_rounds = 32
) {
    CpuLowWorkerAugmentedFixedPointStats out;
    auto t0 = std::chrono::steady_clock::now();
    out.max_worker_cells_before = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
    bool have_before = false;
    bool converged = false;

    for (uint32_t round = 0; round < max_rounds; ++round) {
        CpuLowWorkerBestExactStats exact = cpu_low_apply_best_exact_fixedpoint(
            pool, jobs, sparse, ws, max_run, max_swap);
        if (!have_before) { out.before = exact.before; have_before = true; }
        if (exact.schedule_changed) ++out.exact_schedule_changes;
        if (exact.exact_improved) ++out.exact_primary_improvements;
        if (exact.profile_improved) ++out.exact_profile_improvements;

        CpuLowWorkerNeutralBalanceStats neutral = cpu_low_apply_worker_neutral_balance(
            pool, jobs, sparse, ws);
        if (!cpu_low_worker_score_equal_aug(exact.after, neutral.exact_before)
            || !cpu_low_worker_score_equal_aug(neutral.exact_before, neutral.exact_after)) {
            std::cerr << "cpu LOW augmented neutral exact mismatch\n";
            std::exit(305);
        }
        out.after = neutral.exact_after;
        out.neutral_moves += neutral.accepted_moves;
        out.neutral_candidates += neutral.candidate_evaluations;
        if (neutral.move_limit_hit) out.neutral_limit_hit = true;
        out.rounds = round + 1;
        if (out.neutral_limit_hit) break;
        if (!exact.schedule_changed && !neutral.accepted_moves) {
            converged = true;
            break;
        }
    }
    if (!converged && !out.neutral_limit_hit)
        out.round_limit_hit = true;

    out.max_worker_cells_after = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
    if (cpu_low_worker_unique_score_less(out.before, out.after)
        || out.max_worker_cells_after > out.max_worker_cells_before) {
        std::cerr << "cpu LOW augmented fixedpoint regression\n";
        std::exit(306);
    }
    out.build_s = ram_seconds_since(t0);
    std::cerr << "cpu_low_worker_augmented_fixedpoint"
              << " objective=exact-neutral-augmented-v5.36-plan"
              << " rounds=" << out.rounds
              << " exact_schedule_changes=" << out.exact_schedule_changes
              << " exact_primary_improvements=" << out.exact_primary_improvements
              << " exact_profile_improvements=" << out.exact_profile_improvements
              << " neutral_moves=" << out.neutral_moves
              << " neutral_candidates=" << out.neutral_candidates
              << " neutral_limit_hit=" << (out.neutral_limit_hit ? 1 : 0)
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
