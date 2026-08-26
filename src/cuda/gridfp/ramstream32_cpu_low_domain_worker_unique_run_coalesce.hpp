#pragma once

#include "ramstream32_cpu_low_domain_worker_exact_workspace.hpp"
#include "ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// Research-only v5.32 bounded-run exact coalescing.
//
// Single-job v5.27/v5.31 search can be trapped when moving one job would make
// the exact page tuple worse even though moving two or more adjacent jobs as one
// atomic step improves it.  v5.32 evaluates prefix/suffix chunks of a maximal
// same-owner run, up to max_run jobs, and commits only a final exact-objective
// improvement.  Intermediate one-job states are never materialized.

struct CpuLowDomainWorkerRunCoalesceStats {
    uint64_t candidate_evaluations = 0;
    uint64_t cap_rejections = 0;
    uint64_t accepted_runs = 0;
    uint64_t moved_jobs = 0;
    uint64_t moved_cells = 0;
    uint64_t page_improving_runs = 0;
    uint64_t transition_only_runs = 0;
    uint32_t max_run_requested = 0;
    uint32_t max_run_used = 0;
    bool move_limit_hit = false;
    uint64_t unique_pages_2m_before = 0;
    uint64_t unique_pages_2m_after = 0;
    uint64_t unique_pages_4k_before = 0;
    uint64_t unique_pages_4k_after = 0;
    uint64_t owner_transitions_before = 0;
    uint64_t owner_transitions_after = 0;
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    size_t dense_delta_peak_entries = 0;
    double build_s = 0.0;
};

struct CpuLowWorkerRunCandidate {
    bool valid = false;
    size_t begin = 0;
    size_t end = 0;
    int src = -1;
    int dst = -1;
    uint64_t cells = 0;
    CpuLowDomainWorkerUniqueScore score{};
    CpuLowWorkerDenseDelta d2;
    CpuLowWorkerDenseDelta d4;
    int64_t transition_delta = 0;
};

static CpuLowDomainWorkerRunCoalesceStats
cpu_low_apply_domain_worker_unique_run_coalesce(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const CpuLowWorkerExactWorkspace& ws,
    uint32_t max_run = 4,
    uint64_t max_accepted = 4096
) {
    CpuLowDomainWorkerRunCoalesceStats stats;
    auto t0 = std::chrono::steady_clock::now();
    cpu_low_validate_worker_exact_workspace(ws, jobs, sparse);
    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || pool.domain_size <= 0) {
        std::cerr << "cpu LOW run coalesce requires domain schedule\n";
        std::exit(256);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW run coalesce requires prepared schedule\n";
        std::exit(257);
    }
    if (!max_run) {
        std::cerr << "cpu LOW run coalesce max_run must be positive\n";
        std::exit(258);
    }
    stats.max_run_requested = max_run;

    const auto& ordered = ws.ordered;
    const auto& dense = ws.dense;
    std::vector<int> owner(ordered.size(), -1);
    std::vector<int> original_domain(ordered.size(), -1);
    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size() || ws.ordered_pos[q] == size_t(-1)) {
                std::cerr << "cpu LOW run coalesce bad job owner\n";
                std::exit(259);
            }
            size_t k = ws.ordered_pos[q];
            if (owner[k] >= 0) {
                std::cerr << "cpu LOW run coalesce duplicate job owner\n";
                std::exit(260);
            }
            owner[k] = w;
            original_domain[k] = d;
        }
    }

    const int domains = (pool.workers + pool.domain_size - 1) / pool.domain_size;
    std::vector<std::pair<size_t,size_t>> domain_segs(
        size_t(domains), {ordered.size(), ordered.size()});
    int last_domain = -1;
    for (size_t k = 0; k < ordered.size(); ++k) {
        int d = original_domain[k];
        if (owner[k] < 0 || d < 0 || d >= domains || d < last_domain) {
            std::cerr << "cpu LOW run coalesce lost domain ordering\n";
            std::exit(261);
        }
        last_domain = d;
        if (domain_segs[size_t(d)].first == ordered.size())
            domain_segs[size_t(d)].first = k;
        domain_segs[size_t(d)].second = k + 1;
    }

    std::vector<uint64_t> loads = pool.sticky_worker_cells;
    std::vector<uint64_t> domain_cap(size_t(domains), 0);
    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        domain_cap[size_t(d)] = std::max(domain_cap[size_t(d)], loads[size_t(w)]);
    }
    stats.max_worker_cells_before = loads.empty() ? 0
        : *std::max_element(loads.begin(), loads.end());

    std::vector<uint32_t> refs2m(dense.universe_2m.size(), 0);
    std::vector<uint32_t> refs4k(dense.universe_4k.size(), 0);
    uint64_t transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        cpu_low_worker_dense_ref_add(refs2m, dense.boundary[k].pages_2m, +1);
        cpu_low_worker_dense_ref_add(refs4k, dense.boundary[k].pages_4k, +1);
        transitions += ws.transition_weight[k];
    }
    uint64_t unique2m = cpu_low_worker_dense_ref_unique(refs2m);
    uint64_t unique4k = cpu_low_worker_dense_ref_unique(refs4k);
    stats.unique_pages_2m_before = unique2m;
    stats.unique_pages_4k_before = unique4k;
    stats.owner_transitions_before = transitions;

    std::vector<uint64_t> prefix(ordered.size() + 1, 0);
    for (size_t i = 0; i < ordered.size(); ++i)
        prefix[i + 1] = prefix[i] + ordered[i].cells;
    auto range_cells = [&](size_t a, size_t b) -> uint64_t {
        return prefix[b] - prefix[a];
    };

    auto add_edge_delta = [&](CpuLowWorkerRunCandidate& cand,
                              size_t boundary,
                              int old_left, int old_right,
                              int new_left, int new_right) {
        if (!boundary || boundary >= ordered.size()) return;
        bool old_active = old_left != old_right;
        bool new_active = new_left != new_right;
        if (old_active == new_active) return;
        int delta = new_active ? +1 : -1;
        cpu_low_worker_dense_delta_add(
            cand.d2, dense.boundary[boundary].pages_2m, delta);
        cpu_low_worker_dense_delta_add(
            cand.d4, dense.boundary[boundary].pages_4k, delta);
        cand.transition_delta += int64_t(delta) * ws.transition_weight[boundary];
    };

    for (;;) {
        CpuLowDomainWorkerUniqueScore current{unique2m, unique4k, transitions};
        CpuLowWorkerRunCandidate best;
        best.score = current;

        auto consider = [&](size_t begin, size_t end, int dst) {
            if (begin >= end) return;
            int src = owner[begin];
            for (size_t p = begin + 1; p < end; ++p)
                if (owner[p] != src) return;
            if (src == dst || dst < 0 || dst >= pool.workers) return;
            int d = original_domain[begin];
            if (dst / pool.domain_size != d) return;
            for (size_t p = begin; p < end; ++p)
                if (original_domain[p] != d) return;

            ++stats.candidate_evaluations;
            uint64_t cells = range_cells(begin, end);
            if (loads[size_t(src)] < cells) {
                std::cerr << "cpu LOW run coalesce source load underflow\n";
                std::exit(323);
            }
            if (cells > domain_cap[size_t(d)]
                || loads[size_t(dst)] > domain_cap[size_t(d)] - cells) {
                ++stats.cap_rejections;
                return;
            }

            CpuLowWorkerRunCandidate cand;
            cand.valid = true;
            cand.begin = begin;
            cand.end = end;
            cand.src = src;
            cand.dst = dst;
            cand.cells = cells;
            cand.score = current;
            cand.d2.entries.clear();
            cand.d4.entries.clear();

            if (begin > 0) {
                int old_l = owner[begin - 1];
                int new_l = old_l;
                add_edge_delta(cand, begin, old_l, src, new_l, dst);
            }
            if (end < ordered.size()) {
                int old_r = owner[end];
                int new_r = old_r;
                add_edge_delta(cand, end, src, old_r, dst, new_r);
            }
            stats.dense_delta_peak_entries = std::max(
                stats.dense_delta_peak_entries,
                std::max(cand.d2.entries.size(), cand.d4.entries.size()));
            cpu_low_worker_dense_delta_normalize(cand.d2);
            cpu_low_worker_dense_delta_normalize(cand.d4);

            int64_t tr = int64_t(transitions) + cand.transition_delta;
            if (tr < 0) {
                std::cerr << "cpu LOW run coalesce transition underflow\n";
                std::exit(262);
            }
            cand.score = {
                cpu_low_worker_dense_unique_after_delta(refs2m, unique2m, cand.d2),
                cpu_low_worker_dense_unique_after_delta(refs4k, unique4k, cand.d4),
                uint64_t(tr)};
            if (!cpu_low_worker_unique_score_less(cand.score, current)) return;

            bool take = !best.valid
                || cpu_low_worker_unique_score_less(cand.score, best.score)
                || (!cpu_low_worker_unique_score_less(best.score, cand.score)
                    && !cpu_low_worker_unique_score_less(cand.score, best.score)
                    && (end - begin < best.end - best.begin
                        || (end - begin == best.end - best.begin
                            && (begin < best.begin
                                || (begin == best.begin && dst < best.dst)))));
            if (take) best = std::move(cand);
        };

        for (int d = 0; d < domains; ++d) {
            auto seg = domain_segs[size_t(d)];
            if (seg.first >= seg.second) continue;
            size_t a = seg.first;
            while (a < seg.second) {
                int src = owner[a];
                size_t b = a + 1;
                while (b < seg.second && owner[b] == src) ++b;
                size_t len = b - a;
                uint32_t lim = uint32_t(std::min<size_t>(len, max_run));
                if (a > seg.first) {
                    int dst = owner[a - 1];
                    for (uint32_t r = 1; r <= lim; ++r)
                        consider(a, a + r, dst);
                }
                if (b < seg.second) {
                    int dst = owner[b];
                    for (uint32_t r = 1; r <= lim; ++r)
                        consider(b - r, b, dst);
                }
                a = b;
            }
        }

        if (!best.valid) break;
        cpu_low_worker_dense_apply_delta(refs2m, unique2m, best.d2);
        cpu_low_worker_dense_apply_delta(refs4k, unique4k, best.d4);
        int64_t next_transitions = int64_t(transitions) + best.transition_delta;
        if (next_transitions < 0) {
            std::cerr << "cpu LOW run coalesce accepted transition underflow\n";
            std::exit(263);
        }
        transitions = uint64_t(next_transitions);
        if (loads[size_t(best.src)] < best.cells) {
            std::cerr << "cpu LOW run coalesce accepted source load underflow\n";
            std::exit(324);
        }
        loads[size_t(best.src)] -= best.cells;
        loads[size_t(best.dst)] += best.cells;
        int d = best.dst / pool.domain_size;
        if (loads[size_t(best.dst)] > domain_cap[size_t(d)]) {
            std::cerr << "cpu LOW run coalesce cap violation\n";
            std::exit(264);
        }
        for (size_t p = best.begin; p < best.end; ++p)
            owner[p] = best.dst;

        ++stats.accepted_runs;
        stats.moved_jobs += best.end - best.begin;
        stats.moved_cells += best.cells;
        stats.max_run_used = std::max<uint32_t>(
            stats.max_run_used, uint32_t(best.end - best.begin));
        if (best.score.pages_2m < current.pages_2m
            || (best.score.pages_2m == current.pages_2m
                && best.score.pages_4k < current.pages_4k))
            ++stats.page_improving_runs;
        else
            ++stats.transition_only_runs;

        if (stats.accepted_runs >= max_accepted) {
            stats.move_limit_hit = true;
            break;
        }
    }

    uint64_t expected_cells = 0;
    for (const auto& x : ordered) expected_cells += x.cells;
    std::vector<std::vector<size_t>> next_jobs(size_t(pool.workers));
    std::vector<uint64_t> next_cells(size_t(pool.workers), 0);
    for (size_t k = 0; k < ordered.size(); ++k) {
        int w = owner[k];
        if (w < 0 || w >= pool.workers || w / pool.domain_size != original_domain[k]) {
            std::cerr << "cpu LOW run coalesce final owner mismatch\n";
            std::exit(265);
        }
        next_jobs[size_t(w)].push_back(ordered[k].index);
        next_cells[size_t(w)] += ordered[k].cells;
    }
    uint64_t assigned = 0;
    for (uint64_t x : next_cells) assigned += x;
    if (assigned != expected_cells) {
        std::cerr << "cpu LOW run coalesce cell accounting mismatch\n";
        std::exit(266);
    }
    stats.max_worker_cells_after = next_cells.empty() ? 0
        : *std::max_element(next_cells.begin(), next_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW run coalesce max-worker regression\n";
        std::exit(267);
    }

    std::vector<uint32_t> verify2m(refs2m.size(), 0), verify4k(refs4k.size(), 0);
    uint64_t verify_transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        cpu_low_worker_dense_ref_add(verify2m, dense.boundary[k].pages_2m, +1);
        cpu_low_worker_dense_ref_add(verify4k, dense.boundary[k].pages_4k, +1);
        verify_transitions += ws.transition_weight[k];
    }
    if (verify2m != refs2m || verify4k != refs4k
        || cpu_low_worker_dense_ref_unique(verify2m) != unique2m
        || cpu_low_worker_dense_ref_unique(verify4k) != unique4k
        || verify_transitions != transitions) {
        std::cerr << "cpu LOW run coalesce exact accounting mismatch\n";
        std::exit(268);
    }

    stats.unique_pages_2m_after = unique2m;
    stats.unique_pages_4k_after = unique4k;
    stats.owner_transitions_after = transitions;
    CpuLowDomainWorkerUniqueScore before{
        stats.unique_pages_2m_before,
        stats.unique_pages_4k_before,
        stats.owner_transitions_before};
    CpuLowDomainWorkerUniqueScore after{
        stats.unique_pages_2m_after,
        stats.unique_pages_4k_after,
        stats.owner_transitions_after};
    if (cpu_low_worker_unique_score_less(before, after)) {
        std::cerr << "cpu LOW run coalesce objective regression\n";
        std::exit(269);
    }

    pool.sticky_worker_jobs.swap(next_jobs);
    pool.sticky_worker_cells.swap(next_cells);
    stats.build_s = ram_seconds_since(t0);
    std::cerr << "cpu_low_domain_worker_unique_run_coalesce"
              << " objective=bounded-run-global-unique-v5.32-plan"
              << " max_run=" << stats.max_run_requested
              << " max_run_used=" << stats.max_run_used
              << " candidate_evaluations=" << stats.candidate_evaluations
              << " cap_rejections=" << stats.cap_rejections
              << " accepted_runs=" << stats.accepted_runs
              << " moved_jobs=" << stats.moved_jobs
              << " moved_cells=" << stats.moved_cells
              << " page_improving_runs=" << stats.page_improving_runs
              << " transition_only_runs=" << stats.transition_only_runs
              << " move_limit_hit=" << (stats.move_limit_hit ? 1 : 0)
              << " unique_pages_2m_before=" << stats.unique_pages_2m_before
              << " unique_pages_2m_after=" << stats.unique_pages_2m_after
              << " unique_pages_4k_before=" << stats.unique_pages_4k_before
              << " unique_pages_4k_after=" << stats.unique_pages_4k_after
              << " owner_transitions_before=" << stats.owner_transitions_before
              << " owner_transitions_after=" << stats.owner_transitions_after
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
