#pragma once

#include "ramstream32_cpu_low_domain_worker_exact_workspace.hpp"
#include "ramstream32_cpu_low_domain_worker_unique_run_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Research-only v5.33 bounded adjacent-run swaps.
// A one-way move may be blocked by the destination's domain cap.  Swapping a
// short suffix/prefix pair across one worker boundary can preserve both worker
// caps while changing the outer ownership boundaries atomically.

struct CpuLowDomainWorkerSwapStats {
    uint64_t candidate_evaluations = 0;
    uint64_t cap_rejections = 0;
    uint64_t accepted_swaps = 0;
    uint64_t moved_jobs = 0;
    uint64_t moved_cells = 0;
    uint64_t page_improving_swaps = 0;
    uint64_t transition_only_swaps = 0;
    uint32_t max_swap_requested = 0;
    uint32_t max_left_used = 0;
    uint32_t max_right_used = 0;
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

struct CpuLowWorkerSwapCandidate {
    bool valid = false;
    size_t left_begin = 0;
    size_t boundary = 0;
    size_t right_end = 0;
    int left_owner = -1;
    int right_owner = -1;
    uint64_t left_cells = 0;
    uint64_t right_cells = 0;
    CpuLowDomainWorkerUniqueScore score{};
    CpuLowWorkerDenseDelta d2;
    CpuLowWorkerDenseDelta d4;
    int64_t transition_delta = 0;
};

static CpuLowDomainWorkerSwapStats
cpu_low_apply_domain_worker_unique_swap_coalesce(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const CpuLowWorkerExactWorkspace& ws,
    uint32_t max_swap = 4,
    uint64_t max_accepted = 4096
) {
    CpuLowDomainWorkerSwapStats stats;
    auto t0 = std::chrono::steady_clock::now();
    cpu_low_validate_worker_exact_workspace(ws, jobs, sparse);
    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || pool.domain_size <= 0) {
        std::cerr << "cpu LOW swap coalesce requires domain schedule\n";
        std::exit(270);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW swap coalesce requires prepared schedule\n";
        std::exit(271);
    }
    if (!max_swap) {
        std::cerr << "cpu LOW swap coalesce max_swap must be positive\n";
        std::exit(272);
    }
    stats.max_swap_requested = max_swap;

    const auto& ordered = ws.ordered;
    const auto& dense = ws.dense;
    std::vector<int> owner(ordered.size(), -1), original_domain(ordered.size(), -1);
    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size() || ws.ordered_pos[q] == size_t(-1)) {
                std::cerr << "cpu LOW swap coalesce bad job owner\n";
                std::exit(273);
            }
            size_t k = ws.ordered_pos[q];
            if (owner[k] >= 0) {
                std::cerr << "cpu LOW swap coalesce duplicate job owner\n";
                std::exit(274);
            }
            owner[k] = w;
            original_domain[k] = d;
        }
    }

    int domains = (pool.workers + pool.domain_size - 1) / pool.domain_size;
    std::vector<std::pair<size_t,size_t>> domain_segs(
        size_t(domains), {ordered.size(), ordered.size()});
    int last_domain = -1;
    for (size_t k = 0; k < ordered.size(); ++k) {
        int d = original_domain[k];
        if (owner[k] < 0 || d < 0 || d >= domains || d < last_domain) {
            std::cerr << "cpu LOW swap coalesce lost domain ordering\n";
            std::exit(275);
        }
        last_domain = d;
        if (domain_segs[size_t(d)].first == ordered.size())
            domain_segs[size_t(d)].first = k;
        domain_segs[size_t(d)].second = k + 1;
    }

    std::vector<uint64_t> loads = pool.sticky_worker_cells;
    std::vector<uint64_t> domain_cap(size_t(domains), 0);
    for (int w = 0; w < pool.workers; ++w)
        domain_cap[size_t(w / pool.domain_size)] = std::max(
            domain_cap[size_t(w / pool.domain_size)], loads[size_t(w)]);
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
    auto range_cells = [&](size_t a, size_t b) { return prefix[b] - prefix[a]; };

    auto add_edge_delta = [&](CpuLowWorkerSwapCandidate& cand,
                              size_t boundary,
                              int old_left, int old_right,
                              int new_left, int new_right) {
        if (!boundary || boundary >= ordered.size()) return;
        bool old_active = old_left != old_right;
        bool new_active = new_left != new_right;
        if (old_active == new_active) return;
        int delta = new_active ? +1 : -1;
        cpu_low_worker_dense_delta_add(cand.d2, dense.boundary[boundary].pages_2m, delta);
        cpu_low_worker_dense_delta_add(cand.d4, dense.boundary[boundary].pages_4k, delta);
        cand.transition_delta += int64_t(delta) * ws.transition_weight[boundary];
    };

    for (;;) {
        CpuLowDomainWorkerUniqueScore current{unique2m, unique4k, transitions};
        CpuLowWorkerSwapCandidate best;
        best.score = current;

        for (int d = 0; d < domains; ++d) {
            auto seg = domain_segs[size_t(d)];
            if (seg.first >= seg.second) continue;
            size_t k = seg.first + 1;
            while (k < seg.second) {
                if (owner[k - 1] == owner[k]) { ++k; continue; }
                int a_owner = owner[k - 1];
                int b_owner = owner[k];
                size_t a = k - 1;
                while (a > seg.first && owner[a - 1] == a_owner) --a;
                size_t b = k + 1;
                while (b < seg.second && owner[b] == b_owner) ++b;
                uint32_t la_max = uint32_t(std::min<size_t>(k - a, max_swap));
                uint32_t lb_max = uint32_t(std::min<size_t>(b - k, max_swap));

                for (uint32_t la = 1; la <= la_max; ++la) {
                    size_t left_begin = k - la;
                    uint64_t left_cells = range_cells(left_begin, k);
                    for (uint32_t lb = 1; lb <= lb_max; ++lb) {
                        ++stats.candidate_evaluations;
                        size_t right_end = k + lb;
                        uint64_t right_cells = range_cells(k, right_end);
                        uint64_t load_a = loads[size_t(a_owner)];
                        uint64_t load_b = loads[size_t(b_owner)];
                        if (load_a < left_cells || load_b < right_cells) {
                            std::cerr << "cpu LOW swap coalesce source load underflow\n";
                            std::exit(325);
                        }
                        uint64_t base_a = load_a - left_cells;
                        uint64_t base_b = load_b - right_cells;
                        uint64_t cap = domain_cap[size_t(d)];
                        if (base_a > cap || base_b > cap) {
                            std::cerr << "cpu LOW swap coalesce preexisting cap violation\n";
                            std::exit(326);
                        }
                        if (right_cells > cap - base_a || left_cells > cap - base_b) {
                            ++stats.cap_rejections;
                            continue;
                        }
                        uint64_t new_a = base_a + right_cells;
                        uint64_t new_b = base_b + left_cells;

                        CpuLowWorkerSwapCandidate cand;
                        cand.valid = true;
                        cand.left_begin = left_begin;
                        cand.boundary = k;
                        cand.right_end = right_end;
                        cand.left_owner = a_owner;
                        cand.right_owner = b_owner;
                        cand.left_cells = left_cells;
                        cand.right_cells = right_cells;
                        cand.score = current;

                        auto new_owner_at = [&](size_t p) -> int {
                            if (p >= left_begin && p < k) return b_owner;
                            if (p >= k && p < right_end) return a_owner;
                            return owner[p];
                        };
                        size_t edges[3] = {left_begin, k, right_end};
                        for (size_t edge : edges) {
                            if (!edge || edge >= ordered.size()) continue;
                            add_edge_delta(
                                cand, edge,
                                owner[edge - 1], owner[edge],
                                new_owner_at(edge - 1), new_owner_at(edge));
                        }
                        stats.dense_delta_peak_entries = std::max(
                            stats.dense_delta_peak_entries,
                            std::max(cand.d2.entries.size(), cand.d4.entries.size()));
                        cpu_low_worker_dense_delta_normalize(cand.d2);
                        cpu_low_worker_dense_delta_normalize(cand.d4);
                        int64_t tr = int64_t(transitions) + cand.transition_delta;
                        if (tr < 0) {
                            std::cerr << "cpu LOW swap coalesce transition underflow\n";
                            std::exit(276);
                        }
                        cand.score = {
                            cpu_low_worker_dense_unique_after_delta(refs2m, unique2m, cand.d2),
                            cpu_low_worker_dense_unique_after_delta(refs4k, unique4k, cand.d4),
                            uint64_t(tr)};
                        if (!cpu_low_worker_unique_score_less(cand.score, current)) continue;

                        uint32_t cand_jobs = la + lb;
                        uint32_t best_jobs = best.valid
                            ? uint32_t((best.boundary - best.left_begin)
                                + (best.right_end - best.boundary)) : 0;
                        bool take = !best.valid
                            || cpu_low_worker_unique_score_less(cand.score, best.score)
                            || (!cpu_low_worker_unique_score_less(best.score, cand.score)
                                && !cpu_low_worker_unique_score_less(cand.score, best.score)
                                && (cand_jobs < best_jobs
                                    || (cand_jobs == best_jobs
                                        && (left_begin < best.left_begin
                                            || (left_begin == best.left_begin
                                                && right_end < best.right_end)))));
                        if (take) best = std::move(cand);
                        (void)new_a;
                        (void)new_b;
                    }
                }
                k = b;
            }
        }

        if (!best.valid) break;
        cpu_low_worker_dense_apply_delta(refs2m, unique2m, best.d2);
        cpu_low_worker_dense_apply_delta(refs4k, unique4k, best.d4);
        int64_t tr = int64_t(transitions) + best.transition_delta;
        if (tr < 0) {
            std::cerr << "cpu LOW swap coalesce accepted transition underflow\n";
            std::exit(277);
        }
        transitions = uint64_t(tr);
        uint64_t load_a = loads[size_t(best.left_owner)];
        uint64_t load_b = loads[size_t(best.right_owner)];
        if (load_a < best.left_cells || load_b < best.right_cells) {
            std::cerr << "cpu LOW swap coalesce accepted source load underflow\n";
            std::exit(327);
        }
        uint64_t base_a = load_a - best.left_cells;
        uint64_t base_b = load_b - best.right_cells;
        int d = best.left_owner / pool.domain_size;
        if (best.right_owner / pool.domain_size != d
            || base_a > domain_cap[size_t(d)] || base_b > domain_cap[size_t(d)]
            || best.right_cells > domain_cap[size_t(d)] - base_a
            || best.left_cells > domain_cap[size_t(d)] - base_b) {
            std::cerr << "cpu LOW swap coalesce accepted cap/domain violation\n";
            std::exit(328);
        }
        loads[size_t(best.left_owner)] = base_a + best.right_cells;
        loads[size_t(best.right_owner)] = base_b + best.left_cells;
        if (loads[size_t(best.left_owner)] > domain_cap[size_t(d)]
            || loads[size_t(best.right_owner)] > domain_cap[size_t(d)]) {
            std::cerr << "cpu LOW swap coalesce cap/domain violation\n";
            std::exit(278);
        }
        for (size_t p = best.left_begin; p < best.boundary; ++p)
            owner[p] = best.right_owner;
        for (size_t p = best.boundary; p < best.right_end; ++p)
            owner[p] = best.left_owner;

        ++stats.accepted_swaps;
        uint32_t lj = uint32_t(best.boundary - best.left_begin);
        uint32_t rj = uint32_t(best.right_end - best.boundary);
        stats.moved_jobs += lj + rj;
        stats.moved_cells += best.left_cells + best.right_cells;
        stats.max_left_used = std::max(stats.max_left_used, lj);
        stats.max_right_used = std::max(stats.max_right_used, rj);
        if (best.score.pages_2m < current.pages_2m
            || (best.score.pages_2m == current.pages_2m
                && best.score.pages_4k < current.pages_4k))
            ++stats.page_improving_swaps;
        else
            ++stats.transition_only_swaps;
        if (stats.accepted_swaps >= max_accepted) {
            stats.move_limit_hit = true;
            break;
        }
    }

    uint64_t expected = 0;
    for (const auto& x : ordered) expected += x.cells;
    std::vector<std::vector<size_t>> next_jobs(size_t(pool.workers));
    std::vector<uint64_t> next_cells(size_t(pool.workers), 0);
    for (size_t k = 0; k < ordered.size(); ++k) {
        int w = owner[k];
        if (w < 0 || w >= pool.workers || w / pool.domain_size != original_domain[k]) {
            std::cerr << "cpu LOW swap coalesce final owner mismatch\n";
            std::exit(279);
        }
        next_jobs[size_t(w)].push_back(ordered[k].index);
        next_cells[size_t(w)] += ordered[k].cells;
    }
    uint64_t assigned = 0;
    for (uint64_t x : next_cells) assigned += x;
    if (assigned != expected) {
        std::cerr << "cpu LOW swap coalesce accounting mismatch\n";
        std::exit(280);
    }
    stats.max_worker_cells_after = next_cells.empty() ? 0
        : *std::max_element(next_cells.begin(), next_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW swap coalesce max-worker regression\n";
        std::exit(281);
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
        std::cerr << "cpu LOW swap coalesce exact accounting mismatch\n";
        std::exit(282);
    }

    stats.unique_pages_2m_after = unique2m;
    stats.unique_pages_4k_after = unique4k;
    stats.owner_transitions_after = transitions;
    CpuLowDomainWorkerUniqueScore before{
        stats.unique_pages_2m_before, stats.unique_pages_4k_before,
        stats.owner_transitions_before};
    CpuLowDomainWorkerUniqueScore after{
        stats.unique_pages_2m_after, stats.unique_pages_4k_after,
        stats.owner_transitions_after};
    if (cpu_low_worker_unique_score_less(before, after)) {
        std::cerr << "cpu LOW swap coalesce objective regression\n";
        std::exit(283);
    }

    pool.sticky_worker_jobs.swap(next_jobs);
    pool.sticky_worker_cells.swap(next_cells);
    stats.build_s = ram_seconds_since(t0);
    std::cerr << "cpu_low_domain_worker_unique_swap_coalesce"
              << " objective=bounded-swap-global-unique-v5.33-plan"
              << " max_swap=" << stats.max_swap_requested
              << " max_left_used=" << stats.max_left_used
              << " max_right_used=" << stats.max_right_used
              << " candidate_evaluations=" << stats.candidate_evaluations
              << " cap_rejections=" << stats.cap_rejections
              << " accepted_swaps=" << stats.accepted_swaps
              << " moved_jobs=" << stats.moved_jobs
              << " moved_cells=" << stats.moved_cells
              << " page_improving_swaps=" << stats.page_improving_swaps
              << " transition_only_swaps=" << stats.transition_only_swaps
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
