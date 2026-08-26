#pragma once

#include "ramstream32_cpu_low_domain_worker_dense_page.hpp"
#include "ramstream32_cpu_low_domain_worker_unique_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// Research-only v5.30 exact coalescer using dense page IDs and vector refcounts.
// Control flow, candidate order, legal moves and score ordering intentionally
// mirror v5.29. Only page representation/refcount storage changes.

struct CpuLowDomainWorkerUniqueDenseStats {
    int domains = 0;
    int nonempty_domains = 0;
    int noncontiguous_domains_before = 0;
    int improved_domains = 0;
    int contiguous_domains_after = 0;
    uint64_t candidate_evaluations = 0;
    uint64_t cap_rejections = 0;
    uint64_t accepted_moves = 0;
    uint64_t unique_page_improving_moves = 0;
    uint64_t transition_only_moves = 0;
    uint64_t moved_cells = 0;
    uint64_t unique_pages_2m_before = 0;
    uint64_t unique_pages_2m_after = 0;
    uint64_t unique_pages_4k_before = 0;
    uint64_t unique_pages_4k_after = 0;
    uint64_t owner_transitions_before = 0;
    uint64_t owner_transitions_after = 0;
    uint64_t dense_delta_normalizations = 0;
    size_t dense_delta_peak_entries = 0;
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    size_t dense_index_bytes = 0;
    double dense_index_build_s = 0.0;
    double mask_index_mib = 0.0;
    double mask_index_build_s = 0.0;
    double build_s = 0.0;
};

static CpuLowDomainWorkerUniqueScore cpu_low_worker_unique_dense_after_score(
    const CpuLowDomainWorkerUniqueDenseStats& s
) {
    return {s.unique_pages_2m_after,
            s.unique_pages_4k_after,
            s.owner_transitions_after};
}

static CpuLowDomainWorkerUniqueDenseStats
cpu_low_apply_domain_worker_unique_dense_coalesce(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    constexpr int PASSES = 12;
    CpuLowDomainWorkerUniqueDenseStats stats;
    auto t0 = std::chrono::steady_clock::now();

    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || pool.domain_size <= 0) {
        std::cerr << "cpu LOW dense unique coalesce requires domain schedule\n";
        std::exit(223);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW dense unique coalesce requires prepared schedule\n";
        std::exit(224);
    }

    std::vector<CpuLowStaticJobCost> ordered;
    ordered.reserve(jobs.size());
    uint64_t expected_cells = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].main_size && !jobs[i].block_size) continue;
        uint64_t cells = cpu_low_sparse_job_cells(jobs[i], sparse);
        ordered.push_back({i, jobs[i].mask, cells});
        expected_cells += cells;
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        if (a.mask != b.mask) return a.mask < b.mask;
        return a.index < b.index;
    });

    stats.domains = (pool.workers + pool.domain_size - 1) / pool.domain_size;
    stats.max_worker_cells_before = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(
            pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());

    std::vector<int> owner(ordered.size(), -1);
    std::vector<int> original_domain(ordered.size(), -1);
    std::vector<size_t> ordered_pos(jobs.size(), size_t(-1));
    for (size_t i = 0; i < ordered.size(); ++i)
        ordered_pos[ordered[i].index] = i;

    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size() || ordered_pos[q] == size_t(-1)) {
                std::cerr << "cpu LOW dense unique bad job owner\n";
                std::exit(225);
            }
            size_t k = ordered_pos[q];
            if (owner[k] >= 0) {
                std::cerr << "cpu LOW dense unique duplicate job owner\n";
                std::exit(226);
            }
            owner[k] = w;
            original_domain[k] = d;
        }
    }

    std::vector<std::pair<size_t,size_t>> domain_segs(
        size_t(stats.domains), {ordered.size(), ordered.size()});
    int last_domain = -1;
    for (size_t k = 0; k < ordered.size(); ++k) {
        int d = original_domain[k];
        if (owner[k] < 0 || d < 0 || d >= stats.domains || d < last_domain) {
            std::cerr << "cpu LOW dense unique lost domain ordering\n";
            std::exit(227);
        }
        last_domain = d;
        if (domain_segs[size_t(d)].first == ordered.size())
            domain_segs[size_t(d)].first = k;
        domain_segs[size_t(d)].second = k + 1;
    }

    auto index_t0 = std::chrono::steady_clock::now();
    CpuLowDomainPageMaskIndex mask_index = cpu_low_build_domain_page_mask_index();
    stats.mask_index_build_s = ram_seconds_since(index_t0);
    stats.mask_index_mib = double(
        mask_index.first_nonempty.size() * sizeof(uint32_t)
        + mask_index.next_nonempty.size() * sizeof(uint32_t)) / double(1 << 20);

    CpuLowWorkerDensePageIndex dense = cpu_low_build_worker_dense_page_index(
        ordered, layout, storage, mask_index);
    stats.dense_index_bytes = dense.bytes();
    stats.dense_index_build_s = dense.build_s;

    std::vector<uint32_t> transition_weight_cache(ordered.size() + 1, 0);
    std::vector<uint8_t> transition_weight_ready(ordered.size() + 1, 0);
    auto transition_weight = [&](size_t boundary) -> uint32_t {
        if (!boundary || boundary >= ordered.size()) {
            std::cerr << "cpu LOW dense unique transition boundary out of range\n";
            std::exit(228);
        }
        if (!transition_weight_ready[boundary]) {
            transition_weight_cache[boundary] = cpu_low_worker_unique_transition_weight(
                layout, mask_index, ordered[boundary].mask);
            transition_weight_ready[boundary] = 1;
        }
        return transition_weight_cache[boundary];
    };

    std::vector<uint32_t> refs2m(dense.universe_2m.size(), 0);
    std::vector<uint32_t> refs4k(dense.universe_4k.size(), 0);
    uint64_t unique2m = 0, unique4k = 0, transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        const auto& sig = dense.boundary[k];
        cpu_low_worker_dense_ref_add(refs2m, sig.pages_2m, +1);
        cpu_low_worker_dense_ref_add(refs4k, sig.pages_4k, +1);
        transitions += transition_weight(k);
    }
    unique2m = cpu_low_worker_dense_ref_unique(refs2m);
    unique4k = cpu_low_worker_dense_ref_unique(refs4k);
    stats.unique_pages_2m_before = unique2m;
    stats.unique_pages_4k_before = unique4k;
    stats.owner_transitions_before = transitions;

    CpuLowWorkerDenseDelta candidate_d2, candidate_d4, best_d2, best_d4;
    size_t reserve2 = 0, reserve4 = 0;
    for (size_t k = 1; k < dense.boundary.size(); ++k) {
        reserve2 = std::max(reserve2, dense.boundary[k].pages_2m.size());
        reserve4 = std::max(reserve4, dense.boundary[k].pages_4k.size());
    }
    candidate_d2.entries.reserve(2 * reserve2);
    candidate_d4.entries.reserve(2 * reserve4);
    best_d2.entries.reserve(2 * reserve2);
    best_d4.entries.reserve(2 * reserve4);

    std::vector<uint64_t> loads = pool.sticky_worker_cells;
    for (int d = 0; d < stats.domains; ++d) {
        auto seg = domain_segs[size_t(d)];
        if (seg.first >= seg.second) continue;
        ++stats.nonempty_domains;
        int first_worker = d * pool.domain_size;
        int nworkers = std::min(pool.domain_size, pool.workers - first_worker);
        if (nworkers <= 1 || seg.second - seg.first <= 1) {
            ++stats.contiguous_domains_after;
            continue;
        }
        if (cpu_low_worker_domain_is_contiguous(
                owner, seg.first, seg.second, first_worker, nworkers)) {
            ++stats.contiguous_domains_after;
            continue;
        }
        ++stats.noncontiguous_domains_before;

        uint64_t cap = 0;
        for (int w = first_worker; w < first_worker + nworkers; ++w)
            cap = std::max(cap, loads[size_t(w)]);

        bool domain_improved = false;
        for (int pass = 0; pass < PASSES; ++pass) {
            bool changed = false;
            size_t count = seg.second - seg.first;
            for (size_t qq = 0; qq < count; ++qq) {
                size_t i = (pass & 1) ? (seg.second - 1 - qq) : (seg.first + qq);
                int src = owner[i];
                uint64_t cells = ordered[i].cells;

                int candidates[2] = {-1, -1};
                int nc = 0;
                if (i > seg.first && owner[i - 1] != src)
                    candidates[nc++] = owner[i - 1];
                if (i + 1 < seg.second && owner[i + 1] != src
                    && (nc == 0 || owner[i + 1] != candidates[0]))
                    candidates[nc++] = owner[i + 1];
                if (!nc) continue;

                CpuLowDomainWorkerUniqueScore current{unique2m, unique4k, transitions};
                CpuLowDomainWorkerUniqueScore best = current;
                int best_dst = -1;
                int64_t best_transition_delta = 0;
                best_d2.entries.clear();
                best_d4.entries.clear();

                for (int c = 0; c < nc; ++c) {
                    int dst = candidates[c];
                    ++stats.candidate_evaluations;
                    if (dst < first_worker || dst >= first_worker + nworkers) {
                        std::cerr << "cpu LOW dense unique crossed domain\n";
                        std::exit(229);
                    }
                    if (loads[size_t(dst)] > cap - cells) {
                        ++stats.cap_rejections;
                        continue;
                    }

                    candidate_d2.entries.clear();
                    candidate_d4.entries.clear();
                    int64_t transition_delta = 0;
                    auto change_edge = [&](size_t boundary,
                                           int old_left, int old_right,
                                           int new_left, int new_right) {
                        bool old_active = old_left != old_right;
                        bool new_active = new_left != new_right;
                        if (old_active == new_active) return;
                        int delta = new_active ? +1 : -1;
                        const auto& sig = dense.boundary[boundary];
                        cpu_low_worker_dense_delta_add(candidate_d2, sig.pages_2m, delta);
                        cpu_low_worker_dense_delta_add(candidate_d4, sig.pages_4k, delta);
                        transition_delta += int64_t(delta) * transition_weight(boundary);
                    };

                    if (i > 0)
                        change_edge(i,
                            owner[i - 1], src,
                            owner[i - 1], dst);
                    if (i + 1 < ordered.size())
                        change_edge(i + 1,
                            src, owner[i + 1],
                            dst, owner[i + 1]);

                    stats.dense_delta_peak_entries = std::max(
                        stats.dense_delta_peak_entries,
                        std::max(candidate_d2.entries.size(), candidate_d4.entries.size()));
                    cpu_low_worker_dense_delta_normalize(candidate_d2);
                    cpu_low_worker_dense_delta_normalize(candidate_d4);
                    stats.dense_delta_normalizations += 2;

                    int64_t candidate_transitions = int64_t(transitions) + transition_delta;
                    if (candidate_transitions < 0) {
                        std::cerr << "cpu LOW dense unique transition underflow\n";
                        std::exit(230);
                    }
                    CpuLowDomainWorkerUniqueScore candidate{
                        cpu_low_worker_dense_unique_after_delta(
                            refs2m, unique2m, candidate_d2),
                        cpu_low_worker_dense_unique_after_delta(
                            refs4k, unique4k, candidate_d4),
                        uint64_t(candidate_transitions)};
                    if (cpu_low_worker_unique_score_less(candidate, best)) {
                        best = candidate;
                        best_dst = dst;
                        best_d2 = candidate_d2;
                        best_d4 = candidate_d4;
                        best_transition_delta = transition_delta;
                    }
                }

                if (best_dst >= 0) {
                    bool page_improved = best.pages_2m < current.pages_2m
                        || (best.pages_2m == current.pages_2m
                            && best.pages_4k < current.pages_4k);
                    cpu_low_worker_dense_apply_delta(refs2m, unique2m, best_d2);
                    cpu_low_worker_dense_apply_delta(refs4k, unique4k, best_d4);
                    int64_t next_transitions = int64_t(transitions) + best_transition_delta;
                    if (next_transitions < 0) {
                        std::cerr << "cpu LOW dense unique accepted transition underflow\n";
                        std::exit(231);
                    }
                    transitions = uint64_t(next_transitions);
                    loads[size_t(src)] -= cells;
                    loads[size_t(best_dst)] += cells;
                    if (loads[size_t(best_dst)] > cap) {
                        std::cerr << "cpu LOW dense unique cap violation\n";
                        std::exit(232);
                    }
                    owner[i] = best_dst;
                    ++stats.accepted_moves;
                    stats.moved_cells += cells;
                    if (page_improved) ++stats.unique_page_improving_moves;
                    else ++stats.transition_only_moves;
                    changed = true;
                    domain_improved = true;
                }
            }
            if (!changed) break;
        }
        if (domain_improved) ++stats.improved_domains;
        if (cpu_low_worker_domain_is_contiguous(
                owner, seg.first, seg.second, first_worker, nworkers))
            ++stats.contiguous_domains_after;
    }

    std::vector<std::vector<size_t>> next_jobs(size_t(pool.workers));
    std::vector<uint64_t> next_cells(size_t(pool.workers), 0);
    for (size_t k = 0; k < ordered.size(); ++k) {
        int w = owner[k];
        if (w < 0 || w >= pool.workers) {
            std::cerr << "cpu LOW dense unique invalid worker owner\n";
            std::exit(233);
        }
        int d = w / pool.domain_size;
        if (d != original_domain[k]) {
            std::cerr << "cpu LOW dense unique domain provenance mismatch\n";
            std::exit(234);
        }
        next_jobs[size_t(w)].push_back(ordered[k].index);
        next_cells[size_t(w)] += ordered[k].cells;
    }

    uint64_t assigned_cells = 0;
    for (uint64_t x : next_cells) assigned_cells += x;
    if (assigned_cells != expected_cells) {
        std::cerr << "cpu LOW dense unique accounting mismatch"
                  << " assigned=" << assigned_cells
                  << " expected=" << expected_cells << '\n';
        std::exit(235);
    }

    stats.max_worker_cells_after = next_cells.empty() ? 0
        : *std::max_element(next_cells.begin(), next_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW dense unique increased max worker load\n";
        std::exit(236);
    }

    std::vector<uint32_t> verify2m(refs2m.size(), 0), verify4k(refs4k.size(), 0);
    uint64_t verify_transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        const auto& sig = dense.boundary[k];
        cpu_low_worker_dense_ref_add(verify2m, sig.pages_2m, +1);
        cpu_low_worker_dense_ref_add(verify4k, sig.pages_4k, +1);
        verify_transitions += transition_weight(k);
    }
    if (verify2m != refs2m || verify4k != refs4k
        || verify_transitions != transitions
        || cpu_low_worker_dense_ref_unique(verify2m) != unique2m
        || cpu_low_worker_dense_ref_unique(verify4k) != unique4k) {
        std::cerr << "cpu LOW dense unique incremental accounting mismatch\n";
        std::exit(237);
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
        std::cerr << "cpu LOW dense unique exact objective regression\n";
        std::exit(238);
    }

    pool.sticky_worker_jobs.swap(next_jobs);
    pool.sticky_worker_cells.swap(next_cells);
    stats.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_domain_worker_unique_dense_coalesce"
              << " objective=global-unique-neighbor-coalesce-v5.27-plan"
              << " implementation=dense-page-ref-v5.30"
              << " domains=" << stats.domains
              << " nonempty_domains=" << stats.nonempty_domains
              << " noncontiguous_domains_before=" << stats.noncontiguous_domains_before
              << " improved_domains=" << stats.improved_domains
              << " contiguous_domains_after=" << stats.contiguous_domains_after
              << " candidate_evaluations=" << stats.candidate_evaluations
              << " cap_rejections=" << stats.cap_rejections
              << " accepted_moves=" << stats.accepted_moves
              << " unique_page_improving_moves=" << stats.unique_page_improving_moves
              << " transition_only_moves=" << stats.transition_only_moves
              << " moved_cells=" << stats.moved_cells
              << " unique_pages_2m_before=" << stats.unique_pages_2m_before
              << " unique_pages_2m_after=" << stats.unique_pages_2m_after
              << " unique_pages_4k_before=" << stats.unique_pages_4k_before
              << " unique_pages_4k_after=" << stats.unique_pages_4k_after
              << " owner_transitions_before=" << stats.owner_transitions_before
              << " owner_transitions_after=" << stats.owner_transitions_after
              << " dense_delta_normalizations=" << stats.dense_delta_normalizations
              << " dense_delta_peak_entries=" << stats.dense_delta_peak_entries
              << " dense_index_mib="
              << double(stats.dense_index_bytes) / double(1 << 20)
              << " dense_index_build_s=" << stats.dense_index_build_s
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " mask_index_mib=" << stats.mask_index_mib
              << " mask_index_build_s=" << stats.mask_index_build_s
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
