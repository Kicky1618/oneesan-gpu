#pragma once

#include "ramstream32_cpu_low_domain_page.hpp"
#include "ramstream32_cpu_low_domain_worker_locality.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// Research-only v5.26 worker-boundary coalescing stage.
//
// v5.25 converts a whole domain to contiguous worker intervals when that is
// possible under the domain's existing LPT maximum.  Some domains cannot be
// fully converted without exceeding that cap.  This stage attacks those
// remaining non-contiguous owner patterns locally.
//
// A move reassigns one ordered HIGH-mask job only to the owner of its immediate
// left or right neighbour.  The destination must be in the same domain and its
// new load must remain <= the domain's pre-pass maximum.  Therefore domain
// boundaries never move and the global maximum worker load cannot increase.
//
// Only the two owner boundaries adjacent to the moved job can change.  Among
// cap-safe moves, the pass accepts a move iff the affected local tuple
//
//   (sum of 2 MiB boundary-page penalties,
//    sum of 4 KiB boundary-page penalties,
//    owner-transition count)
//
// decreases lexicographically.  Since all other boundaries are unchanged, the
// complete objective decreases monotonically after every accepted move.

struct CpuLowDomainWorkerCoalesceScore {
    uint64_t pages_2m = 0;
    uint64_t pages_4k = 0;
    uint64_t transitions = 0;
};

struct CpuLowDomainWorkerCoalesceStats {
    int domains = 0;
    int nonempty_domains = 0;
    int noncontiguous_domains_before = 0;
    int improved_domains = 0;
    int contiguous_domains_after = 0;
    uint64_t candidate_evaluations = 0;
    uint64_t cap_rejections = 0;
    uint64_t accepted_moves = 0;
    uint64_t page_improving_moves = 0;
    uint64_t transition_only_moves = 0;
    uint64_t moved_cells = 0;
    uint64_t penalty_2m_before = 0;
    uint64_t penalty_2m_after = 0;
    uint64_t penalty_4k_before = 0;
    uint64_t penalty_4k_after = 0;
    uint64_t owner_transitions_before = 0;
    uint64_t owner_transitions_after = 0;
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double mask_index_mib = 0.0;
    double mask_index_build_s = 0.0;
    double build_s = 0.0;
};

static bool cpu_low_worker_coalesce_score_less(
    const CpuLowDomainWorkerCoalesceScore& a,
    const CpuLowDomainWorkerCoalesceScore& b
) {
    return a.pages_2m < b.pages_2m
        || (a.pages_2m == b.pages_2m && a.pages_4k < b.pages_4k)
        || (a.pages_2m == b.pages_2m && a.pages_4k == b.pages_4k
            && a.transitions < b.transitions);
}

static CpuLowDomainWorkerCoalesceScore cpu_low_worker_coalesce_score_add(
    CpuLowDomainWorkerCoalesceScore a,
    const CpuLowDomainWorkerCoalesceScore& b
) {
    a.pages_2m += b.pages_2m;
    a.pages_4k += b.pages_4k;
    a.transitions += b.transitions;
    return a;
}

static CpuLowDomainWorkerCoalesceScore cpu_low_worker_boundary_score(
    size_t boundary,
    const std::vector<CpuLowStaticJobCost>& ordered,
    const std::vector<int>& owner,
    const StorageLayout& layout,
    const StorageFactorHost& storage,
    const CpuLowDomainPageMaskIndex& mask_index,
    std::vector<CpuLowDomainBoundaryPagePenalty>& penalty_cache,
    std::vector<uint8_t>& penalty_ready
) {
    if (!boundary || boundary >= ordered.size()) return {};
    if (owner[boundary - 1] == owner[boundary]) return {};
    if (!penalty_ready[boundary]) {
        penalty_cache[boundary] = cpu_low_domain_boundary_page_penalty(
            layout, storage, mask_index, ordered[boundary].mask);
        penalty_ready[boundary] = 1;
    }
    const auto& p = penalty_cache[boundary];
    return {p.pages_2m, p.pages_4k, 1};
}

static CpuLowDomainWorkerCoalesceScore cpu_low_worker_local_score_with_owner(
    size_t i, size_t begin, size_t end, int replacement_owner,
    const std::vector<CpuLowStaticJobCost>& ordered,
    const std::vector<int>& owner,
    const StorageLayout& layout,
    const StorageFactorHost& storage,
    const CpuLowDomainPageMaskIndex& mask_index,
    std::vector<CpuLowDomainBoundaryPagePenalty>& penalty_cache,
    std::vector<uint8_t>& penalty_ready
) {
    CpuLowDomainWorkerCoalesceScore z;
    auto edge = [&](size_t boundary, int left_owner, int right_owner) {
        if (left_owner == right_owner) return;
        if (!penalty_ready[boundary]) {
            penalty_cache[boundary] = cpu_low_domain_boundary_page_penalty(
                layout, storage, mask_index, ordered[boundary].mask);
            penalty_ready[boundary] = 1;
        }
        const auto& p = penalty_cache[boundary];
        z.pages_2m += p.pages_2m;
        z.pages_4k += p.pages_4k;
        ++z.transitions;
    };

    if (i > begin)
        edge(i, owner[i - 1], replacement_owner);
    if (i + 1 < end)
        edge(i + 1, replacement_owner, owner[i + 1]);
    return z;
}

static bool cpu_low_worker_domain_is_contiguous(
    const std::vector<int>& owner,
    size_t begin, size_t end,
    int first_worker, int nworkers
) {
    if (begin >= end) return true;
    std::vector<uint8_t> closed(size_t(nworkers), 0);
    int prev = owner[begin];
    if (prev < first_worker || prev >= first_worker + nworkers) return false;
    for (size_t i = begin + 1; i < end; ++i) {
        int cur = owner[i];
        if (cur < first_worker || cur >= first_worker + nworkers) return false;
        if (cur == prev) continue;
        closed[size_t(prev - first_worker)] = 1;
        if (closed[size_t(cur - first_worker)]) return false;
        prev = cur;
    }
    return true;
}

static CpuLowDomainWorkerCoalesceStats cpu_low_apply_domain_worker_coalesce(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    constexpr int PASSES = 12;
    CpuLowDomainWorkerCoalesceStats stats;
    auto t0 = std::chrono::steady_clock::now();

    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || pool.domain_size <= 0) {
        std::cerr << "cpu LOW domain worker coalesce requires domain schedule\n";
        std::exit(179);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW domain worker coalesce requires prepared schedule\n";
        std::exit(180);
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
                std::cerr << "cpu LOW domain worker coalesce bad job owner\n";
                std::exit(181);
            }
            size_t k = ordered_pos[q];
            if (owner[k] >= 0) {
                std::cerr << "cpu LOW domain worker coalesce duplicate job owner\n";
                std::exit(182);
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
            std::cerr << "cpu LOW domain worker coalesce lost domain ordering\n";
            std::exit(183);
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

    std::vector<CpuLowDomainBoundaryPagePenalty> penalty_cache(ordered.size() + 1);
    std::vector<uint8_t> penalty_ready(ordered.size() + 1, 0);
    auto total_score = [&]() {
        CpuLowDomainWorkerCoalesceScore z;
        for (int d = 0; d < stats.domains; ++d) {
            auto seg = domain_segs[size_t(d)];
            if (seg.first >= seg.second) continue;
            for (size_t k = seg.first + 1; k < seg.second; ++k)
                z = cpu_low_worker_coalesce_score_add(
                    z, cpu_low_worker_boundary_score(
                        k, ordered, owner, layout, storage, mask_index,
                        penalty_cache, penalty_ready));
        }
        return z;
    };

    auto before = total_score();
    stats.penalty_2m_before = before.pages_2m;
    stats.penalty_4k_before = before.pages_4k;
    stats.owner_transitions_before = before.transitions;

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

                auto current_local = cpu_low_worker_local_score_with_owner(
                    i, seg.first, seg.second, src,
                    ordered, owner, layout, storage, mask_index,
                    penalty_cache, penalty_ready);
                int best_dst = -1;
                CpuLowDomainWorkerCoalesceScore best_local = current_local;

                for (int c = 0; c < nc; ++c) {
                    int dst = candidates[c];
                    ++stats.candidate_evaluations;
                    if (dst < first_worker || dst >= first_worker + nworkers) {
                        std::cerr << "cpu LOW domain worker coalesce crossed domain\n";
                        std::exit(184);
                    }
                    if (loads[size_t(dst)] > cap - cells) {
                        ++stats.cap_rejections;
                        continue;
                    }
                    auto candidate_local = cpu_low_worker_local_score_with_owner(
                        i, seg.first, seg.second, dst,
                        ordered, owner, layout, storage, mask_index,
                        penalty_cache, penalty_ready);
                    if (cpu_low_worker_coalesce_score_less(candidate_local, best_local)) {
                        best_local = candidate_local;
                        best_dst = dst;
                    }
                }

                if (best_dst >= 0) {
                    bool page_improved = best_local.pages_2m < current_local.pages_2m
                        || (best_local.pages_2m == current_local.pages_2m
                            && best_local.pages_4k < current_local.pages_4k);
                    loads[size_t(src)] -= cells;
                    loads[size_t(best_dst)] += cells;
                    if (loads[size_t(best_dst)] > cap) {
                        std::cerr << "cpu LOW domain worker coalesce cap violation\n";
                        std::exit(185);
                    }
                    owner[i] = best_dst;
                    ++stats.accepted_moves;
                    stats.moved_cells += cells;
                    if (page_improved) ++stats.page_improving_moves;
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
        int d = w / pool.domain_size;
        if (w < 0 || w >= pool.workers || d != original_domain[k]) {
            std::cerr << "cpu LOW domain worker coalesce domain provenance mismatch\n";
            std::exit(186);
        }
        next_jobs[size_t(w)].push_back(ordered[k].index);
        next_cells[size_t(w)] += ordered[k].cells;
    }

    uint64_t assigned_cells = 0;
    for (uint64_t x : next_cells) assigned_cells += x;
    if (assigned_cells != expected_cells) {
        std::cerr << "cpu LOW domain worker coalesce accounting mismatch"
                  << " assigned=" << assigned_cells
                  << " expected=" << expected_cells << '\n';
        std::exit(187);
    }

    stats.max_worker_cells_after = next_cells.empty() ? 0
        : *std::max_element(next_cells.begin(), next_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW domain worker coalesce increased max worker load"
                  << " before=" << stats.max_worker_cells_before
                  << " after=" << stats.max_worker_cells_after << '\n';
        std::exit(188);
    }

    auto after = total_score();
    stats.penalty_2m_after = after.pages_2m;
    stats.penalty_4k_after = after.pages_4k;
    stats.owner_transitions_after = after.transitions;
    if (cpu_low_worker_coalesce_score_less(before, after)) {
        std::cerr << "cpu LOW domain worker coalesce objective regression\n";
        std::exit(189);
    }

    pool.sticky_worker_jobs.swap(next_jobs);
    pool.sticky_worker_cells.swap(next_cells);
    stats.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_domain_worker_coalesce"
              << " objective=neighbor-page-coalesce-under-domain-cap-v5.26-plan"
              << " domains=" << stats.domains
              << " nonempty_domains=" << stats.nonempty_domains
              << " noncontiguous_domains_before=" << stats.noncontiguous_domains_before
              << " improved_domains=" << stats.improved_domains
              << " contiguous_domains_after=" << stats.contiguous_domains_after
              << " candidate_evaluations=" << stats.candidate_evaluations
              << " cap_rejections=" << stats.cap_rejections
              << " accepted_moves=" << stats.accepted_moves
              << " page_improving_moves=" << stats.page_improving_moves
              << " transition_only_moves=" << stats.transition_only_moves
              << " moved_cells=" << stats.moved_cells
              << " penalty_2m_before=" << stats.penalty_2m_before
              << " penalty_2m_after=" << stats.penalty_2m_after
              << " penalty_4k_before=" << stats.penalty_4k_before
              << " penalty_4k_after=" << stats.penalty_4k_after
              << " owner_transitions_before=" << stats.owner_transitions_before
              << " owner_transitions_after=" << stats.owner_transitions_after
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " mask_index_mib=" << stats.mask_index_mib
              << " mask_index_build_s=" << stats.mask_index_build_s
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
