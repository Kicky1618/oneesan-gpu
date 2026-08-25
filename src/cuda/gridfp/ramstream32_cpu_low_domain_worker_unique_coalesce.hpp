#pragma once

#include "ramstream32_cpu_low_domain_page_global.hpp"
#include "ramstream32_cpu_low_domain_worker_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

// Research-only v5.27 exact-unique worker-boundary coalescing stage.
//
// v5.26 minimizes a sum of per-boundary page cardinalities.  Different worker
// boundaries can expose the same VM page, so that local sum is not identical to
// the exact global unique cross-worker page set.  v5.27 keeps a reference count
// for every page ID currently exposed by every worker transition, including the
// fixed transitions at domain boundaries.  A one-job neighbour move changes at
// most two transition boundaries, so its exact effect on the global unique page
// set can be evaluated incrementally.
//
// The legal move set and load proof are the same as v5.26: a job may move only
// to the owner of its immediate left/right neighbour inside the same domain,
// and the destination must remain below the domain's pre-pass worker maximum.
// Therefore domain ownership and the global max-worker bound cannot regress.

struct CpuLowDomainWorkerUniqueScore {
    uint64_t pages_2m = 0;
    uint64_t pages_4k = 0;
    uint64_t transitions = 0;
};

struct CpuLowDomainWorkerUniqueCoalesceStats {
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
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double mask_index_mib = 0.0;
    double mask_index_build_s = 0.0;
    double build_s = 0.0;
};

static bool cpu_low_worker_unique_score_less(
    const CpuLowDomainWorkerUniqueScore& a,
    const CpuLowDomainWorkerUniqueScore& b
) {
    return a.pages_2m < b.pages_2m
        || (a.pages_2m == b.pages_2m && a.pages_4k < b.pages_4k)
        || (a.pages_2m == b.pages_2m && a.pages_4k == b.pages_4k
            && a.transitions < b.transitions);
}

static void cpu_low_worker_unique_ref_add(
    std::unordered_map<uint64_t,uint32_t>& refs,
    const std::vector<uint64_t>& pages,
    int delta
) {
    for (uint64_t page : pages) {
        auto it = refs.find(page);
        uint32_t old = it == refs.end() ? 0u : it->second;
        if (delta < 0) {
            if (!old) {
                std::cerr << "cpu LOW worker unique page ref underflow\n";
                std::exit(190);
            }
            if (old == 1) refs.erase(it);
            else it->second = old - 1;
        } else {
            if (old == std::numeric_limits<uint32_t>::max()) {
                std::cerr << "cpu LOW worker unique page ref overflow\n";
                std::exit(191);
            }
            if (it == refs.end()) refs.emplace(page, 1u);
            else it->second = old + 1;
        }
    }
}

static uint64_t cpu_low_worker_unique_after_delta(
    const std::unordered_map<uint64_t,uint32_t>& refs,
    const std::unordered_map<uint64_t,int>& delta
) {
    uint64_t unique = refs.size();
    for (const auto& kv : delta) {
        auto it = refs.find(kv.first);
        int64_t old = it == refs.end() ? 0 : int64_t(it->second);
        int64_t now = old + kv.second;
        if (now < 0) {
            std::cerr << "cpu LOW worker unique candidate ref underflow\n";
            std::exit(192);
        }
        if (old == 0 && now > 0) ++unique;
        else if (old > 0 && now == 0) --unique;
    }
    return unique;
}

static CpuLowDomainWorkerUniqueCoalesceStats
cpu_low_apply_domain_worker_unique_coalesce(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    constexpr int PASSES = 12;
    CpuLowDomainWorkerUniqueCoalesceStats stats;
    auto t0 = std::chrono::steady_clock::now();

    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || pool.domain_size <= 0) {
        std::cerr << "cpu LOW domain worker unique coalesce requires domain schedule\n";
        std::exit(193);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW domain worker unique coalesce requires prepared schedule\n";
        std::exit(194);
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
                std::cerr << "cpu LOW worker unique coalesce bad job owner\n";
                std::exit(195);
            }
            size_t k = ordered_pos[q];
            if (owner[k] >= 0) {
                std::cerr << "cpu LOW worker unique coalesce duplicate job owner\n";
                std::exit(196);
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
            std::cerr << "cpu LOW worker unique coalesce lost domain ordering\n";
            std::exit(197);
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

    std::vector<CpuLowDomainGlobalPageSignature> signature_cache(ordered.size() + 1);
    std::vector<uint8_t> signature_ready(ordered.size() + 1, 0);
    auto signature = [&](size_t boundary) -> const CpuLowDomainGlobalPageSignature& {
        if (!boundary || boundary >= ordered.size()) {
            std::cerr << "cpu LOW worker unique signature boundary out of range\n";
            std::exit(198);
        }
        if (!signature_ready[boundary]) {
            signature_cache[boundary] = cpu_low_domain_boundary_page_signature(
                layout, storage, mask_index, ordered[boundary].mask);
            signature_ready[boundary] = 1;
        }
        return signature_cache[boundary];
    };

    std::unordered_map<uint64_t,uint32_t> refs2m, refs4k;
    uint64_t transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        const auto& sig = signature(k);
        cpu_low_worker_unique_ref_add(refs2m, sig.pages_2m, +1);
        cpu_low_worker_unique_ref_add(refs4k, sig.pages_4k, +1);
        ++transitions;
    }

    stats.unique_pages_2m_before = refs2m.size();
    stats.unique_pages_4k_before = refs4k.size();
    stats.owner_transitions_before = transitions;

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

                CpuLowDomainWorkerUniqueScore current{
                    uint64_t(refs2m.size()), uint64_t(refs4k.size()), transitions};
                CpuLowDomainWorkerUniqueScore best = current;
                int best_dst = -1;
                std::unordered_map<uint64_t,int> best_d2, best_d4;
                int best_transition_delta = 0;

                for (int c = 0; c < nc; ++c) {
                    int dst = candidates[c];
                    ++stats.candidate_evaluations;
                    if (dst < first_worker || dst >= first_worker + nworkers) {
                        std::cerr << "cpu LOW worker unique coalesce crossed domain\n";
                        std::exit(199);
                    }
                    if (loads[size_t(dst)] > cap - cells) {
                        ++stats.cap_rejections;
                        continue;
                    }

                    std::unordered_map<uint64_t,int> d2, d4;
                    int transition_delta = 0;
                    auto change_edge = [&](size_t boundary,
                                           int old_left, int old_right,
                                           int new_left, int new_right) {
                        bool old_active = old_left != old_right;
                        bool new_active = new_left != new_right;
                        if (old_active == new_active) return;
                        int delta = new_active ? +1 : -1;
                        const auto& sig = signature(boundary);
                        for (uint64_t page : sig.pages_2m) d2[page] += delta;
                        for (uint64_t page : sig.pages_4k) d4[page] += delta;
                        transition_delta += delta;
                    };

                    if (i > 0)
                        change_edge(i,
                            owner[i - 1], src,
                            owner[i - 1], dst);
                    if (i + 1 < ordered.size())
                        change_edge(i + 1,
                            src, owner[i + 1],
                            dst, owner[i + 1]);

                    CpuLowDomainWorkerUniqueScore candidate{
                        cpu_low_worker_unique_after_delta(refs2m, d2),
                        cpu_low_worker_unique_after_delta(refs4k, d4),
                        uint64_t(int64_t(transitions) + transition_delta)};
                    if (cpu_low_worker_unique_score_less(candidate, best)) {
                        best = candidate;
                        best_dst = dst;
                        best_d2.swap(d2);
                        best_d4.swap(d4);
                        best_transition_delta = transition_delta;
                    }
                }

                if (best_dst >= 0) {
                    bool page_improved = best.pages_2m < current.pages_2m
                        || (best.pages_2m == current.pages_2m
                            && best.pages_4k < current.pages_4k);
                    for (const auto& kv : best_d2) {
                        if (kv.second < 0) {
                            for (int j = 0; j < -kv.second; ++j)
                                cpu_low_worker_unique_ref_add(
                                    refs2m, std::vector<uint64_t>{kv.first}, -1);
                        } else {
                            for (int j = 0; j < kv.second; ++j)
                                cpu_low_worker_unique_ref_add(
                                    refs2m, std::vector<uint64_t>{kv.first}, +1);
                        }
                    }
                    for (const auto& kv : best_d4) {
                        if (kv.second < 0) {
                            for (int j = 0; j < -kv.second; ++j)
                                cpu_low_worker_unique_ref_add(
                                    refs4k, std::vector<uint64_t>{kv.first}, -1);
                        } else {
                            for (int j = 0; j < kv.second; ++j)
                                cpu_low_worker_unique_ref_add(
                                    refs4k, std::vector<uint64_t>{kv.first}, +1);
                        }
                    }
                    transitions = uint64_t(int64_t(transitions) + best_transition_delta);
                    loads[size_t(src)] -= cells;
                    loads[size_t(best_dst)] += cells;
                    if (loads[size_t(best_dst)] > cap) {
                        std::cerr << "cpu LOW worker unique coalesce cap violation\n";
                        std::exit(200);
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
        int d = w / pool.domain_size;
        if (w < 0 || w >= pool.workers || d != original_domain[k]) {
            std::cerr << "cpu LOW worker unique coalesce domain provenance mismatch\n";
            std::exit(201);
        }
        next_jobs[size_t(w)].push_back(ordered[k].index);
        next_cells[size_t(w)] += ordered[k].cells;
    }

    uint64_t assigned_cells = 0;
    for (uint64_t x : next_cells) assigned_cells += x;
    if (assigned_cells != expected_cells) {
        std::cerr << "cpu LOW worker unique coalesce accounting mismatch"
                  << " assigned=" << assigned_cells
                  << " expected=" << expected_cells << '\n';
        std::exit(202);
    }

    stats.max_worker_cells_after = next_cells.empty() ? 0
        : *std::max_element(next_cells.begin(), next_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW worker unique coalesce increased max worker load"
                  << " before=" << stats.max_worker_cells_before
                  << " after=" << stats.max_worker_cells_after << '\n';
        std::exit(203);
    }

    // Rebuild the exact page reference maps from the final owner vector.  This
    // validates the incremental two-edge accounting independently.
    std::unordered_map<uint64_t,uint32_t> verify2m, verify4k;
    uint64_t verify_transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        const auto& sig = signature(k);
        cpu_low_worker_unique_ref_add(verify2m, sig.pages_2m, +1);
        cpu_low_worker_unique_ref_add(verify4k, sig.pages_4k, +1);
        ++verify_transitions;
    }
    if (verify2m != refs2m || verify4k != refs4k
        || verify_transitions != transitions) {
        std::cerr << "cpu LOW worker unique coalesce incremental accounting mismatch\n";
        std::exit(204);
    }

    stats.unique_pages_2m_after = refs2m.size();
    stats.unique_pages_4k_after = refs4k.size();
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
        std::cerr << "cpu LOW worker unique coalesce exact objective regression\n";
        std::exit(205);
    }

    pool.sticky_worker_jobs.swap(next_jobs);
    pool.sticky_worker_cells.swap(next_cells);
    stats.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_domain_worker_unique_coalesce"
              << " objective=global-unique-neighbor-coalesce-v5.27-plan"
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
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " mask_index_mib=" << stats.mask_index_mib
              << " mask_index_build_s=" << stats.mask_index_build_s
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
