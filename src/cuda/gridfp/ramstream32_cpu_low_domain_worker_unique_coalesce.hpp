#pragma once

#include "ramstream32_cpu_low_domain_page_global.hpp"
#include "ramstream32_cpu_low_domain_worker_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

// Research-only v5.27 exact-unique worker-boundary coalescing objective.
// v5.29 keeps the same exact objective and legal moves, but replaces the two
// per-candidate unordered_map page-delta tables with reusable sorted flat
// vectors. A move touches at most two boundary signatures, so each temporary
// delta contains only O(number of factor blocks) entries. This removes heap
// hash-table construction from the n=27 candidate hot path without changing
// the accepted-move semantics.

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
    uint64_t flat_delta_normalizations = 0;
    size_t flat_delta_peak_entries = 0;
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double mask_index_mib = 0.0;
    double mask_index_build_s = 0.0;
    double build_s = 0.0;
};

using CpuLowWorkerUniqueFlatDelta = std::vector<std::pair<uint64_t,int>>;

static bool cpu_low_worker_unique_score_less(
    const CpuLowDomainWorkerUniqueScore& a,
    const CpuLowDomainWorkerUniqueScore& b
) {
    return a.pages_2m < b.pages_2m
        || (a.pages_2m == b.pages_2m && a.pages_4k < b.pages_4k)
        || (a.pages_2m == b.pages_2m && a.pages_4k == b.pages_4k
            && a.transitions < b.transitions);
}

static void cpu_low_worker_unique_ref_add_one(
    std::unordered_map<uint64_t,uint32_t>& refs,
    uint64_t page, int delta
) {
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

static void cpu_low_worker_unique_ref_add(
    std::unordered_map<uint64_t,uint32_t>& refs,
    const std::vector<uint64_t>& pages,
    int delta
) {
    for (uint64_t page : pages)
        cpu_low_worker_unique_ref_add_one(refs, page, delta);
}

static void cpu_low_worker_unique_flat_delta_append(
    CpuLowWorkerUniqueFlatDelta& delta,
    const std::vector<uint64_t>& pages,
    int sign
) {
    for (uint64_t page : pages) delta.push_back({page, sign});
}

static void cpu_low_worker_unique_flat_delta_normalize(
    CpuLowWorkerUniqueFlatDelta& delta
) {
    if (delta.size() <= 1) return;
    std::sort(delta.begin(), delta.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    size_t out = 0;
    for (size_t i = 0; i < delta.size();) {
        uint64_t page = delta[i].first;
        int sum = 0;
        do {
            sum += delta[i].second;
            ++i;
        } while (i < delta.size() && delta[i].first == page);
        if (sum) delta[out++] = {page, sum};
    }
    delta.resize(out);
}

static uint64_t cpu_low_worker_unique_after_flat_delta(
    const std::unordered_map<uint64_t,uint32_t>& refs,
    const CpuLowWorkerUniqueFlatDelta& delta
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

// Compatibility helper retained for the synthetic W10 test. The production
// candidate path uses the normalized flat representation above.
static uint64_t cpu_low_worker_unique_after_delta(
    const std::unordered_map<uint64_t,uint32_t>& refs,
    const std::unordered_map<uint64_t,int>& delta
) {
    CpuLowWorkerUniqueFlatDelta flat;
    flat.reserve(delta.size());
    for (const auto& kv : delta) if (kv.second) flat.push_back(kv);
    cpu_low_worker_unique_flat_delta_normalize(flat);
    return cpu_low_worker_unique_after_flat_delta(refs, flat);
}

static void cpu_low_worker_unique_apply_flat_delta(
    std::unordered_map<uint64_t,uint32_t>& refs,
    const CpuLowWorkerUniqueFlatDelta& delta
) {
    for (const auto& kv : delta) {
        if (kv.second < 0) {
            for (int j = 0; j < -kv.second; ++j)
                cpu_low_worker_unique_ref_add_one(refs, kv.first, -1);
        } else {
            for (int j = 0; j < kv.second; ++j)
                cpu_low_worker_unique_ref_add_one(refs, kv.first, +1);
        }
    }
}

static uint32_t cpu_low_worker_unique_transition_weight_blocks(
    const std::vector<StorageBlock>& blocks,
    const CpuLowDomainPageMaskIndex& mask_index,
    uint32_t threshold_mask
) {
    uint32_t z = 0;
    for (const StorageBlock& sb : blocks) {
        if (!sb.valid || !sb.rows || !sb.cols) continue;
        if (sb.he >= mask_index.stride) {
            std::cerr << "cpu LOW worker unique transition height out of range\n";
            std::exit(193);
        }
        if (mask_index.first_nonempty[sb.he] >= threshold_mask) continue;
        if (mask_index.next(sb.he, threshold_mask) >= mask_index.nmasks) continue;
        ++z;
    }
    return z;
}

static uint32_t cpu_low_worker_unique_transition_weight(
    const StorageLayout& layout,
    const CpuLowDomainPageMaskIndex& mask_index,
    uint32_t threshold_mask
) {
    uint64_t z = uint64_t(cpu_low_worker_unique_transition_weight_blocks(
        layout.main_blocks, mask_index, threshold_mask))
        + cpu_low_worker_unique_transition_weight_blocks(
            layout.block_blocks, mask_index, threshold_mask);
    if (z > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "cpu LOW worker unique transition weight overflow\n";
        std::exit(194);
    }
    return uint32_t(z);
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
        std::exit(195);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW domain worker unique coalesce requires prepared schedule\n";
        std::exit(196);
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
                std::exit(197);
            }
            size_t k = ordered_pos[q];
            if (owner[k] >= 0) {
                std::cerr << "cpu LOW worker unique coalesce duplicate job owner\n";
                std::exit(198);
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
            std::exit(199);
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
    std::vector<uint32_t> transition_weight_cache(ordered.size() + 1, 0);
    std::vector<uint8_t> transition_weight_ready(ordered.size() + 1, 0);
    auto signature = [&](size_t boundary) -> const CpuLowDomainGlobalPageSignature& {
        if (!boundary || boundary >= ordered.size()) {
            std::cerr << "cpu LOW worker unique signature boundary out of range\n";
            std::exit(200);
        }
        if (!signature_ready[boundary]) {
            signature_cache[boundary] = cpu_low_domain_boundary_page_signature(
                layout, storage, mask_index, ordered[boundary].mask);
            signature_ready[boundary] = 1;
        }
        return signature_cache[boundary];
    };
    auto transition_weight = [&](size_t boundary) -> uint32_t {
        if (!boundary || boundary >= ordered.size()) {
            std::cerr << "cpu LOW worker unique transition boundary out of range\n";
            std::exit(201);
        }
        if (!transition_weight_ready[boundary]) {
            transition_weight_cache[boundary] = cpu_low_worker_unique_transition_weight(
                layout, mask_index, ordered[boundary].mask);
            transition_weight_ready[boundary] = 1;
        }
        return transition_weight_cache[boundary];
    };

    std::unordered_map<uint64_t,uint32_t> refs2m, refs4k;
    uint64_t transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        const auto& sig = signature(k);
        cpu_low_worker_unique_ref_add(refs2m, sig.pages_2m, +1);
        cpu_low_worker_unique_ref_add(refs4k, sig.pages_4k, +1);
        transitions += transition_weight(k);
    }

    stats.unique_pages_2m_before = refs2m.size();
    stats.unique_pages_4k_before = refs4k.size();
    stats.owner_transitions_before = transitions;

    // Reusable candidate/best buffers. A move changes at most two signatures.
    const size_t flat_reserve = 2 * (
        layout.main_blocks.size() + layout.block_blocks.size());
    CpuLowWorkerUniqueFlatDelta candidate_d2, candidate_d4, best_d2, best_d4;
    candidate_d2.reserve(flat_reserve);
    candidate_d4.reserve(flat_reserve);
    best_d2.reserve(flat_reserve);
    best_d4.reserve(flat_reserve);

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
                int64_t best_transition_delta = 0;
                best_d2.clear();
                best_d4.clear();

                for (int c = 0; c < nc; ++c) {
                    int dst = candidates[c];
                    ++stats.candidate_evaluations;
                    if (dst < first_worker || dst >= first_worker + nworkers) {
                        std::cerr << "cpu LOW worker unique coalesce crossed domain\n";
                        std::exit(202);
                    }
                    if (loads[size_t(dst)] > cap - cells) {
                        ++stats.cap_rejections;
                        continue;
                    }

                    candidate_d2.clear();
                    candidate_d4.clear();
                    int64_t transition_delta = 0;
                    auto change_edge = [&](size_t boundary,
                                           int old_left, int old_right,
                                           int new_left, int new_right) {
                        bool old_active = old_left != old_right;
                        bool new_active = new_left != new_right;
                        if (old_active == new_active) return;
                        int delta = new_active ? +1 : -1;
                        const auto& sig = signature(boundary);
                        cpu_low_worker_unique_flat_delta_append(
                            candidate_d2, sig.pages_2m, delta);
                        cpu_low_worker_unique_flat_delta_append(
                            candidate_d4, sig.pages_4k, delta);
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

                    stats.flat_delta_peak_entries = std::max(
                        stats.flat_delta_peak_entries,
                        std::max(candidate_d2.size(), candidate_d4.size()));
                    cpu_low_worker_unique_flat_delta_normalize(candidate_d2);
                    cpu_low_worker_unique_flat_delta_normalize(candidate_d4);
                    stats.flat_delta_normalizations += 2;

                    int64_t candidate_transitions = int64_t(transitions) + transition_delta;
                    if (candidate_transitions < 0) {
                        std::cerr << "cpu LOW worker unique transition underflow\n";
                        std::exit(203);
                    }
                    CpuLowDomainWorkerUniqueScore candidate{
                        cpu_low_worker_unique_after_flat_delta(refs2m, candidate_d2),
                        cpu_low_worker_unique_after_flat_delta(refs4k, candidate_d4),
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
                    cpu_low_worker_unique_apply_flat_delta(refs2m, best_d2);
                    cpu_low_worker_unique_apply_flat_delta(refs4k, best_d4);
                    int64_t next_transitions = int64_t(transitions) + best_transition_delta;
                    if (next_transitions < 0) {
                        std::cerr << "cpu LOW worker unique accepted transition underflow\n";
                        std::exit(204);
                    }
                    transitions = uint64_t(next_transitions);
                    loads[size_t(src)] -= cells;
                    loads[size_t(best_dst)] += cells;
                    if (loads[size_t(best_dst)] > cap) {
                        std::cerr << "cpu LOW worker unique coalesce cap violation\n";
                        std::exit(205);
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
            std::cerr << "cpu LOW worker unique coalesce invalid worker owner\n";
            std::exit(206);
        }
        int d = w / pool.domain_size;
        if (d != original_domain[k]) {
            std::cerr << "cpu LOW worker unique coalesce domain provenance mismatch\n";
            std::exit(207);
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
        std::exit(208);
    }

    stats.max_worker_cells_after = next_cells.empty() ? 0
        : *std::max_element(next_cells.begin(), next_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW worker unique coalesce increased max worker load"
                  << " before=" << stats.max_worker_cells_before
                  << " after=" << stats.max_worker_cells_after << '\n';
        std::exit(209);
    }

    // Rebuild the exact page reference maps and weighted transition count from
    // the final owner vector. This independently validates all incremental
    // two-edge deltas, including duplicate page IDs shared by both edges.
    std::unordered_map<uint64_t,uint32_t> verify2m, verify4k;
    uint64_t verify_transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        const auto& sig = signature(k);
        cpu_low_worker_unique_ref_add(verify2m, sig.pages_2m, +1);
        cpu_low_worker_unique_ref_add(verify4k, sig.pages_4k, +1);
        verify_transitions += transition_weight(k);
    }
    if (verify2m != refs2m || verify4k != refs4k
        || verify_transitions != transitions) {
        std::cerr << "cpu LOW worker unique coalesce incremental accounting mismatch\n";
        std::exit(210);
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
        std::exit(211);
    }

    pool.sticky_worker_jobs.swap(next_jobs);
    pool.sticky_worker_cells.swap(next_cells);
    stats.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_domain_worker_unique_coalesce"
              << " objective=global-unique-neighbor-coalesce-v5.27-plan"
              << " implementation=flat-page-delta-v5.29"
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
              << " flat_delta_normalizations=" << stats.flat_delta_normalizations
              << " flat_delta_peak_entries=" << stats.flat_delta_peak_entries
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " mask_index_mib=" << stats.mask_index_mib
              << " mask_index_build_s=" << stats.mask_index_build_s
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
