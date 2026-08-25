#pragma once

#include "ramstream32_cpu_low_domain_page.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

// Research-only global-unique page optimizer for CPU LOW domain boundaries.
// Unlike the production v5.23 local page-penalty pass, this objective unions
// the actual page IDs exposed by every domain boundary before counting them.
// It is intentionally kept out of production until preflight results justify
// the additional schedule-build complexity.

struct CpuLowDomainGlobalPageSignature {
    std::vector<uint64_t> pages_2m;
    std::vector<uint64_t> pages_4k;
};

struct CpuLowDomainGlobalPageScore {
    uint64_t pages_2m = 0;
    uint64_t pages_4k = 0;
};

struct CpuLowDomainGlobalPageStats {
    int boundary_moves = 0;
    uint64_t moved_jobs = 0;
    uint64_t candidate_evaluations = 0;
    uint64_t max_guard_rejections = 0;
    int page_improving_moves = 0;
    int page_tie_load_moves = 0;
    int page_improve_sum_increase_moves = 0;
    uint64_t pages_2m_before = 0;
    uint64_t pages_2m_after = 0;
    uint64_t pages_4k_before = 0;
    uint64_t pages_4k_after = 0;
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double build_s = 0.0;
};

static void cpu_low_domain_boundary_page_signature_blocks(
    CpuLowDomainGlobalPageSignature& out,
    const std::vector<StorageBlock>& blocks,
    const StorageFactorHost& storage,
    uint32_t threshold_mask,
    uint64_t array_tag
) {
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr uint64_t PAGE4K = 4096ull;
    constexpr uint64_t PAGE2M = 2ull << 20;
    constexpr uint64_t ARRAY_TAG = 1ull << 63;
    const uint32_t nmasks = 1u << HIGH_LUT_K;
    uint64_t tag = array_tag ? ARRAY_TAG : 0;

    for (const StorageBlock& sb : blocks) {
        if (!sb.valid || !sb.rows || !sb.cols) continue;
        bool have_left = false;
        for (uint32_t mask = 0; mask < threshold_mask; ++mask) {
            size_t ix = size_t(mask) * S + sb.he;
            if (G_FACTOR.high_mask_off[ix + 1] != G_FACTOR.high_mask_off[ix]) {
                have_left = true;
                break;
            }
        }
        if (!have_left) continue;

        for (uint32_t mask = threshold_mask; mask < nmasks; ++mask) {
            size_t ix = size_t(mask) * S + sb.he;
            uint32_t rows = G_FACTOR.high_mask_off[ix + 1] - G_FACTOR.high_mask_off[ix];
            if (!rows) continue;
            uint32_t row0 = storage.high_mask_begin[
                size_t(mask) * StorageFactorHost::S + sb.he];
            uint64_t begin_elem = uint64_t(sb.off) + uint64_t(row0) * sb.cols;
            uint64_t begin_byte = begin_elem * sizeof(Count);
            if (begin_byte % PAGE4K)
                out.pages_4k.push_back(tag | (begin_byte / PAGE4K));
            if (begin_byte % PAGE2M)
                out.pages_2m.push_back(tag | (begin_byte / PAGE2M));
            break;
        }
    }
}

static CpuLowDomainGlobalPageSignature cpu_low_domain_boundary_page_signature(
    const StorageLayout& layout,
    const StorageFactorHost& storage,
    uint32_t threshold_mask
) {
    CpuLowDomainGlobalPageSignature out;
    out.pages_2m.reserve(layout.main_blocks.size() + layout.block_blocks.size());
    out.pages_4k.reserve(layout.main_blocks.size() + layout.block_blocks.size());
    cpu_low_domain_boundary_page_signature_blocks(
        out, layout.main_blocks, storage, threshold_mask, 0);
    cpu_low_domain_boundary_page_signature_blocks(
        out, layout.block_blocks, storage, threshold_mask, 1);
    std::sort(out.pages_2m.begin(), out.pages_2m.end());
    out.pages_2m.erase(
        std::unique(out.pages_2m.begin(), out.pages_2m.end()), out.pages_2m.end());
    std::sort(out.pages_4k.begin(), out.pages_4k.end());
    out.pages_4k.erase(
        std::unique(out.pages_4k.begin(), out.pages_4k.end()), out.pages_4k.end());
    return out;
}

static uint64_t cpu_low_domain_global_unique_count(std::vector<uint64_t>& pages) {
    std::sort(pages.begin(), pages.end());
    auto it = std::unique(pages.begin(), pages.end());
    return uint64_t(it - pages.begin());
}

static bool cpu_low_domain_global_page_score_less(
    const CpuLowDomainGlobalPageScore& a,
    const CpuLowDomainGlobalPageScore& b
) {
    return a.pages_2m < b.pages_2m
        || (a.pages_2m == b.pages_2m && a.pages_4k < b.pages_4k);
}

static bool cpu_low_domain_global_page_score_equal(
    const CpuLowDomainGlobalPageScore& a,
    const CpuLowDomainGlobalPageScore& b
) {
    return a.pages_2m == b.pages_2m && a.pages_4k == b.pages_4k;
}

static CpuLowDomainGlobalPageScore cpu_low_domain_global_page_score_from_boundaries(
    const std::vector<size_t>& boundaries,
    const std::vector<CpuLowStaticJobCost>& ordered,
    const StorageLayout& layout,
    const StorageFactorHost& storage,
    std::vector<CpuLowDomainGlobalPageSignature>& cache,
    std::vector<uint8_t>& ready
) {
    std::vector<uint64_t> pages2m, pages4k;
    for (size_t boundary : boundaries) {
        if (!boundary || boundary >= ordered.size()) continue;
        if (!ready[boundary]) {
            cache[boundary] = cpu_low_domain_boundary_page_signature(
                layout, storage, ordered[boundary].mask);
            ready[boundary] = 1;
        }
        const auto& sig = cache[boundary];
        pages2m.insert(pages2m.end(), sig.pages_2m.begin(), sig.pages_2m.end());
        pages4k.insert(pages4k.end(), sig.pages_4k.begin(), sig.pages_4k.end());
    }
    return {
        cpu_low_domain_global_unique_count(pages2m),
        cpu_low_domain_global_unique_count(pages4k)
    };
}

static std::vector<size_t> cpu_low_domain_boundaries_from_segments(
    const std::vector<std::pair<size_t,size_t>>& segs
) {
    std::vector<size_t> out;
    if (segs.size() < 2) return out;
    out.reserve(segs.size() - 1);
    for (size_t d = 0; d + 1 < segs.size(); ++d) {
        if (segs[d].first >= segs[d].second
            || segs[d + 1].first >= segs[d + 1].second) continue;
        if (segs[d].second != segs[d + 1].first) continue;
        out.push_back(segs[d].second);
    }
    return out;
}

static CpuLowDomainGlobalPageScore cpu_low_domain_global_page_score_for_pool(
    const CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    std::vector<std::pair<uint32_t,int>> owned;
    for (int w = 0; w < pool.workers; ++w) {
        int domain = pool.domain_size > 0 ? w / pool.domain_size : w;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size()) {
                std::cerr << "cpu LOW global page score bad job index\n";
                std::exit(154);
            }
            owned.push_back({jobs[q].mask, domain});
        }
    }
    std::sort(owned.begin(), owned.end());
    CpuLowDomainGlobalPageSignature all;
    for (size_t i = 1; i < owned.size(); ++i) {
        if (owned[i].second < owned[i - 1].second) {
            std::cerr << "cpu LOW global page score lost domain ordering\n";
            std::exit(155);
        }
        if (owned[i].second == owned[i - 1].second) continue;
        auto sig = cpu_low_domain_boundary_page_signature(
            layout, storage, owned[i].first);
        all.pages_2m.insert(
            all.pages_2m.end(), sig.pages_2m.begin(), sig.pages_2m.end());
        all.pages_4k.insert(
            all.pages_4k.end(), sig.pages_4k.begin(), sig.pages_4k.end());
    }
    return {
        cpu_low_domain_global_unique_count(all.pages_2m),
        cpu_low_domain_global_unique_count(all.pages_4k)
    };
}

static CpuLowDomainGlobalPageStats cpu_low_apply_domain_global_page_tiebreak(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    constexpr size_t RADIUS = 32;
    constexpr int PASSES = 2;
    CpuLowDomainGlobalPageStats stats;
    auto t0 = std::chrono::steady_clock::now();

    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || !pool.domain_refine) {
        std::cerr << "cpu LOW global page tie-break requires refined domain schedule\n";
        std::exit(156);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW global page tie-break requires prepared schedule\n";
        std::exit(157);
    }

    std::vector<CpuLowStaticJobCost> ordered;
    ordered.reserve(jobs.size());
    uint64_t total_cells = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].main_size && !jobs[i].block_size) continue;
        uint64_t cells = cpu_low_sparse_job_cells(jobs[i], sparse);
        ordered.push_back({i, jobs[i].mask, cells});
        total_cells += cells;
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        if (a.mask != b.mask) return a.mask < b.mask;
        return a.index < b.index;
    });
    if (ordered.size() < 2) {
        stats.build_s = ram_seconds_since(t0);
        return stats;
    }

    const int domains = (pool.workers + pool.domain_size - 1) / pool.domain_size;
    std::vector<int> job_domain(jobs.size(), -1);
    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size() || job_domain[q] >= 0) {
                std::cerr << "cpu LOW global page tie-break duplicate job owner\n";
                std::exit(158);
            }
            job_domain[q] = d;
        }
    }

    std::vector<std::pair<size_t,size_t>> segs(size_t(domains), {ordered.size(), ordered.size()});
    int last_domain = -1;
    for (size_t k = 0; k < ordered.size(); ++k) {
        int d = job_domain[ordered[k].index];
        if (d < 0 || d >= domains || d < last_domain) {
            std::cerr << "cpu LOW global page tie-break lost ordered ownership\n";
            std::exit(159);
        }
        last_domain = d;
        if (segs[size_t(d)].first == ordered.size()) segs[size_t(d)].first = k;
        segs[size_t(d)].second = k + 1;
    }

    std::vector<CpuLowDomainGlobalPageSignature> cache(ordered.size() + 1);
    std::vector<uint8_t> ready(ordered.size() + 1, 0);
    auto score = [&] {
        return cpu_low_domain_global_page_score_from_boundaries(
            cpu_low_domain_boundaries_from_segments(segs), ordered,
            layout, storage, cache, ready);
    };

    stats.max_worker_cells_before = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
    auto before = score();
    stats.pages_2m_before = before.pages_2m;
    stats.pages_4k_before = before.pages_4k;

    for (int pass = 0; pass < PASSES; ++pass) {
        bool changed = false;
        for (int qq = 0; qq + 1 < domains; ++qq) {
            int d = pass == 0 ? qq : (domains - 2 - qq);
            auto left = segs[size_t(d)];
            auto right = segs[size_t(d + 1)];
            if (left.first >= left.second || right.first >= right.second) continue;
            if (left.second != right.first) continue;

            int left_workers = std::min(pool.domain_size, pool.workers - d * pool.domain_size);
            int right_workers = std::min(pool.domain_size, pool.workers - (d + 1) * pool.domain_size);
            if (left_workers <= 0 || right_workers <= 0) continue;

            size_t old_boundary = left.second;
            size_t min_boundary = left.first + 1;
            size_t max_boundary = right.second - 1;
            size_t search_lo = old_boundary > RADIUS ? old_boundary - RADIUS : 0;
            search_lo = std::max(search_lo, min_boundary);
            size_t search_hi = std::min(max_boundary, old_boundary + RADIUS);
            if (search_lo > search_hi) continue;

            uint64_t current_left = cpu_low_lpt_range_max_cells(
                ordered, left.first, old_boundary, left_workers, jobs);
            uint64_t current_right = cpu_low_lpt_range_max_cells(
                ordered, old_boundary, right.second, right_workers, jobs);
            uint64_t current_max = std::max(current_left, current_right);
            uint64_t current_sum = cpu_low_domain_pair_sum(current_left, current_right);
            auto current_score = score();

            size_t best_boundary = old_boundary;
            auto best_score = current_score;
            uint64_t best_sum = current_sum;
            uint64_t best_max = current_max;
            size_t best_distance = 0;

            for (size_t candidate = search_lo; candidate <= search_hi; ++candidate) {
                if (candidate == old_boundary) continue;
                ++stats.candidate_evaluations;
                uint64_t lm = cpu_low_lpt_range_max_cells(
                    ordered, left.first, candidate, left_workers, jobs);
                uint64_t rm = cpu_low_lpt_range_max_cells(
                    ordered, candidate, right.second, right_workers, jobs);
                uint64_t candidate_max = std::max(lm, rm);
                if (candidate_max > current_max) {
                    ++stats.max_guard_rejections;
                    continue;
                }
                uint64_t candidate_sum = cpu_low_domain_pair_sum(lm, rm);
                segs[size_t(d)].second = candidate;
                segs[size_t(d + 1)].first = candidate;
                auto candidate_score = score();
                segs[size_t(d)].second = old_boundary;
                segs[size_t(d + 1)].first = old_boundary;
                size_t distance = candidate > old_boundary
                    ? candidate - old_boundary : old_boundary - candidate;

                bool better = cpu_low_domain_global_page_score_less(
                    candidate_score, best_score);
                if (!better && cpu_low_domain_global_page_score_equal(
                        candidate_score, best_score)) {
                    better = candidate_sum < best_sum
                        || (candidate_sum == best_sum && candidate_max < best_max)
                        || (candidate_sum == best_sum && candidate_max == best_max
                            && best_boundary != old_boundary && distance < best_distance);
                }
                if (better) {
                    best_boundary = candidate;
                    best_score = candidate_score;
                    best_sum = candidate_sum;
                    best_max = candidate_max;
                    best_distance = distance;
                }
            }

            if (best_boundary != old_boundary) {
                bool page_improved = cpu_low_domain_global_page_score_less(
                    best_score, current_score);
                bool load_improved = cpu_low_domain_global_page_score_equal(
                    best_score, current_score)
                    && (best_sum < current_sum
                        || (best_sum == current_sum && best_max < current_max));
                if (!page_improved && !load_improved) {
                    std::cerr << "cpu LOW global page tie-break selected non-improving boundary\n";
                    std::exit(160);
                }
                if (best_max > current_max) {
                    std::cerr << "cpu LOW global page tie-break violated pair-max guard\n";
                    std::exit(161);
                }
                size_t distance = best_boundary > old_boundary
                    ? best_boundary - old_boundary : old_boundary - best_boundary;
                segs[size_t(d)].second = best_boundary;
                segs[size_t(d + 1)].first = best_boundary;
                ++stats.boundary_moves;
                stats.moved_jobs += distance;
                if (page_improved) {
                    ++stats.page_improving_moves;
                    if (best_sum > current_sum)
                        ++stats.page_improve_sum_increase_moves;
                } else {
                    ++stats.page_tie_load_moves;
                }
                changed = true;
            }
        }
        if (!changed) break;
    }

    pool.sticky_worker_jobs.assign(size_t(pool.workers), {});
    pool.sticky_worker_cells.assign(size_t(pool.workers), 0);
    for (int d = 0; d < domains; ++d) {
        auto seg = segs[size_t(d)];
        if (seg.first >= seg.second) continue;
        int first_worker = d * pool.domain_size;
        int nworkers = std::min(pool.domain_size, pool.workers - first_worker);
        cpu_low_lpt_assign_range(
            ordered, seg.first, seg.second, first_worker, nworkers,
            jobs, pool.sticky_worker_jobs, pool.sticky_worker_cells);
    }

    uint64_t assigned_cells = 0;
    for (uint64_t x : pool.sticky_worker_cells) assigned_cells += x;
    if (assigned_cells != total_cells) {
        std::cerr << "cpu LOW global page tie-break accounting mismatch\n";
        std::exit(162);
    }

    stats.max_worker_cells_after = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW global page tie-break increased max worker load\n";
        std::exit(163);
    }
    auto after = score();
    stats.pages_2m_after = after.pages_2m;
    stats.pages_4k_after = after.pages_4k;
    if (cpu_low_domain_global_page_score_less(before, after)) {
        std::cerr << "cpu LOW global page tie-break increased global unique page score\n";
        std::exit(164);
    }
    stats.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_domain_global_page_tiebreak"
              << " objective=global-unique-max-guard-page-sum-v5.24-plan"
              << " boundary_moves=" << stats.boundary_moves
              << " moved_jobs=" << stats.moved_jobs
              << " candidate_evaluations=" << stats.candidate_evaluations
              << " max_guard_rejections=" << stats.max_guard_rejections
              << " page_improving_moves=" << stats.page_improving_moves
              << " page_tie_load_moves=" << stats.page_tie_load_moves
              << " page_improve_sum_increase_moves=" << stats.page_improve_sum_increase_moves
              << " pages_2m_before=" << stats.pages_2m_before
              << " pages_2m_after=" << stats.pages_2m_after
              << " pages_4k_before=" << stats.pages_4k_before
              << " pages_4k_after=" << stats.pages_4k_after
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
