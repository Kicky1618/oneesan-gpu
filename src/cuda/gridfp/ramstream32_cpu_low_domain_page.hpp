#pragma once

#include "ramstream32_cpu_low_domain_page_index.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

// Optional second-stage refinement for CPU LOW domain scheduling.
// It starts from the ordinary refined-domain assignment. A candidate boundary
// is eligible whenever the affected two-domain LPT maximum does not increase.
// Among eligible boundaries it minimizes, in order:
//   1. local 2 MiB boundary-page exposure,
//   2. local 4 KiB boundary-page exposure,
//   3. the sum of the two domain LPT maxima,
//   4. the pair maximum,
//   5. displacement from the current boundary.
// The current boundary is always the baseline candidate, so accepted moves
// cannot worsen the page tuple. The pair-max guard proves that enabling this
// pass cannot increase the global maximum worker load.

static bool cpu_low_domain_page_tiebreak_from_env() {
    const char* s = std::getenv("CPU_LOW_DOMAIN_PAGE_TIEBREAK");
    if (!s || !*s || std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0
        || std::strcmp(s, "no") == 0 || std::strcmp(s, "off") == 0) return false;
    if (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0
        || std::strcmp(s, "yes") == 0 || std::strcmp(s, "on") == 0) return true;
    std::cerr << "CPU_LOW_DOMAIN_PAGE_TIEBREAK must be 0/1, false/true, no/yes, or off/on\n";
    std::exit(143);
}

struct CpuLowDomainBoundaryPagePenalty {
    uint32_t pages_2m = 0;
    uint32_t pages_4k = 0;
};

struct CpuLowDomainPageTieStats {
    int boundary_moves = 0;
    uint64_t moved_jobs = 0;
    uint64_t candidate_evaluations = 0;
    uint64_t max_guard_rejections = 0;
    int page_improving_moves = 0;
    int page_tie_load_moves = 0;
    int page_improve_sum_increase_moves = 0;
    uint64_t penalty_2m_before = 0;
    uint64_t penalty_2m_after = 0;
    uint64_t penalty_4k_before = 0;
    uint64_t penalty_4k_after = 0;
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double mask_index_mib = 0.0;
    double mask_index_build_s = 0.0;
    double build_s = 0.0;
};

static uint32_t cpu_low_unique_pages(std::vector<uint64_t>& pages) {
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    if (pages.size() > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "cpu LOW domain page penalty overflow\n";
        std::exit(144);
    }
    return uint32_t(pages.size());
}

static CpuLowDomainBoundaryPagePenalty cpu_low_domain_boundary_page_penalty_blocks(
    const std::vector<StorageBlock>& blocks,
    const StorageFactorHost& storage,
    const CpuLowDomainPageMaskIndex& mask_index,
    uint32_t threshold_mask
) {
    constexpr uint64_t PAGE4K = 4096ull;
    constexpr uint64_t PAGE2M = 2ull << 20;
    std::vector<uint64_t> pages4k, pages2m;
    pages4k.reserve(blocks.size());
    pages2m.reserve(blocks.size());

    for (const StorageBlock& sb : blocks) {
        if (!sb.valid || !sb.rows || !sb.cols) continue;
        if (sb.he >= mask_index.stride) {
            std::cerr << "cpu LOW domain page penalty height out of range\n";
            std::exit(167);
        }
        if (mask_index.first_nonempty[sb.he] >= threshold_mask) continue;
        uint32_t mask = mask_index.next(sb.he, threshold_mask);
        if (mask >= mask_index.nmasks) continue;

        uint32_t row0 = storage.high_mask_begin[
            size_t(mask) * StorageFactorHost::S + sb.he];
        uint64_t begin_elem = uint64_t(sb.off) + uint64_t(row0) * sb.cols;
        uint64_t begin_byte = begin_elem * sizeof(Count);
        if (begin_byte % PAGE4K) pages4k.push_back(begin_byte / PAGE4K);
        if (begin_byte % PAGE2M) pages2m.push_back(begin_byte / PAGE2M);
    }

    CpuLowDomainBoundaryPagePenalty z;
    z.pages_4k = cpu_low_unique_pages(pages4k);
    z.pages_2m = cpu_low_unique_pages(pages2m);
    return z;
}

static CpuLowDomainBoundaryPagePenalty cpu_low_domain_boundary_page_penalty(
    const StorageLayout& layout, const StorageFactorHost& storage,
    const CpuLowDomainPageMaskIndex& mask_index,
    uint32_t threshold_mask
) {
    auto a = cpu_low_domain_boundary_page_penalty_blocks(
        layout.main_blocks, storage, mask_index, threshold_mask);
    auto b = cpu_low_domain_boundary_page_penalty_blocks(
        layout.block_blocks, storage, mask_index, threshold_mask);
    return {uint32_t(a.pages_2m + b.pages_2m),
            uint32_t(a.pages_4k + b.pages_4k)};
}

static bool cpu_low_domain_page_penalty_less(
    const CpuLowDomainBoundaryPagePenalty& a,
    const CpuLowDomainBoundaryPagePenalty& b
) {
    return a.pages_2m < b.pages_2m
        || (a.pages_2m == b.pages_2m && a.pages_4k < b.pages_4k);
}

static bool cpu_low_domain_page_penalty_equal(
    const CpuLowDomainBoundaryPagePenalty& a,
    const CpuLowDomainBoundaryPagePenalty& b
) {
    return a.pages_2m == b.pages_2m && a.pages_4k == b.pages_4k;
}

static uint64_t cpu_low_domain_pair_sum(uint64_t a, uint64_t b) {
    return a > std::numeric_limits<uint64_t>::max() - b
        ? std::numeric_limits<uint64_t>::max() : a + b;
}

static CpuLowDomainPageTieStats cpu_low_apply_domain_page_tiebreak(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    constexpr size_t RADIUS = 32;
    constexpr int PASSES = 2;
    CpuLowDomainPageTieStats stats;
    auto t0 = std::chrono::steady_clock::now();

    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || !pool.domain_refine) {
        std::cerr << "cpu LOW domain page tie-break requires refined domain schedule\n";
        std::exit(145);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW domain page tie-break requires prepared schedule\n";
        std::exit(146);
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

    auto index_t0 = std::chrono::steady_clock::now();
    CpuLowDomainPageMaskIndex mask_index = cpu_low_build_domain_page_mask_index();
    stats.mask_index_build_s = ram_seconds_since(index_t0);
    stats.mask_index_mib = double(
        mask_index.first_nonempty.size() * sizeof(uint32_t)
        + mask_index.next_nonempty.size() * sizeof(uint32_t)) / double(1 << 20);

    const int domains = (pool.workers + pool.domain_size - 1) / pool.domain_size;
    std::vector<int> job_domain(jobs.size(), -1);
    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size() || job_domain[q] >= 0) {
                std::cerr << "cpu LOW domain page tie-break duplicate job owner\n";
                std::exit(147);
            }
            job_domain[q] = d;
        }
    }

    std::vector<std::pair<size_t,size_t>> segs(size_t(domains), {ordered.size(), ordered.size()});
    int last_domain = -1;
    for (size_t k = 0; k < ordered.size(); ++k) {
        int d = job_domain[ordered[k].index];
        if (d < 0 || d >= domains || d < last_domain) {
            std::cerr << "cpu LOW domain page tie-break lost ordered domain ownership\n";
            std::exit(148);
        }
        last_domain = d;
        if (segs[size_t(d)].first == ordered.size()) segs[size_t(d)].first = k;
        segs[size_t(d)].second = k + 1;
    }

    std::vector<CpuLowDomainBoundaryPagePenalty> cache(ordered.size() + 1);
    std::vector<uint8_t> ready(ordered.size() + 1, 0);
    auto penalty = [&](size_t boundary) -> CpuLowDomainBoundaryPagePenalty {
        if (boundary == 0 || boundary >= ordered.size()) return {};
        if (!ready[boundary]) {
            cache[boundary] = cpu_low_domain_boundary_page_penalty(
                layout, storage, mask_index, ordered[boundary].mask);
            ready[boundary] = 1;
        }
        return cache[boundary];
    };

    auto sum_penalties = [&](uint64_t& p2, uint64_t& p4) {
        p2 = p4 = 0;
        for (int d = 0; d + 1 < domains; ++d) {
            if (segs[size_t(d)].first >= segs[size_t(d)].second
                || segs[size_t(d + 1)].first >= segs[size_t(d + 1)].second) continue;
            if (segs[size_t(d)].second != segs[size_t(d + 1)].first) continue;
            auto p = penalty(segs[size_t(d)].second);
            p2 += p.pages_2m;
            p4 += p.pages_4k;
        }
    };

    stats.max_worker_cells_before = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
    sum_penalties(stats.penalty_2m_before, stats.penalty_4k_before);

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
            auto current_penalty = penalty(old_boundary);

            size_t best_boundary = old_boundary;
            auto best_penalty = current_penalty;
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
                auto cp = penalty(candidate);
                size_t distance = candidate > old_boundary
                    ? candidate - old_boundary : old_boundary - candidate;

                bool better = cpu_low_domain_page_penalty_less(cp, best_penalty);
                if (!better && cpu_low_domain_page_penalty_equal(cp, best_penalty)) {
                    better = candidate_sum < best_sum
                        || (candidate_sum == best_sum && candidate_max < best_max)
                        || (candidate_sum == best_sum && candidate_max == best_max
                            && best_boundary != old_boundary && distance < best_distance);
                }
                if (better) {
                    best_boundary = candidate;
                    best_penalty = cp;
                    best_sum = candidate_sum;
                    best_max = candidate_max;
                    best_distance = distance;
                }
            }

            if (best_boundary != old_boundary) {
                bool page_improved = cpu_low_domain_page_penalty_less(
                    best_penalty, current_penalty);
                bool load_improved = cpu_low_domain_page_penalty_equal(
                    best_penalty, current_penalty)
                    && (best_sum < current_sum
                        || (best_sum == current_sum && best_max < current_max));
                if (!page_improved && !load_improved) {
                    std::cerr << "cpu LOW domain page tie-break selected non-improving boundary\n";
                    std::exit(151);
                }
                if (best_max > current_max) {
                    std::cerr << "cpu LOW domain page tie-break violated pair-max guard\n";
                    std::exit(152);
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
        std::cerr << "cpu LOW domain page tie-break accounting mismatch\n";
        std::exit(149);
    }
    stats.max_worker_cells_after = pool.sticky_worker_cells.empty() ? 0
        : *std::max_element(pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW domain page tie-break increased max worker load before="
                  << stats.max_worker_cells_before << " after=" << stats.max_worker_cells_after << '\n';
        std::exit(150);
    }
    sum_penalties(stats.penalty_2m_after, stats.penalty_4k_after);
    if (stats.penalty_2m_after > stats.penalty_2m_before
        || (stats.penalty_2m_after == stats.penalty_2m_before
            && stats.penalty_4k_after > stats.penalty_4k_before)) {
        std::cerr << "cpu LOW domain page tie-break increased page penalty"
                  << " before_2m=" << stats.penalty_2m_before
                  << " after_2m=" << stats.penalty_2m_after
                  << " before_4k=" << stats.penalty_4k_before
                  << " after_4k=" << stats.penalty_4k_after << '\n';
        std::exit(153);
    }
    stats.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_domain_page_tiebreak"
              << " objective=max_guard-page-sum-v5.23"
              << " boundary_moves=" << stats.boundary_moves
              << " moved_jobs=" << stats.moved_jobs
              << " candidate_evaluations=" << stats.candidate_evaluations
              << " max_guard_rejections=" << stats.max_guard_rejections
              << " page_improving_moves=" << stats.page_improving_moves
              << " page_tie_load_moves=" << stats.page_tie_load_moves
              << " page_improve_sum_increase_moves=" << stats.page_improve_sum_increase_moves
              << " penalty_2m_before=" << stats.penalty_2m_before
              << " penalty_2m_after=" << stats.penalty_2m_after
              << " penalty_4k_before=" << stats.penalty_4k_before
              << " penalty_4k_after=" << stats.penalty_4k_after
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " mask_index_mib=" << stats.mask_index_mib
              << " mask_index_build_s=" << stats.mask_index_build_s
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
