#pragma once

#include "ramstream32_cpu_low_sparse_persistent.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

// Research-only v5.25 worker-locality stage.
//
// Domain scheduling already guarantees that each NUMA domain owns one ordered
// HIGH-mask interval, but exact-cell LPT inside the domain can scatter adjacent
// masks across many workers.  This stage leaves every domain boundary fixed and
// asks whether the jobs inside a domain can instead be split into contiguous
// worker intervals without exceeding that domain's current LPT maximum.
//
// If yes, the domain is converted to contiguous worker ownership.  If no, its
// original LPT assignment is preserved.  Therefore every converted domain has
// max worker load <= its previous LPT max, every fallback domain is identical,
// and the global max worker load cannot increase.

struct CpuLowDomainWorkerLocalityStats {
    int domains = 0;
    int nonempty_domains = 0;
    int converted_domains = 0;
    int fallback_domains = 0;
    int unchanged_trivial_domains = 0;
    uint64_t converted_jobs = 0;
    uint64_t contiguous_worker_segments = 0;
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double build_s = 0.0;
};

static uint64_t cpu_low_range_cells(
    const std::vector<CpuLowStaticJobCost>& ordered,
    size_t begin, size_t end
) {
    uint64_t z = 0;
    for (size_t i = begin; i < end; ++i) z += ordered[i].cells;
    return z;
}

static bool cpu_low_domain_contiguous_segments_under_cap(
    const std::vector<CpuLowStaticJobCost>& ordered,
    size_t begin, size_t end,
    int nworkers, uint64_t cap,
    std::vector<std::pair<size_t,size_t>>& segs
) {
    segs.clear();
    if (begin >= end) return true;
    if (nworkers <= 0) return false;

    size_t seg_begin = begin;
    uint64_t acc = 0;
    for (size_t i = begin; i < end; ++i) {
        uint64_t cells = ordered[i].cells;
        if (cells > cap) return false;
        if (acc && acc > cap - cells) {
            segs.push_back({seg_begin, i});
            seg_begin = i;
            acc = 0;
        }
        acc += cells;
    }
    segs.push_back({seg_begin, end});
    if (segs.size() > size_t(nworkers)) return false;

    const size_t target = std::min<size_t>(size_t(nworkers), end - begin);
    while (segs.size() < target) {
        size_t best = size_t(-1);
        uint64_t best_cells = 0;
        size_t best_len = 0;
        for (size_t s = 0; s < segs.size(); ++s) {
            size_t len = segs[s].second - segs[s].first;
            if (len <= 1) continue;
            uint64_t cells = cpu_low_range_cells(
                ordered, segs[s].first, segs[s].second);
            if (best == size_t(-1) || cells > best_cells
                || (cells == best_cells && len > best_len)) {
                best = s;
                best_cells = cells;
                best_len = len;
            }
        }
        if (best == size_t(-1)) return false;

        auto seg = segs[best];
        uint64_t prefix = 0;
        uint64_t best_delta = std::numeric_limits<uint64_t>::max();
        size_t split = seg.first + 1;
        for (size_t i = seg.first; i + 1 < seg.second; ++i) {
            prefix += ordered[i].cells;
            uint64_t right = best_cells - prefix;
            uint64_t delta = prefix > right ? prefix - right : right - prefix;
            if (delta < best_delta) {
                best_delta = delta;
                split = i + 1;
            }
        }
        segs[best] = {seg.first, split};
        segs.insert(segs.begin() + best + 1, {split, seg.second});
    }

    for (const auto& seg : segs) {
        if (cpu_low_range_cells(ordered, seg.first, seg.second) > cap)
            return false;
    }
    return true;
}

static CpuLowDomainWorkerLocalityStats cpu_low_apply_domain_worker_locality(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse
) {
    CpuLowDomainWorkerLocalityStats stats;
    auto t0 = std::chrono::steady_clock::now();

    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || pool.domain_size <= 0) {
        std::cerr << "cpu LOW domain worker locality requires domain schedule\n";
        std::exit(169);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW domain worker locality requires prepared schedule\n";
        std::exit(170);
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

    std::vector<int> job_domain(jobs.size(), -1);
    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size() || job_domain[q] >= 0) {
                std::cerr << "cpu LOW domain worker locality duplicate job owner\n";
                std::exit(171);
            }
            job_domain[q] = d;
        }
    }

    std::vector<std::pair<size_t,size_t>> domain_segs(
        size_t(stats.domains), {ordered.size(), ordered.size()});
    int last_domain = -1;
    for (size_t k = 0; k < ordered.size(); ++k) {
        int d = job_domain[ordered[k].index];
        if (d < 0 || d >= stats.domains || d < last_domain) {
            std::cerr << "cpu LOW domain worker locality lost domain ordering\n";
            std::exit(172);
        }
        last_domain = d;
        if (domain_segs[size_t(d)].first == ordered.size())
            domain_segs[size_t(d)].first = k;
        domain_segs[size_t(d)].second = k + 1;
    }

    std::vector<std::vector<size_t>> next_jobs = pool.sticky_worker_jobs;
    std::vector<uint64_t> next_cells = pool.sticky_worker_cells;
    std::vector<std::pair<size_t,size_t>> worker_segs;

    for (int d = 0; d < stats.domains; ++d) {
        auto domain_seg = domain_segs[size_t(d)];
        if (domain_seg.first >= domain_seg.second) continue;
        ++stats.nonempty_domains;
        int first_worker = d * pool.domain_size;
        int nworkers = std::min(pool.domain_size, pool.workers - first_worker);
        uint64_t cap = 0;
        for (int w = first_worker; w < first_worker + nworkers; ++w)
            cap = std::max(cap, pool.sticky_worker_cells[size_t(w)]);

        if (domain_seg.second - domain_seg.first <= 1 || nworkers <= 1) {
            ++stats.unchanged_trivial_domains;
            continue;
        }

        if (!cpu_low_domain_contiguous_segments_under_cap(
                ordered, domain_seg.first, domain_seg.second,
                nworkers, cap, worker_segs)) {
            ++stats.fallback_domains;
            continue;
        }

        for (int w = first_worker; w < first_worker + nworkers; ++w) {
            next_jobs[size_t(w)].clear();
            next_cells[size_t(w)] = 0;
        }
        for (size_t s = 0; s < worker_segs.size(); ++s) {
            int w = first_worker + int(s);
            for (size_t i = worker_segs[s].first; i < worker_segs[s].second; ++i) {
                next_jobs[size_t(w)].push_back(ordered[i].index);
                next_cells[size_t(w)] += ordered[i].cells;
            }
            if (next_cells[size_t(w)] > cap) {
                std::cerr << "cpu LOW domain worker locality cap violation"
                          << " domain=" << d
                          << " worker=" << w
                          << " cells=" << next_cells[size_t(w)]
                          << " cap=" << cap << '\n';
                std::exit(173);
            }
        }
        ++stats.converted_domains;
        stats.converted_jobs += domain_seg.second - domain_seg.first;
        stats.contiguous_worker_segments += worker_segs.size();
    }

    uint64_t assigned_cells = 0;
    for (uint64_t x : next_cells) assigned_cells += x;
    if (assigned_cells != expected_cells) {
        std::cerr << "cpu LOW domain worker locality accounting mismatch"
                  << " assigned=" << assigned_cells
                  << " expected=" << expected_cells << '\n';
        std::exit(174);
    }

    stats.max_worker_cells_after = next_cells.empty() ? 0
        : *std::max_element(next_cells.begin(), next_cells.end());
    if (stats.max_worker_cells_after > stats.max_worker_cells_before) {
        std::cerr << "cpu LOW domain worker locality increased max worker load"
                  << " before=" << stats.max_worker_cells_before
                  << " after=" << stats.max_worker_cells_after << '\n';
        std::exit(175);
    }

    pool.sticky_worker_jobs.swap(next_jobs);
    pool.sticky_worker_cells.swap(next_cells);
    stats.build_s = ram_seconds_since(t0);

    std::cerr << "cpu_low_domain_worker_locality"
              << " objective=contiguous-under-lpt-cap-v5.25-plan"
              << " domains=" << stats.domains
              << " nonempty_domains=" << stats.nonempty_domains
              << " converted_domains=" << stats.converted_domains
              << " fallback_domains=" << stats.fallback_domains
              << " unchanged_trivial_domains=" << stats.unchanged_trivial_domains
              << " converted_jobs=" << stats.converted_jobs
              << " contiguous_worker_segments=" << stats.contiguous_worker_segments
              << " max_worker_cells_before=" << stats.max_worker_cells_before
              << " max_worker_cells_after=" << stats.max_worker_cells_after
              << " build_s=" << stats.build_s << '\n';
    return stats;
}
