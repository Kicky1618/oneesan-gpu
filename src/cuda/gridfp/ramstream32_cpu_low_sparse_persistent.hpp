#pragma once

#include "ramstream32_cpu_low_sparse.hpp"
#include "ramstream32_cpu_affinity.hpp"

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

// Persistent worker wrapper for the sparse zero-scratch CPU LOW executor.
// The LOW recurrence and per-group body remain in ramstream32_cpu_low_sparse.hpp.
// Dynamic mode preserves the historical atomic work queue. Sticky mode builds a
// one-time exact-cell LPT partition. Contiguous mode keeps HIGH-occupancy masks
// in numeric order and computes the min-max optimal contiguous worker partition.
// Domain mode keeps numeric mask ranges contiguous only across worker domains,
// then restores exact-cell LPT inside each domain. Every static mode gives each
// persistent worker the same occupancy groups on every row.
enum CpuLowScheduleMode : uint8_t {
    CPU_LOW_SCHEDULE_DYNAMIC = 0,
    CPU_LOW_SCHEDULE_STICKY = 1,
    CPU_LOW_SCHEDULE_CONTIGUOUS = 2,
    CPU_LOW_SCHEDULE_DOMAIN = 3,
};

static CpuLowScheduleMode cpu_low_schedule_mode_from_env() {
    const char* s = std::getenv("CPU_LOW_SCHEDULE");
    if (!s || !*s || std::strcmp(s, "dynamic") == 0)
        return CPU_LOW_SCHEDULE_DYNAMIC;
    if (std::strcmp(s, "sticky") == 0)
        return CPU_LOW_SCHEDULE_STICKY;
    if (std::strcmp(s, "contiguous") == 0)
        return CPU_LOW_SCHEDULE_CONTIGUOUS;
    if (std::strcmp(s, "domain") == 0)
        return CPU_LOW_SCHEDULE_DOMAIN;
    std::cerr << "CPU_LOW_SCHEDULE must be dynamic, sticky, contiguous, or domain\n";
    std::exit(135);
}

static const char* cpu_low_schedule_name(CpuLowScheduleMode mode) {
    if (mode == CPU_LOW_SCHEDULE_STICKY) return "sticky";
    if (mode == CPU_LOW_SCHEDULE_CONTIGUOUS) return "contiguous";
    if (mode == CPU_LOW_SCHEDULE_DOMAIN) return "domain";
    return "dynamic";
}

static int cpu_low_domain_size_from_env() {
    const char* s = std::getenv("CPU_LOW_DOMAIN_SIZE");
    if (!s || !*s) return 0;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end || v <= 0 || v > 1'000'000) {
        std::cerr << "CPU_LOW_DOMAIN_SIZE must be a positive integer\n";
        std::exit(138);
    }
    return int(v);
}

static bool cpu_low_domain_refine_from_env() {
    const char* s = std::getenv("CPU_LOW_DOMAIN_REFINE");
    if (!s || !*s || std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0
        || std::strcmp(s, "yes") == 0 || std::strcmp(s, "on") == 0) return true;
    if (std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0
        || std::strcmp(s, "no") == 0 || std::strcmp(s, "off") == 0) return false;
    std::cerr << "CPU_LOW_DOMAIN_REFINE must be 0/1, false/true, no/yes, or off/on\n";
    std::exit(142);
}

static uint64_t cpu_low_sparse_job_cells(
    const CpuLowJob& job, const CpuLowSparseHost& sparse
) {
    uint64_t total = 0;
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
            const FBlock& x = job.main_blocks[bid];
            if (!x.stride) continue;
            uint64_t rows = uint64_t((x.end - x.off) / x.stride);
            auto [na, nb] = cpu_sparse_range(
                sparse.nn_orbit_off, sparse.nblocks, pi, bid);
            auto [ra, rb] = cpu_sparse_range(
                sparse.nr_orbit_off, sparse.nblocks, pi, bid);
            auto [la, lb] = cpu_sparse_range(
                sparse.nl_orbit_off, sparse.nblocks, pi, bid);
            auto [ba, bb] = cpu_sparse_range(
                sparse.local_closure_off, sparse.nblocks, pi, bid);
            auto [ca, cb] = cpu_sparse_range(
                sparse.cross_closure_off, sparse.nblocks, pi, bid);
            uint64_t ops = uint64_t(nb - na) + uint64_t(rb - ra)
                + uint64_t(lb - la) + uint64_t(bb - ba) + uint64_t(cb - ca);
            total += rows * ops;
        }
    }
    return total;
}

struct CpuLowStaticJobCost {
    size_t index = 0;
    uint32_t mask = 0;
    uint64_t cells = 0;
};

static size_t cpu_low_contiguous_segments_needed(
    const std::vector<CpuLowStaticJobCost>& ordered, uint64_t cap
) {
    if (ordered.empty()) return 0;
    size_t segments = 1;
    uint64_t acc = 0;
    for (const auto& x : ordered) {
        if (x.cells > cap) return std::numeric_limits<size_t>::max();
        if (acc && acc > cap - x.cells) {
            ++segments;
            acc = 0;
        }
        acc += x.cells;
    }
    return segments;
}

static uint64_t cpu_low_contiguous_segment_cells(
    const std::vector<CpuLowStaticJobCost>& ordered,
    const std::pair<size_t,size_t>& seg
) {
    uint64_t z = 0;
    for (size_t i = seg.first; i < seg.second; ++i) z += ordered[i].cells;
    return z;
}

static uint64_t cpu_low_build_contiguous_schedule(
    const std::vector<CpuLowJob>& jobs, const CpuLowSparseHost& sparse,
    int workers, std::vector<std::vector<size_t>>& worker_jobs,
    std::vector<uint64_t>& worker_cells
) {
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

    worker_jobs.assign(size_t(workers), {});
    worker_cells.assign(size_t(workers), 0);
    if (ordered.empty()) return 0;

    size_t target_segments = std::min<size_t>(size_t(workers), ordered.size());
    uint64_t lo = 0;
    for (const auto& x : ordered) lo = std::max(lo, x.cells);
    uint64_t hi = total_cells;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (cpu_low_contiguous_segments_needed(ordered, mid) <= target_segments)
            hi = mid;
        else
            lo = mid + 1;
    }
    uint64_t optimal_cap = lo;

    std::vector<std::pair<size_t,size_t>> segs;
    size_t begin = 0;
    uint64_t acc = 0;
    for (size_t i = 0; i < ordered.size(); ++i) {
        uint64_t cells = ordered[i].cells;
        if (acc && acc > optimal_cap - cells) {
            segs.push_back({begin, i});
            begin = i;
            acc = 0;
        }
        acc += cells;
    }
    segs.push_back({begin, ordered.size()});

    while (segs.size() < target_segments) {
        size_t best_seg = size_t(-1);
        uint64_t best_cells = 0;
        for (size_t s = 0; s < segs.size(); ++s) {
            if (segs[s].second - segs[s].first <= 1) continue;
            uint64_t cells = cpu_low_contiguous_segment_cells(ordered, segs[s]);
            if (best_seg == size_t(-1) || cells > best_cells) {
                best_seg = s;
                best_cells = cells;
            }
        }
        if (best_seg == size_t(-1)) {
            std::cerr << "cpu LOW contiguous split reconstruction failed\n";
            std::exit(136);
        }

        auto seg = segs[best_seg];
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
        segs[best_seg] = {seg.first, split};
        segs.insert(segs.begin() + best_seg + 1, {split, seg.second});
    }

    for (size_t w = 0; w < segs.size(); ++w) {
        for (size_t i = segs[w].first; i < segs[w].second; ++i) {
            worker_jobs[w].push_back(ordered[i].index);
            worker_cells[w] += ordered[i].cells;
        }
        if (worker_cells[w] > optimal_cap) {
            std::cerr << "cpu LOW contiguous cap violation worker=" << w
                      << " cells=" << worker_cells[w]
                      << " cap=" << optimal_cap << '\n';
            std::exit(137);
        }
    }
    return optimal_cap;
}

static uint64_t cpu_low_saturating_mul(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a)
        return std::numeric_limits<uint64_t>::max();
    return a * b;
}

static void cpu_low_sort_lpt_range(
    std::vector<CpuLowStaticJobCost>& local,
    const std::vector<CpuLowJob>& jobs
) {
    std::sort(local.begin(), local.end(), [&](const auto& a, const auto& b) {
        if (a.cells != b.cells) return a.cells > b.cells;
        if (jobs[a.index].scratch_bytes != jobs[b.index].scratch_bytes)
            return jobs[a.index].scratch_bytes > jobs[b.index].scratch_bytes;
        return jobs[a.index].g < jobs[b.index].g;
    });
}

static uint64_t cpu_low_lpt_range_max_cells(
    const std::vector<CpuLowStaticJobCost>& ordered,
    size_t begin, size_t end, int nworkers,
    const std::vector<CpuLowJob>& jobs
) {
    if (begin >= end || nworkers <= 0) return 0;
    std::vector<CpuLowStaticJobCost> local(
        ordered.begin() + begin, ordered.begin() + end);
    cpu_low_sort_lpt_range(local, jobs);
    std::vector<uint64_t> loads(size_t(nworkers), 0);
    for (const auto& x : local) {
        int best = 0;
        for (int w = 1; w < nworkers; ++w) {
            if (loads[size_t(w)] < loads[size_t(best)]) best = w;
        }
        loads[size_t(best)] += x.cells;
    }
    return *std::max_element(loads.begin(), loads.end());
}

static void cpu_low_lpt_assign_range(
    const std::vector<CpuLowStaticJobCost>& ordered,
    size_t begin, size_t end, int first_worker, int nworkers,
    const std::vector<CpuLowJob>& jobs,
    std::vector<std::vector<size_t>>& worker_jobs,
    std::vector<uint64_t>& worker_cells
) {
    if (begin >= end || nworkers <= 0) return;
    std::vector<CpuLowStaticJobCost> local(
        ordered.begin() + begin, ordered.begin() + end);
    cpu_low_sort_lpt_range(local, jobs);
    for (const auto& x : local) {
        int best = first_worker;
        for (int w = first_worker + 1; w < first_worker + nworkers; ++w) {
            if (worker_cells[size_t(w)] < worker_cells[size_t(best)]) best = w;
        }
        worker_jobs[size_t(best)].push_back(x.index);
        worker_cells[size_t(best)] += x.cells;
    }
}

static void cpu_low_refine_domain_boundaries(
    const std::vector<CpuLowStaticJobCost>& ordered,
    const std::vector<CpuLowJob>& jobs,
    int workers, int domain_size,
    std::vector<std::pair<size_t,size_t>>& segs,
    int& refined_boundaries, uint64_t& moved_jobs
) {
    constexpr size_t RADIUS = 32;
    constexpr int PASSES = 2;
    refined_boundaries = 0;
    moved_jobs = 0;
    if (segs.size() < 2) return;

    for (int pass = 0; pass < PASSES; ++pass) {
        bool changed = false;
        for (size_t q = 0; q + 1 < segs.size(); ++q) {
            size_t d = pass == 0 ? q : (segs.size() - 2 - q);
            auto left = segs[d];
            auto right = segs[d + 1];
            if (left.first >= left.second || right.first >= right.second) continue;
            if (left.second != right.first) {
                std::cerr << "cpu LOW domain boundary lost contiguity\n";
                std::exit(141);
            }

            int left_first_worker = int(d) * domain_size;
            int right_first_worker = int(d + 1) * domain_size;
            int left_workers = std::min(domain_size, workers - left_first_worker);
            int right_workers = std::min(domain_size, workers - right_first_worker);
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
            uint64_t current_sum = current_left + current_right;

            size_t best_boundary = old_boundary;
            uint64_t best_max = current_max;
            uint64_t best_sum = current_sum;
            size_t best_distance = 0;

            for (size_t candidate = search_lo; candidate <= search_hi; ++candidate) {
                if (candidate == old_boundary) continue;
                uint64_t lm = cpu_low_lpt_range_max_cells(
                    ordered, left.first, candidate, left_workers, jobs);
                uint64_t rm = cpu_low_lpt_range_max_cells(
                    ordered, candidate, right.second, right_workers, jobs);
                uint64_t pm = std::max(lm, rm);
                uint64_t ps = lm + rm;
                size_t distance = candidate > old_boundary
                    ? candidate - old_boundary : old_boundary - candidate;
                bool better = pm < best_max
                    || (pm == best_max && ps < best_sum)
                    || (pm == best_max && ps == best_sum
                        && best_boundary != old_boundary && distance < best_distance);
                if (better) {
                    best_boundary = candidate;
                    best_max = pm;
                    best_sum = ps;
                    best_distance = distance;
                }
            }

            if (best_boundary != old_boundary
                && (best_max < current_max
                    || (best_max == current_max && best_sum < current_sum))) {
                size_t distance = best_boundary > old_boundary
                    ? best_boundary - old_boundary : old_boundary - best_boundary;
                segs[d].second = best_boundary;
                segs[d + 1].first = best_boundary;
                ++refined_boundaries;
                moved_jobs += distance;
                changed = true;
            }
        }
        if (!changed) break;
    }
}

static uint64_t cpu_low_build_domain_schedule(
    const std::vector<CpuLowJob>& jobs, const CpuLowSparseHost& sparse,
    int workers, int domain_size, bool refine_boundaries,
    std::vector<std::vector<size_t>>& worker_jobs,
    std::vector<uint64_t>& worker_cells,
    int& active_domains, int& refined_boundaries,
    uint64_t& moved_jobs
) {
    worker_jobs.assign(size_t(workers), {});
    worker_cells.assign(size_t(workers), 0);
    active_domains = 0;
    refined_boundaries = 0;
    moved_jobs = 0;
    int domains = (workers + domain_size - 1) / domain_size;

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
    if (ordered.empty()) return 0;

    auto feasible = [&](uint64_t per_worker_cap) {
        size_t i = 0;
        for (int d = 0; d < domains && i < ordered.size(); ++d) {
            int first_worker = d * domain_size;
            int nworkers = std::min(domain_size, workers - first_worker);
            uint64_t cap = cpu_low_saturating_mul(
                per_worker_cap, uint64_t(nworkers));
            uint64_t acc = 0;
            while (i < ordered.size() && ordered[i].cells <= cap - acc) {
                acc += ordered[i].cells;
                ++i;
            }
        }
        return i == ordered.size();
    };

    uint64_t lo = 0, hi = total_cells;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (feasible(mid)) hi = mid;
        else lo = mid + 1;
    }
    uint64_t normalized_cap = lo;

    std::vector<std::pair<size_t,size_t>> segs(size_t(domains));
    size_t i = 0;
    for (int d = 0; d < domains; ++d) {
        int first_worker = d * domain_size;
        int nworkers = std::min(domain_size, workers - first_worker);
        uint64_t cap = cpu_low_saturating_mul(normalized_cap, uint64_t(nworkers));
        uint64_t acc = 0;
        size_t begin = i;
        while (i < ordered.size() && ordered[i].cells <= cap - acc) {
            acc += ordered[i].cells;
            ++i;
        }
        segs[size_t(d)] = {begin, i};
    }
    if (i != ordered.size()) {
        std::cerr << "cpu LOW domain schedule reconstruction failed assigned="
                  << i << " jobs=" << ordered.size() << '\n';
        std::exit(139);
    }

    if (refine_boundaries) {
        cpu_low_refine_domain_boundaries(
            ordered, jobs, workers, domain_size, segs,
            refined_boundaries, moved_jobs);
    }

    for (int d = 0; d < domains; ++d) {
        auto seg = segs[size_t(d)];
        if (seg.first >= seg.second) continue;
        ++active_domains;
        int first_worker = d * domain_size;
        int nworkers = std::min(domain_size, workers - first_worker);
        cpu_low_lpt_assign_range(
            ordered, seg.first, seg.second, first_worker, nworkers,
            jobs, worker_jobs, worker_cells);
    }

    return normalized_cap;
}

struct CpuLowSparsePersistentPool {
    int workers = 1;
    CpuLowScheduleMode schedule_mode = CPU_LOW_SCHEDULE_DYNAMIC;
    int domain_size = 0;
    bool domain_refine = true;
    std::vector<CpuLowSparseStats> stats;
    double wall_s = 0.0;
    double worker_start_s = 0.0;
    double schedule_build_s = 0.0;
    uint64_t contiguous_optimal_cap = 0;
    uint64_t domain_normalized_cap = 0;
    int domain_active_domains = 0;
    int domain_refined_boundaries = 0;
    uint64_t domain_refined_job_moves = 0;

    std::mutex mu;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    std::vector<std::thread> threads;
    bool stopping = false;
    bool in_flight = false;
    uint64_t generation = 0;
    int pending = 0;
    std::atomic<size_t> next{0};
    std::chrono::steady_clock::time_point run_start{};

    const std::vector<CpuLowJob>* run_jobs = nullptr;
    RamCounts* run_main = nullptr;
    RamCounts* run_block = nullptr;
    const StorageFactorHost* run_storage = nullptr;
    const StorageLayout* run_layout = nullptr;
    const CpuLowSparseHost* run_sparse = nullptr;
    Count run_mod = 0;

    const std::vector<CpuLowJob>* sticky_source_jobs = nullptr;
    const CpuLowSparseHost* sticky_source_sparse = nullptr;
    std::vector<std::vector<size_t>> sticky_worker_jobs;
    std::vector<uint64_t> sticky_worker_cells;

    explicit CpuLowSparsePersistentPool(
        int n,
        CpuLowScheduleMode mode = cpu_low_schedule_mode_from_env(),
        int requested_domain_size = cpu_low_domain_size_from_env(),
        bool requested_domain_refine = cpu_low_domain_refine_from_env()
    ) : workers(std::max(1, n)), schedule_mode(mode),
        domain_size(requested_domain_size), domain_refine(requested_domain_refine),
        stats(size_t(std::max(1, n))) {
        if (schedule_mode == CPU_LOW_SCHEDULE_DOMAIN
            && (domain_size <= 0 || domain_size > workers)) {
            std::cerr << "CPU_LOW_DOMAIN_SIZE must be in 1..CPU_WORKERS for domain schedule\n";
            std::exit(140);
        }
    }

    CpuLowSparsePersistentPool(const CpuLowSparsePersistentPool&) = delete;
    CpuLowSparsePersistentPool& operator=(const CpuLowSparsePersistentPool&) = delete;

    ~CpuLowSparsePersistentPool() { shutdown(); }

    void ensure_started() {
        if (!threads.empty()) return;
        auto t0 = std::chrono::steady_clock::now();
        threads.reserve(workers);
        for (int w = 0; w < workers; ++w)
            threads.emplace_back([this, w] { worker_loop(w); });
        worker_start_s += ram_seconds_since(t0);
        std::cerr << "cpu_low_sparse_persistent workers=" << workers
                  << " schedule=" << cpu_low_schedule_name(schedule_mode);
        if (schedule_mode == CPU_LOW_SCHEDULE_DOMAIN)
            std::cerr << " domain_size=" << domain_size
                      << " refine=" << int(domain_refine);
        std::cerr << " start_s=" << worker_start_s << '\n';
    }

    void prepare_static_schedule(
        const std::vector<CpuLowJob>& jobs, const CpuLowSparseHost& sparse
    ) {
        if (schedule_mode == CPU_LOW_SCHEDULE_DYNAMIC) return;
        if (sticky_source_jobs == &jobs && sticky_source_sparse == &sparse) return;

        wait_run();

        auto t0 = std::chrono::steady_clock::now();
        uint64_t total_cells = 0;
        size_t nonempty_jobs = 0;
        contiguous_optimal_cap = 0;
        domain_normalized_cap = 0;
        domain_active_domains = 0;
        domain_refined_boundaries = 0;
        domain_refined_job_moves = 0;

        if (schedule_mode == CPU_LOW_SCHEDULE_DOMAIN) {
            domain_normalized_cap = cpu_low_build_domain_schedule(
                jobs, sparse, workers, domain_size, domain_refine,
                sticky_worker_jobs, sticky_worker_cells, domain_active_domains,
                domain_refined_boundaries, domain_refined_job_moves);
            for (size_t i = 0; i < jobs.size(); ++i) {
                if (!jobs[i].main_size && !jobs[i].block_size) continue;
                ++nonempty_jobs;
                total_cells += cpu_low_sparse_job_cells(jobs[i], sparse);
            }
        } else if (schedule_mode == CPU_LOW_SCHEDULE_CONTIGUOUS) {
            contiguous_optimal_cap = cpu_low_build_contiguous_schedule(
                jobs, sparse, workers, sticky_worker_jobs, sticky_worker_cells);
            for (size_t i = 0; i < jobs.size(); ++i) {
                if (!jobs[i].main_size && !jobs[i].block_size) continue;
                ++nonempty_jobs;
                total_cells += cpu_low_sparse_job_cells(jobs[i], sparse);
            }
        } else {
            std::vector<std::pair<size_t, uint64_t>> ranked;
            ranked.reserve(jobs.size());
            for (size_t i = 0; i < jobs.size(); ++i) {
                if (!jobs[i].main_size && !jobs[i].block_size) continue;
                uint64_t cells = cpu_low_sparse_job_cells(jobs[i], sparse);
                ranked.push_back({i, cells});
                total_cells += cells;
            }
            nonempty_jobs = ranked.size();
            std::sort(ranked.begin(), ranked.end(), [&](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second > b.second;
                if (jobs[a.first].scratch_bytes != jobs[b.first].scratch_bytes)
                    return jobs[a.first].scratch_bytes > jobs[b.first].scratch_bytes;
                return jobs[a.first].g < jobs[b.first].g;
            });

            sticky_worker_jobs.assign(size_t(workers), {});
            sticky_worker_cells.assign(size_t(workers), 0);
            for (const auto& [index, cells] : ranked) {
                int best = 0;
                for (int w = 1; w < workers; ++w) {
                    if (sticky_worker_cells[size_t(w)] < sticky_worker_cells[size_t(best)])
                        best = w;
                }
                sticky_worker_jobs[size_t(best)].push_back(index);
                sticky_worker_cells[size_t(best)] += cells;
            }
        }

        sticky_source_jobs = &jobs;
        sticky_source_sparse = &sparse;
        double dt = ram_seconds_since(t0);
        schedule_build_s += dt;

        uint64_t min_cells = sticky_worker_cells.empty() ? 0 : sticky_worker_cells[0];
        uint64_t max_cells = 0;
        for (uint64_t x : sticky_worker_cells) {
            min_cells = std::min(min_cells, x);
            max_cells = std::max(max_cells, x);
        }
        double avg = workers ? double(total_cells) / workers : 0.0;
        const char* prefix = "cpu_low_sticky_schedule";
        if (schedule_mode == CPU_LOW_SCHEDULE_CONTIGUOUS)
            prefix = "cpu_low_contiguous_schedule";
        else if (schedule_mode == CPU_LOW_SCHEDULE_DOMAIN)
            prefix = "cpu_low_domain_schedule";
        std::cerr << prefix
                  << " jobs=" << nonempty_jobs
                  << " workers=" << workers
                  << " total_cells=" << total_cells
                  << " min_worker_cells=" << min_cells
                  << " max_worker_cells=" << max_cells
                  << " imbalance=" << (avg > 0.0 ? double(max_cells) / avg : 0.0);
        if (schedule_mode == CPU_LOW_SCHEDULE_CONTIGUOUS)
            std::cerr << " optimal_cap=" << contiguous_optimal_cap;
        if (schedule_mode == CPU_LOW_SCHEDULE_DOMAIN)
            std::cerr << " domain_size=" << domain_size
                      << " refine=" << int(domain_refine)
                      << " active_domains=" << domain_active_domains
                      << " outer_normalized_cap=" << domain_normalized_cap
                      << " refined_boundaries=" << domain_refined_boundaries
                      << " refined_job_moves=" << domain_refined_job_moves;
        std::cerr << " build_s=" << dt << '\n';
    }

    void prepare_sticky_schedule(
        const std::vector<CpuLowJob>& jobs, const CpuLowSparseHost& sparse
    ) {
        prepare_static_schedule(jobs, sparse);
    }

    void shutdown() {
        if (threads.empty()) return;
        wait_run();
        {
            std::lock_guard<std::mutex> lock(mu);
            if (stopping) return;
            stopping = true;
            ++generation;
        }
        start_cv.notify_all();
        for (auto& t : threads) if (t.joinable()) t.join();
        threads.clear();
    }

    void process_job(int w, size_t q) {
        const CpuLowJob& job = (*run_jobs)[q];
        if (!job.main_size && !job.block_size) return;
        process_cpu_low_group_sparse(
            stats[size_t(w)], job, *run_main, *run_block,
            *run_storage, *run_layout, *run_sparse, run_mod);
    }

    void worker_loop(int w) {
        cpu_low_bind_worker(w);
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mu);
                start_cv.wait(lock, [&] { return stopping || generation != seen; });
                if (stopping) return;
                seen = generation;
            }

            if (schedule_mode != CPU_LOW_SCHEDULE_DYNAMIC) {
                for (size_t q : sticky_worker_jobs[size_t(w)]) process_job(w, q);
            } else {
                for (;;) {
                    size_t q = next.fetch_add(1, std::memory_order_relaxed);
                    if (q >= run_jobs->size()) break;
                    process_job(w, q);
                }
            }

            {
                std::lock_guard<std::mutex> lock(mu);
                if (--pending == 0) {
                    wall_s += ram_seconds_since(run_start);
                    in_flight = false;
                    done_cv.notify_all();
                }
            }
        }
    }

    bool start_run(
        const std::vector<CpuLowJob>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const CpuLowSparseHost& sparse, Count mod
    ) {
        if (jobs.empty()) return false;
        prepare_static_schedule(jobs, sparse);
        ensure_started();
        {
            std::lock_guard<std::mutex> lock(mu);
            if (in_flight || pending != 0) {
                std::cerr << "cpu low persistent start while previous run is active\n";
                std::exit(134);
            }
            run_jobs = &jobs;
            run_main = &main_auth;
            run_block = &block_auth;
            run_storage = &storage;
            run_layout = &layout;
            run_sparse = &sparse;
            run_mod = mod;
            next.store(0, std::memory_order_relaxed);
            pending = workers;
            in_flight = true;
            run_start = std::chrono::steady_clock::now();
            ++generation;
        }
        start_cv.notify_all();
        return true;
    }

    void wait_run() {
        std::unique_lock<std::mutex> lock(mu);
        done_cv.wait(lock, [&] { return !in_flight && pending == 0; });
    }

    void run(
        const std::vector<CpuLowJob>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const CpuLowSparseHost& sparse, Count mod
    ) {
        if (start_run(jobs, main_auth, block_auth, storage, layout, sparse, mod))
            wait_run();
    }

    double kernel_s() const {
        double z = 0.0;
        for (const auto& x : stats) z += x.kernel_s;
        return z;
    }
    uint64_t groups() const {
        uint64_t z = 0;
        for (const auto& x : stats) z += x.groups;
        return z;
    }
};
