#pragma once

#include "ramstream32_cpu_low_sparse.hpp"
#include "ramstream32_cpu_affinity.hpp"

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

// Persistent worker wrapper for the sparse zero-scratch CPU LOW executor.
// The LOW recurrence and per-group body remain in ramstream32_cpu_low_sparse.hpp.
// Dynamic mode preserves the historical atomic work queue. Sticky mode builds a
// one-time exact-cell LPT partition and gives each persistent worker the same
// occupancy groups on every row, so CPU affinity can preserve group ownership
// across generations.
enum CpuLowScheduleMode : uint8_t {
    CPU_LOW_SCHEDULE_DYNAMIC = 0,
    CPU_LOW_SCHEDULE_STICKY = 1,
};

static CpuLowScheduleMode cpu_low_schedule_mode_from_env() {
    const char* s = std::getenv("CPU_LOW_SCHEDULE");
    if (!s || !*s || std::strcmp(s, "dynamic") == 0)
        return CPU_LOW_SCHEDULE_DYNAMIC;
    if (std::strcmp(s, "sticky") == 0)
        return CPU_LOW_SCHEDULE_STICKY;
    std::cerr << "CPU_LOW_SCHEDULE must be dynamic or sticky\n";
    std::exit(135);
}

static const char* cpu_low_schedule_name(CpuLowScheduleMode mode) {
    return mode == CPU_LOW_SCHEDULE_STICKY ? "sticky" : "dynamic";
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

struct CpuLowSparsePersistentPool {
    int workers = 1;
    CpuLowScheduleMode schedule_mode = CPU_LOW_SCHEDULE_DYNAMIC;
    std::vector<CpuLowSparseStats> stats;
    double wall_s = 0.0;
    double worker_start_s = 0.0;
    double schedule_build_s = 0.0;

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
        int n, CpuLowScheduleMode mode = cpu_low_schedule_mode_from_env()
    ) : workers(std::max(1, n)), schedule_mode(mode),
        stats(size_t(std::max(1, n))) {}

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
                  << " schedule=" << cpu_low_schedule_name(schedule_mode)
                  << " start_s=" << worker_start_s << '\n';
    }

    void prepare_sticky_schedule(
        const std::vector<CpuLowJob>& jobs, const CpuLowSparseHost& sparse
    ) {
        if (schedule_mode != CPU_LOW_SCHEDULE_STICKY) return;
        if (sticky_source_jobs == &jobs && sticky_source_sparse == &sparse) return;

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::pair<size_t, uint64_t>> ranked;
        ranked.reserve(jobs.size());
        uint64_t total_cells = 0;
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (!jobs[i].main_size && !jobs[i].block_size) continue;
            uint64_t cells = cpu_low_sparse_job_cells(jobs[i], sparse);
            ranked.push_back({i, cells});
            total_cells += cells;
        }
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
        sticky_source_jobs = &jobs;
        sticky_source_sparse = &sparse;
        schedule_build_s += ram_seconds_since(t0);

        uint64_t min_cells = sticky_worker_cells.empty() ? 0 : sticky_worker_cells[0];
        uint64_t max_cells = 0;
        for (uint64_t x : sticky_worker_cells) {
            min_cells = std::min(min_cells, x);
            max_cells = std::max(max_cells, x);
        }
        double avg = workers ? double(total_cells) / workers : 0.0;
        std::cerr << "cpu_low_sticky_schedule jobs=" << ranked.size()
                  << " workers=" << workers
                  << " total_cells=" << total_cells
                  << " min_worker_cells=" << min_cells
                  << " max_worker_cells=" << max_cells
                  << " imbalance=" << (avg > 0.0 ? double(max_cells) / avg : 0.0)
                  << " build_s=" << schedule_build_s << '\n';
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

            if (schedule_mode == CPU_LOW_SCHEDULE_STICKY) {
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
        prepare_sticky_schedule(jobs, sparse);
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
