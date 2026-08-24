#pragma once

#include "ramstream32_cpu_high_direct.hpp"

#include <condition_variable>
#include <mutex>

// Persistent worker wrapper for the zero-scratch CPU HIGH direct executor.
// The recurrence and job body remain in ramstream32_cpu_high_direct.hpp; this
// class only removes per-row std::thread construction/destruction. Workers bind
// CPU affinity once and sleep between HIGH rows.
struct CpuHighDirectPersistentPool {
    int workers = 1;
    std::vector<CpuHighDirectStats> stats;
    double wall_s = 0.0;
    double schedule_build_s = 0.0;
    double worker_start_s = 0.0;

    std::vector<const CpuHighJob*> schedule_source;
    std::vector<const CpuHighJob*> scheduled_jobs;

    std::mutex mu;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    std::vector<std::thread> threads;
    bool stopping = false;
    uint64_t generation = 0;
    int pending = 0;
    std::atomic<size_t> next{0};

    RamCounts* run_main = nullptr;
    RamCounts* run_block = nullptr;
    const StorageFactorHost* run_storage = nullptr;
    const StorageLayout* run_layout = nullptr;
    const CpuHighDirectHost* run_direct = nullptr;
    const CpuHighCrossHost* run_cross = nullptr;
    Count run_mod = 0;

    explicit CpuHighDirectPersistentPool(int n)
        : workers(std::max(1, n)), stats(size_t(std::max(1, n))) {
        auto t0 = std::chrono::steady_clock::now();
        threads.reserve(workers);
        for (int w = 0; w < workers; ++w) {
            threads.emplace_back([this, w] { worker_loop(w); });
        }
        worker_start_s = ram_seconds_since(t0);
        std::cerr << "cpu_high_direct_persistent workers=" << workers
                  << " start_s=" << worker_start_s << '\n';
    }

    CpuHighDirectPersistentPool(const CpuHighDirectPersistentPool&) = delete;
    CpuHighDirectPersistentPool& operator=(const CpuHighDirectPersistentPool&) = delete;

    ~CpuHighDirectPersistentPool() { shutdown(); }

    void shutdown() {
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

    void worker_loop(int w) {
        cpu_high_bind_worker(w);
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mu);
                start_cv.wait(lock, [&] { return stopping || generation != seen; });
                if (stopping) return;
                seen = generation;
            }

            for (;;) {
                size_t q = next.fetch_add(1, std::memory_order_relaxed);
                if (q >= scheduled_jobs.size()) break;
                process_cpu_high_group_direct(
                    stats[size_t(w)], *scheduled_jobs[q],
                    *run_main, *run_block, *run_storage, *run_layout,
                    *run_direct, *run_cross, run_mod);
            }

            {
                std::lock_guard<std::mutex> lock(mu);
                if (--pending == 0) done_cv.notify_one();
            }
        }
    }

    void prepare_schedule(
        const std::vector<const CpuHighJob*>& jobs,
        const CpuHighDirectHost& direct
    ) {
        if (schedule_source.size() == jobs.size()
            && std::equal(schedule_source.begin(), schedule_source.end(), jobs.begin()))
            return;

        auto t0 = std::chrono::steady_clock::now();
        schedule_source = jobs;
        std::vector<std::pair<const CpuHighJob*,uint64_t>> ranked;
        ranked.reserve(jobs.size());
        uint64_t total_cells = 0;
        uint64_t max_cells = 0;
        uint64_t min_cells = UINT64_MAX;
        for (const CpuHighJob* job : jobs) {
            uint64_t cells = cpu_high_direct_job_cells(*job, direct);
            ranked.push_back({job, cells});
            total_cells += cells;
            max_cells = std::max(max_cells, cells);
            min_cells = std::min(min_cells, cells);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            if (a.first->scratch_bytes != b.first->scratch_bytes)
                return a.first->scratch_bytes > b.first->scratch_bytes;
            return a.first->g < b.first->g;
        });
        scheduled_jobs.clear();
        scheduled_jobs.reserve(ranked.size());
        for (const auto& x : ranked) scheduled_jobs.push_back(x.first);
        if (jobs.empty()) min_cells = 0;
        double dt = ram_seconds_since(t0);
        schedule_build_s += dt;
        std::cerr << "cpu_high_direct_schedule jobs=" << jobs.size()
                  << " total_cells=" << total_cells
                  << " max_cells=" << max_cells
                  << " min_cells=" << min_cells
                  << " build_s=" << dt
                  << " persistent=1\n";
    }

    void run(
        const std::vector<const CpuHighJob*>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const CpuHighDirectHost& direct, const CpuHighCrossHost& cross, Count mod
    ) {
        prepare_schedule(jobs, direct);
        if (scheduled_jobs.empty()) return;

        auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mu);
            run_main = &main_auth;
            run_block = &block_auth;
            run_storage = &storage;
            run_layout = &layout;
            run_direct = &direct;
            run_cross = &cross;
            run_mod = mod;
            next.store(0, std::memory_order_relaxed);
            pending = workers;
            ++generation;
        }
        start_cv.notify_all();
        {
            std::unique_lock<std::mutex> lock(mu);
            done_cv.wait(lock, [&] { return pending == 0; });
        }
        wall_s += ram_seconds_since(t0);
    }

    double kernel_s() const {
        double z = 0; for (const auto& x : stats) z += x.kernel_s; return z;
    }
    uint64_t groups() const {
        uint64_t z = 0; for (const auto& x : stats) z += x.groups; return z;
    }
};
