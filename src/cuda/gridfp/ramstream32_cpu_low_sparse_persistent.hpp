#pragma once

#include "ramstream32_cpu_low_sparse.hpp"

#include <condition_variable>
#include <mutex>

// Persistent worker wrapper for the sparse zero-scratch CPU LOW executor.
// The LOW recurrence and per-group body remain in ramstream32_cpu_low_sparse.hpp;
// this class only removes per-row std::thread construction/destruction. Jobs are
// fixed for the lifetime of a production run, so workers sleep between rows and
// consume the same atomic dynamic queue on every generation.
struct CpuLowSparsePersistentPool {
    int workers = 1;
    std::vector<CpuLowSparseStats> stats;
    double wall_s = 0.0;
    double worker_start_s = 0.0;

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

    explicit CpuLowSparsePersistentPool(int n)
        : workers(std::max(1, n)), stats(size_t(std::max(1, n))) {}

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
                  << " start_s=" << worker_start_s << '\n';
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

    void worker_loop(int w) {
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
                if (q >= run_jobs->size()) break;
                const CpuLowJob& job = (*run_jobs)[q];
                if (!job.main_size && !job.block_size) continue;
                process_cpu_low_group_sparse(
                    stats[size_t(w)], job, *run_main, *run_block,
                    *run_storage, *run_layout, *run_sparse, run_mod);
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
