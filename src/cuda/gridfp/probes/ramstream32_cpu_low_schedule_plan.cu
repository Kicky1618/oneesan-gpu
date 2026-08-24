#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_sparse_persistent.hpp"

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    int workers = argc > 2 ? std::max(1, std::atoi(argv[2])) : 32;
    bool dump_workers = argc > 3 && std::strcmp(argv[3], "--workers") == 0;
    int W = n + 1;
    if (W != TARGET_W || n < 2 || W > MAXW || workers <= 0) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);

    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);
    CpuLowSparsePersistentPool pool(workers, CPU_LOW_SCHEDULE_STICKY);
    pool.prepare_sticky_schedule(jobs, sparse);

    size_t nonempty_jobs = 0;
    uint64_t exact_total = 0;
    for (const auto& job : jobs) {
        if (!job.main_size && !job.block_size) continue;
        ++nonempty_jobs;
        exact_total += cpu_low_sparse_job_cells(job, sparse);
    }

    uint64_t assigned_total = std::accumulate(
        pool.sticky_worker_cells.begin(), pool.sticky_worker_cells.end(), uint64_t(0));
    size_t assigned_jobs = 0;
    for (const auto& v : pool.sticky_worker_jobs) assigned_jobs += v.size();
    if (assigned_total != exact_total || assigned_jobs != nonempty_jobs) {
        std::cerr << "cpu LOW sticky plan accounting mismatch jobs="
                  << assigned_jobs << '/' << nonempty_jobs
                  << " cells=" << assigned_total << '/' << exact_total << '\n';
        return 2;
    }

    uint64_t min_cells = pool.sticky_worker_cells.empty() ? 0 : pool.sticky_worker_cells[0];
    uint64_t max_cells = 0;
    for (uint64_t x : pool.sticky_worker_cells) {
        min_cells = std::min(min_cells, x);
        max_cells = std::max(max_cells, x);
    }
    double avg = workers ? double(exact_total) / workers : 0.0;
    double imbalance = avg > 0.0 ? double(max_cells) / avg : 0.0;

    std::cout << std::setprecision(12)
              << "cpu_low_schedule_plan OK"
              << " n=" << n
              << " workers=" << workers
              << " jobs=" << nonempty_jobs
              << " total_cells=" << exact_total
              << " min_worker_cells=" << min_cells
              << " max_worker_cells=" << max_cells
              << " avg_worker_cells=" << avg
              << " imbalance=" << imbalance
              << " build_s=" << pool.schedule_build_s
              << '\n';

    if (dump_workers) {
        std::cout << "worker\tjobs\tcells\tfraction\n";
        for (int w = 0; w < workers; ++w) {
            uint64_t cells = pool.sticky_worker_cells[size_t(w)];
            double fraction = exact_total ? double(cells) / double(exact_total) : 0.0;
            std::cout << w << '\t'
                      << pool.sticky_worker_jobs[size_t(w)].size() << '\t'
                      << cells << '\t' << fraction << '\n';
        }
    }
    return 0;
}
