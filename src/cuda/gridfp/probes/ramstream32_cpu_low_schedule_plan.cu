#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_sparse_persistent.hpp"

struct CpuLowBoundaryPages {
    std::unordered_set<uint64_t> shared_4k;
    std::unordered_set<uint64_t> cross_worker_4k;
    std::unordered_set<uint64_t> shared_2m;
    std::unordered_set<uint64_t> cross_worker_2m;
};

static void cpu_low_add_boundary_page(
    CpuLowBoundaryPages& out, uint64_t boundary_byte,
    int left_owner, int right_owner
) {
    constexpr uint64_t PAGE4K = 4096ull;
    constexpr uint64_t PAGE2M = 2ull << 20;
    if (boundary_byte % PAGE4K) {
        uint64_t page = boundary_byte / PAGE4K;
        out.shared_4k.insert(page);
        if (left_owner != right_owner) out.cross_worker_4k.insert(page);
    }
    if (boundary_byte % PAGE2M) {
        uint64_t page = boundary_byte / PAGE2M;
        out.shared_2m.insert(page);
        if (left_owner != right_owner) out.cross_worker_2m.insert(page);
    }
}

static CpuLowBoundaryPages cpu_low_boundary_pages(
    const std::vector<StorageBlock>& blocks,
    const StorageFactorHost& storage,
    const std::vector<int>& owner
) {
    constexpr int S = FactorTablesHost::STRIDE;
    const uint32_t nmasks = 1u << HIGH_LUT_K;
    CpuLowBoundaryPages out;

    for (const StorageBlock& sb : blocks) {
        if (!sb.valid || !sb.rows || !sb.cols) continue;
        bool have_prev = false;
        uint64_t prev_end_byte = 0;
        int prev_owner = -1;

        for (uint32_t mask = 0; mask < nmasks; ++mask) {
            size_t ix = size_t(mask) * S + sb.he;
            uint32_t rows = G_FACTOR.high_mask_off[ix + 1] - G_FACTOR.high_mask_off[ix];
            if (!rows) continue;
            if (owner[mask] < 0) {
                std::cerr << "cpu LOW page plan missing owner mask=" << mask
                          << " h=" << unsigned(sb.he) << '\n';
                std::exit(3);
            }

            uint32_t row0 = storage.high_mask_begin[size_t(mask) * StorageFactorHost::S + sb.he];
            uint64_t begin_elem = uint64_t(sb.off) + uint64_t(row0) * sb.cols;
            uint64_t begin_byte = begin_elem * sizeof(Count);
            uint64_t end_byte = begin_byte + uint64_t(rows) * sb.cols * sizeof(Count);

            if (have_prev) {
                if (begin_byte != prev_end_byte) {
                    std::cerr << "cpu LOW storage mask ranges not contiguous h="
                              << unsigned(sb.he) << " begin=" << begin_byte
                              << " previous_end=" << prev_end_byte << '\n';
                    std::exit(4);
                }
                cpu_low_add_boundary_page(
                    out, begin_byte, prev_owner, owner[mask]);
            }
            have_prev = true;
            prev_end_byte = end_byte;
            prev_owner = owner[mask];
        }
    }
    return out;
}

static uint64_t cpu_low_page_count(uint64_t bytes, uint64_t page) {
    return bytes ? (bytes + page - 1) / page : 0;
}

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

    std::vector<int> owner(size_t(1) << HIGH_LUT_K, -1);
    for (int w = 0; w < workers; ++w) {
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            uint32_t mask = jobs[q].mask;
            if (owner[mask] >= 0) {
                std::cerr << "cpu LOW sticky duplicate mask owner mask=" << mask << '\n';
                return 5;
            }
            owner[mask] = w;
        }
    }

    CpuLowBoundaryPages main_pages = cpu_low_boundary_pages(
        layout.main_blocks, storage, owner);
    CpuLowBoundaryPages block_pages = cpu_low_boundary_pages(
        layout.block_blocks, storage, owner);

    uint64_t shared_4k = main_pages.shared_4k.size() + block_pages.shared_4k.size();
    uint64_t cross_4k = main_pages.cross_worker_4k.size() + block_pages.cross_worker_4k.size();
    uint64_t shared_2m = main_pages.shared_2m.size() + block_pages.shared_2m.size();
    uint64_t cross_2m = main_pages.cross_worker_2m.size() + block_pages.cross_worker_2m.size();
    uint64_t main_bytes = uint64_t(layout.main_size) * sizeof(Count);
    uint64_t block_bytes = uint64_t(layout.block_size) * sizeof(Count);
    uint64_t total_pages_4k = cpu_low_page_count(main_bytes, 4096)
        + cpu_low_page_count(block_bytes, 4096);
    uint64_t total_pages_2m = cpu_low_page_count(main_bytes, 2ull << 20)
        + cpu_low_page_count(block_bytes, 2ull << 20);

    double cross_of_shared_4k = shared_4k ? double(cross_4k) / shared_4k : 0.0;
    double cross_of_shared_2m = shared_2m ? double(cross_2m) / shared_2m : 0.0;
    double cross_auth_4k = total_pages_4k ? double(cross_4k) / total_pages_4k : 0.0;
    double cross_auth_2m = total_pages_2m ? double(cross_2m) / total_pages_2m : 0.0;

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
              << " shared_pages_4k=" << shared_4k
              << " cross_worker_pages_4k=" << cross_4k
              << " cross_worker_of_shared_4k=" << cross_of_shared_4k
              << " cross_worker_auth_page_fraction_4k=" << cross_auth_4k
              << " shared_pages_2m=" << shared_2m
              << " cross_worker_pages_2m=" << cross_2m
              << " cross_worker_of_shared_2m=" << cross_of_shared_2m
              << " cross_worker_auth_page_fraction_2m=" << cross_auth_2m
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
