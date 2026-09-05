#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

// Reuse the tested factorized topology codec and local transition kernels, but
// replace its HBM authoritative arrays and main() with the RAM backend below.
#define main oneesan_factorized_hbm_unused_main
#include "../b300/oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "ramstream32_factorized_storage.hpp"

static WindowPlan make_forced_factor_window(bool high_window) {
    WindowPlan wp;
    if (high_window) {
        // Exact same split as the B300 forced2 profile:
        // process p = W-1 .. L+1, fixing all LOW positions [0,L).
        wp.p_hi = TARGET_W - 1;
        wp.p_lo = LOW_LUT_K + 1;
    } else {
        // process p = L .. 1, fixing all HIGH positions [L+1,W).
        wp.p_hi = LOW_LUT_K;
        wp.p_lo = 1;
    }
    wp.fixed_pos = window_candidates(TARGET_W, wp.p_hi, wp.p_lo);

    Code max_main = 0, max_block = 0;
    size_t max_bytes = 0;
    int groups = 1 << int(wp.fixed_pos.size());
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(TARGET_W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g),
                     mf, mo, bf, bo);
        auto ms = make_spec(TARGET_W, mf, mo);
        auto ds = make_spec(TARGET_W - 1, bf, bo);
        size_t bytes = size_t(2 * ms.size + 2 * ds.size) * sizeof(Count);
        if (bytes > max_bytes) {
            max_bytes = bytes;
            max_main = ms.size;
            max_block = ds.size;
        }
    }
    wp.max_bytes = max_bytes;
    wp.max_main = max_main;
    wp.max_block = max_block;
    return wp;
}

struct ForcedJob {
    int g = 0;
    Code work = 0;
};

static std::vector<ForcedJob> make_forced_jobs(const WindowPlan& wp) {
    int groups = 1 << int(wp.fixed_pos.size());
    std::vector<ForcedJob> jobs;
    jobs.reserve(groups);
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(TARGET_W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g),
                     mf, mo, bf, bo);
        auto ms = make_spec(TARGET_W, mf, mo);
        auto ds = make_spec(TARGET_W - 1, bf, bo);
        jobs.push_back({g, 2 * ms.size + 2 * ds.size});
    }
    std::sort(jobs.begin(), jobs.end(), [](const ForcedJob& a, const ForcedJob& b) {
        return a.work > b.work;
    });
    return jobs;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int target_mib = argc > 3 ? std::atoi(argv[3]) : 16384;
    int cpu_threads = argc > 4
        ? std::atoi(argv[4])
        : int(std::max(1u, std::thread::hardware_concurrency()));
    int W = n + 1;

    if (W != TARGET_W || n < 2 || W > MAXW) {
        std::cerr << "binary specialized for n=" << (TARGET_W - 1) << "\n";
        return 1;
    }
    if (target_mib <= 0 || cpu_threads <= 0) {
        std::cerr << "target_mib and cpu_threads must be positive\n";
        return 1;
    }
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) {
        std::cerr << "forced2 requires LOW_LUT_K + HIGH_LUT_K == TARGET_W - 1\n";
        return 1;
    }

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);

    WindowPlan high_wp = make_forced_factor_window(true);
    WindowPlan low_wp = make_forced_factor_window(false);
    size_t target = size_t(target_mib) << 20;
    std::cerr
        << "forced2 high_window_max_gib=" << double(high_wp.max_bytes) / double(1ULL << 30)
        << " low_window_max_gib=" << double(low_wp.max_bytes) / double(1ULL << 30)
        << " scratch_target_gib=" << double(target) / double(1ULL << 30) << "\n";
    if (high_wp.max_bytes > target || low_wp.max_bytes > target) {
        std::cerr << "forced two-window scratch does not fit. Increase scratch_target_mib "
                     "or use a future three-window profile.\n";
        return 4;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) {
        std::cerr << "need a CUDA GPU\n";
        return 2;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");

    StorageDeviceTables device_tables;
    device_tables.install(storage, G_FACTOR);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "mmap factorized main");
    block_auth.alloc(layout.block_size, "mmap factorized block");

    MateID init = MateID(R) << (2 * (W - 1));
    main_auth.ptr[storage_rank_main_host(init, storage, layout)] = 1;

    RamFactorCtx ctx;
    ctx.init(mod);

    std::vector<ForcedJob> high_jobs = make_forced_jobs(high_wp);
    std::vector<ForcedJob> low_jobs = make_forced_jobs(low_wp);
    int gpu_threads = 256;
    auto wall0 = std::chrono::steady_clock::now();

    for (int row = 0; row < W; ++row) {
        for (const ForcedJob& job : high_jobs) {
            if (!job.work) continue;
            process_group_ramfactor(ctx, main_auth, block_auth, storage, layout,
                                    W, high_wp, job.g, gpu_threads, cpu_threads);
        }
        for (const ForcedJob& job : low_jobs) {
            if (!job.work) continue;
            process_group_ramfactor(ctx, main_auth, block_auth, storage, layout,
                                    W, low_wp, job.g, gpu_threads, cpu_threads);
        }
        std::cerr << "row " << (row + 1) << "/" << W
                  << " groups=" << ctx.groups
                  << " memcpy_runs=" << ctx.memcpy_runs << "\n";
    }

    double wall_s = ram_seconds_since(wall0);
    Count answer = main_auth.ptr[storage_rank_main_host(MateID(R), storage, layout)];
    double auth_gib = double(layout.main_size + layout.block_size) * sizeof(Count) /
                      double(1ULL << 30);
    double avg_memcpy_elems = ctx.memcpy_runs
        ? double(ctx.memcpy_elems / ctx.memcpy_runs)
        : 0.0;

    std::cout
        << "backend=gridfp-ramstream32-factorized-forced2-v3"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " main_states=" << layout.main_size
        << " blocked_states=" << layout.block_size
        << " auth_gib=" << auth_gib
        << " scratch_target_mib=" << target_mib
        << " high_window_max_gib=" << double(high_wp.max_bytes) / double(1ULL << 30)
        << " low_window_max_gib=" << double(low_wp.max_bytes) / double(1ULL << 30)
        << " cpu_threads=" << cpu_threads
        << " low_lut_k=" << LOW_LUT_K
        << " high_lut_k=" << HIGH_LUT_K
        << " groups=" << ctx.groups
        << " memcpy_runs=" << ctx.memcpy_runs
        << " avg_memcpy_elems=" << avg_memcpy_elems
        << " pack_s=" << ctx.pack_s
        << " h2d_s=" << ctx.h2d_s
        << " kernel_s=" << ctx.kernel_s
        << " d2h_s=" << ctx.d2h_s
        << " unpack_s=" << ctx.unpack_s
        << " wall_s=" << wall_s
        << "\n";

    ctx.destroy();
    device_tables.release();
    main_auth.release();
    block_auth.release();
    return 0;
}
