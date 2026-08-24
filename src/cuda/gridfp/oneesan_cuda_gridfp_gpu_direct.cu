#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN

#include "ramstream32_cpu_low_inplace.hpp"
#include "ramstream32_cpu_high_direct.hpp"
#include "ramstream32_gpu_direct.cuh"

static bool gpu_direct_has_arg(int argc, char** argv, const char* needle) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], needle) == 0) return true;
    return false;
}

static double gpu_direct_seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    int grid_x = argc > 4 ? std::atoi(argv[4]) : 16;
    int grid_y = argc > 5 ? std::atoi(argv[5]) : 8;
    bool plan_only = gpu_direct_has_arg(argc, argv, "--plan-only");
    int W = n + 1;

    if (W != TARGET_W || n < 2 || W > MAXW) {
        std::cerr << "binary specialized for n=" << (TARGET_W - 1) << '\n';
        return 1;
    }
    if constexpr (LOW_LUT_K + HIGH_LUT_K + 1 != TARGET_W) {
        std::cerr << "GPU direct requires LOW+HIGH+1=W\n";
        return 1;
    }
    if (threads <= 0 || threads > 1024 || grid_x <= 0 || grid_y <= 0) {
        std::cerr << "invalid launch geometry\n";
        return 2;
    }

    auto prep0 = std::chrono::steady_clock::now();
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectCrossHost cross = build_gpu_direct_cross(storage);
    double prepare_s = gpu_direct_seconds_since(prep0);

    size_t authoritative_bytes = size_t(layout.main_size + layout.block_size) * sizeof(Count);
    size_t metadata_bytes = 0;
    metadata_bytes += (lowdesc.main_desc.size() + lowdesc.block_desc.size()) * sizeof(uint32_t);
    metadata_bytes += loworbit.rec.size() * sizeof(uint64_t);
    metadata_bytes += highdirect.orbit_ops.size() * sizeof(CpuHighOrbitOp);
    metadata_bytes += highdirect.closure_ops.size() * sizeof(CpuHighClosureOp);
    metadata_bytes += (highdirect.orbit_off.nn.size() + highdirect.orbit_off.nrnl.size()
        + highdirect.closure_off.block.size() + highdirect.closure_off.cross.size()) * sizeof(uint32_t);
    metadata_bytes += (cross.high_rank.size() + cross.low_rank.size()) * sizeof(uint32_t);
    size_t total_device_bytes = authoritative_bytes + metadata_bytes;

    if (plan_only) {
        std::cout
            << "backend=gridfp-gpu-direct-v0.1-plan"
            << " n=" << n
            << " main_states=" << layout.main_size
            << " blocked_states=" << layout.block_size
            << " authoritative_gib=" << double(authoritative_bytes) / double(1ULL << 30)
            << " metadata_mib=" << double(metadata_bytes) / double(1ULL << 20)
            << " total_device_gib=" << double(total_device_bytes) / double(1ULL << 30)
            << " low_launches_per_row=" << (2 * LOW_LUT_K)
            << " high_launches_per_row=" << (2 * HIGH_LUT_K)
            << " scratch_bytes=0"
            << " prepare_s=" << prepare_s
            << '\n';
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "gpu direct device count");
    if (visible < 1) {
        std::cerr << "no CUDA device\n";
        return 3;
    }
    ck(cudaSetDevice(0), "gpu direct set device");

    size_t free_bytes = 0, total_bytes = 0;
    ck(cudaMemGetInfo(&free_bytes, &total_bytes), "gpu direct mem info");
    if (total_device_bytes > free_bytes) {
        std::cerr << "insufficient HBM: need_gib="
                  << double(total_device_bytes) / double(1ULL << 30)
                  << " free_gib=" << double(free_bytes) / double(1ULL << 30)
                  << '\n';
        return 4;
    }

    Count* dmain = nullptr;
    Count* dblock = nullptr;
    ck(cudaMalloc(&dmain, size_t(layout.main_size) * sizeof(Count)), "gpu direct alloc main");
    ck(cudaMalloc(&dblock, size_t(layout.block_size) * sizeof(Count)), "gpu direct alloc block");
    ck(cudaMemset(dmain, 0, size_t(layout.main_size) * sizeof(Count)), "gpu direct zero main");
    ck(cudaMemset(dblock, 0, size_t(layout.block_size) * sizeof(Count)), "gpu direct zero block");
    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "gpu direct modulus");

    GpuDirectDeviceTables tables;
    tables.install(storage, layout, lowdesc, loworbit, highdirect, cross);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = storage_rank_main_host(init, storage, layout);
    Count one = 1;
    ck(cudaMemcpy(dmain + init_rank, &one, sizeof(one), cudaMemcpyHostToDevice),
       "gpu direct init state");

    double high_s = 0.0, low_s = 0.0;
    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        auto t = std::chrono::steady_clock::now();
        gpu_direct_run_high(dmain, dblock, layout, threads, grid_x, grid_y);
        high_s += gpu_direct_seconds_since(t);

        t = std::chrono::steady_clock::now();
        gpu_direct_run_low(dmain, dblock, layout, threads, grid_x, grid_y);
        low_s += gpu_direct_seconds_since(t);

        std::cerr << "row " << row + 1 << '/' << W
                  << " high_s=" << high_s << " low_s=" << low_s << '\n';
    }
    double wall_s = gpu_direct_seconds_since(wall0);

    Code final_rank = storage_rank_main_host(MateID(R), storage, layout);
    Count answer = 0;
    ck(cudaMemcpy(&answer, dmain + final_rank, sizeof(answer), cudaMemcpyDeviceToHost),
       "gpu direct answer");

    std::cout
        << "backend=gridfp-gpu-direct-v0.1"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " main_states=" << layout.main_size
        << " blocked_states=" << layout.block_size
        << " authoritative_gib=" << double(authoritative_bytes) / double(1ULL << 30)
        << " metadata_mib=" << double(metadata_bytes) / double(1ULL << 20)
        << " threads=" << threads
        << " grid_x=" << grid_x
        << " grid_y=" << grid_y
        << " high_s=" << high_s
        << " low_s=" << low_s
        << " prepare_s=" << prepare_s
        << " wall_s=" << wall_s
        << " scratch_bytes=0"
        << '\n';

    tables.release();
    cudaFree(dmain);
    cudaFree(dblock);
    return 0;
}
