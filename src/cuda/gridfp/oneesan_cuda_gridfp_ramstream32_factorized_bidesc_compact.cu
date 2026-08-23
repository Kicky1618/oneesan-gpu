#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc.cu"
#undef RAMSTREAM_BIDESC_NO_MAIN

#include "ramstream32_bidesc_compact.cuh"

static void process_group_bidesc_compact(
    Direct2DCtx& c, RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    int W, const WindowPlan& wp, int g, int gpu_threads
) {
    GroupSpec ms, ds;
    bool fix_low = false;
    uint32_t mask = 0;
    std::vector<FBlock> fmb, fdb;
    configure_factor_group(W, wp, g, ms, ds, fix_low, mask, fmb, fdb);
    if (!ms.size && !ds.size) return;

    c.ensure(ms.size, ds.size);
    auto t = std::chrono::steady_clock::now();
    if (ms.size) auth_to_device_main(c, main_auth, c.dA, fmb, fix_low, mask, storage, layout);
    if (ds.size) auth_to_device_block(c, block_auth, c.dD, fdb, fix_low, mask, storage, layout);
    c.h2d_s += ram_seconds_since(t);

    int bm = int(std::min<Code>(65535, (ms.size + gpu_threads - 1) / gpu_threads));
    int bd = int(std::min<Code>(65535, (ds.size + gpu_threads - 1) / gpu_threads));

    t = std::chrono::steady_clock::now();
    Count* cur = c.dA;
    Count* nxt = c.dB;
    Count* dcur = c.dD;
    Count* dnext = c.dE;
    for (int p = wp.p_hi; p >= wp.p_lo; --p) {
        if (ms.size) ck(cudaMemcpy(nxt, cur, size_t(ms.size) * sizeof(Count),
                                   cudaMemcpyDeviceToDevice), "compact identity");
        if (ds.size) ck(cudaMemset(dnext, 0, size_t(ds.size) * sizeof(Count)),
                        "compact clear block");
        if (fix_low) {
            if (ms.size) main_group_highdesc_compact_kernel<<<bm, gpu_threads>>>(
                cur, ms.size, nxt, dnext, p);
            if (ds.size) blocked_group_highdesc_kernel<<<bd, gpu_threads>>>(
                dcur, ds.size, nxt, p);
        } else {
            if (ms.size) main_group_lowdesc_compact_kernel<<<bm, gpu_threads>>>(
                cur, ms.size, nxt, dnext, p);
            if (ds.size) blocked_group_lowdesc_kernel<<<bd, gpu_threads>>>(
                dcur, ds.size, nxt, p);
        }
        ck(cudaGetLastError(), "compact transition");
        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }
    ck(cudaDeviceSynchronize(), "compact transition sync");
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) device_to_auth_main(c, main_auth, cur, fmb, fix_low, mask, storage, layout);
    if (ds.size) device_to_auth_block(c, block_auth, dcur, fdb, fix_low, mask, storage, layout);
    c.d2h_s += ram_seconds_since(t);
    ++c.groups;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int target_mib = argc > 3 ? std::atoi(argv[3]) : 16384;
    bool plan_only = argc > 4 && std::strcmp(argv[4], "--plan-only") == 0;
    int W = n + 1;
    if (W != TARGET_W || n < 2 || W > MAXW) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    auto desc0 = std::chrono::steady_clock::now();
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    double desc_build_s = ram_seconds_since(desc0);

    WindowPlan high_wp = make_direct2d_window(true);
    WindowPlan low_wp = make_direct2d_window(false);
    size_t target = size_t(target_mib) << 20;
    if (high_wp.max_bytes > target || low_wp.max_bytes > target) return 4;

    double descriptor_mib = double((lowdesc.main_desc.size() + lowdesc.block_desc.size()
        + highdesc.main_desc.size() + highdesc.block_desc.size()) * sizeof(uint32_t)) / (1 << 20);
    double mask_mib = double((G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
        + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t)) / (1 << 20);
    double dense_rank_saved_mib = double((storage.low_packed_rank.size()
        + storage.high_packed_rank.size()) * sizeof(uint32_t)) / (1 << 20);

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-bidesc-compact-v3.5-plan"
            << " n=" << n
            << " descriptor_mib=" << descriptor_mib
            << " mask_tables_mib=" << mask_mib
            << " dense_rank_saved_mib=" << dense_rank_saved_mib
            << " desc_build_s=" << desc_build_s
            << " high_window_max_gib=" << double(high_wp.max_bytes) / double(1ULL << 30)
            << " low_window_max_gib=" << double(low_wp.max_bytes) / double(1ULL << 30)
            << "\n";
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) return 2;
    ck(cudaSetDevice(0), "cudaSetDevice");

    BidescMaskDeviceTables mask_tables;
    mask_tables.install(G_FACTOR);
    LowDescDeviceTables lowdesc_tables;
    lowdesc_tables.install(lowdesc);
    HighDescDeviceTables highdesc_tables;
    highdesc_tables.install(highdesc);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "mmap compact main");
    block_auth.alloc(layout.block_size, "mmap compact block");
    MateID init = MateID(R) << (2 * (W - 1));
    main_auth.ptr[storage_rank_main_host(init, storage, layout)] = 1;

    Direct2DCtx ctx;
    ctx.init(mod);
    auto high_jobs = make_direct2d_jobs(high_wp);
    auto low_jobs = make_direct2d_jobs(low_wp);
    int gpu_threads = 256;
    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        for (const auto& job : high_jobs)
            if (job.work) process_group_bidesc_compact(ctx, main_auth, block_auth, storage, layout,
                                                       W, high_wp, job.g, gpu_threads);
        for (const auto& job : low_jobs)
            if (job.work) process_group_bidesc_compact(ctx, main_auth, block_auth, storage, layout,
                                                       W, low_wp, job.g, gpu_threads);
        std::cerr << "row " << row + 1 << "/" << W << " groups=" << ctx.groups << "\n";
    }

    double wall_s = ram_seconds_since(wall0);
    Count answer = main_auth.ptr[storage_rank_main_host(MateID(R), storage, layout)];
    std::cout
        << "backend=gridfp-ramstream32-factorized-bidesc-compact-v3.5"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " descriptor_mib=" << descriptor_mib
        << " mask_tables_mib=" << mask_mib
        << " dense_rank_saved_mib=" << dense_rank_saved_mib
        << " groups=" << ctx.groups
        << " h2d_s=" << ctx.h2d_s
        << " kernel_s=" << ctx.kernel_s
        << " d2h_s=" << ctx.d2h_s
        << " wall_s=" << wall_s
        << "\n";

    ctx.destroy();
    highdesc_tables.release();
    lowdesc_tables.release();
    mask_tables.release();
    main_auth.release();
    block_auth.release();
    return 0;
}
