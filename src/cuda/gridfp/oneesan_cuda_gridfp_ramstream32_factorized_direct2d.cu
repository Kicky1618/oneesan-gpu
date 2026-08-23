#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "../b300/oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "ramstream32_factorized_storage.hpp"

struct Direct2DCtx {
    uint8_t* arena = nullptr;
    size_t cap_arena = 0;
    Count *dA = nullptr, *dB = nullptr, *dD = nullptr, *dE = nullptr;
    double h2d_s = 0, kernel_s = 0, d2h_s = 0;
    uint64_t groups = 0;
    uint64_t copy1d = 0, copy2d = 0;
    long double copy_elems = 0;

    void init(Count mod) {
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "copy modulus");
    }

    void ensure(Code m, Code d) {
        auto align256 = [](size_t x) { return (x + 255) & ~size_t(255); };
        size_t mb = align256(size_t(m) * sizeof(Count));
        size_t db = align256(size_t(d) * sizeof(Count));
        size_t need = 2 * mb + 2 * db;
        if (need > cap_arena) {
            if (arena) cudaFree(arena);
            cap_arena = need;
            ck(cudaMalloc(&arena, cap_arena), "direct2d scratch");
        }
        size_t off = 0;
        dA = reinterpret_cast<Count*>(arena + off); off += mb;
        dB = reinterpret_cast<Count*>(arena + off); off += mb;
        dD = reinterpret_cast<Count*>(arena + off); off += db;
        dE = reinterpret_cast<Count*>(arena + off);
    }

    void destroy() {
        if (arena) cudaFree(arena);
        arena = nullptr;
        cap_arena = 0;
    }
};

static void auth_to_device_rect(
    Direct2DCtx& c, const RamCounts& auth, Count* dst,
    const StorageBlock& sb, const FBlock& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& storage
) {
    if (fb.end == fb.off) return;
    constexpr int S = StorageFactorHost::S;

    if (!fix_low) {
        // HIGH occupancy is fixed.  Its rows form one contiguous run and all LOW
        // columns are present, so the factor block is a single linear DMA.
        uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + fb.he];
        Code elems = fb.end - fb.off;
        ck(cudaMemcpy(dst + fb.off,
                      auth.ptr + sb.off + Code(row0) * sb.cols,
                      size_t(elems) * sizeof(Count),
                      cudaMemcpyHostToDevice),
           "direct2d H2D contiguous");
        ++c.copy1d;
        c.copy_elems += elems;
        return;
    }

    // LOW occupancy is fixed.  Every HIGH row contributes the same-width LOW
    // slice.  cudaMemcpy2D expresses the whole strided gather in one copy call;
    // CUDA may internally stage pageable host memory, but we no longer allocate
    // a multi-GiB pinned group buffer or issue one CPU memcpy per row.
    uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + fb.hs];
    size_t width_bytes = size_t(fb.stride) * sizeof(Count);
    Code rows = fb.stride ? (fb.end - fb.off) / fb.stride : 0;
    if (!rows || !width_bytes) return;
    ck(cudaMemcpy2D(dst + fb.off, width_bytes,
                    auth.ptr + sb.off + col0, size_t(sb.cols) * sizeof(Count),
                    width_bytes, size_t(rows), cudaMemcpyHostToDevice),
       "direct2d H2D strided");
    ++c.copy2d;
    c.copy_elems += Code(rows) * fb.stride;
}

static void device_to_auth_rect(
    Direct2DCtx& c, RamCounts& auth, const Count* src,
    const StorageBlock& sb, const FBlock& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& storage
) {
    if (fb.end == fb.off) return;
    constexpr int S = StorageFactorHost::S;

    if (!fix_low) {
        uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + fb.he];
        Code elems = fb.end - fb.off;
        ck(cudaMemcpy(auth.ptr + sb.off + Code(row0) * sb.cols,
                      src + fb.off,
                      size_t(elems) * sizeof(Count),
                      cudaMemcpyDeviceToHost),
           "direct2d D2H contiguous");
        ++c.copy1d;
        c.copy_elems += elems;
        return;
    }

    uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + fb.hs];
    size_t width_bytes = size_t(fb.stride) * sizeof(Count);
    Code rows = fb.stride ? (fb.end - fb.off) / fb.stride : 0;
    if (!rows || !width_bytes) return;
    ck(cudaMemcpy2D(auth.ptr + sb.off + col0, size_t(sb.cols) * sizeof(Count),
                    src + fb.off, width_bytes,
                    width_bytes, size_t(rows), cudaMemcpyDeviceToHost),
       "direct2d D2H strided");
    ++c.copy2d;
    c.copy_elems += Code(rows) * fb.stride;
}

static void auth_to_device_main(
    Direct2DCtx& c, const RamCounts& auth, Count* dst,
    const std::vector<FBlock>& blocks, bool fix_low, uint32_t mask,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    for (size_t i = 0; i < blocks.size(); ++i)
        auth_to_device_rect(c, auth, dst, layout.main_blocks[i], blocks[i],
                            fix_low, mask, storage);
}

static void auth_to_device_block(
    Direct2DCtx& c, const RamCounts& auth, Count* dst,
    const std::vector<FBlock>& blocks, bool fix_low, uint32_t mask,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    for (size_t i = 0; i < blocks.size(); ++i)
        auth_to_device_rect(c, auth, dst, layout.block_blocks[i], blocks[i],
                            fix_low, mask, storage);
}

static void device_to_auth_main(
    Direct2DCtx& c, RamCounts& auth, const Count* src,
    const std::vector<FBlock>& blocks, bool fix_low, uint32_t mask,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    for (size_t i = 0; i < blocks.size(); ++i)
        device_to_auth_rect(c, auth, src, layout.main_blocks[i], blocks[i],
                            fix_low, mask, storage);
}

static void device_to_auth_block(
    Direct2DCtx& c, RamCounts& auth, const Count* src,
    const std::vector<FBlock>& blocks, bool fix_low, uint32_t mask,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    for (size_t i = 0; i < blocks.size(); ++i)
        device_to_auth_rect(c, auth, src, layout.block_blocks[i], blocks[i],
                            fix_low, mask, storage);
}

static void configure_factor_group(
    int W, const WindowPlan& wp, int g,
    GroupSpec& ms, GroupSpec& ds, bool& fix_low, uint32_t& mask,
    std::vector<FBlock>& fmb, std::vector<FBlock>& fdb
) {
    uint32_t mf, mo, bf, bo;
    window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
    ms = make_spec(W, mf, mo);
    ds = make_spec(W - 1, bf, bo);

    fix_low = wp.p_hi > LOW_LUT_K;
    mask = fix_low
        ? (mo & ((1u << LOW_LUT_K) - 1u))
        : ((mo >> (LOW_LUT_K + 1)) & ((1u << HIGH_LUT_K) - 1u));
    fmb = make_factor_main_blocks(fix_low, mask);
    fdb = make_factor_block_blocks(fix_low, mask);

    if (fmb.empty() || fdb.empty() ||
        fmb.back().end != ms.size || fdb.back().end != ds.size) {
        std::cerr << "direct2d group size mismatch main="
                  << (fmb.empty() ? 0 : fmb.back().end) << "/" << ms.size
                  << " block=" << (fdb.empty() ? 0 : fdb.back().end) << "/" << ds.size
                  << " fix_low=" << fix_low << " mask=" << mask << "\n";
        std::exit(40);
    }

    int fm = int(fmb.size()), fd = int(fdb.size()), fl = fix_low ? 1 : 0;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, fmb.data(), fmb.size() * sizeof(FBlock)),
       "direct2d main blocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, fdb.data(), fdb.size() * sizeof(FBlock)),
       "direct2d block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &fm, sizeof(fm)), "direct2d main nblocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &fd, sizeof(fd)), "direct2d block nblocks");
    ck(cudaMemcpyToSymbol(D_F_MASK, &mask, sizeof(mask)), "direct2d mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fl, sizeof(fl)), "direct2d mode");
    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "direct2d main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "direct2d block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &mf, sizeof(mf)), "direct2d main fixed");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &mo, sizeof(mo)), "direct2d main occ");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &bf, sizeof(bf)), "direct2d block fixed");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &bo, sizeof(bo)), "direct2d block occ");
}

static void process_group_direct2d(
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
                                   cudaMemcpyDeviceToDevice), "direct2d identity");
        if (ds.size) ck(cudaMemset(dnext, 0, size_t(ds.size) * sizeof(Count)),
                        "direct2d clear block");
        if (ms.size) main_group_kernel<<<bm, gpu_threads>>>(cur, nullptr, ms.size, nxt, dnext, p);
        if (ds.size) blocked_group_kernel<<<bd, gpu_threads>>>(dcur, ds.size, nxt, p);
        ck(cudaGetLastError(), "direct2d transition");
        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }
    ck(cudaDeviceSynchronize(), "direct2d transition sync");
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) device_to_auth_main(c, main_auth, cur, fmb, fix_low, mask, storage, layout);
    if (ds.size) device_to_auth_block(c, block_auth, dcur, fdb, fix_low, mask, storage, layout);
    c.d2h_s += ram_seconds_since(t);
    ++c.groups;
}

static WindowPlan make_direct2d_window(bool high_window) {
    WindowPlan wp;
    wp.p_hi = high_window ? TARGET_W - 1 : LOW_LUT_K;
    wp.p_lo = high_window ? LOW_LUT_K + 1 : 1;
    wp.fixed_pos = window_candidates(TARGET_W, wp.p_hi, wp.p_lo);

    size_t mx = 0;
    Code mm = 0, md = 0;
    int groups = 1 << int(wp.fixed_pos.size());
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(TARGET_W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g),
                     mf, mo, bf, bo);
        auto ms = make_spec(TARGET_W, mf, mo);
        auto ds = make_spec(TARGET_W - 1, bf, bo);
        size_t bytes = size_t(2 * ms.size + 2 * ds.size) * sizeof(Count);
        if (bytes > mx) { mx = bytes; mm = ms.size; md = ds.size; }
    }
    wp.max_bytes = mx;
    wp.max_main = mm;
    wp.max_block = md;
    return wp;
}

struct DirectJob { int g; Code work; };

static std::vector<DirectJob> make_direct2d_jobs(const WindowPlan& wp) {
    int groups = 1 << int(wp.fixed_pos.size());
    std::vector<DirectJob> jobs;
    jobs.reserve(groups);
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(TARGET_W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g),
                     mf, mo, bf, bo);
        auto ms = make_spec(TARGET_W, mf, mo);
        auto ds = make_spec(TARGET_W - 1, bf, bo);
        jobs.push_back({g, 2 * ms.size + 2 * ds.size});
    }
    std::sort(jobs.begin(), jobs.end(), [](const DirectJob& a, const DirectJob& b) {
        return a.work > b.work;
    });
    return jobs;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int target_mib = argc > 3 ? std::atoi(argv[3]) : 16384;
    int W = n + 1;

    if (W != TARGET_W || n < 2 || W > MAXW) {
        std::cerr << "binary specialized for n=" << (TARGET_W - 1) << "\n";
        return 1;
    }
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) {
        std::cerr << "forced2 requires LOW+HIGH=W-1\n";
        return 1;
    }

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);

    WindowPlan high_wp = make_direct2d_window(true);
    WindowPlan low_wp = make_direct2d_window(false);
    size_t target = size_t(target_mib) << 20;
    if (high_wp.max_bytes > target || low_wp.max_bytes > target) {
        std::cerr << "direct2d forced window does not fit: high_gib="
                  << double(high_wp.max_bytes) / double(1ULL << 30)
                  << " low_gib=" << double(low_wp.max_bytes) / double(1ULL << 30)
                  << " target_gib=" << double(target) / double(1ULL << 30) << "\n";
        return 4;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) return 2;
    ck(cudaSetDevice(0), "cudaSetDevice");

    StorageDeviceTables device_tables;
    device_tables.install(storage, G_FACTOR);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "mmap direct2d main");
    block_auth.alloc(layout.block_size, "mmap direct2d block");
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
            if (job.work) process_group_direct2d(ctx, main_auth, block_auth, storage, layout,
                                                 W, high_wp, job.g, gpu_threads);
        for (const auto& job : low_jobs)
            if (job.work) process_group_direct2d(ctx, main_auth, block_auth, storage, layout,
                                                 W, low_wp, job.g, gpu_threads);
        std::cerr << "row " << row + 1 << "/" << W
                  << " groups=" << ctx.groups
                  << " copy1d=" << ctx.copy1d
                  << " copy2d=" << ctx.copy2d << "\n";
    }

    double wall_s = ram_seconds_since(wall0);
    Count answer = main_auth.ptr[storage_rank_main_host(MateID(R), storage, layout)];
    double auth_gib = double(layout.main_size + layout.block_size) * sizeof(Count) /
                      double(1ULL << 30);
    double avg_copy_elems = (ctx.copy1d + ctx.copy2d)
        ? double(ctx.copy_elems / (ctx.copy1d + ctx.copy2d)) : 0.0;

    std::cout
        << "backend=gridfp-ramstream32-factorized-direct2d-v3.1"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " auth_gib=" << auth_gib
        << " scratch_target_mib=" << target_mib
        << " high_window_max_gib=" << double(high_wp.max_bytes) / double(1ULL << 30)
        << " low_window_max_gib=" << double(low_wp.max_bytes) / double(1ULL << 30)
        << " groups=" << ctx.groups
        << " copy1d=" << ctx.copy1d
        << " copy2d=" << ctx.copy2d
        << " avg_copy_elems=" << avg_copy_elems
        << " h2d_s=" << ctx.h2d_s
        << " kernel_s=" << ctx.kernel_s
        << " d2h_s=" << ctx.d2h_s
        << " wall_s=" << wall_s
        << "\n";

    ctx.destroy();
    device_tables.release();
    main_auth.release();
    block_auth.release();
    return 0;
}
