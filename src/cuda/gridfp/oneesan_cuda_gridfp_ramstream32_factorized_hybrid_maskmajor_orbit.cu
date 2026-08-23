#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "ramstream32_cpu_low_maskmajor.hpp"
#include "ramstream32_high_orbit.cuh"

static void maskmajor_orbit_release_dense_host(StorageFactorHost& storage) {
    storage.low_packed_rank.clear(); storage.low_packed_rank.shrink_to_fit();
    storage.high_packed_rank.clear(); storage.high_packed_rank.shrink_to_fit();
    storage.low_all_codes.clear(); storage.low_all_codes.shrink_to_fit();
    storage.high_all_codes.clear(); storage.high_all_codes.shrink_to_fit();
    storage.low_mask_begin.clear(); storage.low_mask_begin.shrink_to_fit();
    G_FACTOR.low_packed_rank.clear(); G_FACTOR.low_packed_rank.shrink_to_fit();
    G_FACTOR.high_packed_rank.clear(); G_FACTOR.high_packed_rank.shrink_to_fit();
    G_FACTOR.low_all_codes.clear(); G_FACTOR.low_all_codes.shrink_to_fit();
    G_FACTOR.high_all_codes.clear(); G_FACTOR.high_all_codes.shrink_to_fit();
    G_FACTOR.high_main_base.clear(); G_FACTOR.high_main_base.shrink_to_fit();
    G_FACTOR.high_block_base.clear(); G_FACTOR.high_block_base.shrink_to_fit();
    G_FACTOR.low_mask_codes.clear(); G_FACTOR.low_mask_codes.shrink_to_fit();
}

struct OrbitMaskBatch {
    uint32_t first = 0, last = 0; // [first,last)
    Code main_begin = 0, main_size = 0;
    Code block_begin = 0, block_size = 0;
    size_t arena_bytes = 0;
};

static std::vector<int> make_orbit_g_for_low_mask(const WindowPlan& wp) {
    const uint32_t NM = 1u << LOW_LUT_K;
    std::vector<int> out(NM, -1);
    int groups = 1 << int(wp.fixed_pos.size());
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(TARGET_W, wp.p_hi, wp.p_lo, wp.fixed_pos,
                     uint32_t(g), mf, mo, bf, bo);
        uint32_t mask = mo & (NM - 1u);
        if (out[mask] != -1) {
            std::cerr << "duplicate LOW mask in HIGH schedule mask=" << mask << '\n';
            std::exit(170);
        }
        out[mask] = g;
    }
    for (uint32_t m = 0; m < NM; ++m)
        if (out[m] < 0) {
            std::cerr << "missing LOW mask in HIGH schedule mask=" << m << '\n';
            std::exit(171);
        }
    return out;
}

struct OrbitMaskBatchPlan {
    std::vector<OrbitMaskBatch> batches;
    Code raw_peak = 0;
    size_t arena_peak_bytes = 0;
};

static OrbitMaskBatchPlan make_orbit_mask_batches(
    const LowMaskMajorLayout& mm, size_t target_bytes
) {
    OrbitMaskBatchPlan plan;
    uint32_t a = 0;
    while (a < mm.masks) {
        uint32_t b = a;
        Code raw = 0;
        while (b < mm.masks) {
            Code gm = mm.main_mask_off[b + 1] - mm.main_mask_off[b];
            Code gd = mm.block_mask_off[b + 1] - mm.block_mask_off[b];
            Code nr = raw + gm + gd;
            size_t need = size_t(nr) * sizeof(Count);
            if (need > target_bytes && b > a) break;
            if (need > target_bytes) {
                std::cerr << "single LOW mask exceeds v5.3 raw target mask=" << b
                          << " need_gib=" << double(need) / double(1ULL << 30)
                          << " target_gib=" << double(target_bytes) / double(1ULL << 30)
                          << '\n';
                std::exit(172);
            }
            raw = nr;
            ++b;
        }
        OrbitMaskBatch z;
        z.first = a; z.last = b;
        z.main_begin = mm.main_mask_off[a];
        z.main_size = mm.main_mask_off[b] - mm.main_mask_off[a];
        z.block_begin = mm.block_mask_off[a];
        z.block_size = mm.block_mask_off[b] - mm.block_mask_off[a];
        z.arena_bytes = size_t(z.main_size + z.block_size) * sizeof(Count);
        if (z.arena_bytes > target_bytes) std::exit(173);
        plan.raw_peak = std::max(plan.raw_peak, z.main_size + z.block_size);
        plan.arena_peak_bytes = std::max(plan.arena_peak_bytes, z.arena_bytes);
        plan.batches.push_back(z);
        a = b;
    }
    return plan;
}

struct OrbitMaskBatchCtx {
    Count* raw = nullptr;
    Code raw_cap = 0;
    double h2d_s = 0, kernel_s = 0, d2h_s = 0;
    uint64_t batches = 0, groups = 0, pcie_calls = 0;

    void init(const OrbitMaskBatchPlan& plan, Count mod) {
        raw_cap = plan.raw_peak;
        if (raw_cap)
            ck(cudaMalloc(&raw, size_t(raw_cap) * sizeof(Count)), "v5.3 raw arena");
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "v5.3 modulus");
    }
    void release() {
        if (raw) cudaFree(raw);
        raw = nullptr;
        raw_cap = 0;
    }
};

static void process_orbit_mask_batch(
    OrbitMaskBatchCtx& c, const OrbitMaskBatch& batch,
    const std::vector<int>& g_for_mask,
    RamCounts& main_auth, RamCounts& block_auth,
    const LowMaskMajorLayout& mm, int W, const WindowPlan& wp, int gpu_threads
) {
    Count* raw_m = c.raw;
    Count* raw_d = c.raw + batch.main_size;

    auto t = std::chrono::steady_clock::now();
    if (batch.main_size) {
        ck(cudaMemcpy(raw_m, main_auth.ptr + batch.main_begin,
                      size_t(batch.main_size) * sizeof(Count), cudaMemcpyHostToDevice),
           "v5.3 H2D main");
        ++c.pcie_calls;
    }
    // The p=1 LOW transition at the end of the previous row consumes every
    // blocked state.  Start HIGH with an HBM memset instead of blocked H2D.
    if (batch.block_size)
        ck(cudaMemset(raw_d, 0, size_t(batch.block_size) * sizeof(Count)),
           "v5.3 zero blocked");
    c.h2d_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    for (uint32_t mask = batch.first; mask < batch.last; ++mask) {
        int g = g_for_mask[mask];
        GroupSpec ms, ds;
        bool fix_low = false;
        uint32_t got_mask = 0;
        std::vector<FBlock> fmb, fdb;
        configure_factor_group(W, wp, g, ms, ds, fix_low, got_mask, fmb, fdb);
        if (!fix_low || got_mask != mask) std::exit(174);

        Code moff = mm.main_mask_off[mask] - batch.main_begin;
        Code doff = mm.block_mask_off[mask] - batch.block_begin;
        if (moff + ms.size > batch.main_size || doff + ds.size > batch.block_size)
            std::exit(175);
        Count* gm = raw_m + moff;
        Count* gd = raw_d + doff;
        int bm = int(std::min<Code>(65535, (ms.size + gpu_threads - 1) / gpu_threads));

        for (int p = wp.p_hi; p >= wp.p_lo; --p) {
            if (p <= 1) std::exit(176); // HIGH window must never need p=1 rules.
            if (ms.size) {
                main_group_high_orbit_inplace_kernel<<<bm, gpu_threads>>>(gm, ms.size, gd, p);
                main_group_high_closure_inplace_kernel<<<bm, gpu_threads>>>(gm, ms.size, gd, p);
                ck(cudaGetLastError(), "v5.3 HIGH orbit transition");
            }
        }
        ck(cudaDeviceSynchronize(), "v5.3 group sync");
        ++c.groups;
    }
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (batch.main_size) {
        ck(cudaMemcpy(main_auth.ptr + batch.main_begin, raw_m,
                      size_t(batch.main_size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "v5.3 D2H main");
        ++c.pcie_calls;
    }
    if (batch.block_size) {
        ck(cudaMemcpy(block_auth.ptr + batch.block_begin, raw_d,
                      size_t(batch.block_size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "v5.3 D2H block");
        ++c.pcie_calls;
    }
    c.d2h_s += ram_seconds_since(t);
    ++c.batches;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int gpu_target_mib = argc > 3 ? std::atoi(argv[3]) : 12288;
    int cpu_workers = argc > 4 ? std::max(1, std::atoi(argv[4])) : 4;
    bool plan_only = argc > 5 && std::strcmp(argv[5], "--plan-only") == 0;
    int W = n + 1;
    if (W != TARGET_W || n < 2 || W > MAXW) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout logical = build_storage_layout(storage);
    LowMaskMajorLayout mm = build_lowmask_major_layout(storage, logical);

    auto meta0 = std::chrono::steady_clock::now();
    LowDescHost lowdesc = build_low_descriptors(storage, logical);
    HighDescHost highdesc = build_high_descriptors(storage, logical);
    HighOrbitHost highorbit = build_high_orbit(storage, logical);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, logical, lowdesc);
    CpuLowMaskSparseHost sparse = build_cpu_low_maskmajor_sparse(storage, logical, lowdesc, loworbit);
    double meta_build_s = ram_seconds_since(meta0);

    WindowPlan high_wp = make_direct2d_window(true);
    WindowPlan low_wp = make_direct2d_window(false);
    size_t gpu_target = size_t(gpu_target_mib) << 20;
    OrbitMaskBatchPlan batch_plan = make_orbit_mask_batches(mm, gpu_target);
    const auto& batches = batch_plan.batches;
    auto g_for_mask = make_orbit_g_for_low_mask(high_wp);
    auto cpu_jobs = make_cpu_low_jobs(W, low_wp);

    uint32_t max_masks = 0;
    for (const auto& b : batches) max_masks = std::max(max_masks, b.last - b.first);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = lowmask_major_rank_main_host(init, storage, logical, mm);
    Code answer_rank = lowmask_major_rank_main_host(MateID(R), storage, logical, mm);

    double main_bytes = double(mm.main_size) * sizeof(Count);
    double block_bytes = double(mm.block_size) * sizeof(Count);
    double pcie_bytes = double(W) * (2.0 * main_bytes + block_bytes);
    double pcie_tib = pcie_bytes / double(1ULL << 40);
    double pcie_50gib_s = pcie_bytes / (50.0 * double(1ULL << 30));
    uint64_t batch_calls = uint64_t(batches.size()) * uint64_t(W) * 3;
    double avg_copy_mib = batch_calls
        ? pcie_bytes / double(batch_calls) / double(1 << 20) : 0.0;
    double highdesc_mib = double((highdesc.main_desc.size() + highdesc.block_desc.size())
                                 * sizeof(uint32_t)) / double(1 << 20);
    double highorbit_mib = double(highorbit.rec.size() * sizeof(uint64_t)) / double(1 << 20);
    double mask_mib = double((G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
        + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t))
        / double(1 << 20);
    double sparse_mib = double(sparse.orbit_ops.size() * sizeof(CpuLowMaskOrbitOp)
        + sparse.closure_ops.size() * sizeof(CpuLowMaskClosureOp)
        + (sparse.orbit_off.size() + sparse.closure_off.size()) * sizeof(uint32_t))
        / double(1 << 20);

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-orbit-v5.3-plan"
            << " n=" << n
            << " batches_per_row=" << batches.size()
            << " max_masks_per_batch=" << max_masks
            << " gpu_data_target_gib=" << double(gpu_target) / double(1ULL << 30)
            << " gpu_arena_peak_gib=" << double(batch_plan.arena_peak_bytes) / double(1ULL << 30)
            << " raw_peak_gib=" << double(batch_plan.raw_peak * sizeof(Count)) / double(1ULL << 30)
            << " alt_main_gib=0 alt_block_gib=0"
            << " gpu_high_desc_mib=" << highdesc_mib
            << " gpu_high_orbit_mib=" << highorbit_mib
            << " gpu_mask_mib=" << mask_mib
            << " cpu_sparse_mib=" << sparse_mib
            << " meta_build_s=" << meta_build_s
            << " cpu_scratch_gib=0"
            << " row_boundary_blocked_zero=1"
            << " pcie_tib_per_residue=" << pcie_tib
            << " pcie_50gib_s=" << pcie_50gib_s
            << " pcie_calls_per_residue=" << batch_calls
            << " avg_pcie_copy_mib=" << avg_copy_mib
            << '\n';
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) return 2;
    ck(cudaSetDevice(0), "cudaSetDevice");

    BidescMaskDeviceTables mask_tables; mask_tables.install(G_FACTOR);
    HighDescDeviceTables highdesc_tables; highdesc_tables.install(highdesc);
    HighOrbitDeviceTables highorbit_tables; highorbit_tables.install(highorbit);
    maskmajor_orbit_release_dense_host(storage);
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    highorbit.rec.clear(); highorbit.rec.shrink_to_fit();
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    loworbit.rec.clear(); loworbit.rec.shrink_to_fit();

    RamCounts main_auth, block_auth;
    main_auth.alloc(mm.main_size, "mmap maskmajor-orbit main");
    block_auth.alloc(mm.block_size, "mmap maskmajor-orbit block");
    main_auth.ptr[init_rank] = 1;

    OrbitMaskBatchCtx gpu; gpu.init(batch_plan, mod);
    CpuLowMaskMajorPool cpu(cpu_workers);
    int gpu_threads = 256;
    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        for (const auto& b : batches)
            process_orbit_mask_batch(gpu, b, g_for_mask, main_auth, block_auth,
                                     mm, W, high_wp, gpu_threads);
        cpu.run(cpu_jobs, main_auth, block_auth, storage, logical, mm, sparse, mod);
        std::cerr << "row " << row + 1 << '/' << W
                  << " gpu_batches=" << gpu.batches
                  << " gpu_groups=" << gpu.groups
                  << " cpu_groups=" << cpu.groups()
                  << " pcie_calls=" << gpu.pcie_calls << '\n';
    }

    double wall_s = ram_seconds_since(wall0);
    Count answer = main_auth.ptr[answer_rank];
    std::cout
        << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-orbit-v5.3"
        << " n=" << n << " residue=" << answer << " modulus=" << mod
        << " batches=" << gpu.batches << " gpu_groups=" << gpu.groups
        << " cpu_groups=" << cpu.groups() << " cpu_workers=" << cpu_workers
        << " cpu_scratch_gib=0"
        << " pcie_calls=" << gpu.pcie_calls
        << " h2d_s=" << gpu.h2d_s << " gpu_kernel_s=" << gpu.kernel_s
        << " d2h_s=" << gpu.d2h_s
        << " cpu_kernel_sum_s=" << cpu.kernel_s() << " cpu_wall_s=" << cpu.wall_s
        << " cpu_orbit_columns=" << cpu.orbit_columns()
        << " pcie_tib_per_residue=" << pcie_tib << " wall_s=" << wall_s << '\n';

    gpu.release();
    highorbit_tables.release(); highdesc_tables.release(); mask_tables.release();
    main_auth.release(); block_auth.release();
    return 0;
}
