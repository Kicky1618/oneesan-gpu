#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define main oneesan_maskmajor_v51_unused_main
#include "oneesan_cuda_gridfp_ramstream32_factorized_hybrid_maskmajor.cu"
#undef main

struct MaskBatch {
    uint32_t first = 0, last = 0; // [first,last)
    Code main_begin = 0, main_size = 0;
    Code block_begin = 0, block_size = 0;
    Code alt_main = 0, alt_block = 0;
    size_t data_bytes = 0;
};

static std::vector<int> make_g_for_low_mask(const WindowPlan& wp) {
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
            std::exit(150);
        }
        out[mask] = g;
    }
    for (uint32_t m = 0; m < NM; ++m)
        if (out[m] < 0) {
            std::cerr << "missing LOW mask in HIGH schedule mask=" << m << '\n';
            std::exit(151);
        }
    return out;
}

static std::vector<MaskBatch> make_mask_batches(
    const LowMaskMajorLayout& mm, size_t target_bytes
) {
    std::vector<MaskBatch> out;
    const uint32_t NM = mm.masks;
    uint32_t a = 0;
    while (a < NM) {
        uint32_t b = a;
        Code raw_m = 0, raw_d = 0, alt_m = 0, alt_d = 0;
        while (b < NM) {
            Code gm = mm.main_mask_off[b + 1] - mm.main_mask_off[b];
            Code gd = mm.block_mask_off[b + 1] - mm.block_mask_off[b];
            Code nm = raw_m + gm;
            Code nd = raw_d + gd;
            Code am = std::max(alt_m, gm);
            Code ad = std::max(alt_d, gd);
            size_t need = size_t(nm + nd + am + ad) * sizeof(Count);
            if (need > target_bytes && b > a) break;
            if (need > target_bytes) {
                std::cerr << "single mask does not fit batched target mask=" << b
                          << " need_gib=" << double(need) / double(1ULL << 30)
                          << " target_gib=" << double(target_bytes) / double(1ULL << 30)
                          << '\n';
                std::exit(152);
            }
            raw_m = nm; raw_d = nd; alt_m = am; alt_d = ad;
            ++b;
        }
        MaskBatch z;
        z.first = a; z.last = b;
        z.main_begin = mm.main_mask_off[a];
        z.main_size = mm.main_mask_off[b] - mm.main_mask_off[a];
        z.block_begin = mm.block_mask_off[a];
        z.block_size = mm.block_mask_off[b] - mm.block_mask_off[a];
        z.alt_main = alt_m; z.alt_block = alt_d;
        z.data_bytes = size_t(z.main_size + z.block_size + alt_m + alt_d) * sizeof(Count);
        out.push_back(z);
        a = b;
    }
    return out;
}

struct MaskBatchCtx {
    Count *raw_m = nullptr, *raw_d = nullptr, *alt_m = nullptr, *alt_d = nullptr;
    Code raw_m_cap = 0, raw_d_cap = 0, alt_m_cap = 0, alt_d_cap = 0;
    double h2d_s = 0, kernel_s = 0, d2h_s = 0;
    uint64_t batches = 0, groups = 0, pcie_calls = 0;

    void init(const std::vector<MaskBatch>& batches, Count mod) {
        for (const auto& b : batches) {
            raw_m_cap = std::max(raw_m_cap, b.main_size);
            raw_d_cap = std::max(raw_d_cap, b.block_size);
            alt_m_cap = std::max(alt_m_cap, b.alt_main);
            alt_d_cap = std::max(alt_d_cap, b.alt_block);
        }
        if (raw_m_cap) ck(cudaMalloc(&raw_m, size_t(raw_m_cap) * sizeof(Count)), "batch raw main");
        if (raw_d_cap) ck(cudaMalloc(&raw_d, size_t(raw_d_cap) * sizeof(Count)), "batch raw block");
        if (alt_m_cap) ck(cudaMalloc(&alt_m, size_t(alt_m_cap) * sizeof(Count)), "batch alt main");
        if (alt_d_cap) ck(cudaMalloc(&alt_d, size_t(alt_d_cap) * sizeof(Count)), "batch alt block");
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "batch modulus");
    }
    void release() {
        if (raw_m) cudaFree(raw_m); if (raw_d) cudaFree(raw_d);
        if (alt_m) cudaFree(alt_m); if (alt_d) cudaFree(alt_d);
        raw_m = raw_d = alt_m = alt_d = nullptr;
    }
};

static void process_mask_batch(
    MaskBatchCtx& c, const MaskBatch& batch, const std::vector<int>& g_for_mask,
    RamCounts& main_auth, RamCounts& block_auth,
    const LowMaskMajorLayout& mm, int W, const WindowPlan& wp, int gpu_threads
) {
    auto t = std::chrono::steady_clock::now();
    if (batch.main_size) {
        ck(cudaMemcpy(c.raw_m, main_auth.ptr + batch.main_begin,
                      size_t(batch.main_size) * sizeof(Count), cudaMemcpyHostToDevice),
           "batch H2D main");
        ++c.pcie_calls;
    }
    // Row-boundary blocked is exactly zero; initialize the whole raw blocked
    // batch in HBM instead of transferring it from host RAM.
    if (batch.block_size)
        ck(cudaMemset(c.raw_d, 0, size_t(batch.block_size) * sizeof(Count)),
           "batch zero blocked");
    c.h2d_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    for (uint32_t mask = batch.first; mask < batch.last; ++mask) {
        int g = g_for_mask[mask];
        GroupSpec ms, ds;
        bool fix_low = false;
        uint32_t got_mask = 0;
        std::vector<FBlock> fmb, fdb;
        configure_factor_group(W, wp, g, ms, ds, fix_low, got_mask, fmb, fdb);
        if (!fix_low || got_mask != mask) std::exit(153);

        Code moff = mm.main_mask_off[mask] - batch.main_begin;
        Code doff = mm.block_mask_off[mask] - batch.block_begin;
        if (moff + ms.size > batch.main_size || doff + ds.size > batch.block_size)
            std::exit(154);

        Count* rawm = c.raw_m + moff;
        Count* rawd = c.raw_d + doff;
        Count* cur = rawm;
        Count* nxt = c.alt_m;
        Count* dcur = rawd;
        Count* dnext = c.alt_d;
        int bm = int(std::min<Code>(65535, (ms.size + gpu_threads - 1) / gpu_threads));
        int bd = int(std::min<Code>(65535, (ds.size + gpu_threads - 1) / gpu_threads));

        for (int p = wp.p_hi; p >= wp.p_lo; --p) {
            if (ms.size)
                ck(cudaMemcpy(nxt, cur, size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToDevice),
                   "batch identity");
            if (ds.size)
                ck(cudaMemset(dnext, 0, size_t(ds.size) * sizeof(Count)), "batch clear block");
            if (ms.size)
                main_group_highdesc_compact_kernel<<<bm, gpu_threads>>>(cur, ms.size, nxt, dnext, p);
            if (ds.size)
                blocked_group_highdesc_kernel<<<bd, gpu_threads>>>(dcur, ds.size, nxt, p);
            ck(cudaGetLastError(), "batch transition");
            std::swap(cur, nxt);
            std::swap(dcur, dnext);
        }
        // W=28 uses 13 HIGH edges (odd), but keep this generic.  Put the final
        // state back into the raw batch before moving to the next group.
        if (ms.size && cur != rawm)
            ck(cudaMemcpy(rawm, cur, size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToDevice),
               "batch commit main");
        if (ds.size && dcur != rawd)
            ck(cudaMemcpy(rawd, dcur, size_t(ds.size) * sizeof(Count), cudaMemcpyDeviceToDevice),
               "batch commit block");
        ck(cudaDeviceSynchronize(), "batch group sync");
        ++c.groups;
    }
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (batch.main_size) {
        ck(cudaMemcpy(main_auth.ptr + batch.main_begin, c.raw_m,
                      size_t(batch.main_size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "batch D2H main");
        ++c.pcie_calls;
    }
    if (batch.block_size) {
        ck(cudaMemcpy(block_auth.ptr + batch.block_begin, c.raw_d,
                      size_t(batch.block_size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "batch D2H block");
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
    LowOrbitHost orbit = build_cpu_low_orbit(storage, logical, lowdesc);
    CpuLowMaskSparseHost sparse = build_cpu_low_maskmajor_sparse(storage, logical, lowdesc, orbit);
    double meta_build_s = ram_seconds_since(meta0);

    WindowPlan high_wp = make_direct2d_window(true);
    WindowPlan low_wp = make_direct2d_window(false);
    size_t gpu_target = size_t(gpu_target_mib) << 20;
    auto batches = make_mask_batches(mm, gpu_target);
    auto g_for_mask = make_g_for_low_mask(high_wp);
    auto cpu_jobs = make_cpu_low_jobs(W, low_wp);

    size_t max_data = 0;
    Code max_raw_m = 0, max_raw_d = 0, max_alt_m = 0, max_alt_d = 0;
    uint32_t max_masks = 0;
    for (const auto& b : batches) {
        max_data = std::max(max_data, b.data_bytes);
        max_raw_m = std::max(max_raw_m, b.main_size);
        max_raw_d = std::max(max_raw_d, b.block_size);
        max_alt_m = std::max(max_alt_m, b.alt_main);
        max_alt_d = std::max(max_alt_d, b.alt_block);
        max_masks = std::max(max_masks, b.last - b.first);
    }

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = lowmask_major_rank_main_host(init, storage, logical, mm);
    Code answer_rank = lowmask_major_rank_main_host(MateID(R), storage, logical, mm);

    double main_bytes = double(mm.main_size) * sizeof(Count);
    double block_bytes = double(mm.block_size) * sizeof(Count);
    double pcie_bytes = double(W) * (2.0 * main_bytes + block_bytes);
    double pcie_tib = pcie_bytes / double(1ULL << 40);
    double pcie_50gib_s = pcie_bytes / (50.0 * double(1ULL << 30));
    uint64_t batch_calls = uint64_t(batches.size()) * uint64_t(W) * 3;
    double avg_copy_mib = batch_calls ? pcie_bytes / double(batch_calls) / double(1 << 20) : 0.0;
    double highdesc_mib = double((highdesc.main_desc.size() + highdesc.block_desc.size())
                                 * sizeof(uint32_t)) / (1 << 20);
    double mask_mib = double((G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
        + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t)) / (1 << 20);
    double sparse_mib = double(sparse.orbit_ops.size() * sizeof(CpuLowMaskOrbitOp)
        + sparse.closure_ops.size() * sizeof(CpuLowMaskClosureOp)
        + (sparse.orbit_off.size() + sparse.closure_off.size()) * sizeof(uint32_t)) / (1 << 20);

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-batch-v5.2-plan"
            << " n=" << n
            << " batches_per_row=" << batches.size()
            << " max_masks_per_batch=" << max_masks
            << " gpu_data_target_gib=" << double(gpu_target) / double(1ULL << 30)
            << " gpu_data_peak_gib=" << double(max_data) / double(1ULL << 30)
            << " raw_main_peak_gib=" << double(max_raw_m * sizeof(Count)) / double(1ULL << 30)
            << " raw_block_peak_gib=" << double(max_raw_d * sizeof(Count)) / double(1ULL << 30)
            << " alt_main_gib=" << double(max_alt_m * sizeof(Count)) / double(1ULL << 30)
            << " alt_block_gib=" << double(max_alt_d * sizeof(Count)) / double(1ULL << 30)
            << " gpu_high_desc_mib=" << highdesc_mib
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
    maskmajor_release_dense_host(storage);
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    orbit.rec.clear(); orbit.rec.shrink_to_fit();

    RamCounts main_auth, block_auth;
    main_auth.alloc(mm.main_size, "mmap maskmajor-batch main");
    block_auth.alloc(mm.block_size, "mmap maskmajor-batch block");
    main_auth.ptr[init_rank] = 1;

    MaskBatchCtx gpu; gpu.init(batches, mod);
    CpuLowMaskMajorPool cpu(cpu_workers);
    int gpu_threads = 256;
    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        for (const auto& b : batches)
            process_mask_batch(gpu, b, g_for_mask, main_auth, block_auth, mm,
                               W, high_wp, gpu_threads);
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
        << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-batch-v5.2"
        << " n=" << n << " residue=" << answer << " modulus=" << mod
        << " batches=" << gpu.batches << " gpu_groups=" << gpu.groups
        << " cpu_groups=" << cpu.groups() << " cpu_workers=" << cpu_workers
        << " cpu_scratch_gib=0"
        << " pcie_calls=" << gpu.pcie_calls
        << " h2d_s=" << gpu.h2d_s << " gpu_kernel_s=" << gpu.kernel_s
        << " d2h_s=" << gpu.d2h_s
        << " cpu_kernel_sum_s=" << cpu.kernel_s() << " cpu_wall_s=" << cpu.wall_s
        << " pcie_tib_per_residue=" << pcie_tib << " wall_s=" << wall_s << '\n';

    gpu.release();
    highdesc_tables.release(); mask_tables.release();
    main_auth.release(); block_auth.release();
    return 0;
}
