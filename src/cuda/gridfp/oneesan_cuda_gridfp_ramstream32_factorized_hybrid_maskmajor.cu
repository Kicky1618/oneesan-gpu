#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN

#include "ramstream32_cpu_low_maskmajor.hpp"

static void maskmajor_release_dense_host(StorageFactorHost& storage) {
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
    // LOW mask codes are only needed on the GPU after installation; the CPU
    // LOW executor needs offsets but not the code list.  HIGH mask codes remain
    // resident because CROSS closures rank the modified HIGH topology.
    G_FACTOR.low_mask_codes.clear(); G_FACTOR.low_mask_codes.shrink_to_fit();
}

static void process_group_bidesc_maskmajor(
    Direct2DCtx& c, RamCounts& main_auth, RamCounts& block_auth,
    const LowMaskMajorLayout& mm,
    int W, const WindowPlan& wp, int g, int gpu_threads
) {
    GroupSpec ms, ds;
    bool fix_low = false;
    uint32_t mask = 0;
    std::vector<FBlock> fmb, fdb;
    configure_factor_group(W, wp, g, ms, ds, fix_low, mask, fmb, fdb);
    if (!fix_low) {
        std::cerr << "mask-major GPU transport requires HIGH window\n";
        std::exit(140);
    }
    if (!ms.size && !ds.size) return;
    if (mask + 1 >= mm.main_mask_off.size() || mask + 1 >= mm.block_mask_off.size())
        std::exit(141);
    if (Code(mm.main_mask_off[mask + 1] - mm.main_mask_off[mask]) != ms.size
        || Code(mm.block_mask_off[mask + 1] - mm.block_mask_off[mask]) != ds.size) {
        std::cerr << "mask-major transfer/group size mismatch mask=" << mask << '\n';
        std::exit(142);
    }

    c.ensure(ms.size, ds.size);
    auto t = std::chrono::steady_clock::now();
    if (ms.size) {
        ck(cudaMemcpy(c.dA, main_auth.ptr + mm.main_mask_off[mask],
                      size_t(ms.size) * sizeof(Count), cudaMemcpyHostToDevice),
           "maskmajor H2D main");
        ++c.copy1d; c.copy_elems += ms.size;
    }
    if (ds.size) {
        ck(cudaMemcpy(c.dD, block_auth.ptr + mm.block_mask_off[mask],
                      size_t(ds.size) * sizeof(Count), cudaMemcpyHostToDevice),
           "maskmajor H2D block");
        ++c.copy1d; c.copy_elems += ds.size;
    }
    c.h2d_s += ram_seconds_since(t);

    int bm = int(std::min<Code>(65535, (ms.size + gpu_threads - 1) / gpu_threads));
    int bd = int(std::min<Code>(65535, (ds.size + gpu_threads - 1) / gpu_threads));
    t = std::chrono::steady_clock::now();
    Count* cur = c.dA;
    Count* nxt = c.dB;
    Count* dcur = c.dD;
    Count* dnext = c.dE;
    for (int p = wp.p_hi; p >= wp.p_lo; --p) {
        if (ms.size)
            ck(cudaMemcpy(nxt, cur, size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToDevice),
               "maskmajor identity");
        if (ds.size)
            ck(cudaMemset(dnext, 0, size_t(ds.size) * sizeof(Count)), "maskmajor clear block");
        if (ms.size)
            main_group_highdesc_compact_kernel<<<bm, gpu_threads>>>(cur, ms.size, nxt, dnext, p);
        if (ds.size)
            blocked_group_highdesc_kernel<<<bd, gpu_threads>>>(dcur, ds.size, nxt, p);
        ck(cudaGetLastError(), "maskmajor transition");
        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }
    ck(cudaDeviceSynchronize(), "maskmajor transition sync");
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) {
        ck(cudaMemcpy(main_auth.ptr + mm.main_mask_off[mask], cur,
                      size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "maskmajor D2H main");
        ++c.copy1d; c.copy_elems += ms.size;
    }
    if (ds.size) {
        ck(cudaMemcpy(block_auth.ptr + mm.block_mask_off[mask], dcur,
                      size_t(ds.size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "maskmajor D2H block");
        ++c.copy1d; c.copy_elems += ds.size;
    }
    c.d2h_s += ram_seconds_since(t);
    ++c.groups;
}

struct HighCopyModel {
    uint64_t old_calls_per_row = 0;
    uint64_t maskmajor_calls_per_row = 0;
    uint64_t nonempty_groups = 0;
};

static HighCopyModel build_high_copy_model(const WindowPlan& wp) {
    HighCopyModel z;
    int groups = 1 << int(wp.fixed_pos.size());
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(TARGET_W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
        GroupSpec ms = make_spec(TARGET_W, mf, mo);
        GroupSpec ds = make_spec(TARGET_W - 1, bf, bo);
        uint32_t mask = mo & ((1u << LOW_LUT_K) - 1u);
        auto mb = make_factor_main_blocks(true, mask);
        auto db = make_factor_block_blocks(true, mask);
        uint64_t nmb = 0, ndb = 0;
        for (const auto& b : mb) if (b.end != b.off) ++nmb;
        for (const auto& b : db) if (b.end != b.off) ++ndb;
        z.old_calls_per_row += 2 * (nmb + ndb);
        if (ms.size) z.maskmajor_calls_per_row += 2;
        if (ds.size) z.maskmajor_calls_per_row += 2;
        if (ms.size || ds.size) ++z.nonempty_groups;
    }
    return z;
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
    if (high_wp.max_bytes > gpu_target) {
        std::cerr << "mask-major high window does not fit: high_gib="
                  << double(high_wp.max_bytes) / double(1ULL << 30)
                  << " target_gib=" << double(gpu_target) / double(1ULL << 30) << '\n';
        return 4;
    }

    auto high_jobs = make_direct2d_jobs(high_wp);
    auto cpu_jobs = make_cpu_low_jobs(W, low_wp);
    HighCopyModel copy_model = build_high_copy_model(high_wp);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = lowmask_major_rank_main_host(init, storage, logical, mm);
    Code answer_rank = lowmask_major_rank_main_host(MateID(R), storage, logical, mm);

    double highdesc_mib = double((highdesc.main_desc.size() + highdesc.block_desc.size())
                                 * sizeof(uint32_t)) / (1 << 20);
    double mask_mib = double((G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
        + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size())
        * sizeof(uint32_t)) / (1 << 20);
    double sparse_orbit_mib = double(sparse.orbit_ops.size() * sizeof(CpuLowMaskOrbitOp)) / (1 << 20);
    double sparse_closure_mib = double(sparse.closure_ops.size() * sizeof(CpuLowMaskClosureOp)) / (1 << 20);
    double sparse_offsets_mib = double((sparse.orbit_off.size() + sparse.closure_off.size())
                                       * sizeof(uint32_t)) / (1 << 20);
    double old_dense_cpu_mib = double((lowdesc.main_desc.size() + lowdesc.block_desc.size()) * sizeof(uint32_t)
                                      + orbit.rec.size() * sizeof(uint64_t)) / (1 << 20);
    double dense_host_release_mib = double((storage.low_packed_rank.size() + storage.high_packed_rank.size()
        + G_FACTOR.low_packed_rank.size() + G_FACTOR.high_packed_rank.size()) * sizeof(uint32_t)) / (1 << 20);
    double mm_layout_mib = double((mm.main_mask_off.size() + mm.block_mask_off.size()
        + mm.main_block_off.size() + mm.block_block_off.size()) * sizeof(Code)) / (1 << 20);

    double auth_bytes = double(mm.main_size + mm.block_size) * sizeof(Count);
    double pcie_bytes = 2.0 * W * auth_bytes;
    double pcie_tib = pcie_bytes / double(1ULL << 40);
    double pcie_50gib_s = pcie_bytes / (50.0 * double(1ULL << 30));
    uint64_t mm_calls_residue = copy_model.maskmajor_calls_per_row * uint64_t(W);
    uint64_t old_calls_residue = copy_model.old_calls_per_row * uint64_t(W);
    double avg_copy_mib = copy_model.maskmajor_calls_per_row
        ? (2.0 * auth_bytes / double(copy_model.maskmajor_calls_per_row)) / double(1 << 20)
        : 0.0;

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-v5-plan"
            << " n=" << n
            << " gpu_high_desc_mib=" << highdesc_mib
            << " gpu_mask_mib=" << mask_mib
            << " cpu_sparse_orbit_mib=" << sparse_orbit_mib
            << " cpu_sparse_closure_mib=" << sparse_closure_mib
            << " cpu_sparse_offsets_mib=" << sparse_offsets_mib
            << " cpu_dense_meta_replaced_mib=" << old_dense_cpu_mib
            << " maskmajor_layout_mib=" << mm_layout_mib
            << " dense_host_release_mib=" << dense_host_release_mib
            << " meta_build_s=" << meta_build_s
            << " gpu_high_window_max_gib=" << double(high_wp.max_bytes) / double(1ULL << 30)
            << " cpu_workers=" << cpu_workers
            << " cpu_scratch_gib=0"
            << " pcie_tib_per_residue=" << pcie_tib
            << " pcie_50gib_s=" << pcie_50gib_s
            << " old_copy_calls_per_residue=" << old_calls_residue
            << " maskmajor_copy_calls_per_residue=" << mm_calls_residue
            << " copy_call_reduction=" << (mm_calls_residue ? double(old_calls_residue)/mm_calls_residue : 0.0)
            << " avg_maskmajor_copy_mib=" << avg_copy_mib
            << '\n';
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) return 2;
    ck(cudaSetDevice(0), "cudaSetDevice");

    BidescMaskDeviceTables mask_tables;
    mask_tables.install(G_FACTOR);
    HighDescDeviceTables highdesc_tables;
    highdesc_tables.install(highdesc);

    maskmajor_release_dense_host(storage);
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    orbit.rec.clear(); orbit.rec.shrink_to_fit();

    RamCounts main_auth, block_auth;
    main_auth.alloc(mm.main_size, "mmap maskmajor main");
    block_auth.alloc(mm.block_size, "mmap maskmajor block");
    main_auth.ptr[init_rank] = 1;

    Direct2DCtx gpu;
    gpu.init(mod);
    CpuLowMaskMajorPool cpu(cpu_workers);
    int gpu_threads = 256;
    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        for (const auto& job : high_jobs)
            if (job.work)
                process_group_bidesc_maskmajor(gpu, main_auth, block_auth, mm,
                                               W, high_wp, job.g, gpu_threads);
        cpu.run(cpu_jobs, main_auth, block_auth, storage, logical, mm, sparse, mod);
        std::cerr << "row " << row + 1 << '/' << W
                  << " gpu_groups=" << gpu.groups
                  << " cpu_groups=" << cpu.groups()
                  << " pci_copy_calls=" << gpu.copy1d << '\n';
    }

    double wall_s = ram_seconds_since(wall0);
    Count answer = main_auth.ptr[answer_rank];
    std::cout
        << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-v5"
        << " n=" << n << " residue=" << answer << " modulus=" << mod
        << " gpu_high_desc_mib=" << highdesc_mib
        << " cpu_sparse_orbit_mib=" << sparse_orbit_mib
        << " cpu_sparse_closure_mib=" << sparse_closure_mib
        << " gpu_groups=" << gpu.groups << " cpu_groups=" << cpu.groups()
        << " cpu_workers=" << cpu_workers << " cpu_scratch_gib=0"
        << " pci_copy_calls=" << gpu.copy1d
        << " h2d_s=" << gpu.h2d_s << " gpu_kernel_s=" << gpu.kernel_s
        << " d2h_s=" << gpu.d2h_s
        << " cpu_kernel_sum_s=" << cpu.kernel_s() << " cpu_wall_s=" << cpu.wall_s
        << " pcie_tib_per_residue=" << pcie_tib
        << " wall_s=" << wall_s << '\n';

    gpu.destroy();
    highdesc_tables.release();
    mask_tables.release();
    main_auth.release();
    block_auth.release();
    return 0;
}
