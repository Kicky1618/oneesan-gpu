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
#include "ramstream32_high_rawbatch.cuh"

static void rawbatch_release_dense_host(StorageFactorHost& storage) {
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

struct RawMaskBatch {
    uint32_t first = 0, last = 0;
    Code main_begin = 0, main_size = 0;
    Code block_begin = 0, block_size = 0;
    size_t arena_bytes = 0;
    HighRawBatchTasks high;
};

struct RawMaskBatchPlan {
    std::vector<RawMaskBatch> batches;
    Code raw_peak = 0;
    size_t arena_peak_bytes = 0;
    uint64_t tasks_per_row = 0;
    uint32_t max_tasks = 0;
    uint64_t task_bytes_per_row = 0;
};

static RawMaskBatchPlan make_raw_mask_batches(
    const LowMaskMajorLayout& mm, const StorageLayout& logical,
    size_t target_bytes
) {
    RawMaskBatchPlan plan;
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
                std::cerr << "single LOW mask exceeds v5.4 target mask=" << b
                          << " need_gib=" << double(need) / double(1ULL << 30)
                          << " target_gib=" << double(target_bytes) / double(1ULL << 30)
                          << '\n';
                std::exit(210);
            }
            raw = nr;
            ++b;
        }
        RawMaskBatch z;
        z.first = a; z.last = b;
        z.main_begin = mm.main_mask_off[a];
        z.main_size = mm.main_mask_off[b] - mm.main_mask_off[a];
        z.block_begin = mm.block_mask_off[a];
        z.block_size = mm.block_mask_off[b] - mm.block_mask_off[a];
        z.arena_bytes = size_t(z.main_size + z.block_size) * sizeof(Count);
        z.high = build_high_raw_tasks(mm, logical, a, b);
        if (z.arena_bytes > target_bytes) std::exit(211);
        plan.raw_peak = std::max(plan.raw_peak, z.main_size + z.block_size);
        plan.arena_peak_bytes = std::max(plan.arena_peak_bytes, z.arena_bytes);
        plan.tasks_per_row += z.high.tasks.size();
        plan.max_tasks = std::max<uint32_t>(plan.max_tasks, uint32_t(z.high.tasks.size()));
        plan.task_bytes_per_row += z.high.tasks.size() * sizeof(HighRawTask);
        plan.batches.push_back(std::move(z));
        a = b;
    }
    return plan;
}

struct RawBatchCtx {
    Count* raw = nullptr;
    Code raw_cap = 0;
    HighRawTaskDeviceBuffer taskbuf;
    double h2d_s = 0, kernel_s = 0, d2h_s = 0;
    uint64_t batches = 0, pcie_calls = 0, task_uploads = 0, kernel_launches = 0;

    void init(const RawMaskBatchPlan& plan, Count mod) {
        raw_cap = plan.raw_peak;
        if (raw_cap) ck(cudaMalloc(&raw, size_t(raw_cap) * sizeof(Count)), "v5.4 raw arena");
        taskbuf.ensure(plan.max_tasks);
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "v5.4 modulus");
    }
    void release() {
        taskbuf.release();
        if (raw) cudaFree(raw);
        raw = nullptr; raw_cap = 0;
    }
};

static void process_raw_batch(
    RawBatchCtx& c, const RawMaskBatch& batch,
    RamCounts& main_auth, RamCounts& block_auth,
    const WindowPlan& high_wp, int gpu_threads
) {
    Count* raw_m = c.raw;
    Count* raw_d = c.raw + batch.main_size;

    auto t = std::chrono::steady_clock::now();
    if (batch.main_size) {
        ck(cudaMemcpy(raw_m, main_auth.ptr + batch.main_begin,
                      size_t(batch.main_size) * sizeof(Count), cudaMemcpyHostToDevice),
           "v5.4 H2D main");
        ++c.pcie_calls;
    }
    if (batch.block_size)
        ck(cudaMemset(raw_d, 0, size_t(batch.block_size) * sizeof(Count)),
           "v5.4 zero blocked");
    c.taskbuf.upload(batch.high.tasks);
    ++c.task_uploads;
    c.h2d_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    uint32_t nt = uint32_t(batch.high.tasks.size());
    for (int p = high_wp.p_hi; p >= high_wp.p_lo; --p) {
        if (p <= 1) std::exit(212);
        if (nt) {
            high_rawbatch_orbit_kernel<<<nt, gpu_threads>>>(
                raw_m, raw_d, c.taskbuf.ptr, nt, batch.main_begin, batch.block_begin, p);
            high_rawbatch_closure_kernel<<<nt, gpu_threads>>>(
                raw_m, raw_d, c.taskbuf.ptr, nt, batch.main_begin, batch.block_begin, p);
            c.kernel_launches += 2;
            ck(cudaGetLastError(), "v5.4 HIGH batch transition");
        }
    }
    // One synchronization per multi-mask batch, replacing one synchronization
    // and O(H) launches/configuration per LOW mask in v5.3.
    ck(cudaDeviceSynchronize(), "v5.4 batch sync");
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (batch.main_size) {
        ck(cudaMemcpy(main_auth.ptr + batch.main_begin, raw_m,
                      size_t(batch.main_size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "v5.4 D2H main");
        ++c.pcie_calls;
    }
    if (batch.block_size) {
        ck(cudaMemcpy(block_auth.ptr + batch.block_begin, raw_d,
                      size_t(batch.block_size) * sizeof(Count), cudaMemcpyDeviceToHost),
           "v5.4 D2H block");
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
    RawMaskBatchPlan batch_plan = make_raw_mask_batches(mm, logical, gpu_target);
    auto cpu_jobs = make_cpu_low_jobs(W, low_wp);

    uint32_t max_masks = 0;
    for (const auto& b : batch_plan.batches)
        max_masks = std::max(max_masks, b.last - b.first);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = lowmask_major_rank_main_host(init, storage, logical, mm);
    Code answer_rank = lowmask_major_rank_main_host(MateID(R), storage, logical, mm);

    double main_bytes = double(mm.main_size) * sizeof(Count);
    double block_bytes = double(mm.block_size) * sizeof(Count);
    double pcie_bytes = double(W) * (2.0 * main_bytes + block_bytes);
    double pcie_tib = pcie_bytes / double(1ULL << 40);
    double pcie_50gib_s = pcie_bytes / (50.0 * double(1ULL << 30));
    uint64_t data_calls = uint64_t(batch_plan.batches.size()) * uint64_t(W) * 3;
    uint64_t task_calls = uint64_t(batch_plan.batches.size()) * uint64_t(W);
    uint64_t launches = uint64_t(batch_plan.batches.size()) * uint64_t(W)
                      * uint64_t(HIGH_LUT_K) * 2;
    uint64_t v53_launches = uint64_t(1u << LOW_LUT_K) * uint64_t(W)
                          * uint64_t(HIGH_LUT_K) * 2;
    double task_gib_residue = double(batch_plan.task_bytes_per_row) * W / double(1ULL << 30);
    double highdesc_mib = double((highdesc.main_desc.size() + highdesc.block_desc.size())
                                 * sizeof(uint32_t)) / double(1 << 20);
    double highorbit_mib = double(highorbit.rec.size() * sizeof(uint64_t)) / double(1 << 20);
    double mask_mib = double((G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
        + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t))
        / double(1 << 20);
    double sparse_mib = double(cpu_mm_sparse_bytes(sparse)) / double(1 << 20);
    double layout_device_mib = double((mm.main_block_off.size() + mm.block_block_off.size())
                                      * sizeof(Code)) / double(1 << 20);

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-rawbatch-v5.4-plan"
            << " n=" << n
            << " batches_per_row=" << batch_plan.batches.size()
            << " max_masks_per_batch=" << max_masks
            << " high_tasks_per_row=" << batch_plan.tasks_per_row
            << " max_tasks_per_batch=" << batch_plan.max_tasks
            << " task_metadata_gib_per_residue=" << task_gib_residue
            << " gpu_data_target_gib=" << double(gpu_target) / double(1ULL << 30)
            << " gpu_arena_peak_gib=" << double(batch_plan.arena_peak_bytes) / double(1ULL << 30)
            << " raw_peak_gib=" << double(batch_plan.raw_peak * sizeof(Count)) / double(1ULL << 30)
            << " alt_main_gib=0 alt_block_gib=0"
            << " gpu_high_desc_mib=" << highdesc_mib
            << " gpu_high_orbit_mib=" << highorbit_mib
            << " gpu_mask_mib=" << mask_mib
            << " gpu_layout_mib=" << layout_device_mib
            << " cpu_sparse_mib=" << sparse_mib
            << " meta_build_s=" << meta_build_s
            << " cpu_scratch_gib=0"
            << " row_boundary_blocked_zero=1"
            << " pcie_tib_per_residue=" << pcie_tib
            << " pcie_50gib_s=" << pcie_50gib_s
            << " data_pcie_calls_per_residue=" << data_calls
            << " task_upload_calls_per_residue=" << task_calls
            << " kernel_launches_per_residue=" << launches
            << " v53_kernel_launches_per_residue=" << v53_launches
            << " kernel_launch_reduction=" << (launches ? double(v53_launches) / launches : 0.0)
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
    HighRawLayoutDeviceTables raw_layout; raw_layout.install(mm, logical);
    rawbatch_release_dense_host(storage);
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    highorbit.rec.clear(); highorbit.rec.shrink_to_fit();
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    loworbit.rec.clear(); loworbit.rec.shrink_to_fit();

    RamCounts main_auth, block_auth;
    main_auth.alloc(mm.main_size, "mmap maskmajor-rawbatch main");
    block_auth.alloc(mm.block_size, "mmap maskmajor-rawbatch block");
    main_auth.ptr[init_rank] = 1;

    RawBatchCtx gpu; gpu.init(batch_plan, mod);
    CpuLowMaskMajorPool cpu(cpu_workers);
    int gpu_threads = 256;
    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        for (const auto& b : batch_plan.batches)
            process_raw_batch(gpu, b, main_auth, block_auth, high_wp, gpu_threads);
        cpu.run(cpu_jobs, main_auth, block_auth, storage, logical, mm, sparse, mod);
        std::cerr << "row " << row + 1 << '/' << W
                  << " gpu_batches=" << gpu.batches
                  << " cpu_groups=" << cpu.groups()
                  << " pcie_calls=" << gpu.pcie_calls
                  << " kernel_launches=" << gpu.kernel_launches << '\n';
    }

    double wall_s = ram_seconds_since(wall0);
    Count answer = main_auth.ptr[answer_rank];
    std::cout
        << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-rawbatch-v5.4"
        << " n=" << n << " residue=" << answer << " modulus=" << mod
        << " batches=" << gpu.batches << " cpu_groups=" << cpu.groups()
        << " cpu_workers=" << cpu_workers << " cpu_scratch_gib=0"
        << " pcie_calls=" << gpu.pcie_calls << " task_uploads=" << gpu.task_uploads
        << " kernel_launches=" << gpu.kernel_launches
        << " h2d_s=" << gpu.h2d_s << " gpu_kernel_s=" << gpu.kernel_s
        << " d2h_s=" << gpu.d2h_s
        << " cpu_kernel_sum_s=" << cpu.kernel_s() << " cpu_wall_s=" << cpu.wall_s
        << " cpu_orbit_columns=" << cpu.orbit_columns()
        << " cpu_closure_columns=" << cpu.closure_columns()
        << " pcie_tib_per_residue=" << pcie_tib << " wall_s=" << wall_s << '\n';

    gpu.release();
    raw_layout.release(); highorbit_tables.release(); highdesc_tables.release(); mask_tables.release();
    main_auth.release(); block_auth.release();
    return 0;
}
