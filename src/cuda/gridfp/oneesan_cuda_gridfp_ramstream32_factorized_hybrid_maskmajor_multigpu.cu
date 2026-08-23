#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "ramstream32_cpu_low_maskmajor.hpp"
#include "ramstream32_high_rawbatch.cuh"

static void v6_release_dense_host(StorageFactorHost& storage) {
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

struct V6Batch {
    uint32_t first = 0, last = 0;
    Code main_begin = 0, main_size = 0;
    Code block_begin = 0, block_size = 0;
    size_t arena_bytes = 0;
    HighRawBatchTasks high;
};

struct V6BatchPlan {
    std::vector<V6Batch> batches;
    Code raw_peak = 0;
    size_t arena_peak_bytes = 0;
    uint64_t tasks_per_row = 0;
    uint64_t task_bytes_per_row = 0;
};

static V6BatchPlan make_v6_batches(
    const LowMaskMajorLayout& mm, const StorageLayout& logical, size_t target_bytes
) {
    V6BatchPlan plan;
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
                std::cerr << "single LOW mask exceeds v6 target mask=" << b
                          << " need_gib=" << double(need) / double(1ULL << 30)
                          << " target_gib=" << double(target_bytes) / double(1ULL << 30) << '\n';
                std::exit(240);
            }
            raw = nr;
            ++b;
        }
        V6Batch z;
        z.first = a; z.last = b;
        z.main_begin = mm.main_mask_off[a];
        z.main_size = mm.main_mask_off[b] - mm.main_mask_off[a];
        z.block_begin = mm.block_mask_off[a];
        z.block_size = mm.block_mask_off[b] - mm.block_mask_off[a];
        z.arena_bytes = size_t(z.main_size + z.block_size) * sizeof(Count);
        z.high = build_high_raw_tasks(mm, logical, a, b);
        plan.raw_peak = std::max(plan.raw_peak, z.main_size + z.block_size);
        plan.arena_peak_bytes = std::max(plan.arena_peak_bytes, z.arena_bytes);
        plan.tasks_per_row += z.high.tasks.size();
        plan.task_bytes_per_row += z.high.tasks.size() * sizeof(HighRawTask);
        plan.batches.push_back(std::move(z));
        a = b;
    }
    return plan;
}

struct V6Range {
    size_t begin = 0, end = 0;
    uint64_t arena_sum = 0;
    uint64_t transfer_bytes_per_row = 0;
    uint64_t tasks = 0;
};

// Keep each GPU on one contiguous interval of LOW masks for better NUMA/page
// locality.  Batch sizes are already nearly equal, so a contiguous byte target
// is enough to balance both PCIe bytes and HIGH work.
static std::vector<V6Range> partition_v6_batches(const V6BatchPlan& p, int ngpu) {
    ngpu = std::max(1, std::min<int>(ngpu, p.batches.size()));
    std::vector<V6Range> out(size_t(ngpu));
    uint64_t total = 0;
    for (const auto& b : p.batches) total += b.arena_bytes;
    size_t pos = 0;
    uint64_t remaining = total;
    for (int g = 0; g < ngpu; ++g) {
        V6Range r;
        r.begin = pos;
        int left_gpus = ngpu - g;
        uint64_t target = left_gpus ? (remaining + left_gpus - 1) / left_gpus : remaining;
        while (pos < p.batches.size()) {
            size_t batches_left_after = p.batches.size() - (pos + 1);
            if (batches_left_after < size_t(left_gpus - 1)) break;
            uint64_t nb = p.batches[pos].arena_bytes;
            if (pos > r.begin && r.arena_sum + nb > target) break;
            r.arena_sum += nb;
            r.transfer_bytes_per_row += uint64_t(
                (Code(2) * p.batches[pos].main_size + p.batches[pos].block_size)
                * sizeof(Count));
            r.tasks += p.batches[pos].high.tasks.size();
            ++pos;
        }
        if (pos == r.begin && pos < p.batches.size()) {
            const auto& b = p.batches[pos++];
            r.arena_sum += b.arena_bytes;
            r.transfer_bytes_per_row += uint64_t((Code(2) * b.main_size + b.block_size) * sizeof(Count));
            r.tasks += b.high.tasks.size();
        }
        r.end = pos;
        remaining -= r.arena_sum;
        out[size_t(g)] = r;
    }
    if (pos != p.batches.size()) std::exit(241);
    return out;
}

struct V6GpuCtx {
    int dev = 0;
    V6Range range;
    Count* raw = nullptr;
    Code raw_cap = 0;
    HighRawTask* tasks = nullptr;
    size_t task_count = 0;
    std::vector<uint32_t> task_off; // local-batch prefix

    BidescMaskDeviceTables mask_tables;
    HighDescDeviceTables highdesc_tables;
    HighOrbitDeviceTables highorbit_tables;
    HighRawLayoutDeviceTables raw_layout;

    double h2d_s = 0, kernel_s = 0, d2h_s = 0;
    uint64_t batches_done = 0, pcie_calls = 0, kernel_launches = 0;

    void init(
        int d, V6Range r, const V6BatchPlan& plan,
        const StorageLayout& logical, const LowMaskMajorLayout& mm,
        const HighDescHost& highdesc, const HighOrbitHost& highorbit, Count mod
    ) {
        dev = d; range = r;
        ck(cudaSetDevice(dev), "v6 set device init");
        mask_tables.install(G_FACTOR);
        highdesc_tables.install(highdesc);
        highorbit_tables.install(highorbit);
        raw_layout.install(mm, logical);
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "v6 modulus");

        for (size_t q = r.begin; q < r.end; ++q)
            raw_cap = std::max(raw_cap, plan.batches[q].main_size + plan.batches[q].block_size);
        if (raw_cap) ck(cudaMalloc(&raw, size_t(raw_cap) * sizeof(Count)), "v6 raw arena");

        task_off.reserve(r.end - r.begin + 1);
        task_off.push_back(0);
        std::vector<HighRawTask> flat;
        flat.reserve(size_t(r.tasks));
        for (size_t q = r.begin; q < r.end; ++q) {
            flat.insert(flat.end(), plan.batches[q].high.tasks.begin(), plan.batches[q].high.tasks.end());
            if (flat.size() > 0xffffffffULL) std::exit(242);
            task_off.push_back(uint32_t(flat.size()));
        }
        task_count = flat.size();
        if (task_count) {
            ck(cudaMalloc(&tasks, task_count * sizeof(HighRawTask)), "v6 persistent tasks alloc");
            ck(cudaMemcpy(tasks, flat.data(), task_count * sizeof(HighRawTask), cudaMemcpyHostToDevice),
               "v6 persistent tasks copy");
        }
    }

    void release() {
        ck(cudaSetDevice(dev), "v6 set device release");
        if (tasks) cudaFree(tasks);
        if (raw) cudaFree(raw);
        tasks = nullptr; raw = nullptr;
        raw_layout.release(); highorbit_tables.release(); highdesc_tables.release(); mask_tables.release();
    }
};

static void process_v6_gpu_row(
    V6GpuCtx& c, const V6BatchPlan& plan,
    RamCounts& main_auth, RamCounts& block_auth,
    const WindowPlan& high_wp, int gpu_threads
) {
    ck(cudaSetDevice(c.dev), "v6 set device worker");
    for (size_t q = c.range.begin; q < c.range.end; ++q) {
        const V6Batch& b = plan.batches[q];
        size_t local = q - c.range.begin;
        uint32_t ta = c.task_off[local], tb = c.task_off[local + 1];
        uint32_t nt = tb - ta;
        Count* raw_m = c.raw;
        Count* raw_d = c.raw + b.main_size;

        auto t = std::chrono::steady_clock::now();
        if (b.main_size) {
            ck(cudaMemcpy(raw_m, main_auth.ptr + b.main_begin,
                          size_t(b.main_size) * sizeof(Count), cudaMemcpyHostToDevice),
               "v6 H2D main");
            ++c.pcie_calls;
        }
        if (b.block_size)
            ck(cudaMemset(raw_d, 0, size_t(b.block_size) * sizeof(Count)), "v6 zero blocked");
        c.h2d_s += ram_seconds_since(t);

        t = std::chrono::steady_clock::now();
        for (int p = high_wp.p_hi; p >= high_wp.p_lo; --p) {
            if (nt) {
                high_rawbatch_orbit_kernel<<<nt, gpu_threads>>>(
                    raw_m, raw_d, c.tasks + ta, nt, b.main_begin, b.block_begin, p);
                high_rawbatch_closure_kernel<<<nt, gpu_threads>>>(
                    raw_m, raw_d, c.tasks + ta, nt, b.main_begin, b.block_begin, p);
                c.kernel_launches += 2;
            }
        }
        ck(cudaGetLastError(), "v6 HIGH batch launch");
        ck(cudaDeviceSynchronize(), "v6 batch sync");
        c.kernel_s += ram_seconds_since(t);

        t = std::chrono::steady_clock::now();
        if (b.main_size) {
            ck(cudaMemcpy(main_auth.ptr + b.main_begin, raw_m,
                          size_t(b.main_size) * sizeof(Count), cudaMemcpyDeviceToHost),
               "v6 D2H main");
            ++c.pcie_calls;
        }
        if (b.block_size) {
            ck(cudaMemcpy(block_auth.ptr + b.block_begin, raw_d,
                          size_t(b.block_size) * sizeof(Count), cudaMemcpyDeviceToHost),
               "v6 D2H block");
            ++c.pcie_calls;
        }
        c.d2h_s += ram_seconds_since(t);
        ++c.batches_done;
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int gpu_target_mib = argc > 3 ? std::atoi(argv[3]) : 12288;
    int cpu_workers = argc > 4 ? std::max(1, std::atoi(argv[4])) : 16;
    int requested_gpus = argc > 5 ? std::max(1, std::atoi(argv[5])) : 8;
    bool plan_only = argc > 6 && std::strcmp(argv[6], "--plan-only") == 0;
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
    V6BatchPlan batch_plan = make_v6_batches(mm, logical, gpu_target);
    int plan_gpus = std::min<int>(requested_gpus, batch_plan.batches.size());
    auto ranges = partition_v6_batches(batch_plan, plan_gpus);
    auto cpu_jobs = make_cpu_low_jobs(W, low_wp);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = lowmask_major_rank_main_host(init, storage, logical, mm);
    Code answer_rank = lowmask_major_rank_main_host(MateID(R), storage, logical, mm);

    double main_bytes = double(mm.main_size) * sizeof(Count);
    double block_bytes = double(mm.block_size) * sizeof(Count);
    double pcie_bytes = double(W) * (2.0 * main_bytes + block_bytes);
    double pcie_tib = pcie_bytes / double(1ULL << 40);
    double pcie_50gib_s = pcie_bytes / (50.0 * double(1ULL << 30));
    double ideal_parallel_s = pcie_50gib_s / plan_gpus;
    uint64_t max_transfer = 0, min_transfer = ~uint64_t(0), sum_transfer = 0;
    size_t max_batches = 0, min_batches = size_t(-1);
    for (const auto& r : ranges) {
        max_transfer = std::max(max_transfer, r.transfer_bytes_per_row);
        min_transfer = std::min(min_transfer, r.transfer_bytes_per_row);
        sum_transfer += r.transfer_bytes_per_row;
        max_batches = std::max(max_batches, r.end - r.begin);
        min_batches = std::min(min_batches, r.end - r.begin);
    }
    double avg_transfer = ranges.empty() ? 0.0 : double(sum_transfer) / ranges.size();
    double balance = avg_transfer ? double(max_transfer) / avg_transfer : 0.0;
    uint64_t launches = uint64_t(batch_plan.batches.size()) * W * HIGH_LUT_K * 2;
    double persistent_task_mib_total = double(batch_plan.task_bytes_per_row) / double(1 << 20);
    double sparse_mib = double(cpu_mm_sparse_bytes(sparse)) / double(1 << 20);

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-multigpu-v6-plan"
            << " n=" << n << " gpus=" << plan_gpus
            << " batches_per_row=" << batch_plan.batches.size()
            << " batches_per_gpu_min=" << min_batches
            << " batches_per_gpu_max=" << max_batches
            << " transfer_balance_max_over_avg=" << balance
            << " gpu_data_target_gib=" << double(gpu_target) / double(1ULL << 30)
            << " gpu_arena_peak_gib=" << double(batch_plan.arena_peak_bytes) / double(1ULL << 30)
            << " persistent_task_mib_all_gpus=" << persistent_task_mib_total
            << " cpu_sparse_mib=" << sparse_mib
            << " meta_build_s=" << meta_build_s
            << " pcie_tib_per_residue=" << pcie_tib
            << " pcie_50gib_s_single=" << pcie_50gib_s
            << " pcie_50gib_s_ideal_parallel=" << ideal_parallel_s
            << " kernel_launches_total_per_residue=" << launches
            << " row_boundary_blocked_zero=1 cpu_scratch_gib=0"
            << '\n';
        for (size_t g = 0; g < ranges.size(); ++g) {
            const auto& r = ranges[g];
            std::cout << "gpu_plan dev=" << g
                      << " batch_begin=" << r.begin << " batch_end=" << r.end
                      << " batches=" << (r.end-r.begin)
                      << " tasks=" << r.tasks
                      << " transfer_gib_per_row="
                      << double(r.transfer_bytes_per_row) / double(1ULL << 30) << '\n';
        }
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < requested_gpus) {
        std::cerr << "requested " << requested_gpus << " GPUs but only " << visible << " visible\n";
        return 2;
    }
    int ngpu = std::min<int>(requested_gpus, batch_plan.batches.size());
    if (ngpu != plan_gpus) return 3;

    std::vector<std::unique_ptr<V6GpuCtx>> gpus;
    gpus.reserve(ngpu);
    for (int g = 0; g < ngpu; ++g) {
        auto c = std::make_unique<V6GpuCtx>();
        c->init(g, ranges[size_t(g)], batch_plan, logical, mm, highdesc, highorbit, mod);
        gpus.push_back(std::move(c));
    }

    v6_release_dense_host(storage);
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    highorbit.rec.clear(); highorbit.rec.shrink_to_fit();
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    loworbit.rec.clear(); loworbit.rec.shrink_to_fit();

    RamCounts main_auth, block_auth;
    main_auth.alloc(mm.main_size, "mmap v6 main");
    block_auth.alloc(mm.block_size, "mmap v6 block");
    main_auth.ptr[init_rank] = 1;

    CpuLowMaskMajorPool cpu(cpu_workers);
    const int gpu_threads = 256;
    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        std::vector<std::thread> ts;
        ts.reserve(ngpu);
        for (int g = 0; g < ngpu; ++g) {
            ts.emplace_back([&, g] {
                process_v6_gpu_row(*gpus[size_t(g)], batch_plan,
                                   main_auth, block_auth, high_wp, gpu_threads);
            });
        }
        for (auto& t : ts) t.join();
        cpu.run(cpu_jobs, main_auth, block_auth, storage, logical, mm, sparse, mod);
        std::cerr << "row " << row + 1 << '/' << W << " gpus=" << ngpu
                  << " cpu_groups=" << cpu.groups() << '\n';
    }

    double wall_s = ram_seconds_since(wall0);
    Count answer = main_auth.ptr[answer_rank];
    double gpu_h2d = 0, gpu_kernel = 0, gpu_d2h = 0;
    uint64_t batches_done = 0, pcie_calls = 0, kernel_launches = 0;
    for (const auto& c : gpus) {
        gpu_h2d = std::max(gpu_h2d, c->h2d_s);
        gpu_kernel = std::max(gpu_kernel, c->kernel_s);
        gpu_d2h = std::max(gpu_d2h, c->d2h_s);
        batches_done += c->batches_done;
        pcie_calls += c->pcie_calls;
        kernel_launches += c->kernel_launches;
    }
    std::cout
        << "backend=gridfp-ramstream32-factorized-hybrid-maskmajor-multigpu-v6"
        << " n=" << n << " residue=" << answer << " modulus=" << mod
        << " gpus=" << ngpu << " batches=" << batches_done
        << " pcie_calls=" << pcie_calls << " kernel_launches=" << kernel_launches
        << " gpu_h2d_max_s=" << gpu_h2d << " gpu_kernel_max_s=" << gpu_kernel
        << " gpu_d2h_max_s=" << gpu_d2h
        << " cpu_workers=" << cpu_workers << " cpu_wall_s=" << cpu.wall_s
        << " cpu_kernel_sum_s=" << cpu.kernel_s()
        << " pcie_tib_per_residue=" << pcie_tib << " wall_s=" << wall_s << '\n';

    for (auto& c : gpus) c->release();
    main_auth.release(); block_auth.release();
    return 0;
}
