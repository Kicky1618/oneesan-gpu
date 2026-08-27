#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../gridfp/ramstream32_factorized_storage.hpp"
#include "../gridfp/ramstream32_highdesc.cuh"
#include "../gridfp/ramstream32_lowdesc.cuh"
#include "maskshard_layout.hpp"
#include "maskshard_lowlocal.cuh"
#include "maskshard_highio.cuh"
#include "maskshard_lowclosure.cuh"
#include "maskshard_highorbit.cuh"
#include "maskshard_loworbit.cuh"
#include "maskshard_lowclosure_kernel.cuh"

#ifdef MASKSHARD_HIGH_PERTHREAD_STREAM
#define MASKSHARD_HIGH_LAUNCH(grid, block) \
    grid, block, 0, maskshard_high_execution_stream()
#else
#define MASKSHARD_HIGH_LAUNCH(grid, block) grid, block
#endif

#if defined(MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH) && !defined(MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH)
#error "row-capped BLOCKED orbit launch requires tight BLOCKED-domain launch"
#endif
#if defined(MASKSHARD_ROW_DEPTH_ORBIT_COMPACT) && !defined(MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH)
#error "exact compact BLOCKED orbit launch requires tight BLOCKED-domain launch"
#endif
#if defined(MASKSHARD_HIGH_CLOSURE_TASK_LAUNCH) && !defined(MASKSHARD_HIGH_CLOSURE_ROWPACK)
#error "task-sized HIGH closure launch requires HIGH closure row packing"
#endif
#if defined(MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH) && !defined(MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT)
#error "exact LOW closure launch requires exact LOW closure task mapping"
#endif
#if defined(MASKSHARD_LOW_BLOCK_ORBIT_TIGHT_LAUNCH) && !defined(MASKSHARD_BLOCK_ORBIT)
#error "tight LOW BLOCKED-domain launch requires BLOCKED-domain orbit"
#endif
#if defined(MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT_LAUNCH) && !defined(MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT)
#error "exact LOW orbit launch requires exact compact LOW orbit task mapping"
#endif

struct FullOrbitBatchAddress {
    int owner = -1;
    Code offset = 0;
};

static FullOrbitBatchAddress fullorbit_batch_main_address_host(
    MateID m,
    const StorageFactorHost& storage,
    const StorageLayout& layout,
    const MaskShardLayout& shard
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LC = (1u << (2 * L)) - 1u;
    constexpr uint32_t HC = (1u << (2 * H)) - 1u;
    constexpr uint32_t HR = (1u << H) - 1u;
    const uint32_t lc = uint32_t(m) & LC;
    const uint32_t hc = uint32_t((m >> (2 * (L + 1))) & HC);
    const int he = seg_end_height_host(hc, H);
    const int cv = int(mget(m, L));
    const uint32_t bid = uint32_t(3 * he + cv);
    const uint32_t mask = seg_occ(hc, H);
    const uint32_t hp = storage.high_packed_rank[hc];
    const uint32_t lp = storage.low_packed_rank[lc];
    if (hp == 0xffffffffu || lp == 0xffffffffu || bid >= layout.main_blocks.size()) {
        std::cerr << "fullorbit-batch invalid host main address\n";
        std::exit(200);
    }
    const Code off = shard.main_base[mask]
        + shard.main_block_off[size_t(mask) * shard.main_nblocks + bid]
        + Code(hp & HR) * layout.main_blocks[bid].cols + (lp >> L);
    return {int(shard.owner[mask]), off};
}

static std::vector<uint32_t> build_fullorbit_batch_high_route(
    const StorageFactorHost& storage
) {
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t HM = (1u << H) - 1u;
    std::vector<uint32_t> route(storage.high_all_codes.size(), 0xffffffffu);
    for (int he = 0; he <= H + 1; ++he) {
        const uint32_t a = storage.high_all_off[he];
        const uint32_t b = storage.high_all_off[he + 1];
        for (uint32_t r = 0; r < b - a; ++r) {
            const uint32_t code = storage.high_all_codes[a + r];
            const uint32_t packed = storage.high_packed_rank[code];
            if (packed == 0xffffffffu) {
                std::cerr << "fullorbit-batch missing HIGH packed rank\n";
                std::exit(201);
            }
            route[a + r] = seg_occ(code, H) | ((packed & HM) << H);
        }
    }
    return route;
}

struct FullOrbitBatchWorker {
    int dev = -1;
    uint8_t* arena = nullptr;
    size_t cap = 0;
    Count* a = nullptr;
    Count* d = nullptr;
    double high_io_s = 0.0;
    double high_orbit_s = 0.0;
    double high_closure_s = 0.0;
    double low_orbit_s = 0.0;
    double low_closure_s = 0.0;
    uint64_t high_groups = 0;
    uint64_t low_groups = 0;

    static size_t aligned(size_t x) { return (x + 255) & ~size_t(255); }
    void init(int device) { dev = device; }
    void reset_stats() {
        high_io_s = high_orbit_s = high_closure_s = 0.0;
        low_orbit_s = low_closure_s = 0.0;
        high_groups = low_groups = 0;
    }
    void ensure(Code main_n, Code block_n) {
        const size_t mb = aligned(size_t(main_n) * sizeof(Count));
        const size_t db = aligned(size_t(block_n) * sizeof(Count));
        const size_t need = mb + db;
        if (need > cap) {
            if (arena) ck(cudaFree(arena), "fullorbit-batch worker free arena");
            cap = need;
            ck(cudaMalloc(&arena, cap), "fullorbit-batch HIGH scratch");
        }
        a = reinterpret_cast<Count*>(arena);
        d = reinterpret_cast<Count*>(arena + mb);
    }
    void release() {
        if (dev >= 0) cudaSetDevice(dev);
        if (arena) cudaFree(arena);
        arena = nullptr;
        cap = 0;
        a = d = nullptr;
    }
};

struct FullOrbitBatchHighJob {
    uint32_t low_mask = 0;
    Code main_n = 0;
    Code block_n = 0;
    Code work = 0;
#ifdef MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH
    std::array<Code, HIGH_LUT_K + 2> block_depth_end{};
#endif
#ifdef MASKSHARD_HIGH_CLOSURE_ROWS
    std::array<uint32_t, HIGH_LUT_K> closure_rows{};
#endif
};

struct FullOrbitBatchLowJob {
    uint32_t mask = 0;
#ifdef MASKSHARD_LOW_CLOSURE_COLS
    std::array<Code, LOW_LUT_K> closure_tasks{};
#endif
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
    std::array<std::array<Code, (TARGET_W + 1) / 2 + 1>, LOW_LUT_K>
        closure_tasks_by_cap{};
#endif
};

static std::vector<FullOrbitBatchHighJob> build_fullorbit_batch_high_jobs(
    const HighDescHost* high_desc = nullptr
) {
    constexpr uint32_t NM = 1u << LOW_LUT_K;
    std::vector<FullOrbitBatchHighJob> jobs;
    jobs.reserve(NM);
    for (uint32_t mask = 0; mask < NM; ++mask) {
        const auto mb = make_factor_main_blocks(true, mask);
        const auto db = make_factor_block_blocks(true, mask);
        const Code mn = mb.empty() ? 0 : mb.back().end;
        const Code dn = db.empty() ? 0 : db.back().end;
        FullOrbitBatchHighJob job;
        job.low_mask = mask;
        job.main_n = mn;
        job.block_n = dn;
        job.work = mn + dn;
#ifdef MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH
        if (db.size() != size_t(HIGH_LUT_K + 2)) {
            std::cerr << "fullorbit-batch row-cap BLOCKED block count mismatch mask="
                      << mask << " blocks=" << db.size() << '\n';
            std::exit(205);
        }
        Code prev = 0;
        for (int h = 0; h <= HIGH_LUT_K + 1; ++h) {
            const FBlock& b = db[size_t(h)];
            if (int(b.he) != h || b.off != prev || b.end < b.off) {
                std::cerr << "fullorbit-batch row-cap BLOCKED ordering mismatch mask="
                          << mask << " h=" << h << '\n';
                std::exit(206);
            }
            job.block_depth_end[size_t(h)] = b.end;
            prev = b.end;
        }
        if (job.block_depth_end.back() != dn) {
            std::cerr << "fullorbit-batch row-cap BLOCKED final size mismatch mask="
                      << mask << '\n';
            std::exit(207);
        }
#endif
#ifdef MASKSHARD_HIGH_CLOSURE_ROWS
        if (!high_desc) {
            std::cerr << "fullorbit-batch HIGH closure rows require host descriptors\n";
            std::exit(203);
        }
        for (int pi = 0; pi < HIGH_LUT_K; ++pi) {
            uint64_t closure_work = 0;
            for (size_t bid = 0; bid < mb.size(); ++bid) {
                const FBlock& b = mb[bid];
                if (!b.stride) continue;
                const uint32_t a = high_desc->closure_block_off[size_t(pi) * 65 + bid];
                const uint32_t z = high_desc->closure_block_off[size_t(pi) * 65 + bid + 1];
                const uint64_t block_rows = uint64_t(z - a);
#ifdef MASKSHARD_HIGH_CLOSURE_TASK_LAUNCH
#ifdef MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD
                const bool pack = b.stride
                    < uint32_t(MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD);
#else
                const bool pack = true;
#endif
                closure_work += pack
                    ? (block_rows * uint64_t(b.stride) + 31ULL) >> 5
                    : block_rows;
#else
                closure_work += block_rows;
#endif
            }
            if (closure_work > 0xffffffffULL) {
                std::cerr << "fullorbit-batch HIGH closure work count overflow mask="
                          << mask << " pi=" << pi << " work=" << closure_work << '\n';
                std::exit(204);
            }
            job.closure_rows[size_t(pi)] = uint32_t(closure_work);
        }
#else
        (void)high_desc;
#endif
        jobs.push_back(job);
    }
    std::sort(jobs.begin(), jobs.end(), [](const auto& x, const auto& y) {
        return x.work > y.work;
    });
    return jobs;
}

static void configure_fullorbit_batch_high_group(uint32_t mask) {
    constexpr int L = LOW_LUT_K;
    constexpr uint32_t LM = (1u << L) - 1u;
    const uint32_t mf = LM;
    const uint32_t mo = mask;
    const uint32_t bf = LM;
    const uint32_t bo = mask;
    const GroupSpec ms = make_spec(TARGET_W, mf, mo);
    const GroupSpec ds = make_spec(TARGET_W - 1, bf, bo);
    const auto mb = make_factor_main_blocks(true, mask);
    const auto db = make_factor_block_blocks(true, mask);
    if (mb.empty() || db.empty() || mb.back().end != ms.size || db.back().end != ds.size) {
        std::cerr << "fullorbit-batch HIGH factor-size mismatch mask=" << mask << '\n';
        std::exit(202);
    }
    const int mn = int(mb.size());
    const int dn = int(db.size());
    const int fix_low = 1;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, mb.data(), mb.size() * sizeof(FBlock)), "fullorbit-batch high main blocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, db.data(), db.size() * sizeof(FBlock)), "fullorbit-batch high block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &mn, sizeof(mn)), "fullorbit-batch high main n");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &dn, sizeof(dn)), "fullorbit-batch high block n");
    ck(cudaMemcpyToSymbol(D_F_MASK, &mask, sizeof(mask)), "fullorbit-batch high mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fix_low, sizeof(fix_low)), "fullorbit-batch high mode");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &mf, sizeof(mf)), "fullorbit-batch high mf");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &mo, sizeof(mo)), "fullorbit-batch high mo");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &bf, sizeof(bf)), "fullorbit-batch high bf");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &bo, sizeof(bo)), "fullorbit-batch high bo");
    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "fullorbit-batch high main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "fullorbit-batch high block dp");
}

static void process_fullorbit_batch_high_job(
    FullOrbitBatchWorker& w, const FullOrbitBatchHighJob& job, int threads,
    int zero_based_row
) {
    ck(cudaSetDevice(w.dev), "fullorbit-batch high set device");
    if (!job.main_n && !job.block_n) return;
    configure_fullorbit_batch_high_group(job.low_mask);
    w.ensure(job.main_n, job.block_n);
    const int bm = int(std::min<Code>(65535, (job.main_n + threads - 1) / threads));
    const int bd = int(std::min<Code>(65535, (job.block_n + threads - 1) / threads));
#ifdef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
    const int orbit_cap = std::min(zero_based_row + 1, TARGET_W / 2);
    const Code orbit_block_n =
        maskshard_configure_row_depth_compact_group(job.low_mask, orbit_cap);
    if (orbit_cap == TARGET_W / 2 && orbit_block_n != job.block_n) {
        std::cerr << "fullorbit-batch exact compact BLOCKED size mismatch mask="
                  << job.low_mask << " got=" << orbit_block_n
                  << " expected=" << job.block_n << '\n';
        std::exit(208);
    }
    const int bd_orbit = orbit_block_n
        ? int(std::min<Code>(65535, (orbit_block_n + threads - 1) / threads))
        : 0;
#elif defined(MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH)
    const int orbit_cap = std::min(zero_based_row + 1, HIGH_LUT_K + 1);
    const Code orbit_block_n = job.block_depth_end[size_t(orbit_cap)];
    const int bd_orbit = orbit_block_n
        ? int(std::min<Code>(65535, (orbit_block_n + threads - 1) / threads))
        : 0;
#else
    const int bd_orbit = bd;
    (void)zero_based_row;
#endif
#ifdef MASKSHARD_HIGH_CLOSURE_ROWS
    const int warps_per_block = (threads + 31) / 32;
#endif

    auto t = std::chrono::steady_clock::now();
    if (job.main_n)
        maskshard_high_main_io_kernel<false><<<MASKSHARD_HIGH_LAUNCH(bm, threads)>>>(
            w.a, job.main_n);
#ifndef MASKSHARD_LAZY_ZERO_BLOCK_INIT
    if (job.block_n)
        maskshard_high_block_io_kernel<false><<<MASKSHARD_HIGH_LAUNCH(bd, threads)>>>(
            w.d, job.block_n);
#endif
    ck(cudaGetLastError(), "fullorbit-batch high gather");
    ck(cudaDeviceSynchronize(), "fullorbit-batch high gather sync");
    w.high_io_s += ram_seconds_since(t);

    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        t = std::chrono::steady_clock::now();
#ifdef MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH
        if (bd_orbit)
            maskshard_main_block_highorbit_kernel<<<
                MASKSHARD_HIGH_LAUNCH(bd_orbit, threads)>>>(
                w.a, w.d, job.main_n, p);
#else
        if (job.main_n)
            maskshard_main_block_highorbit_kernel<<<
                MASKSHARD_HIGH_LAUNCH(bm, threads)>>>(
                w.a, w.d, job.main_n, p);
#endif
        ck(cudaGetLastError(), "fullorbit-batch high orbit");
        ck(cudaDeviceSynchronize(), "fullorbit-batch high orbit sync");
        w.high_orbit_s += ram_seconds_since(t);

        t = std::chrono::steady_clock::now();
#ifdef MASKSHARD_HIGH_CLOSURE_ROWS
        const uint32_t pi = uint32_t((TARGET_W - 1) - p);
        Code closure_rows = job.closure_rows[size_t(pi)];
#ifdef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
        const int closure_cap = std::min(zero_based_row + 1, (TARGET_W + 1) / 2);
        closure_rows = maskshard_highclosure_rowdepth_compact_launch_tasks(
            job.low_mask, int(pi), closure_cap);
        if (closure_cap == (TARGET_W + 1) / 2
            && closure_rows != job.closure_rows[size_t(pi)]) {
            std::cerr << "fullorbit-batch exact closure full-cap mismatch mask="
                      << job.low_mask << " pi=" << pi
                      << " got=" << closure_rows
                      << " expected=" << job.closure_rows[size_t(pi)] << '\n';
            std::exit(209);
        }
#endif
        const int bc = closure_rows
            ? int(std::min<Code>(65535,
                (closure_rows + Code(warps_per_block) - 1) / Code(warps_per_block)))
            : 0;
        if (bc)
            maskshard_main_highdesc_closure_inplace_kernel<<<
                MASKSHARD_HIGH_LAUNCH(bc, threads)>>>(
                w.a, w.d, job.main_n, p);
#else
        if (job.main_n)
            maskshard_main_highdesc_closure_inplace_kernel<<<
                MASKSHARD_HIGH_LAUNCH(bm, threads)>>>(
                w.a, w.d, job.main_n, p);
#endif
        ck(cudaGetLastError(), "fullorbit-batch high closure");
        ck(cudaDeviceSynchronize(), "fullorbit-batch high closure sync");
        w.high_closure_s += ram_seconds_since(t);
    }

    t = std::chrono::steady_clock::now();
    if (job.main_n)
        maskshard_high_main_io_kernel<true><<<MASKSHARD_HIGH_LAUNCH(bm, threads)>>>(
            w.a, job.main_n);
    if (job.block_n)
        maskshard_high_block_io_kernel<true><<<MASKSHARD_HIGH_LAUNCH(bd, threads)>>>(
            w.d, job.block_n);
    ck(cudaGetLastError(), "fullorbit-batch high scatter");
    ck(cudaDeviceSynchronize(), "fullorbit-batch high scatter sync");
    w.high_io_s += ram_seconds_since(t);
    ++w.high_groups;
}

static void process_fullorbit_batch_low_group(
    FullOrbitBatchWorker& w,
    const FullOrbitBatchLowJob& job,
    const MaskShardLayout& shard,
    Count* authoritative_main,
    Count* authoritative_block,
    int threads,
    int zero_based_row
) {
    const uint32_t mask = job.mask;
    ck(cudaSetDevice(w.dev), "fullorbit-batch low set device");
    const Code main_n = shard.main_group_size[mask];
    const Code block_n = shard.block_group_size[mask];
    if (!main_n && !block_n) return;
    maskshard_configure_low_group(mask);
    Count* mainv = authoritative_main + shard.main_base[mask];
    Count* blockv = authoritative_block + shard.block_base[mask];
    const int bm = int(std::min<Code>(65535, (main_n + threads - 1) / threads));
#if defined(MASKSHARD_LOW_BLOCK_ORBIT_TIGHT_LAUNCH) && !defined(MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT_LAUNCH)
    const int bd = int(std::min<Code>(65535, (block_n + threads - 1) / threads));
#endif
#ifdef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT_LAUNCH
    const Code low_orbit_n = maskshard_loworbit_compact_total_current_device();
    const int low_orbit_cap = std::min(zero_based_row + 1, TARGET_W / 2);
    if (low_orbit_cap == TARGET_W / 2 && low_orbit_n != block_n) {
        std::cerr << "fullorbit-batch exact LOW orbit full-cap mismatch mask="
                  << mask << " got=" << low_orbit_n
                  << " expected=" << block_n << '\n';
        std::exit(212);
    }
    const int bd_low_orbit = low_orbit_n
        ? int(std::min<Code>(65535, (low_orbit_n + threads - 1) / threads))
        : 0;
#endif
#ifdef MASKSHARD_LOW_CLOSURE_COLS
    const int warps_per_block = (threads + 31) / 32;
#else
    (void)zero_based_row;
#endif

    for (int p = LOW_LUT_K; p >= 1; --p) {
        auto t = std::chrono::steady_clock::now();
#ifdef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT_LAUNCH
        if (bd_low_orbit)
            maskshard_main_block_loworbit_kernel<<<bd_low_orbit, threads>>>(mainv, blockv, main_n, p);
#elif defined(MASKSHARD_LOW_BLOCK_ORBIT_TIGHT_LAUNCH)
        if (block_n)
            maskshard_main_block_loworbit_kernel<<<bd, threads>>>(mainv, blockv, main_n, p);
#else
        if (main_n)
            maskshard_main_block_loworbit_kernel<<<bm, threads>>>(mainv, blockv, main_n, p);
#endif
        ck(cudaGetLastError(), "fullorbit-batch low orbit");
        ck(cudaDeviceSynchronize(), "fullorbit-batch low orbit sync");
        w.low_orbit_s += ram_seconds_since(t);

        t = std::chrono::steady_clock::now();
#ifdef MASKSHARD_LOW_CLOSURE_COLS
        const uint32_t pi = uint32_t(LOW_LUT_K - p);
        Code closure_tasks = job.closure_tasks[size_t(pi)];
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
        const int closure_cap = std::min(zero_based_row + 1, (TARGET_W + 1) / 2);
        closure_tasks = job.closure_tasks_by_cap[size_t(pi)][size_t(closure_cap)];
        if (closure_cap == (TARGET_W + 1) / 2
            && closure_tasks != job.closure_tasks[size_t(pi)]) {
            std::cerr << "fullorbit-batch exact LOW closure full-cap mismatch mask="
                      << mask << " pi=" << pi
                      << " got=" << closure_tasks
                      << " expected=" << job.closure_tasks[size_t(pi)] << '\n';
            std::exit(210);
        }
#else
        (void)zero_based_row;
#endif
        const int bc = closure_tasks
            ? int(std::min<Code>(65535,
                (closure_tasks + Code(warps_per_block) - 1) / Code(warps_per_block)))
            : 0;
        if (bc)
            maskshard_main_lowdesc_closure_inplace_kernel<<<bc, threads>>>(
                mainv, blockv, main_n, p);
#else
        if (main_n)
            maskshard_main_lowdesc_closure_inplace_kernel<<<bm, threads>>>(
                mainv, blockv, main_n, p);
#endif
        ck(cudaGetLastError(), "fullorbit-batch low closure");
        ck(cudaDeviceSynchronize(), "fullorbit-batch low closure sync");
        w.low_closure_s += ram_seconds_since(t);
    }
    ++w.low_groups;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    const int requested = argc > 2 ? std::atoi(argv[2]) : 8;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    std::vector<Count> mods;
    for (int i = 4; i < argc; ++i) {
        const unsigned long long raw = std::strtoull(argv[i], nullptr, 10);
        if (raw < 2 || raw > 0xffffffffULL) {
            std::cerr << "fullorbit-batch modulus must be in [2,4294967295], got " << raw << '\n';
            return 1;
        }
        mods.push_back(Count(raw));
    }
    if (mods.empty()) mods.push_back(4294967291u);
    if (n + 1 != TARGET_W || n < 2 || TARGET_W > MAXW) {
        std::cerr << "fullorbit-batch binary specialized for n=" << TARGET_W - 1 << '\n';
        return 1;
    }
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) {
        std::cerr << "fullorbit-batch requires LOW+HIGH=W-1\n";
        return 1;
    }

    const auto setup0 = std::chrono::steady_clock::now();
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost high_desc = build_high_descriptors(storage, layout);
    LowDescHost low_desc = build_low_descriptors(storage, layout);
#ifdef MASKSHARD_LOW_CLOSURE_COLS
    MaskShardLowClosureColsHost low_closure =
        build_maskshard_low_closure_cols(storage, layout, low_desc);
#endif
    const double highdesc_mib = double(
        (high_desc.main_desc.size() + high_desc.block_desc.size()) * sizeof(uint32_t)
    ) / double(1ULL << 20);
    const double lowdesc_mib = double(
        (low_desc.main_desc.size() + low_desc.block_desc.size()) * sizeof(uint32_t)
    ) / double(1ULL << 20);

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "fullorbit-batch device count");
    const int ngpu = std::min({requested, visible, 8});
    if (ngpu < 1) return 2;
    int peers = 0;
    for (int a = 0; a < ngpu; ++a) for (int b = 0; b < ngpu; ++b) if (a != b) {
        int can = 0;
        ck(cudaDeviceCanAccessPeer(&can, a, b), "fullorbit-batch can peer");
        if (!can) continue;
        ck(cudaSetDevice(a), "fullorbit-batch peer set");
        cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else ck(e, "fullorbit-batch enable peer");
        ++peers;
    }
    if (ngpu > 1 && peers != ngpu * (ngpu - 1)) {
        std::cerr << "fullorbit-batch requires full P2P: " << peers << '/' << ngpu * (ngpu - 1) << '\n';
        return 3;
    }

    MaskShardLayout shard = build_high_mask_shard_layout(storage, layout, ngpu);
    const auto high_route = build_fullorbit_batch_high_route(storage);
    report_high_mask_shard_layout(shard);

    Count* mp[8]{};
    Count* bp[8]{};
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "fullorbit-batch auth set");
        if (shard.gpu_main[d])
            ck(cudaMalloc(&mp[d], size_t(shard.gpu_main[d]) * sizeof(Count)), "fullorbit-batch auth main");
        if (shard.gpu_block[d])
            ck(cudaMalloc(&bp[d], size_t(shard.gpu_block[d]) * sizeof(Count)), "fullorbit-batch auth block");
    }

    std::vector<StorageDeviceTables> factor_dev(ngpu);
    std::vector<HighDescDeviceTables> highdesc_dev(ngpu);
    std::vector<LowDescDeviceTables> lowdesc_dev(ngpu);
#ifdef MASKSHARD_LOW_CLOSURE_COLS
    std::vector<MaskShardLowClosureColsDeviceTables> lowclosure_dev(ngpu);
#endif
    std::vector<MaskShardDeviceMeta> shard_dev(ngpu);
    std::vector<FullOrbitBatchWorker> workers(ngpu);
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "fullorbit-batch table set");
        factor_dev[d].install(storage, G_FACTOR);
        highdesc_dev[d].install(high_desc);
        lowdesc_dev[d].install(low_desc);
#ifdef MASKSHARD_LOW_CLOSURE_COLS
        lowclosure_dev[d].install(low_closure);
#endif
        shard_dev[d].install(d, shard, layout, high_route, mp, bp);
        workers[d].init(d);
    }

    const MateID init = MateID(R) << (2 * (TARGET_W - 1));
    const FullOrbitBatchAddress ia = fullorbit_batch_main_address_host(init, storage, layout, shard);
    const FullOrbitBatchAddress oa = fullorbit_batch_main_address_host(MateID(R), storage, layout, shard);
    const auto high_jobs = build_fullorbit_batch_high_jobs(&high_desc);
#ifdef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT
    maskshard_prepare_highclosure_rowdepth_compact(high_desc, ngpu);
#endif
    std::vector<std::vector<FullOrbitBatchLowJob>> low_jobs(ngpu);
    for (uint32_t mask = 0; mask < shard.masks; ++mask) {
        FullOrbitBatchLowJob job;
        job.mask = mask;
#ifdef MASKSHARD_LOW_CLOSURE_COLS
        const auto mb = make_factor_main_blocks(false, mask);
        for (int pi = 0; pi < LOW_LUT_K; ++pi) {
            Code tasks = 0;
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
            constexpr int FULL_CAP = (TARGET_W + 1) / 2;
            constexpr int CAP_STRIDE = FULL_CAP + 1;
            std::array<Code, FULL_CAP + 1> cap_tasks{};
#endif
            for (size_t bid = 0; bid < mb.size(); ++bid) {
                const FBlock& b = mb[bid];
                if (!b.stride || b.end == b.off) continue;
                const uint32_t a = low_closure.block_off[size_t(pi) * 65 + bid];
                const uint32_t z = low_closure.block_off[size_t(pi) * 65 + bid + 1];
                const uint32_t chunks = (z - a + 31u) >> 5;
                const Code rows = (b.end - b.off) / b.stride;
                tasks += rows * Code(chunks);
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
                for (int cap = 1; cap <= FULL_CAP; ++cap) {
                    const uint32_t selected = low_closure.compact_active_count[
                        (size_t(pi) * 65 + bid) * CAP_STRIDE + size_t(cap)];
                    if (!selected) continue;
                    const uint32_t active_rows = low_closure.high_active_count[
                        (size_t(mask) * (HIGH_LUT_K + 2) + b.he) * CAP_STRIDE
                        + size_t(cap)];
                    cap_tasks[size_t(cap)] += Code(active_rows)
                        * Code((selected + 31u) >> 5);
                }
#endif
            }
            job.closure_tasks[size_t(pi)] = tasks;
#ifdef MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
            job.closure_tasks_by_cap[size_t(pi)] = cap_tasks;
            if (cap_tasks[size_t(FULL_CAP)] != tasks) {
                std::cerr << "fullorbit-batch exact LOW closure setup full-cap mismatch mask="
                          << mask << " pi=" << pi
                          << " got=" << cap_tasks[size_t(FULL_CAP)]
                          << " expected=" << tasks << '\n';
                std::exit(211);
            }
#endif
        }
#endif
        low_jobs[shard.owner[mask]].push_back(job);
    }
    for (auto& v : low_jobs)
        std::sort(v.begin(), v.end(), [&](const FullOrbitBatchLowJob& x,
                                         const FullOrbitBatchLowJob& y) {
            return shard.main_group_size[x.mask] + shard.block_group_size[x.mask]
                 > shard.main_group_size[y.mask] + shard.block_group_size[y.mask];
        });
    const double setup_s = ram_seconds_since(setup0);
    std::cerr << "fullorbit-batch setup_s=" << setup_s
              << " residues=" << mods.size()
              << " highdesc_mib_per_gpu=" << highdesc_mib
              << " lowdesc_mib_per_gpu=" << lowdesc_mib << '\n';

    for (size_t ri = 0; ri < mods.size(); ++ri) {
        const Count mod = mods[ri];
        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "fullorbit-batch residue reset device");
            ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "fullorbit-batch set modulus");
            if (mp[d]) ck(cudaMemset(mp[d], 0, size_t(shard.gpu_main[d]) * sizeof(Count)), "fullorbit-batch zero main");
            if (bp[d]) ck(cudaMemset(bp[d], 0, size_t(shard.gpu_block[d]) * sizeof(Count)), "fullorbit-batch zero block");
            ck(cudaDeviceSynchronize(), "fullorbit-batch reset sync");
            workers[d].reset_stats();
        }
        const Count one = 1;
        ck(cudaSetDevice(ia.owner), "fullorbit-batch init device");
        ck(cudaMemcpy(mp[ia.owner] + ia.offset, &one, sizeof(one), cudaMemcpyHostToDevice), "fullorbit-batch init one");

        const auto wall0 = std::chrono::steady_clock::now();
        for (int row = 0; row < TARGET_W; ++row) {
#ifdef MASKSHARD_ROW_DEPTH_FBLOCK_IO
            for (int d = 0; d < ngpu; ++d) {
                ck(cudaSetDevice(d), "fullorbit-batch row-depth set device");
                maskshard_set_row_depth_fblock_io_row(row);
            }
#endif
            std::atomic<size_t> next_high{0};
            std::vector<std::thread> ts;
            ts.reserve(ngpu);
            for (int d = 0; d < ngpu; ++d) {
                ts.emplace_back([&, d] {
                    for (;;) {
                        const size_t q = next_high.fetch_add(1, std::memory_order_relaxed);
                        if (q >= high_jobs.size()) break;
                        process_fullorbit_batch_high_job(
                            workers[d], high_jobs[q], threads, row);
                    }
                });
            }
            for (auto& t : ts) t.join();

            ts.clear();
            for (int d = 0; d < ngpu; ++d) {
                ts.emplace_back([&, d] {
                    for (const FullOrbitBatchLowJob& job : low_jobs[d])
                        process_fullorbit_batch_low_group(
                            workers[d], job, shard, mp[d], bp[d], threads, row);
                });
            }
            for (auto& t : ts) t.join();
            std::cerr << "fullorbit-batch residue " << ri + 1 << '/' << mods.size()
                      << " row " << row + 1 << '/' << TARGET_W << '\n';
        }
        const double wall_s = ram_seconds_since(wall0);

        Count answer = 0;
        ck(cudaSetDevice(oa.owner), "fullorbit-batch answer device");
        ck(cudaMemcpy(&answer, mp[oa.owner] + oa.offset, sizeof(answer), cudaMemcpyDeviceToHost), "fullorbit-batch answer");

        double high_io = 0, high_orbit = 0, high_closure = 0;
        double low_orbit = 0, low_closure = 0;
        uint64_t high_groups_done = 0, low_groups_done = 0;
        size_t max_scratch = 0;
        for (const auto& w : workers) {
            high_io += w.high_io_s;
            high_orbit += w.high_orbit_s;
            high_closure += w.high_closure_s;
            low_orbit += w.low_orbit_s;
            low_closure += w.low_closure_s;
            high_groups_done += w.high_groups;
            low_groups_done += w.low_groups;
            max_scratch = std::max(max_scratch, w.cap);
        }

        std::cout << "backend=b300-factorized-maskshard-v0.4-fullorbit-batch"
                  << " n=" << n
                  << " gpus=" << ngpu
                  << " residue=" << answer
                  << " modulus=" << mod
                  << " residue_index=" << ri
                  << " residues_total=" << mods.size()
                  << " setup_s=" << setup_s
                  << " wall_s=" << wall_s
                  << " high_io_sum_s=" << high_io
                  << " high_orbit_sum_s=" << high_orbit
                  << " high_closure_sum_s=" << high_closure
                  << " low_orbit_sum_s=" << low_orbit
                  << " low_closure_sum_s=" << low_closure
                  << " high_identity_sum_s=0 low_identity_sum_s=0 low_copyback_sum_s=0"
                  << " highdesc_mib_per_gpu=" << highdesc_mib
                  << " lowdesc_mib_per_gpu=" << lowdesc_mib
                  << " high_groups=" << high_groups_done
                  << " low_groups=" << low_groups_done
                  << " max_scratch_gib=" << double(max_scratch) / double(1ULL << 30)
                  << '\n';
    }

    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "fullorbit-batch cleanup set");
        workers[d].release();
        shard_dev[d].release();
#ifdef MASKSHARD_LOW_CLOSURE_COLS
        lowclosure_dev[d].release();
#endif
        lowdesc_dev[d].release();
        highdesc_dev[d].release();
        factor_dev[d].release();
        if (mp[d]) cudaFree(mp[d]);
        if (bp[d]) cudaFree(bp[d]);
    }
#ifdef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT
    maskshard_release_highclosure_rowdepth_compact();
#endif
    return 0;
}
