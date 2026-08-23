#include <cuda_runtime.h>

// Reuse the complete resident backend setup/allocation/topology machinery but
// keep its scalar-state executor available under private names.  This file
// supplies a second executor whose CUDA blocks are factor-row/LOW-column tiles,
// making A/B benchmarking on B300 straightforward.
#define process_resident_group process_resident_group_scalar
#define run_resident_window run_resident_window_scalar
#define main oneesan_compact_resident_scalar_unused_main
#include "oneesan_cuda_gridfp_b300_hbm32_compact_resident.cu"
#undef main
#undef run_resident_window
#undef process_resident_group

#include "../gridfp/ramstream32_b300_rowkernels.cuh"

static std::pair<uint32_t,uint32_t> resident_group_tile_counts(const ResidentGroup& x) {
    uint64_t mt = 0, bt = 0;
    for (const FBlock& b : x.main_blocks) {
        uint64_t rows = b.stride ? (b.end - b.off) / b.stride : 0;
        mt += rows * compact_tiles_per_row(b.stride);
    }
    for (const FBlock& b : x.block_blocks) {
        uint64_t rows = b.stride ? (b.end - b.off) / b.stride : 0;
        bt += rows * compact_tiles_per_row(b.stride);
    }
    if (mt > 0xffffffffULL || bt > 0xffffffffULL) std::exit(310);
    return {uint32_t(mt), uint32_t(bt)};
}

static void process_resident_group_rowtile(
    ResidentGpu& c, const ResidentGroup& x, int p_hi, int p_lo, int threads
) {
    if (!x.main_size && !x.block_size) return;
    ck(cudaSetDevice(c.dev), "rowtile group device");
    install_group_constants(x);
    auto [mt, bt] = install_compact_tile_prefixes(x.main_blocks, x.block_blocks);

    auto t = std::chrono::steady_clock::now();
    if (mt)
        compact_tile_gather_main_kernel<<<mt, threads>>>(c.main_local);
    if (x.fix_low) {
        if (x.block_size)
            ck(cudaMemset(c.block_local, 0, size_t(x.block_size) * sizeof(Count)),
               "rowtile HIGH clear blocked");
    } else if (bt) {
        compact_tile_gather_block_kernel<<<bt, threads>>>(c.block_local);
    }
    ck(cudaGetLastError(), "rowtile gather launch");
    ck(cudaDeviceSynchronize(), "rowtile gather sync");
    c.gather_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    for (int p = p_hi; p >= p_lo; --p) {
        if (!mt) continue;
        if (x.fix_low) {
            compact_tile_high_orbit_kernel<<<mt, threads>>>(c.main_local, c.block_local, p);
            compact_tile_high_closure_kernel<<<mt, threads>>>(c.main_local, c.block_local, p);
        } else {
            compact_tile_low_orbit_kernel<<<mt, threads>>>(c.main_local, c.block_local, p);
            compact_tile_low_closure_kernel<<<mt, threads>>>(c.main_local, c.block_local, p);
        }
        c.launches += 2;
    }
    ck(cudaGetLastError(), "rowtile transition launch");
    ck(cudaDeviceSynchronize(), "rowtile transition sync");
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (mt)
        compact_tile_scatter_main_kernel<<<mt, threads>>>(c.main_local);
    // HIGH emits blocked values for the LOW half.  LOW p=1 consumes all of
    // them, so the LOW path skips zero scatter exactly like the scalar backend.
    if (x.fix_low && bt)
        compact_tile_scatter_block_kernel<<<bt, threads>>>(c.block_local);
    ck(cudaGetLastError(), "rowtile scatter launch");
    ck(cudaDeviceSynchronize(), "rowtile scatter sync");
    c.scatter_s += ram_seconds_since(t);
    ++c.groups;
}

static void run_resident_window_rowtile(
    std::vector<std::unique_ptr<ResidentGpu>>& gpu,
    const ResidentWindow& w, int threads
) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(gpu.size());
    for (size_t d = 0; d < gpu.size(); ++d) {
        workers.emplace_back([&, d] {
            for (;;) {
                size_t q = next.fetch_add(1, std::memory_order_relaxed);
                if (q >= w.groups.size()) break;
                process_resident_group_rowtile(*gpu[d], w.groups[q], w.p_hi, w.p_lo, threads);
            }
        });
    }
    for (auto& t : workers) t.join();
}

static void resident_window_tile_stats(
    const ResidentWindow& w,
    uint64_t& total_main, uint64_t& total_block,
    uint32_t& max_main, uint32_t& max_block
) {
    total_main = total_block = 0;
    max_main = max_block = 0;
    for (const auto& g : w.groups) {
        auto [mt, bt] = resident_group_tile_counts(g);
        total_main += mt;
        total_block += bt;
        max_main = std::max(max_main, mt);
        max_block = std::max(max_block, bt);
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int requested_gpus = argc > 3 ? std::max(1, std::atoi(argv[3])) : 8;
    int gpu_threads = argc > 4 ? std::max(32, std::atoi(argv[4])) : 256;
    bool plan_only = argc > 5 && std::strcmp(argv[5], "--plan-only") == 0;
    int W = n + 1;
    if (W != TARGET_W || W > MAXW || n < 2) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K + 1 != TARGET_W) return 1;
    if (requested_gpus > MAXGPU) return 2;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout logical = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, logical);
    HighDescHost highdesc = build_high_descriptors(storage, logical);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, logical, lowdesc);
    HighOrbitHost highorbit = build_high_orbit(storage, logical);
    CompactCanonicalRankHost canonical = build_compact_canonical_ranks();
    ResidentWindow high = build_resident_window(true);
    ResidentWindow low = build_resident_window(false);

    Code main_n = H_DP[W][1], block_n = H_DP[W - 1][1];
    Code main_chunk = (main_n + requested_gpus - 1) / requested_gpus;
    Code block_chunk = (block_n + requested_gpus - 1) / requested_gpus;
    Code main_shard_max = std::min(main_chunk, main_n);
    Code block_shard_max = std::min(block_chunk, block_n);
    Code scratch_main = std::max(high.max_main, low.max_main);
    Code scratch_block = std::max(high.max_block, low.max_block);

    long double auth_shard_bytes =
        static_cast<long double>(main_shard_max + block_shard_max) * sizeof(Count);
    long double scratch_bytes =
        static_cast<long double>(scratch_main + scratch_block) * sizeof(Count);
    long double desc_bytes = static_cast<long double>(
        lowdesc.main_desc.size() + lowdesc.block_desc.size()
      + highdesc.main_desc.size() + highdesc.block_desc.size()) * sizeof(uint32_t);
    long double orbit_bytes = static_cast<long double>(
        loworbit.rec.size() + highorbit.rec.size()) * sizeof(uint64_t);
    long double mask_bytes = static_cast<long double>(
        G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
      + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t);
    long double canonical_bytes = static_cast<long double>(
        canonical.low_mask_all_rank.size() + canonical.high_mask_all_rank.size()) * sizeof(uint32_t)
        + static_cast<long double>(G_FACTOR.high_main_base.size() + G_FACTOR.high_block_base.size())
          * sizeof(Code);
    long double need = auth_shard_bytes + scratch_bytes + desc_bytes + orbit_bytes
                     + mask_bytes + canonical_bytes;

    uint64_t high_mt, high_bt, low_mt, low_bt;
    uint32_t high_mmax, high_bmax, low_mmax, low_bmax;
    resident_window_tile_stats(high, high_mt, high_bt, high_mmax, high_bmax);
    resident_window_tile_stats(low, low_mt, low_bt, low_mmax, low_bmax);

    if (plan_only) {
        std::cout << std::fixed << std::setprecision(3)
            << "backend=gridfp-b300-hbm32-compact-resident-rowtile-plan"
            << " n=" << n << " gpus=" << requested_gpus
            << " col_tile=" << CF_COL_TILE
            << " auth_total_gib="
            << gib_ld(static_cast<long double>(main_n + block_n) * sizeof(Count))
            << " auth_shard_gib=" << gib_ld(auth_shard_bytes)
            << " high_groups=" << high.groups.size()
            << " low_groups=" << low.groups.size()
            << " high_scratch_peak_gib=" << double(high.max_bytes) / double(1ULL << 30)
            << " low_scratch_peak_gib=" << double(low.max_bytes) / double(1ULL << 30)
            << " allocated_scratch_gib=" << gib_ld(scratch_bytes)
            << " descriptor_mib=" << static_cast<double>(desc_bytes / (1 << 20))
            << " orbit_mib=" << static_cast<double>(orbit_bytes / (1 << 20))
            << " mask_mib=" << static_cast<double>(mask_bytes / (1 << 20))
            << " canonical_mib=" << static_cast<double>(canonical_bytes / (1 << 20))
            << " estimated_need_gib=" << gib_ld(need)
            << " high_main_tiles_total=" << high_mt
            << " high_block_tiles_total=" << high_bt
            << " high_main_tiles_max_group=" << high_mmax
            << " high_block_tiles_max_group=" << high_bmax
            << " low_main_tiles_total=" << low_mt
            << " low_block_tiles_total=" << low_bt
            << " low_main_tiles_max_group=" << low_mmax
            << " low_block_tiles_max_group=" << low_bmax
            << '\n';
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "rowtile cudaGetDeviceCount");
    if (visible < requested_gpus) {
        std::cerr << "need " << requested_gpus << " GPUs, visible=" << visible << '\n';
        return 3;
    }
    int ngpu = requested_gpus;

    std::vector<std::unique_ptr<ResidentGpu>> gpu;
    gpu.reserve(ngpu);
    for (int g = 0; g < ngpu; ++g) {
        auto c = std::make_unique<ResidentGpu>();
        c->dev = g;
        Code moff = Code(g) * main_chunk;
        Code boff = Code(g) * block_chunk;
        Code mc = moff < main_n ? std::min(main_chunk, main_n - moff) : 0;
        Code bc = boff < block_n ? std::min(block_chunk, block_n - boff) : 0;
        c->allocate_large(mc, bc, scratch_main, scratch_block);
        gpu.push_back(std::move(c));
    }

    enable_full_peer_mesh(ngpu);
    install_global_shards(gpu, main_chunk, block_chunk);
    for (int g = 0; g < ngpu; ++g)
        gpu[size_t(g)]->install_tables(lowdesc, highdesc, loworbit, highorbit, canonical, mod);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = canonical_rank_host(init, W);
    int init_owner = std::min<int>(ngpu - 1, int(init_rank / main_chunk));
    Code init_local = init_rank - Code(init_owner) * main_chunk;
    Count one = 1;
    ck(cudaSetDevice(init_owner), "rowtile init owner");
    ck(cudaMemcpy(gpu[size_t(init_owner)]->main_shard + init_local, &one, sizeof(one),
                  cudaMemcpyHostToDevice), "rowtile init state");

    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        run_resident_window_rowtile(gpu, high, gpu_threads);
        run_resident_window_rowtile(gpu, low, gpu_threads);
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "rowtile clear block device");
            if (gpu[size_t(g)]->block_count)
                ck(cudaMemset(gpu[size_t(g)]->block_shard, 0,
                              size_t(gpu[size_t(g)]->block_count) * sizeof(Count)),
                   "rowtile row block clear");
        }
        std::cerr << "rowtile row " << row + 1 << '/' << W << '\n';
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "rowtile final sync device");
        ck(cudaDeviceSynchronize(), "rowtile final sync");
    }
    double wall_s = ram_seconds_since(wall0);

    Code answer_rank = canonical_rank_host(MateID(R), W);
    int ao = std::min<int>(ngpu - 1, int(answer_rank / main_chunk));
    Code al = answer_rank - Code(ao) * main_chunk;
    Count answer = 0;
    ck(cudaSetDevice(ao), "rowtile answer device");
    ck(cudaMemcpy(&answer, gpu[size_t(ao)]->main_shard + al, sizeof(answer),
                  cudaMemcpyDeviceToHost), "rowtile answer copy");

    double max_gather = 0, max_kernel = 0, max_scatter = 0;
    uint64_t groups = 0, launches = 0;
    for (const auto& c : gpu) {
        max_gather = std::max(max_gather, c->gather_s);
        max_kernel = std::max(max_kernel, c->kernel_s);
        max_scatter = std::max(max_scatter, c->scatter_s);
        groups += c->groups;
        launches += c->launches;
    }
    std::cout << "backend=gridfp-b300-hbm32-compact-resident-rowtile"
              << " n=" << n << " residue=" << answer << " modulus=" << mod
              << " gpus=" << ngpu << " groups=" << groups << " launches=" << launches
              << " gather_max_s=" << max_gather << " kernel_max_s=" << max_kernel
              << " scatter_max_s=" << max_scatter << " wall_s=" << wall_s << '\n';

    for (auto& c : gpu) c->release();
    return 0;
}
