#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/ramstream32_high_orbit.cuh"
#include "../gridfp/ramstream32_low_orbit_device.cuh"
#include "../gridfp/ramstream32_b300_compact_io.cuh"
#include "../gridfp/ramstream32_b300_rowkernels.cuh"

struct ResidentGroupRT {
    uint32_t id = 0;
    uint32_t mask = 0;
    bool fix_low = false;
    Code main_size = 0;
    Code block_size = 0;
    std::vector<FBlock> main_blocks;
    std::vector<FBlock> block_blocks;
    Code work = 0;
};

struct ResidentWindowRT {
    int p_hi = 0, p_lo = 0;
    bool fix_low = false;
    std::vector<ResidentGroupRT> groups;
    Code max_main = 0, max_block = 0;
    size_t max_bytes = 0;
};

static ResidentWindowRT build_resident_window_rt(bool high_window) {
    ResidentWindowRT w;
    w.p_hi = high_window ? TARGET_W - 1 : LOW_LUT_K;
    w.p_lo = high_window ? LOW_LUT_K + 1 : 1;
    w.fix_low = high_window;
    std::vector<int> fp = window_candidates(TARGET_W, w.p_hi, w.p_lo);
    if (fp.size() >= 31) std::exit(300);
    uint32_t ng = 1u << fp.size();
    w.groups.reserve(ng);
    for (uint32_t g = 0; g < ng; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(TARGET_W, w.p_hi, w.p_lo, fp, g, mf, mo, bf, bo);
        GroupSpec ms = make_spec(TARGET_W, mf, mo);
        GroupSpec ds = make_spec(TARGET_W - 1, bf, bo);
        uint32_t mask = high_window
            ? (mo & ((1u << LOW_LUT_K) - 1u))
            : ((mo >> (LOW_LUT_K + 1)) & ((1u << HIGH_LUT_K) - 1u));
        auto mb = make_factor_main_blocks(high_window, mask);
        auto db = make_factor_block_blocks(high_window, mask);
        if (mb.empty() || db.empty() || mb.back().end != ms.size || db.back().end != ds.size)
            std::exit(301);
        ResidentGroupRT x;
        x.id = g;
        x.mask = mask;
        x.fix_low = high_window;
        x.main_size = ms.size;
        x.block_size = ds.size;
        x.main_blocks = std::move(mb);
        x.block_blocks = std::move(db);
        x.work = ms.size + ds.size;
        w.max_main = std::max(w.max_main, x.main_size);
        w.max_block = std::max(w.max_block, x.block_size);
        w.max_bytes = std::max(w.max_bytes,
            size_t(x.main_size + x.block_size) * sizeof(Count));
        w.groups.push_back(std::move(x));
    }
    std::sort(w.groups.begin(), w.groups.end(), [](const ResidentGroupRT& a, const ResidentGroupRT& b) {
        return a.work > b.work;
    });
    return w;
}

struct ResidentGpuRT {
    int dev = 0;
    Count* main_shard = nullptr;
    Count* block_shard = nullptr;
    Code main_count = 0, block_count = 0;

    uint8_t* scratch = nullptr;
    Count* main_local = nullptr;
    Count* block_local = nullptr;
    Code main_cap = 0, block_cap = 0;
    size_t scratch_bytes = 0;

    BidescMaskDeviceTables mask_tables;
    LowDescDeviceTables lowdesc_tables;
    HighDescDeviceTables highdesc_tables;
    LowOrbitDeviceTables loworbit_tables;
    HighOrbitDeviceTables highorbit_tables;
    CompactCanonicalDeviceTables canonical_tables;

    double gather_s = 0, kernel_s = 0, scatter_s = 0;
    uint64_t groups = 0, launches = 0;

    void allocate_large(Code mc, Code bc, Code max_m, Code max_b) {
        main_count = mc;
        block_count = bc;
        main_cap = max_m;
        block_cap = max_b;
        ck(cudaSetDevice(dev), "rowtile set alloc device");
        size_t free0 = 0, total0 = 0;
        ck(cudaMemGetInfo(&free0, &total0), "rowtile meminfo before");
        if (main_count)
            ck(cudaMalloc(&main_shard, size_t(main_count) * sizeof(Count)), "rowtile main shard");
        if (block_count)
            ck(cudaMalloc(&block_shard, size_t(block_count) * sizeof(Count)), "rowtile block shard");
        if (main_shard)
            ck(cudaMemset(main_shard, 0, size_t(main_count) * sizeof(Count)), "rowtile zero main shard");
        if (block_shard)
            ck(cudaMemset(block_shard, 0, size_t(block_count) * sizeof(Count)), "rowtile zero block shard");

        auto al = [](size_t n) { return (n + 255) & ~size_t(255); };
        size_t mb = al(size_t(main_cap) * sizeof(Count));
        size_t db = al(size_t(block_cap) * sizeof(Count));
        scratch_bytes = mb + db;
        if (scratch_bytes) ck(cudaMalloc(&scratch, scratch_bytes), "rowtile scratch");
        main_local = reinterpret_cast<Count*>(scratch);
        block_local = reinterpret_cast<Count*>(scratch + mb);

        size_t free1 = 0, total1 = 0;
        ck(cudaMemGetInfo(&free1, &total1), "rowtile meminfo after large");
        std::cerr << "rowtile gpu=" << dev
                  << " total_gib=" << double(total0) / double(1ULL << 30)
                  << " free_before_gib=" << double(free0) / double(1ULL << 30)
                  << " free_after_large_gib=" << double(free1) / double(1ULL << 30)
                  << " auth_gib="
                  << double((size_t(main_count) + size_t(block_count)) * sizeof(Count)) / double(1ULL << 30)
                  << " scratch_gib=" << double(scratch_bytes) / double(1ULL << 30) << '\n';
    }

    void install_tables(
        const LowDescHost& lowdesc, const HighDescHost& highdesc,
        const LowOrbitHost& loworbit, const HighOrbitHost& highorbit,
        const CompactCanonicalRankHost& canonical, Count mod
    ) {
        ck(cudaSetDevice(dev), "rowtile set table device");
        mask_tables.install(G_FACTOR);
        lowdesc_tables.install(lowdesc);
        highdesc_tables.install(highdesc);
        loworbit_tables.install(loworbit);
        highorbit_tables.install(highorbit);
        canonical_tables.install(canonical);
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "rowtile modulus");
        size_t freeb = 0, totalb = 0;
        ck(cudaMemGetInfo(&freeb, &totalb), "rowtile meminfo after tables");
        std::cerr << "rowtile gpu=" << dev
                  << " free_after_tables_gib=" << double(freeb) / double(1ULL << 30) << '\n';
    }

    void release() {
        ck(cudaSetDevice(dev), "rowtile set release device");
        canonical_tables.release();
        highorbit_tables.release();
        loworbit_tables.release();
        highdesc_tables.release();
        lowdesc_tables.release();
        mask_tables.release();
        if (scratch) cudaFree(scratch);
        if (main_shard) cudaFree(main_shard);
        if (block_shard) cudaFree(block_shard);
        scratch = nullptr;
        main_local = block_local = nullptr;
        main_shard = block_shard = nullptr;
    }
};

static void enable_full_peer_mesh_rt(int ngpu) {
    for (int a = 0; a < ngpu; ++a) {
        ck(cudaSetDevice(a), "rowtile peer set source");
        for (int b = 0; b < ngpu; ++b) if (a != b) {
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, a, b), "rowtile cudaDeviceCanAccessPeer");
            if (!can) {
                std::cerr << "GPU " << a << " cannot peer-access GPU " << b << '\n';
                std::exit(302);
            }
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else ck(e, "rowtile cudaDeviceEnablePeerAccess");
        }
    }
}

static void install_global_shards_rt(
    std::vector<std::unique_ptr<ResidentGpuRT>>& gpu,
    Code main_chunk, Code block_chunk
) {
    int ngpu = int(gpu.size());
    std::array<Count*, MAXGPU> mp{}, bp{};
    for (int g = 0; g < ngpu; ++g) {
        mp[g] = gpu[size_t(g)]->main_shard;
        bp[g] = gpu[size_t(g)]->block_shard;
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "rowtile shard symbols device");
        ck(cudaMemcpyToSymbol(D_MAIN_PTR, mp.data(), sizeof(Count*) * MAXGPU), "rowtile main ptrs");
        ck(cudaMemcpyToSymbol(D_BLOCK_PTR, bp.data(), sizeof(Count*) * MAXGPU), "rowtile block ptrs");
        ck(cudaMemcpyToSymbol(D_MAIN_CHUNK, &main_chunk, sizeof(main_chunk)), "rowtile main chunk");
        ck(cudaMemcpyToSymbol(D_BLOCK_CHUNK, &block_chunk, sizeof(block_chunk)), "rowtile block chunk");
        ck(cudaMemcpyToSymbol(D_NGPU, &ngpu, sizeof(ngpu)), "rowtile ngpu");
    }
}

static void install_group_constants_rt(const ResidentGroupRT& x) {
    int fm = int(x.main_blocks.size()), fd = int(x.block_blocks.size());
    int fl = x.fix_low ? 1 : 0;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, x.main_blocks.data(),
                          x.main_blocks.size() * sizeof(FBlock)), "rowtile group main blocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, x.block_blocks.data(),
                          x.block_blocks.size() * sizeof(FBlock)), "rowtile group block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &fm, sizeof(fm)), "rowtile group main n");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &fd, sizeof(fd)), "rowtile group block n");
    ck(cudaMemcpyToSymbol(D_F_MASK, &x.mask, sizeof(x.mask)), "rowtile group mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fl, sizeof(fl)), "rowtile group mode");
}

static std::pair<uint32_t,uint32_t> resident_group_tile_counts_rt(const ResidentGroupRT& x) {
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
    ResidentGpuRT& c, const ResidentGroupRT& x,
    int p_hi, int p_lo, int threads
) {
    if (!x.main_size && !x.block_size) return;
    ck(cudaSetDevice(c.dev), "rowtile group device");
    install_group_constants_rt(x);
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
    if (x.fix_low && bt)
        compact_tile_scatter_block_kernel<<<bt, threads>>>(c.block_local);
    ck(cudaGetLastError(), "rowtile scatter launch");
    ck(cudaDeviceSynchronize(), "rowtile scatter sync");
    c.scatter_s += ram_seconds_since(t);
    ++c.groups;
}

static void run_resident_window_rowtile(
    std::vector<std::unique_ptr<ResidentGpuRT>>& gpu,
    const ResidentWindowRT& w, int threads
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

static void resident_window_tile_stats_rt(
    const ResidentWindowRT& w,
    uint64_t& total_main, uint64_t& total_block,
    uint32_t& max_main, uint32_t& max_block
) {
    total_main = total_block = 0;
    max_main = max_block = 0;
    for (const auto& g : w.groups) {
        auto [mt, bt] = resident_group_tile_counts_rt(g);
        total_main += mt;
        total_block += bt;
        max_main = std::max(max_main, mt);
        max_block = std::max(max_block, bt);
    }
}

static Code canonical_rank_host_rt(MateID m, int width) {
    return rank_full_suffix_host(m, width, 1);
}
static double gib_ld_rt(long double b) {
    return static_cast<double>(b / static_cast<long double>(1ULL << 30));
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
    ResidentWindowRT high = build_resident_window_rt(true);
    ResidentWindowRT low = build_resident_window_rt(false);

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
    resident_window_tile_stats_rt(high, high_mt, high_bt, high_mmax, high_bmax);
    resident_window_tile_stats_rt(low, low_mt, low_bt, low_mmax, low_bmax);

    if (plan_only) {
        std::cout << std::fixed << std::setprecision(3)
            << "backend=gridfp-b300-hbm32-compact-resident-rowtile-plan"
            << " n=" << n << " gpus=" << requested_gpus
            << " col_tile=" << CF_COL_TILE
            << " auth_total_gib="
            << gib_ld_rt(static_cast<long double>(main_n + block_n) * sizeof(Count))
            << " auth_shard_gib=" << gib_ld_rt(auth_shard_bytes)
            << " high_groups=" << high.groups.size()
            << " low_groups=" << low.groups.size()
            << " high_scratch_peak_gib=" << double(high.max_bytes) / double(1ULL << 30)
            << " low_scratch_peak_gib=" << double(low.max_bytes) / double(1ULL << 30)
            << " allocated_scratch_gib=" << gib_ld_rt(scratch_bytes)
            << " descriptor_mib=" << static_cast<double>(desc_bytes / (1 << 20))
            << " orbit_mib=" << static_cast<double>(orbit_bytes / (1 << 20))
            << " mask_mib=" << static_cast<double>(mask_bytes / (1 << 20))
            << " canonical_mib=" << static_cast<double>(canonical_bytes / (1 << 20))
            << " estimated_need_gib=" << gib_ld_rt(need)
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

    std::vector<std::unique_ptr<ResidentGpuRT>> gpu;
    gpu.reserve(ngpu);
    for (int g = 0; g < ngpu; ++g) {
        auto c = std::make_unique<ResidentGpuRT>();
        c->dev = g;
        Code moff = Code(g) * main_chunk;
        Code boff = Code(g) * block_chunk;
        Code mc = moff < main_n ? std::min(main_chunk, main_n - moff) : 0;
        Code bc = boff < block_n ? std::min(block_chunk, block_n - boff) : 0;
        c->allocate_large(mc, bc, scratch_main, scratch_block);
        gpu.push_back(std::move(c));
    }

    enable_full_peer_mesh_rt(ngpu);
    install_global_shards_rt(gpu, main_chunk, block_chunk);
    for (int g = 0; g < ngpu; ++g)
        gpu[size_t(g)]->install_tables(lowdesc, highdesc, loworbit, highorbit, canonical, mod);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = canonical_rank_host_rt(init, W);
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

    Code answer_rank = canonical_rank_host_rt(MateID(R), W);
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
