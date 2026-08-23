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
#include "../gridfp/ramstream32_b300_compact_io.cuh"
#include "../gridfp/ramstream32_b300_rowkernels.cuh"
#include "../gridfp/ramstream32_b300_high_warp.cuh"
#include "../gridfp/ramstream32_b300_sparse_actions.cuh"
#include "../gridfp/ramstream32_b300_sparse_highwarp.cuh"

struct SparseResidentGroup {
    uint32_t mask = 0;
    bool fix_low = false;
    Code main_size = 0;
    Code block_size = 0;
    std::vector<FBlock> main_blocks;
    std::vector<FBlock> block_blocks;
    Code work = 0;
};

struct SparseResidentWindow {
    int p_hi = 0, p_lo = 0;
    bool fix_low = false;
    std::vector<SparseResidentGroup> groups;
    Code max_main = 0, max_block = 0;
    size_t max_bytes = 0;
};

static SparseResidentWindow build_sparse_resident_window(bool high_window) {
    SparseResidentWindow w;
    w.p_hi = high_window ? TARGET_W - 1 : LOW_LUT_K;
    w.p_lo = high_window ? LOW_LUT_K + 1 : 1;
    w.fix_low = high_window;
    std::vector<int> fp = window_candidates(TARGET_W, w.p_hi, w.p_lo);
    if (fp.size() >= 31) std::exit(370);
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
            std::exit(371);
        SparseResidentGroup x;
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
    std::sort(w.groups.begin(), w.groups.end(), [](const SparseResidentGroup& a, const SparseResidentGroup& b) {
        return a.work > b.work;
    });
    return w;
}

struct SparseResidentGpu {
    int dev = 0;
    Count* main_shard = nullptr;
    Count* block_shard = nullptr;
    Code main_count = 0, block_count = 0;
    uint8_t* scratch = nullptr;
    Count* main_local = nullptr;
    Count* block_local = nullptr;
    size_t scratch_bytes = 0;

    BidescMaskDeviceTables mask_tables;
    B300SparseActionsDeviceTables sparse_tables;
    CompactCanonicalDeviceTables canonical_tables;

    double gather_s = 0, kernel_s = 0, scatter_s = 0;
    uint64_t groups = 0, launches = 0;

    void allocate_large(Code mc, Code bc, Code max_m, Code max_b) {
        main_count = mc;
        block_count = bc;
        ck(cudaSetDevice(dev), "sparse resident set alloc device");
        size_t free0 = 0, total0 = 0;
        ck(cudaMemGetInfo(&free0, &total0), "sparse resident meminfo before");
        if (main_count)
            ck(cudaMalloc(&main_shard, size_t(main_count) * sizeof(Count)), "sparse resident main shard");
        if (block_count)
            ck(cudaMalloc(&block_shard, size_t(block_count) * sizeof(Count)), "sparse resident block shard");
        if (main_shard)
            ck(cudaMemset(main_shard, 0, size_t(main_count) * sizeof(Count)), "sparse resident zero main");
        if (block_shard)
            ck(cudaMemset(block_shard, 0, size_t(block_count) * sizeof(Count)), "sparse resident zero block");

        auto al = [](size_t n) { return (n + 255) & ~size_t(255); };
        size_t mb = al(size_t(max_m) * sizeof(Count));
        size_t db = al(size_t(max_b) * sizeof(Count));
        scratch_bytes = mb + db;
        if (scratch_bytes) ck(cudaMalloc(&scratch, scratch_bytes), "sparse resident scratch");
        main_local = reinterpret_cast<Count*>(scratch);
        block_local = reinterpret_cast<Count*>(scratch + mb);
        size_t free1 = 0, total1 = 0;
        ck(cudaMemGetInfo(&free1, &total1), "sparse resident meminfo after large");
        std::cerr << "sparse resident gpu=" << dev
                  << " total_gib=" << double(total0) / double(1ULL << 30)
                  << " free_before_gib=" << double(free0) / double(1ULL << 30)
                  << " free_after_large_gib=" << double(free1) / double(1ULL << 30)
                  << " auth_gib="
                  << double((size_t(main_count) + size_t(block_count)) * sizeof(Count)) / double(1ULL << 30)
                  << " scratch_gib=" << double(scratch_bytes) / double(1ULL << 30) << '\n';
    }

    void install_tables(
        const B300SparseActionsHost& sparse,
        const CompactCanonicalRankHost& canonical,
        Count mod
    ) {
        ck(cudaSetDevice(dev), "sparse resident table device");
        mask_tables.install(G_FACTOR);
        sparse_tables.install(sparse);
        canonical_tables.install(canonical);
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "sparse resident modulus");
        size_t freeb = 0, totalb = 0;
        ck(cudaMemGetInfo(&freeb, &totalb), "sparse resident meminfo after tables");
        std::cerr << "sparse resident gpu=" << dev
                  << " free_after_tables_gib=" << double(freeb) / double(1ULL << 30) << '\n';
    }

    void release() {
        ck(cudaSetDevice(dev), "sparse resident release device");
        canonical_tables.release();
        sparse_tables.release();
        mask_tables.release();
        if (scratch) cudaFree(scratch);
        if (main_shard) cudaFree(main_shard);
        if (block_shard) cudaFree(block_shard);
        scratch = nullptr;
        main_local = block_local = nullptr;
        main_shard = block_shard = nullptr;
    }
};

static void sparse_enable_peer_mesh(int ngpu) {
    for (int a = 0; a < ngpu; ++a) {
        ck(cudaSetDevice(a), "sparse peer source");
        for (int b = 0; b < ngpu; ++b) if (a != b) {
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, a, b), "sparse cudaDeviceCanAccessPeer");
            if (!can) {
                std::cerr << "GPU " << a << " cannot peer-access GPU " << b << '\n';
                std::exit(372);
            }
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else ck(e, "sparse cudaDeviceEnablePeerAccess");
        }
    }
}

static void sparse_install_global_shards(
    std::vector<std::unique_ptr<SparseResidentGpu>>& gpu,
    Code main_chunk, Code block_chunk
) {
    int ngpu = int(gpu.size());
    std::array<Count*, MAXGPU> mp{}, bp{};
    for (int g = 0; g < ngpu; ++g) {
        mp[g] = gpu[size_t(g)]->main_shard;
        bp[g] = gpu[size_t(g)]->block_shard;
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "sparse shard symbol device");
        ck(cudaMemcpyToSymbol(D_MAIN_PTR, mp.data(), sizeof(Count*) * MAXGPU), "sparse main ptrs");
        ck(cudaMemcpyToSymbol(D_BLOCK_PTR, bp.data(), sizeof(Count*) * MAXGPU), "sparse block ptrs");
        ck(cudaMemcpyToSymbol(D_MAIN_CHUNK, &main_chunk, sizeof(main_chunk)), "sparse main chunk");
        ck(cudaMemcpyToSymbol(D_BLOCK_CHUNK, &block_chunk, sizeof(block_chunk)), "sparse block chunk");
        ck(cudaMemcpyToSymbol(D_NGPU, &ngpu, sizeof(ngpu)), "sparse ngpu");
    }
}

static void sparse_install_group(const SparseResidentGroup& x) {
    int fm = int(x.main_blocks.size()), fd = int(x.block_blocks.size());
    int fl = x.fix_low ? 1 : 0;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, x.main_blocks.data(),
                          x.main_blocks.size() * sizeof(FBlock)), "sparse group main blocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, x.block_blocks.data(),
                          x.block_blocks.size() * sizeof(FBlock)), "sparse group block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &fm, sizeof(fm)), "sparse group main n");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &fd, sizeof(fd)), "sparse group block n");
    ck(cudaMemcpyToSymbol(D_F_MASK, &x.mask, sizeof(x.mask)), "sparse group mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fl, sizeof(fl)), "sparse group mode");
}

static void sparse_process_group(
    SparseResidentGpu& c, const SparseResidentGroup& x,
    const B300SparseActionsHost& sparse,
    int p_hi, int p_lo, int threads
) {
    if (!x.main_size && !x.block_size) return;
    ck(cudaSetDevice(c.dev), "sparse group device");
    sparse_install_group(x);

    uint32_t main_io = 0, block_io = 0;
    if (x.fix_low) {
        auto rows = install_high_warp_row_prefixes(x.main_blocks, x.block_blocks);
        main_io = high_warp_blocks(rows.first);
        block_io = high_warp_blocks(rows.second);
    } else {
        auto tiles = install_compact_tile_prefixes(x.main_blocks, x.block_blocks);
        main_io = tiles.first;
        block_io = tiles.second;
    }

    auto t = std::chrono::steady_clock::now();
    if (main_io) {
        if (x.fix_low) high_warp_gather_main_kernel<<<main_io, 256>>>(c.main_local);
        else compact_tile_gather_main_kernel<<<main_io, threads>>>(c.main_local);
    }
    if (x.fix_low) {
        if (x.block_size)
            ck(cudaMemset(c.block_local, 0, size_t(x.block_size) * sizeof(Count)),
               "sparse HIGH clear blocked");
    } else if (block_io) {
        compact_tile_gather_block_kernel<<<block_io, threads>>>(c.block_local);
    }
    ck(cudaGetLastError(), "sparse gather launch");
    ck(cudaDeviceSynchronize(), "sparse gather sync");
    c.gather_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    for (int p = p_hi; p >= p_lo; --p) {
        if (x.fix_low) {
            uint32_t ob = b300_sparse_high_orbit_blocks(sparse, p);
            uint32_t cb = b300_sparse_high_closure_blocks(sparse, p);
            if (ob) {
                b300_sparse_highwarp_orbit_kernel<<<ob, 256>>>(c.main_local, c.block_local, p);
                ++c.launches;
            }
            if (cb) {
                b300_sparse_highwarp_closure_kernel<<<cb, 256>>>(c.main_local, c.block_local, p);
                ++c.launches;
            }
        } else {
            uint32_t on = b300_sparse_low_orbit_count(sparse, p);
            uint32_t cn = b300_sparse_low_closure_count(sparse, p);
            if (on) {
                b300_sparse_low_orbit_kernel<<<on, threads>>>(c.main_local, c.block_local, p);
                ++c.launches;
            }
            if (cn) {
                b300_sparse_low_closure_kernel<<<cn, threads>>>(c.main_local, c.block_local, p);
                ++c.launches;
            }
        }
    }
    ck(cudaGetLastError(), "sparse transition launch");
    ck(cudaDeviceSynchronize(), "sparse transition sync");
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (main_io) {
        if (x.fix_low) high_warp_scatter_main_kernel<<<main_io, 256>>>(c.main_local);
        else compact_tile_scatter_main_kernel<<<main_io, threads>>>(c.main_local);
    }
    if (x.fix_low && block_io)
        high_warp_scatter_block_kernel<<<block_io, 256>>>(c.block_local);
    ck(cudaGetLastError(), "sparse scatter launch");
    ck(cudaDeviceSynchronize(), "sparse scatter sync");
    c.scatter_s += ram_seconds_since(t);
    ++c.groups;
}

static void sparse_run_window(
    std::vector<std::unique_ptr<SparseResidentGpu>>& gpu,
    const SparseResidentWindow& w,
    const B300SparseActionsHost& sparse,
    int threads
) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(gpu.size());
    for (size_t d = 0; d < gpu.size(); ++d) {
        workers.emplace_back([&, d] {
            for (;;) {
                size_t q = next.fetch_add(1, std::memory_order_relaxed);
                if (q >= w.groups.size()) break;
                sparse_process_group(*gpu[d], w.groups[q], sparse, w.p_hi, w.p_lo, threads);
            }
        });
    }
    for (auto& t : workers) t.join();
}

static Code sparse_canonical_rank_host(MateID m, int width) {
    return rank_full_suffix_host(m, width, 1);
}
static double sparse_gib(long double b) {
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
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    HighOrbitHost highorbit = build_high_orbit(storage, layout);
    CompactCanonicalRankHost canonical = build_compact_canonical_ranks();
    B300SparseActionsHost sparse = build_b300_sparse_actions(
        layout, lowdesc, loworbit, highdesc, highorbit);
    SparseResidentWindow high = build_sparse_resident_window(true);
    SparseResidentWindow low = build_sparse_resident_window(false);

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
    long double sparse_bytes = static_cast<long double>(sparse.bytes());
    long double mask_bytes = static_cast<long double>(
        G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
      + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t);
    long double canonical_bytes = static_cast<long double>(
        canonical.low_mask_all_rank.size() + canonical.high_mask_all_rank.size()) * sizeof(uint32_t)
        + static_cast<long double>(G_FACTOR.high_main_base.size() + G_FACTOR.high_block_base.size())
          * sizeof(Code);
    long double dense_meta_bytes = static_cast<long double>(
        lowdesc.main_desc.size() + lowdesc.block_desc.size()
      + highdesc.main_desc.size() + highdesc.block_desc.size()) * sizeof(uint32_t)
      + static_cast<long double>(loworbit.rec.size() + highorbit.rec.size()) * sizeof(uint64_t);
    long double need = auth_shard_bytes + scratch_bytes + sparse_bytes
                     + mask_bytes + canonical_bytes;

    if (plan_only) {
        std::cout << std::fixed << std::setprecision(3)
            << "backend=gridfp-b300-hbm32-compact-resident-sparse-plan"
            << " n=" << n << " gpus=" << requested_gpus
            << " auth_total_gib="
            << sparse_gib(static_cast<long double>(main_n + block_n) * sizeof(Count))
            << " auth_shard_gib=" << sparse_gib(auth_shard_bytes)
            << " high_groups=" << high.groups.size()
            << " low_groups=" << low.groups.size()
            << " high_scratch_peak_gib=" << double(high.max_bytes) / double(1ULL << 30)
            << " low_scratch_peak_gib=" << double(low.max_bytes) / double(1ULL << 30)
            << " allocated_scratch_gib=" << sparse_gib(scratch_bytes)
            << " sparse_actions_mib=" << static_cast<double>(sparse_bytes / (1 << 20))
            << " dense_transition_meta_mib=" << static_cast<double>(dense_meta_bytes / (1 << 20))
            << " transition_meta_saved_mib=" << static_cast<double>((dense_meta_bytes - sparse_bytes) / (1 << 20))
            << " mask_mib=" << static_cast<double>(mask_bytes / (1 << 20))
            << " canonical_mib=" << static_cast<double>(canonical_bytes / (1 << 20))
            << " estimated_need_gib=" << sparse_gib(need)
            << " headroom_288GB_gib=" << sparse_gib(288.0e9L - need)
            << " headroom_279GB_gib=" << sparse_gib(279.0e9L - need)
            << '\n';
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "sparse resident cudaGetDeviceCount");
    if (visible < requested_gpus) {
        std::cerr << "need " << requested_gpus << " GPUs, visible=" << visible << '\n';
        return 3;
    }
    int ngpu = requested_gpus;

    // Dense construction metadata is no longer needed on host after sparse
    // action generation.  Release it before allocating the huge resident job.
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    loworbit.rec.clear(); loworbit.rec.shrink_to_fit();
    highorbit.rec.clear(); highorbit.rec.shrink_to_fit();

    std::vector<std::unique_ptr<SparseResidentGpu>> gpu;
    gpu.reserve(ngpu);
    for (int g = 0; g < ngpu; ++g) {
        auto c = std::make_unique<SparseResidentGpu>();
        c->dev = g;
        Code moff = Code(g) * main_chunk;
        Code boff = Code(g) * block_chunk;
        Code mc = moff < main_n ? std::min(main_chunk, main_n - moff) : 0;
        Code bc = boff < block_n ? std::min(block_chunk, block_n - boff) : 0;
        c->allocate_large(mc, bc, scratch_main, scratch_block);
        gpu.push_back(std::move(c));
    }

    sparse_enable_peer_mesh(ngpu);
    sparse_install_global_shards(gpu, main_chunk, block_chunk);
    for (int g = 0; g < ngpu; ++g)
        gpu[size_t(g)]->install_tables(sparse, canonical, mod);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = sparse_canonical_rank_host(init, W);
    int init_owner = std::min<int>(ngpu - 1, int(init_rank / main_chunk));
    Code init_local = init_rank - Code(init_owner) * main_chunk;
    Count one = 1;
    ck(cudaSetDevice(init_owner), "sparse resident init owner");
    ck(cudaMemcpy(gpu[size_t(init_owner)]->main_shard + init_local, &one, sizeof(one),
                  cudaMemcpyHostToDevice), "sparse resident init state");

    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        sparse_run_window(gpu, high, sparse, gpu_threads);
        sparse_run_window(gpu, low, sparse, gpu_threads);
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "sparse resident clear block device");
            if (gpu[size_t(g)]->block_count)
                ck(cudaMemset(gpu[size_t(g)]->block_shard, 0,
                              size_t(gpu[size_t(g)]->block_count) * sizeof(Count)),
                   "sparse resident row block clear");
        }
        std::cerr << "sparse resident row " << row + 1 << '/' << W << '\n';
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "sparse resident final sync device");
        ck(cudaDeviceSynchronize(), "sparse resident final sync");
    }
    double wall_s = ram_seconds_since(wall0);

    Code answer_rank = sparse_canonical_rank_host(MateID(R), W);
    int ao = std::min<int>(ngpu - 1, int(answer_rank / main_chunk));
    Code al = answer_rank - Code(ao) * main_chunk;
    Count answer = 0;
    ck(cudaSetDevice(ao), "sparse resident answer device");
    ck(cudaMemcpy(&answer, gpu[size_t(ao)]->main_shard + al, sizeof(answer),
                  cudaMemcpyDeviceToHost), "sparse resident answer copy");

    double max_gather = 0, max_kernel = 0, max_scatter = 0;
    uint64_t groups = 0, launches = 0;
    for (const auto& c : gpu) {
        max_gather = std::max(max_gather, c->gather_s);
        max_kernel = std::max(max_kernel, c->kernel_s);
        max_scatter = std::max(max_scatter, c->scatter_s);
        groups += c->groups;
        launches += c->launches;
    }
    std::cout << "backend=gridfp-b300-hbm32-compact-resident-sparse"
              << " n=" << n << " residue=" << answer << " modulus=" << mod
              << " gpus=" << ngpu << " groups=" << groups << " launches=" << launches
              << " gather_max_s=" << max_gather << " kernel_max_s=" << max_kernel
              << " scatter_max_s=" << max_scatter << " wall_s=" << wall_s << '\n';

    for (auto& c : gpu) c->release();
    return 0;
}
