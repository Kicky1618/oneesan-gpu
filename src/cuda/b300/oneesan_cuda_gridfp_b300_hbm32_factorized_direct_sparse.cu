#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/ramstream32_high_orbit.cuh"
#include "../gridfp/ramstream32_cpu_low_inplace.hpp"
#include "../gridfp/ramstream32_b300_direct_rowshard.cuh"

struct DirectPhysicalLoc {
    uint32_t bid = 0, hr = 0, lr = 0;
};

static DirectPhysicalLoc direct_locate(Code rank, const std::vector<StorageBlock>& blocks) {
    for (uint32_t bid = 0; bid < blocks.size(); ++bid) {
        const auto& b = blocks[bid];
        Code n = Code(b.rows) * b.cols;
        if (!b.valid || rank < b.off || rank >= b.off + n) continue;
        Code r = rank - b.off;
        return {bid, uint32_t(r / b.cols), uint32_t(r % b.cols)};
    }
    std::cerr << "direct physical rank not found rank=" << rank << '\n';
    std::exit(450);
}

static Code direct_local_index(
    const DirectPhysicalLoc& x, const B300DirectRowShardHost& shard,
    const StorageBlock& b, bool blocked
) {
    int owner = int(x.hr % uint32_t(shard.ngpu));
    Code base = blocked ? shard.block_off[owner][x.bid] : shard.main_off[owner][x.bid];
    return base + Code(x.hr / uint32_t(shard.ngpu)) * b.cols + x.lr;
}

struct DirectGpu {
    int dev = -1;
    Count* main_shard = nullptr;
    Count* block_shard = nullptr;
    Code main_count = 0, block_count = 0;
    BidescMaskDeviceTables mask_tables;
    B300DirectStorageDeviceTables storage_tables;
    B300DirectSparseDeviceTables sparse_tables;

    void allocate(Code mc, Code bc) {
        main_count = mc; block_count = bc;
        ck(cudaSetDevice(dev), "direct set alloc device");
        size_t free0 = 0, total0 = 0;
        ck(cudaMemGetInfo(&free0, &total0), "direct meminfo before");
        if (mc) ck(cudaMalloc(&main_shard, size_t(mc) * sizeof(Count)), "direct main shard");
        if (bc) ck(cudaMalloc(&block_shard, size_t(bc) * sizeof(Count)), "direct block shard");
        if (mc) ck(cudaMemset(main_shard, 0, size_t(mc) * sizeof(Count)), "direct zero main");
        if (bc) ck(cudaMemset(block_shard, 0, size_t(bc) * sizeof(Count)), "direct zero block");
        size_t free1 = 0, total1 = 0;
        ck(cudaMemGetInfo(&free1, &total1), "direct meminfo after auth");
        std::cerr << "direct gpu=" << dev
                  << " total_gib=" << double(total0) / double(1ull << 30)
                  << " free_before_gib=" << double(free0) / double(1ull << 30)
                  << " free_after_auth_gib=" << double(free1) / double(1ull << 30)
                  << " auth_gib=" << double((mc + bc) * sizeof(Count)) / double(1ull << 30)
                  << '\n';
    }

    void install(
        const StorageFactorHost& storage,
        const StorageLayout& layout,
        const B300DirectRowShardHost& shard,
        const B300SparseActionsHost& sparse,
        const B300DirectSparsePartitionHost& part,
        Count mod, Count** mp, Count** bp
    ) {
        ck(cudaSetDevice(dev), "direct set install device");
        mask_tables.install(G_FACTOR);
        storage_tables.install(storage);
        sparse_tables.install(sparse, part, dev);
        b300_direct_install_layout(layout, shard, dev, mp, bp);
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "direct modulus");
        size_t freeb = 0, totalb = 0;
        ck(cudaMemGetInfo(&freeb, &totalb), "direct meminfo after tables");
        std::cerr << "direct gpu=" << dev
                  << " free_after_tables_gib=" << double(freeb) / double(1ull << 30) << '\n';
    }

    void release() {
        if (dev < 0) return;
        ck(cudaSetDevice(dev), "direct release device");
        sparse_tables.release();
        storage_tables.release();
        mask_tables.release();
        if (main_shard) cudaFree(main_shard);
        if (block_shard) cudaFree(block_shard);
        main_shard = block_shard = nullptr;
    }
};

static void direct_enable_peer_atomics(int ngpu) {
    for (int a = 0; a < ngpu; ++a) {
        ck(cudaSetDevice(a), "direct peer source");
        for (int b = 0; b < ngpu; ++b) if (a != b) {
            int can = 0, native_atomic = 0;
            ck(cudaDeviceCanAccessPeer(&can, a, b), "direct can peer");
            if (!can) {
                std::cerr << "direct backend requires full peer mesh: " << a << " -> " << b << '\n';
                std::exit(451);
            }
            ck(cudaDeviceGetP2PAttribute(&native_atomic,
                  cudaDevP2PAttrNativeAtomicSupported, a, b), "direct peer atomic query");
            if (!native_atomic) {
                std::cerr << "direct backend requires native peer atomics: " << a << " -> " << b << '\n';
                std::exit(452);
            }
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else ck(e, "direct enable peer");
        }
    }
}

static void direct_sync_all(int ngpu, const char* what) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), what);
        ck(cudaDeviceSynchronize(), what);
    }
}

static void direct_run_high_edge(
    const B300DirectSparsePartitionHost& part, int ngpu, int p, int threads
) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "direct high orbit device");
        uint32_t n = b300_direct_high_orbit_count(part, g, p);
        if (n) b300_direct_high_orbit_kernel<<<n, threads>>>(p);
        ck(cudaGetLastError(), "direct high orbit launch");
    }
    direct_sync_all(ngpu, "direct high orbit sync");
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "direct high closure device");
        uint32_t n = b300_direct_high_closure_count(part, g, p);
        if (n) b300_direct_high_closure_kernel<<<n, threads>>>(p);
        ck(cudaGetLastError(), "direct high closure launch");
    }
    direct_sync_all(ngpu, "direct high closure sync");
}

static void direct_run_low_edge(
    const B300SparseActionsHost& sparse, int ngpu, int p, int threads
) {
    uint32_t on = b300_sparse_low_orbit_count(sparse, p);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "direct low orbit device");
        if (on) b300_direct_low_orbit_kernel<<<on, threads>>>(p);
        ck(cudaGetLastError(), "direct low orbit launch");
    }
    direct_sync_all(ngpu, "direct low orbit sync");
    uint32_t cn = b300_sparse_low_closure_count(sparse, p);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "direct low closure device");
        if (cn) b300_direct_low_closure_kernel<<<cn, threads>>>(p);
        ck(cudaGetLastError(), "direct low closure launch");
    }
    direct_sync_all(ngpu, "direct low closure sync");
}

static double direct_gib(long double b) {
    return double(b / static_cast<long double>(1ull << 30));
}
static double direct_mib(long double b) {
    return double(b / static_cast<long double>(1ull << 20));
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int requested_gpus = argc > 3 ? std::max(1, std::atoi(argv[3])) : 8;
    int threads = argc > 4 ? std::max(32, std::atoi(argv[4])) : 256;
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
    B300SparseActionsHost sparse = build_b300_sparse_actions(
        layout, lowdesc, loworbit, highdesc, highorbit);
    B300DirectRowShardHost shard = build_b300_direct_row_shards(layout, requested_gpus);
    B300DirectSparsePartitionHost part = b300_direct_partition_high(sparse, requested_gpus);

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_phys = storage_rank_main_host(init, storage, layout);
    Code answer_phys = storage_rank_main_host(MateID(R), storage, layout);
    DirectPhysicalLoc init_loc = direct_locate(init_phys, layout.main_blocks);
    DirectPhysicalLoc answer_loc = direct_locate(answer_phys, layout.main_blocks);

    long double mask_bytes = static_cast<long double>(
        G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
      + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t);
    long double storage_bytes = static_cast<long double>(
        storage.low_all_codes.size() + storage.high_all_codes.size()
      + storage.low_mask_begin.size() + storage.high_mask_begin.size()) * sizeof(uint32_t);
    long double low_sparse_bytes = static_cast<long double>(sparse.low_orbit.size()) * sizeof(B300SparseOrbitOp)
                                  + static_cast<long double>(sparse.low_closure.size()) * sizeof(uint64_t)
                                  + static_cast<long double>(sparse.low_orbit_off.size()
                                                           + sparse.low_closure_off.size()) * sizeof(uint32_t);
    long double layout_bytes = static_cast<long double>(
        layout.main_blocks.size() + layout.block_blocks.size()) * sizeof(StorageBlock)
        + sizeof(shard.main_off) + sizeof(shard.block_off);

    std::array<long double, MAXGPU> need{};
    std::array<long double, MAXGPU> high_sparse_bytes{};
    long double max_need = 0;
    int max_g = 0;
    for (int g = 0; g < requested_gpus; ++g) {
        high_sparse_bytes[g] = static_cast<long double>(part.high_orbit[g].size()) * sizeof(B300SparseOrbitOp)
                             + static_cast<long double>(part.high_closure[g].size()) * sizeof(uint64_t)
                             + static_cast<long double>(part.high_orbit_off[g].size()
                                                      + part.high_closure_off[g].size()) * sizeof(uint32_t);
        long double auth = static_cast<long double>(shard.main_count[g] + shard.block_count[g]) * sizeof(Count);
        need[g] = auth + mask_bytes + storage_bytes + low_sparse_bytes + high_sparse_bytes[g] + layout_bytes;
        if (need[g] > max_need) { max_need = need[g]; max_g = g; }
    }

    long double canonical_gs = static_cast<long double>(
        4 * layout.main_size + 2 * layout.block_size) * sizeof(Count) * W;
    long double ramstream_pcie = static_cast<long double>(
        2 * layout.main_size + layout.block_size) * sizeof(Count) * W;

    if (plan_only) {
        std::cout << std::fixed << std::setprecision(3)
                  << "backend=gridfp-b300-hbm32-factorized-direct-sparse-plan"
                  << " n=" << n << " gpus=" << requested_gpus
                  << " auth_total_gib=" << direct_gib(
                         static_cast<long double>(layout.main_size + layout.block_size) * sizeof(Count))
                  << " runtime_groups=0 scratch_gib=0.000"
                  << " gather_scatter_tib_per_residue=0.000"
                  << " host_pcie_tib_per_residue=0.000"
                  << " eliminated_canonical_gs_tib=" << double(canonical_gs / (1ull << 40))
                  << " eliminated_ramstream_pcie_tib=" << double(ramstream_pcie / (1ull << 40))
                  << " low_sparse_replicated_mib=" << direct_mib(low_sparse_bytes)
                  << " common_meta_mib=" << direct_mib(mask_bytes + storage_bytes + layout_bytes)
                  << " max_need_gib=" << direct_gib(max_need)
                  << " max_need_gpu=" << max_g
                  << " headroom_288GB_gib=" << direct_gib(288.0e9L - max_need)
                  << " headroom_279GB_gib=" << direct_gib(279.0e9L - max_need)
                  << '\n';
        for (int g = 0; g < requested_gpus; ++g) {
            std::cout << "direct_gpu=" << g
                      << " main_gib=" << direct_gib(static_cast<long double>(shard.main_count[g]) * sizeof(Count))
                      << " block_gib=" << direct_gib(static_cast<long double>(shard.block_count[g]) * sizeof(Count))
                      << " high_sparse_mib=" << direct_mib(high_sparse_bytes[g])
                      << " need_gib=" << direct_gib(need[g]) << '\n';
        }
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "direct cudaGetDeviceCount");
    if (visible < requested_gpus) {
        std::cerr << "need " << requested_gpus << " GPUs, visible=" << visible << '\n';
        return 3;
    }

    // Dense construction tables and dense 4^K rank arrays are host-only now.
    // Release them before allocating the ~1.94 TiB authoritative HBM state.
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
    highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    loworbit.rec.clear(); loworbit.rec.shrink_to_fit();
    highorbit.rec.clear(); highorbit.rec.shrink_to_fit();
    G_FACTOR.low_packed_rank.clear(); G_FACTOR.low_packed_rank.shrink_to_fit();
    G_FACTOR.high_packed_rank.clear(); G_FACTOR.high_packed_rank.shrink_to_fit();
    storage.low_packed_rank.clear(); storage.low_packed_rank.shrink_to_fit();
    storage.high_packed_rank.clear(); storage.high_packed_rank.shrink_to_fit();

    std::vector<std::unique_ptr<DirectGpu>> gpu;
    gpu.reserve(requested_gpus);
    for (int g = 0; g < requested_gpus; ++g) {
        auto c = std::make_unique<DirectGpu>();
        c->dev = g;
        c->allocate(shard.main_count[g], shard.block_count[g]);
        gpu.push_back(std::move(c));
    }

    direct_enable_peer_atomics(requested_gpus);
    std::array<Count*, MAXGPU> mp{}, bp{};
    for (int g = 0; g < requested_gpus; ++g) {
        mp[g] = gpu[g]->main_shard;
        bp[g] = gpu[g]->block_shard;
    }
    for (int g = 0; g < requested_gpus; ++g)
        gpu[g]->install(storage, layout, shard, sparse, part, mod, mp.data(), bp.data());

    int init_owner = int(init_loc.hr % uint32_t(requested_gpus));
    Code init_local = direct_local_index(init_loc, shard, layout.main_blocks[init_loc.bid], false);
    Count one = 1;
    ck(cudaSetDevice(init_owner), "direct init owner");
    ck(cudaMemcpy(gpu[init_owner]->main_shard + init_local, &one, sizeof(one),
                  cudaMemcpyHostToDevice), "direct init state");

    auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p)
            direct_run_high_edge(part, requested_gpus, p, threads);
        for (int p = LOW_LUT_K; p >= 1; --p)
            direct_run_low_edge(sparse, requested_gpus, p, threads);
        std::cerr << "direct row " << row + 1 << '/' << W << '\n';
    }
    direct_sync_all(requested_gpus, "direct final sync");
    double wall_s = ram_seconds_since(wall0);

    int answer_owner = int(answer_loc.hr % uint32_t(requested_gpus));
    Code answer_local = direct_local_index(answer_loc, shard, layout.main_blocks[answer_loc.bid], false);
    Count answer = 0;
    ck(cudaSetDevice(answer_owner), "direct answer owner");
    ck(cudaMemcpy(&answer, gpu[answer_owner]->main_shard + answer_local, sizeof(answer),
                  cudaMemcpyDeviceToHost), "direct answer copy");

    std::cout << "backend=gridfp-b300-hbm32-factorized-direct-sparse"
              << " n=" << n << " residue=" << answer << " modulus=" << mod
              << " gpus=" << requested_gpus
              << " groups=0 bulk_transfer_bytes=0"
              << " wall_s=" << wall_s << '\n';

    for (auto& c : gpu) c->release();
    return 0;
}
