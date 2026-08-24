// W28/B300x8 production candidate: precomputed spectral HIGH-mask sharding plus
// cheapest-executor HIGH orbit scheduling and destination-owner closures.
// LOW window is completely local.  HIGH closures use no cross-GPU atomic RMW.
#define main(...) b300_maskshard_reference_main(__VA_ARGS__)
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_maskshard_sparse.cu"
#undef main

#include "../gridfp/ramstream32_b300_precomputed_w28_partition.cuh"
#include "../gridfp/ramstream32_b300_direct_exec_owner.cuh"

static void preexec_peer_mesh(int n) {
    for (int a = 0; a < n; ++a) {
        ck(cudaSetDevice(a), "preexec peer source");
        for (int b = 0; b < n; ++b) if (a != b) {
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, a, b), "preexec peer access query");
            if (!can) {
                std::cerr << "preexec requires peer access " << a << " -> " << b << '\n';
                std::exit(500);
            }
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else ck(e, "preexec enable peer");
        }
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int ng = argc > 3 ? std::max(1, std::atoi(argv[3])) : 8;
    int th = argc > 4 ? std::max(32, std::atoi(argv[4])) : 256;
    bool plan = argc > 5 && std::strcmp(argv[5], "--plan-only") == 0;
    int W = n + 1;
    if (W != TARGET_W || W > MAXW || n < 2 || ng > MAXGPU) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K + 1 != TARGET_W) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost f = build_storage_factor_tables(G_FACTOR);
    StorageLayout l = build_storage_layout(f);
    LowDescHost ld = build_low_descriptors(f, l);
    HighDescHost hd = build_high_descriptors(f, l);
    LowOrbitHost lo = build_cpu_low_orbit(f, l, ld);
    HighOrbitHost ho = build_high_orbit(f, l);
    B300SparseActionsHost sparse = build_b300_sparse_actions(l, ld, lo, hd, ho);

    B300DirectMaskShardHost shard = build_b300_direct_mask_shards_w28_precomputed(f, l, ng);
    B300DirectExecOwnerStats es{};
    B300DirectSparsePartitionHost part = b300_direct_partition_high_by_exec_owner(
        sparse, f, l, shard, &es);

    Code ip = storage_rank_main_host(MateID(R) << (2 * (W - 1)), f, l);
    Code ap = storage_rank_main_host(MateID(R), f, l);
    MaskLoc il = locate_mask(ip, l.main_blocks), al = locate_mask(ap, l.main_blocks);

    long double maskb = (long double)(G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
        + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t);
    long double storeb = (long double)(f.low_all_codes.size() + f.high_all_codes.size()
        + f.low_mask_begin.size() + f.high_mask_begin.size()) * sizeof(uint32_t);
    long double mapcommon = (long double)shard.high_owner.size()
        + (long double)shard.high_local.size() * sizeof(uint32_t);
    long double lows = (long double)sparse.low_orbit.size() * sizeof(B300SparseOrbitOp)
        + (long double)sparse.low_closure.size() * sizeof(uint64_t)
        + (long double)(sparse.low_orbit_off.size() + sparse.low_closure_off.size()) * sizeof(uint32_t);
    long double layoutb = (long double)(l.main_blocks.size() + l.block_blocks.size()) * sizeof(StorageBlock)
        + sizeof(shard.main_off) + sizeof(shard.block_off);

    std::array<long double, MAXGPU> need{}, hs{};
    long double maxneed = 0, minauth = 1e100L, maxauth = 0;
    for (int g = 0; g < ng; ++g) {
        hs[g] = (long double)part.high_orbit[g].size() * sizeof(B300SparseOrbitOp)
            + (long double)part.high_closure[g].size() * sizeof(uint64_t)
            + (long double)(part.high_orbit_off[g].size() + part.high_closure_off[g].size()) * sizeof(uint32_t);
        long double auth = (long double)(shard.main_count[g] + shard.block_count[g]) * sizeof(Count);
        minauth = std::min(minauth, auth); maxauth = std::max(maxauth, auth);
        need[g] = auth + maskb + storeb + mapcommon
            + (long double)shard.owned_rows[g].size() * sizeof(uint32_t)
            + lows + hs[g] + layoutb;
        maxneed = std::max(maxneed, need[g]);
    }

    auto GiB2 = [](long double x) { return double(x / (1ull << 30)); };
    auto MiB2 = [](long double x) { return double(x / (1ull << 20)); };
    auto TiB2 = [](long double x) { return double(x / (1ull << 40)); };
    long double cgs = (long double)(4 * l.main_size + 2 * l.block_size) * sizeof(Count) * W;
    long double rp = (long double)(2 * l.main_size + l.block_size) * sizeof(Count) * W;
    long double peer_per_residue = (long double)(es.orbit_exec_owner_peer_bytes + es.closure_peer_read_bytes) * W;
    long double source_peer_per_residue = (long double)es.orbit_source_owner_peer_bytes * W;
    uint64_t max_orbit_work = 0, max_closure_work = 0;
    for (int g = 0; g < ng; ++g) {
        max_orbit_work = std::max(max_orbit_work, es.orbit_work[g]);
        max_closure_work = std::max(max_closure_work, es.closure_work[g]);
    }

    if (plan) {
        std::cout << std::fixed << std::setprecision(6)
            << "backend=gridfp-b300-hbm32-factorized-precomputed-exec-sparse-plan"
            << " n=" << n << " gpus=" << ng
            << " partition=precomputed-spectral-lambda1-swap512"
            << " auth_total_gib=" << GiB2((long double)(l.main_size + l.block_size) * sizeof(Count))
            << " auth_min_gib=" << GiB2(minauth) << " auth_max_gib=" << GiB2(maxauth)
            << " auth_imbalance=" << double(maxauth / minauth)
            << " runtime_groups=0 scratch_gib=0.000 low_p2p_bytes=0"
            << " high_remote_system_atomics=" << es.closure_remote_system_atomics
            << " source_owner_orbit_peer_tib_per_residue=" << TiB2(source_peer_per_residue)
            << " exec_owner_orbit_peer_tib_per_residue="
            << TiB2((long double)es.orbit_exec_owner_peer_bytes * W)
            << " destination_owner_closure_read_tib_per_residue="
            << TiB2((long double)es.closure_peer_read_bytes * W)
            << " modeled_peer_tib_per_residue=" << TiB2(peer_per_residue)
            << " orbit_exec_saving_fraction="
            << (es.orbit_source_owner_peer_bytes
                ? 1.0 - double(es.orbit_exec_owner_peer_bytes) / double(es.orbit_source_owner_peer_bytes) : 0.0)
            << " max_orbit_work=" << max_orbit_work
            << " max_closure_work=" << max_closure_work
            << " gather_scatter_tib_per_residue=0.000 host_pcie_tib_per_residue=0.000"
            << " eliminated_canonical_gs_tib=" << TiB2(cgs)
            << " eliminated_ramstream_pcie_tib=" << TiB2(rp)
            << " low_sparse_replicated_mib=" << MiB2(lows)
            << " common_meta_mib=" << MiB2(maskb + storeb + mapcommon + layoutb)
            << " max_need_gib=" << GiB2(maxneed)
            << " headroom_288GB_gib=" << GiB2(288.0e9L - maxneed)
            << " headroom_279GB_gib=" << GiB2(279.0e9L - maxneed)
            << '\n';
        for (int g = 0; g < ng; ++g)
            std::cout << "preexec_gpu=" << g
                      << " auth_gib=" << GiB2((long double)(shard.main_count[g] + shard.block_count[g]) * sizeof(Count))
                      << " owned_high_rows=" << shard.owned_rows[g].size()
                      << " orbit_work=" << es.orbit_work[g]
                      << " closure_work=" << es.closure_work[g]
                      << " high_sparse_mib=" << MiB2(hs[g])
                      << " need_gib=" << GiB2(need[g]) << '\n';
        return 0;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "preexec device count");
    if (visible < ng) { std::cerr << "need " << ng << " GPUs visible=" << visible << '\n'; return 3; }

    ld.main_desc.clear(); ld.block_desc.clear(); hd.main_desc.clear(); hd.block_desc.clear();
    lo.rec.clear(); ho.rec.clear();
    G_FACTOR.low_packed_rank.clear(); G_FACTOR.low_packed_rank.shrink_to_fit();
    G_FACTOR.high_packed_rank.clear(); G_FACTOR.high_packed_rank.shrink_to_fit();
    f.low_packed_rank.clear(); f.low_packed_rank.shrink_to_fit();
    f.high_packed_rank.clear(); f.high_packed_rank.shrink_to_fit();

    std::vector<std::unique_ptr<MaskGpu>> gpu;
    for (int g = 0; g < ng; ++g) {
        auto c = std::make_unique<MaskGpu>(); c->dev = g;
        c->alloc(shard.main_count[g], shard.block_count[g]);
        gpu.push_back(std::move(c));
    }
    preexec_peer_mesh(ng);
    std::array<Count*, MAXGPU> mp{}, bp{};
    for (int g = 0; g < ng; ++g) { mp[g] = gpu[g]->main; bp[g] = gpu[g]->block; }
    for (int g = 0; g < ng; ++g) gpu[g]->install(f, l, shard, sparse, part, mod, mp.data(), bp.data());

    uint32_t iai = f.high_all_off[l.main_blocks[il.bid].he] + il.hr;
    int io = shard.high_owner[iai];
    Code ix = local_mask_index(il, shard, f, l.main_blocks[il.bid], false);
    Count one = 1;
    ck(cudaSetDevice(io), "preexec init dev");
    ck(cudaMemcpy(gpu[io]->main + ix, &one, sizeof(one), cudaMemcpyHostToDevice), "preexec init");

    auto t0 = std::chrono::steady_clock::now();
    for (int row = 0; row < W; ++row) {
        for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) run_high(part, ng, p, th);
        for (int p = LOW_LUT_K; p >= 1; --p) run_low(sparse, ng, p, th);
        std::cerr << "preexec row " << row + 1 << '/' << W << '\n';
    }
    sync_all(ng, "preexec final sync");
    double sec = ram_seconds_since(t0);

    uint32_t aai = f.high_all_off[l.main_blocks[al.bid].he] + al.hr;
    int ao = shard.high_owner[aai];
    Code ax = local_mask_index(al, shard, f, l.main_blocks[al.bid], false);
    Count ans = 0;
    ck(cudaSetDevice(ao), "preexec answer dev");
    ck(cudaMemcpy(&ans, gpu[ao]->main + ax, sizeof(ans), cudaMemcpyDeviceToHost), "preexec answer");
    std::cout << "backend=gridfp-b300-hbm32-factorized-precomputed-exec-sparse"
              << " n=" << n << " residue=" << ans << " modulus=" << mod << " gpus=" << ng
              << " high_remote_system_atomics=0 low_p2p_bytes=0 bulk_transfer_bytes=0"
              << " modeled_peer_tib_per_residue=" << TiB2(peer_per_residue)
              << " wall_s=" << sec << '\n';
    for (auto& c : gpu) c->release();
    return 0;
}
