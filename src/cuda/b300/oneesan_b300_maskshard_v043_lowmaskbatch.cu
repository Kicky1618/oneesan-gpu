// v0.43 executable LOW mask-batch backend.  Reuse every static helper from the
// existing full-orbit translation unit, but replace only its main/LOW row loop.
#define MASKSHARD_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT 1
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#define MASKSHARD_BLOCK_ORBIT_TIGHT_LAUNCH 1
#define MASKSHARD_BLOCK_ORBIT_ROW_CAP_LAUNCH 1
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK 1
#define MASKSHARD_HIGH_CLOSURE_ROWPACK_THRESHOLD 29
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH 1
#define MASKSHARD_HIGH_CLOSURE_TASK_LAUNCH 1
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT 1
#define MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_LOW_CLOSURE_COLS 1
#define MASKSHARD_LOW_CLOSURE_ROW_DEPTH 1
#define MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT 1
#define MASKSHARD_LOW_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_LOW_CLOSURE_TASK_U32 1
#define MASKSHARD_LOW_CLOSURE_PACKED_PREFIX 1
#define MASKSHARD_LOW_CLOSURE_PACKED_META 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH 1
#define MASKSHARD_LOW_BLOCK_ORBIT_TIGHT_LAUNCH 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT_LAUNCH 1
#define MASKSHARD_LOW_ORBIT_ROW_DEPTH_WARP_DECODE 1
#define MASKSHARD_LOW_ORBIT_WARP_DECODE_FULLCAP 1
#define MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS 1
#define MASKSHARD_LOW_ORBIT_WARP_ROW_U32 1
#define MASKSHARD_LOW_GROUP_PACKED_CONFIG 1
#define MASKSHARD_LOW_GROUP_PACKED_CACHE 1
#define MASKSHARD_LOW_GROUP_STATIC_BASE_CACHE 1
#define MASKSHARD_SKIP_ZERO_BLOCK_GATHER 1
#define MASKSHARD_LAZY_ZERO_BLOCK_INIT 1
#define MASKSHARD_ROW_DEPTH_FBLOCK_IO 1
#define MASKSHARD_ROW_DEPTH_EXACT_IO 1
#define MASKSHARD_ROW_DEPTH_ORBIT 1
#define MASKSHARD_ROW_DEPTH_ORBIT_COMPACT 1

#define main oneesan_maskshard_v43_legacy_main
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu"
#undef main

// Bring in the reusable packed config builder only after the legacy translation
// unit is parsed; the new main below sees its low-closure capture/report hooks.
#define cudaMalloc maskshard_guarded_cuda_malloc
#include "maskshard_loworbit_warprow_packed.cuh"
#ifdef MASKSHARD_LOW_MASKBATCH_COMPACT_RANGES
#include "maskshard_low_maskbatch_range_executor.cuh"
#else
#include "maskshard_low_maskbatch_executor.cuh"
#endif

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    const int requested = argc > 2 ? std::atoi(argv[2]) : 8;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    if (threads < 32 || threads > 1024 || (threads & 31)) {
        std::cerr << "v0.43 LOW mask-batch requires threads multiple of 32 in [32,1024]\n";
        return 1;
    }
    std::vector<Count> mods;
    for (int i = 4; i < argc; ++i) {
        const unsigned long long raw = std::strtoull(argv[i], nullptr, 10);
        if (raw < 2 || raw > 0xffffffffULL) return 1;
        mods.push_back(Count(raw));
    }
    if (mods.empty()) mods.push_back(4294967291u);
    if (n + 1 != TARGET_W || n < 2 || TARGET_W > MAXW) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    const auto setup0 = std::chrono::steady_clock::now();
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost high_desc = build_high_descriptors(storage, layout);
    LowDescHost low_desc = build_low_descriptors(storage, layout);
    MaskShardLowClosureColsHost low_closure =
        build_maskshard_low_closure_cols(storage, layout, low_desc);
    const double highdesc_mib = double(
        (high_desc.main_desc.size() + high_desc.block_desc.size()) * sizeof(uint32_t)
    ) / double(1ULL << 20);
    const double lowdesc_mib = double(
        (low_desc.main_desc.size() + low_desc.block_desc.size()) * sizeof(uint32_t)
    ) / double(1ULL << 20);

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "v0.43 device count");
    const int ngpu = std::min({requested, visible, 8});
    if (ngpu < 1) return 2;
    int peers = 0;
    for (int a = 0; a < ngpu; ++a) for (int b = 0; b < ngpu; ++b) if (a != b) {
        int can = 0;
        ck(cudaDeviceCanAccessPeer(&can, a, b), "v0.43 can peer");
        if (!can) continue;
        ck(cudaSetDevice(a), "v0.43 peer set");
        cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else ck(e, "v0.43 enable peer");
        ++peers;
    }
    if (ngpu > 1 && peers != ngpu * (ngpu - 1)) return 3;

    MaskShardLayout shard = build_high_mask_shard_layout(storage, layout, ngpu);
    const auto high_route = build_fullorbit_batch_high_route(storage);
    report_high_mask_shard_layout(shard);

    Count* mp[8]{};
    Count* bp[8]{};
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "v0.43 auth set");
        if (shard.gpu_main[d])
            ck(cudaMalloc(&mp[d], size_t(shard.gpu_main[d]) * sizeof(Count)),
               "v0.43 auth main");
        if (shard.gpu_block[d])
            ck(cudaMalloc(&bp[d], size_t(shard.gpu_block[d]) * sizeof(Count)),
               "v0.43 auth block");
    }

    std::vector<StorageDeviceTables> factor_dev(ngpu);
    std::vector<HighDescDeviceTables> highdesc_dev(ngpu);
    std::vector<LowDescDeviceTables> lowdesc_dev(ngpu);
    std::vector<MaskShardLowClosureColsDeviceTables> lowclosure_dev(ngpu);
    std::vector<MaskShardDeviceMeta> shard_dev(ngpu);
    std::vector<FullOrbitBatchWorker> workers(ngpu);
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "v0.43 table set");
        factor_dev[d].install(storage, G_FACTOR);
        highdesc_dev[d].install(high_desc);
        lowdesc_dev[d].install(low_desc);
        lowclosure_dev[d].install(low_closure);
        shard_dev[d].install(d, shard, layout, high_route, mp, bp);
        workers[d].init(d);
    }

    const MateID init = MateID(R) << (2 * (TARGET_W - 1));
    const FullOrbitBatchAddress ia =
        fullorbit_batch_main_address_host(init, storage, layout, shard);
    const FullOrbitBatchAddress oa =
        fullorbit_batch_main_address_host(MateID(R), storage, layout, shard);
    const auto high_jobs = build_fullorbit_batch_high_jobs(&high_desc);
    maskshard_prepare_highclosure_rowdepth_compact(high_desc, ngpu);

    MaskShardLowMaskBatchExecutor low_batch;
    low_batch.install(shard);

    const double setup_s = ram_seconds_since(setup0);
    std::cerr << "v0.43 setup_s=" << setup_s
              << " residues=" << mods.size()
              << " highdesc_mib_per_gpu=" << highdesc_mib
              << " lowdesc_mib_per_gpu=" << lowdesc_mib << '\n';

    for (size_t ri = 0; ri < mods.size(); ++ri) {
        const Count mod = mods[ri];
        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "v0.43 reset device");
            ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "v0.43 set modulus");
            if (mp[d]) ck(cudaMemset(mp[d], 0,
                size_t(shard.gpu_main[d]) * sizeof(Count)), "v0.43 zero main");
            if (bp[d]) ck(cudaMemset(bp[d], 0,
                size_t(shard.gpu_block[d]) * sizeof(Count)), "v0.43 zero block");
            ck(cudaDeviceSynchronize(), "v0.43 reset sync");
            workers[d].reset_stats();
        }
        const Count one = 1;
        ck(cudaSetDevice(ia.owner), "v0.43 init device");
        ck(cudaMemcpy(mp[ia.owner] + ia.offset, &one, sizeof(one),
                      cudaMemcpyHostToDevice), "v0.43 init one");

        std::array<double, 8> low_batch_s{};
        const auto wall0 = std::chrono::steady_clock::now();
        for (int row = 0; row < TARGET_W; ++row) {
            for (int d = 0; d < ngpu; ++d) {
                ck(cudaSetDevice(d), "v0.43 row-depth device");
                maskshard_set_row_depth_fblock_io_row(row);
            }

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

            low_batch.prepare_row(shard, row);
            ts.clear();
            for (int d = 0; d < ngpu; ++d) {
                ts.emplace_back([&, d] {
                    const auto t0 = std::chrono::steady_clock::now();
                    low_batch.run_device_row(d, mp[d], bp[d], row, threads);
                    low_batch_s[d] += ram_seconds_since(t0);
                });
            }
            for (auto& t : ts) t.join();
            std::cerr << "v0.43 residue " << ri + 1 << '/' << mods.size()
                      << " row " << row + 1 << '/' << TARGET_W << '\n';
        }
        const double wall_s = ram_seconds_since(wall0);

        Count answer = 0;
        ck(cudaSetDevice(oa.owner), "v0.43 answer device");
        ck(cudaMemcpy(&answer, mp[oa.owner] + oa.offset, sizeof(answer),
                      cudaMemcpyDeviceToHost), "v0.43 answer");

        double high_io = 0, high_orbit = 0, high_closure = 0;
        double low_batch_sum = 0;
        size_t max_scratch = 0;
        for (int d = 0; d < ngpu; ++d) {
            high_io += workers[d].high_io_s;
            high_orbit += workers[d].high_orbit_s;
            high_closure += workers[d].high_closure_s;
            low_batch_sum += low_batch_s[d];
            max_scratch = std::max(max_scratch, workers[d].cap);
        }

        std::cout << "backend=b300-factorized-maskshard-v0.43-lowmaskbatch"
                  << " n=" << n << " gpus=" << ngpu
                  << " residue=" << answer << " modulus=" << mod
                  << " residue_index=" << ri
                  << " residues_total=" << mods.size()
                  << " setup_s=" << setup_s << " wall_s=" << wall_s
                  << " high_io_sum_s=" << high_io
                  << " high_orbit_sum_s=" << high_orbit
                  << " high_closure_sum_s=" << high_closure
                  << " low_batch_sum_s=" << low_batch_sum
                  << " low_kernel_launches="
                  << (std::uint64_t(ngpu) * TARGET_W * LOW_LUT_K * 2ULL)
                  << " highdesc_mib_per_gpu=" << highdesc_mib
                  << " lowdesc_mib_per_gpu=" << lowdesc_mib
                  << " max_scratch_gib="
                  << double(max_scratch) / double(1ULL << 30) << '\n';
    }

    low_batch.release();
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "v0.43 cleanup device");
        workers[d].release();
        shard_dev[d].release();
        lowclosure_dev[d].release();
        lowdesc_dev[d].release();
        highdesc_dev[d].release();
        factor_dev[d].release();
        if (mp[d]) cudaFree(mp[d]);
        if (bp[d]) cudaFree(bp[d]);
    }
    maskshard_release_highclosure_rowdepth_compact();
    return 0;
}

#undef cudaMalloc
