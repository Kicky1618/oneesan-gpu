#include <cuda_runtime.h>

#include <algorithm>
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
#include "maskshard_layout.hpp"
#include "maskshard_lowlocal.cuh"
#include "maskshard_highio.cuh"
#include "maskshard_highorbit.cuh"

struct MaskShardOrbitAddress {
    int owner = -1;
    Code offset = 0;
};

static MaskShardOrbitAddress maskshard_orbit_main_address_host(
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
        std::cerr << "maskshard-orbit invalid host main address\n";
        std::exit(180);
    }
    const Code off = shard.main_base[mask]
        + shard.main_block_off[size_t(mask) * shard.main_nblocks + bid]
        + Code(hp & HR) * layout.main_blocks[bid].cols + (lp >> L);
    return {int(shard.owner[mask]), off};
}

static std::vector<uint32_t> build_maskshard_orbit_high_route(
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
                std::cerr << "maskshard-orbit missing HIGH packed rank\n";
                std::exit(181);
            }
            route[a + r] = seg_occ(code, H) | ((packed & HM) << H);
        }
    }
    return route;
}

struct MaskShardOrbitWorker {
    int dev = -1;
    uint8_t* arena = nullptr;
    size_t cap = 0;
    Count* a = nullptr;
    Count* d = nullptr;
    double high_io_s = 0.0;
    double high_orbit_s = 0.0;
    double high_closure_s = 0.0;
    MaskShardLowStats low;
    uint64_t high_groups = 0;

    static size_t aligned(size_t x) { return (x + 255) & ~size_t(255); }
    void init(int device) { dev = device; }

    void ensure_bytes(size_t need) {
        if (need <= cap) return;
        if (arena) ck(cudaFree(arena), "maskshard-orbit worker free arena");
        cap = need;
        ck(cudaMalloc(&arena, cap), "maskshard-orbit worker arena");
    }

    void buffers(Code main_n, Code block_n) {
        const size_t mb = aligned(size_t(main_n) * sizeof(Count));
        const size_t db = aligned(size_t(block_n) * sizeof(Count));
        ensure_bytes(mb + db);
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

struct HighOrbitJob {
    uint32_t low_mask = 0;
    Code main_n = 0;
    Code block_n = 0;
    Code work = 0;
};

static std::vector<HighOrbitJob> build_high_orbit_jobs() {
    constexpr uint32_t NM = 1u << LOW_LUT_K;
    std::vector<HighOrbitJob> jobs;
    jobs.reserve(NM);
    for (uint32_t mask = 0; mask < NM; ++mask) {
        const auto mb = make_factor_main_blocks(true, mask);
        const auto db = make_factor_block_blocks(true, mask);
        const Code mn = mb.empty() ? 0 : mb.back().end;
        const Code dn = db.empty() ? 0 : db.back().end;
        jobs.push_back({mask, mn, dn, mn + dn});
    }
    std::sort(jobs.begin(), jobs.end(), [](const HighOrbitJob& x, const HighOrbitJob& y) {
        return x.work > y.work;
    });
    return jobs;
}

static void maskshard_configure_high_orbit_group(uint32_t mask) {
    constexpr int L = LOW_LUT_K;
    constexpr uint32_t LM = (1u << L) - 1u;
    const uint32_t mf = LM;
    const uint32_t mo = mask;
    const uint32_t bf = LM;
    const uint32_t bo = mask;
    const GroupSpec ms = make_spec(TARGET_W, mf, mo);
    const GroupSpec ds = make_spec(TARGET_W - 1, bf, bo);
    std::vector<FBlock> mb = make_factor_main_blocks(true, mask);
    std::vector<FBlock> db = make_factor_block_blocks(true, mask);
    if (mb.empty() || db.empty() || mb.back().end != ms.size || db.back().end != ds.size) {
        std::cerr << "maskshard-orbit factor-size mismatch mask=" << mask << '\n';
        std::exit(182);
    }
    const int mn = int(mb.size());
    const int dn = int(db.size());
    const int fix_low = 1;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, mb.data(), mb.size() * sizeof(FBlock)),
       "maskshard-orbit main blocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, db.data(), db.size() * sizeof(FBlock)),
       "maskshard-orbit block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &mn, sizeof(mn)), "maskshard-orbit main nblocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &dn, sizeof(dn)), "maskshard-orbit block nblocks");
    ck(cudaMemcpyToSymbol(D_F_MASK, &mask, sizeof(mask)), "maskshard-orbit low-mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fix_low, sizeof(fix_low)), "maskshard-orbit mode");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &mf, sizeof(mf)), "maskshard-orbit main fixed");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &mo, sizeof(mo)), "maskshard-orbit main occ");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &bf, sizeof(bf)), "maskshard-orbit block fixed");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &bo, sizeof(bo)), "maskshard-orbit block occ");
    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "maskshard-orbit main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "maskshard-orbit block dp");
}

static void process_high_orbit_job(
    MaskShardOrbitWorker& w, const HighOrbitJob& job, int threads
) {
    ck(cudaSetDevice(w.dev), "maskshard-orbit set device");
    if (!job.main_n && !job.block_n) return;
    maskshard_configure_high_orbit_group(job.low_mask);
    w.buffers(job.main_n, job.block_n);
    const int bm = int(std::min<Code>(65535, (job.main_n + threads - 1) / threads));
    const int bd = int(std::min<Code>(65535, (job.block_n + threads - 1) / threads));

    auto t = std::chrono::steady_clock::now();
    if (job.main_n)
        maskshard_high_main_io_kernel<false><<<bm, threads>>>(w.a, job.main_n);
    if (job.block_n)
        maskshard_high_block_io_kernel<false><<<bd, threads>>>(w.d, job.block_n);
    ck(cudaGetLastError(), "maskshard-orbit gather");
    ck(cudaDeviceSynchronize(), "maskshard-orbit gather sync");
    w.high_io_s += ram_seconds_since(t);

    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        t = std::chrono::steady_clock::now();
        if (job.main_n)
            maskshard_main_block_highorbit_kernel<<<bm, threads>>>(w.a, w.d, job.main_n, p);
        ck(cudaGetLastError(), "maskshard-orbit core");
        ck(cudaDeviceSynchronize(), "maskshard-orbit core sync");
        w.high_orbit_s += ram_seconds_since(t);

        t = std::chrono::steady_clock::now();
        if (job.main_n)
            maskshard_main_highdesc_closure_inplace_kernel<<<bm, threads>>>(
                w.a, w.d, job.main_n, p);
        ck(cudaGetLastError(), "maskshard-orbit closure");
        ck(cudaDeviceSynchronize(), "maskshard-orbit closure sync");
        w.high_closure_s += ram_seconds_since(t);
    }

    t = std::chrono::steady_clock::now();
    if (job.main_n)
        maskshard_high_main_io_kernel<true><<<bm, threads>>>(w.a, job.main_n);
    if (job.block_n)
        maskshard_high_block_io_kernel<true><<<bd, threads>>>(w.d, job.block_n);
    ck(cudaGetLastError(), "maskshard-orbit scatter");
    ck(cudaDeviceSynchronize(), "maskshard-orbit scatter sync");
    w.high_io_s += ram_seconds_since(t);
    ++w.high_groups;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    const Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    const int requested = argc > 3 ? std::atoi(argv[3]) : 8;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 256;
    if (n + 1 != TARGET_W || n < 2 || TARGET_W > MAXW) {
        std::cerr << "maskshard-orbit binary specialized for n=" << TARGET_W - 1 << '\n';
        return 1;
    }
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) {
        std::cerr << "maskshard-orbit requires LOW+HIGH=W-1\n";
        return 1;
    }

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    const auto desc_t0 = std::chrono::steady_clock::now();
    HighDescHost high_desc = build_high_descriptors(storage, layout);
    const double highdesc_build_s = ram_seconds_since(desc_t0);
    const double highdesc_mib = double(
        (high_desc.main_desc.size() + high_desc.block_desc.size()) * sizeof(uint32_t)
    ) / double(1ULL << 20);

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "maskshard-orbit device count");
    const int ngpu = std::min({requested, visible, 8});
    if (ngpu < 1) return 2;
    int peers = 0;
    for (int a = 0; a < ngpu; ++a) {
        for (int b = 0; b < ngpu; ++b) if (a != b) {
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, a, b), "maskshard-orbit can peer");
            if (!can) continue;
            ck(cudaSetDevice(a), "maskshard-orbit peer set");
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else ck(e, "maskshard-orbit enable peer");
            ++peers;
        }
    }
    if (ngpu > 1 && peers != ngpu * (ngpu - 1)) {
        std::cerr << "maskshard-orbit requires full P2P: "
                  << peers << '/' << ngpu * (ngpu - 1) << '\n';
        return 3;
    }

    MaskShardLayout shard = build_high_mask_shard_layout(storage, layout, ngpu);
    const auto high_route = build_maskshard_orbit_high_route(storage);
    report_high_mask_shard_layout(shard);

    Count* mp[8]{};
    Count* bp[8]{};
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "maskshard-orbit auth set");
        if (shard.gpu_main[d])
            ck(cudaMalloc(&mp[d], size_t(shard.gpu_main[d]) * sizeof(Count)),
               "maskshard-orbit auth main");
        if (shard.gpu_block[d])
            ck(cudaMalloc(&bp[d], size_t(shard.gpu_block[d]) * sizeof(Count)),
               "maskshard-orbit auth block");
        if (mp[d])
            ck(cudaMemset(mp[d], 0, size_t(shard.gpu_main[d]) * sizeof(Count)),
               "maskshard-orbit zero main");
        if (bp[d])
            ck(cudaMemset(bp[d], 0, size_t(shard.gpu_block[d]) * sizeof(Count)),
               "maskshard-orbit zero block");
    }

    std::vector<StorageDeviceTables> factor_dev(ngpu);
    std::vector<HighDescDeviceTables> highdesc_dev(ngpu);
    std::vector<MaskShardDeviceMeta> shard_dev(ngpu);
    std::vector<MaskShardOrbitWorker> workers(ngpu);
    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "maskshard-orbit table set");
        factor_dev[d].install(storage, G_FACTOR);
        highdesc_dev[d].install(high_desc);
        shard_dev[d].install(d, shard, layout, high_route, mp, bp);
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "maskshard-orbit modulus");
        workers[d].init(d);
    }

    const MateID init = MateID(R) << (2 * (TARGET_W - 1));
    const MaskShardOrbitAddress ia = maskshard_orbit_main_address_host(init, storage, layout, shard);
    const MaskShardOrbitAddress oa = maskshard_orbit_main_address_host(MateID(R), storage, layout, shard);
    const Count one = 1;
    ck(cudaSetDevice(ia.owner), "maskshard-orbit init device");
    ck(cudaMemcpy(mp[ia.owner] + ia.offset, &one, sizeof(one), cudaMemcpyHostToDevice),
       "maskshard-orbit init one");

    const auto high_jobs = build_high_orbit_jobs();
    std::vector<std::vector<uint32_t>> low_jobs(ngpu);
    for (uint32_t mask = 0; mask < shard.masks; ++mask)
        low_jobs[shard.owner[mask]].push_back(mask);
    for (auto& v : low_jobs)
        std::sort(v.begin(), v.end(), [&](uint32_t x, uint32_t y) {
            return shard.main_group_size[x] + shard.block_group_size[x]
                 > shard.main_group_size[y] + shard.block_group_size[y];
        });

    const auto wall0 = std::chrono::steady_clock::now();
    for (int row = 0; row < TARGET_W; ++row) {
        std::atomic<size_t> next_high{0};
        std::vector<std::thread> ts;
        ts.reserve(ngpu);
        for (int d = 0; d < ngpu; ++d) {
            ts.emplace_back([&, d] {
                for (;;) {
                    const size_t q = next_high.fetch_add(1, std::memory_order_relaxed);
                    if (q >= high_jobs.size()) break;
                    process_high_orbit_job(workers[d], high_jobs[q], threads);
                }
            });
        }
        for (auto& t : ts) t.join();

        ts.clear();
        for (int d = 0; d < ngpu; ++d) {
            ts.emplace_back([&, d] {
                ck(cudaSetDevice(d), "maskshard-orbit low worker set");
                for (uint32_t mask : low_jobs[d]) {
                    const Code mn = shard.main_group_size[mask];
                    const Code bn = shard.block_group_size[mask];
                    workers[d].buffers(mn, bn);
                    maskshard_process_low_group_buffers(
                        workers[d].low, shard, mask, mp[d], bp[d],
                        workers[d].a, workers[d].d, threads);
                }
            });
        }
        for (auto& t : ts) t.join();
        std::cerr << "maskshard-orbit row " << row + 1 << '/' << TARGET_W << '\n';
    }
    const double wall_s = ram_seconds_since(wall0);

    Count answer = 0;
    ck(cudaSetDevice(oa.owner), "maskshard-orbit answer device");
    ck(cudaMemcpy(&answer, mp[oa.owner] + oa.offset, sizeof(answer), cudaMemcpyDeviceToHost),
       "maskshard-orbit answer");

    double high_io = 0, high_orbit = 0, high_closure = 0;
    double low_id = 0, low_kernel = 0, low_copy = 0;
    uint64_t high_groups_done = 0, low_groups_done = 0, low_copy_groups = 0;
    size_t max_scratch = 0;
    for (const auto& w : workers) {
        high_io += w.high_io_s;
        high_orbit += w.high_orbit_s;
        high_closure += w.high_closure_s;
        low_id += w.low.identity_s;
        low_kernel += w.low.kernel_s;
        low_copy += w.low.copyback_s;
        high_groups_done += w.high_groups;
        low_groups_done += w.low.groups;
        low_copy_groups += w.low.copyback_groups;
        max_scratch = std::max(max_scratch, w.cap);
    }

    const double cross_frac = high_desc.main_observations
        ? double(high_desc.main_cross) / double(high_desc.main_observations) : 0.0;
    std::cout << "backend=b300-factorized-maskshard-v0.3-highorbit"
              << " n=" << n
              << " gpus=" << ngpu
              << " residue=" << answer
              << " modulus=" << mod
              << " wall_s=" << wall_s
              << " high_io_sum_s=" << high_io
              << " high_orbit_sum_s=" << high_orbit
              << " high_closure_sum_s=" << high_closure
              << " high_identity_sum_s=0"
              << " low_identity_sum_s=" << low_id
              << " low_kernel_sum_s=" << low_kernel
              << " low_copyback_sum_s=" << low_copy
              << " highdesc_build_s=" << highdesc_build_s
              << " highdesc_mib_per_gpu=" << highdesc_mib
              << " highdesc_cross_frac=" << cross_frac
              << " high_groups=" << high_groups_done
              << " low_groups=" << low_groups_done
              << " low_copyback_groups=" << low_copy_groups
              << " max_scratch_gib=" << double(max_scratch) / double(1ULL << 30)
              << '\n';

    for (int d = 0; d < ngpu; ++d) {
        ck(cudaSetDevice(d), "maskshard-orbit cleanup set");
        workers[d].release();
        shard_dev[d].release();
        highdesc_dev[d].release();
        factor_dev[d].release();
        if (mp[d]) cudaFree(mp[d]);
        if (bp[d]) cudaFree(bp[d]);
    }
    return 0;
}
