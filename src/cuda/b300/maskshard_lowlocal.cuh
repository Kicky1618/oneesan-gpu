#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

// Shard-local LOW-window executor for MaskShardLayout.
//
// The authoritative HIGH-mask group is already laid out exactly like
// make_factor_*_blocks(false, mask). The authoritative group itself is one
// ping-pong side; only one alternate M+D buffer is required. No group
// gather/scatter and no canonical rank conversion occur.

struct MaskShardLowScratch {
    uint8_t* arena = nullptr;
    size_t cap = 0;
    Count* main_alt = nullptr;
    Count* block_alt = nullptr;

    void ensure(Code main_n, Code block_n) {
        auto al = [](size_t x) { return (x + 255) & ~size_t(255); };
        const size_t mb = al(size_t(main_n) * sizeof(Count));
        const size_t db = al(size_t(block_n) * sizeof(Count));
        const size_t need = mb + db;
        if (need > cap) {
            if (arena) ck(cudaFree(arena), "maskshard low free scratch");
            cap = need;
            ck(cudaMalloc(&arena, cap), "maskshard low scratch");
        }
        main_alt = reinterpret_cast<Count*>(arena);
        block_alt = reinterpret_cast<Count*>(arena + mb);
    }

    void release() {
        if (arena) cudaFree(arena);
        arena = nullptr;
        cap = 0;
        main_alt = block_alt = nullptr;
    }
};

struct MaskShardLowStats {
    double identity_s = 0.0;
    double kernel_s = 0.0;
    double copyback_s = 0.0;
    uint64_t groups = 0;
    uint64_t copyback_groups = 0;
};

static void maskshard_configure_low_group(uint32_t mask) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t HM = (1u << H) - 1u;

    const uint32_t mf = HM << (L + 1);
    const uint32_t mo = mask << (L + 1);
    const uint32_t bf = HM << L;
    const uint32_t bo = mask << L;
    const GroupSpec ms = make_spec(TARGET_W, mf, mo);
    const GroupSpec ds = make_spec(TARGET_W - 1, bf, bo);
    std::vector<FBlock> mb = make_factor_main_blocks(false, mask);
    std::vector<FBlock> db = make_factor_block_blocks(false, mask);

    if (mb.empty() || db.empty() || mb.back().end != ms.size || db.back().end != ds.size) {
        std::cerr << "maskshard low group factor-size mismatch mask=" << mask << '\n';
        std::exit(130);
    }

    const int mn = int(mb.size());
    const int dn = int(db.size());
    const int fix_low = 0;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, mb.data(), mb.size() * sizeof(FBlock)),
       "maskshard low main blocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, db.data(), db.size() * sizeof(FBlock)),
       "maskshard low block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &mn, sizeof(mn)), "maskshard low main nblocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &dn, sizeof(dn)), "maskshard low block nblocks");
    ck(cudaMemcpyToSymbol(D_F_MASK, &mask, sizeof(mask)), "maskshard low mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fix_low, sizeof(fix_low)), "maskshard low mode");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &mf, sizeof(mf)), "maskshard low main fixed");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &mo, sizeof(mo)), "maskshard low main occ");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &bf, sizeof(bf)), "maskshard low block fixed");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &bo, sizeof(bo)), "maskshard low block occ");
    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "maskshard low main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "maskshard low block dp");
}

static void maskshard_process_low_group_buffers(
    MaskShardLowStats& stats,
    const MaskShardLayout& shard,
    uint32_t mask,
    Count* authoritative_main,
    Count* authoritative_block,
    Count* main_alt,
    Count* block_alt,
    int threads = 256
) {
    if (mask >= shard.masks) {
        std::cerr << "maskshard low invalid mask=" << mask << '\n';
        std::exit(131);
    }
    const Code main_n = shard.main_group_size[mask];
    const Code block_n = shard.block_group_size[mask];
    if (!main_n && !block_n) return;

    maskshard_configure_low_group(mask);

    Count* auth_m = authoritative_main + shard.main_base[mask];
    Count* auth_b = authoritative_block + shard.block_base[mask];
    Count* cur = auth_m;
    Count* dcur = auth_b;
    Count* nxt = main_alt;
    Count* dnext = block_alt;

    const int bm = int(std::min<Code>(65535, (main_n + threads - 1) / threads));
    const int bd = int(std::min<Code>(65535, (block_n + threads - 1) / threads));

    for (int p = LOW_LUT_K; p >= 1; --p) {
        auto t = std::chrono::steady_clock::now();
        if (main_n)
            ck(cudaMemcpy(nxt, cur, size_t(main_n) * sizeof(Count), cudaMemcpyDeviceToDevice),
               "maskshard low identity");
        if (block_n)
            ck(cudaMemset(dnext, 0, size_t(block_n) * sizeof(Count)),
               "maskshard low clear blocked");
        stats.identity_s += ram_seconds_since(t);

        t = std::chrono::steady_clock::now();
        if (main_n)
            main_group_kernel<<<bm, threads>>>(cur, nullptr, main_n, nxt, dnext, p);
        if (block_n)
            blocked_group_kernel<<<bd, threads>>>(dcur, block_n, nxt, p);
        ck(cudaGetLastError(), "maskshard low transition");
        ck(cudaDeviceSynchronize(), "maskshard low transition sync");
        stats.kernel_s += ram_seconds_since(t);

        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }

    // LOW_LUT_K=14 (n=27 production) is even, so cur/dcur are authoritative
    // already. Keep a fallback copyback for odd regression widths.
    if (cur != auth_m || dcur != auth_b) {
        auto t = std::chrono::steady_clock::now();
        if (main_n && cur != auth_m)
            ck(cudaMemcpy(auth_m, cur, size_t(main_n) * sizeof(Count), cudaMemcpyDeviceToDevice),
               "maskshard low main copyback");
        if (block_n && dcur != auth_b)
            ck(cudaMemcpy(auth_b, dcur, size_t(block_n) * sizeof(Count), cudaMemcpyDeviceToDevice),
               "maskshard low block copyback");
        stats.copyback_s += ram_seconds_since(t);
        ++stats.copyback_groups;
    }
    ++stats.groups;
}

static void maskshard_process_low_group(
    MaskShardLowScratch& scratch,
    MaskShardLowStats& stats,
    const MaskShardLayout& shard,
    uint32_t mask,
    Count* authoritative_main,
    Count* authoritative_block,
    int threads = 256
) {
    const Code main_n = shard.main_group_size[mask];
    const Code block_n = shard.block_group_size[mask];
    scratch.ensure(main_n, block_n);
    maskshard_process_low_group_buffers(
        stats, shard, mask, authoritative_main, authoritative_block,
        scratch.main_alt, scratch.block_alt, threads);
}
