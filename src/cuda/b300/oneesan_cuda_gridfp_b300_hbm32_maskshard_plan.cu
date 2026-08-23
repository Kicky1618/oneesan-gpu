#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../gridfp/ramstream32_factorized_storage.hpp"
#include "maskshard_layout.hpp"

static void verify_low_window_direct_layout(
    const StorageFactorHost& storage,
    const StorageLayout& layout,
    const MaskShardLayout& shard
) {
    (void)storage;
    for (uint32_t mask = 0; mask < shard.masks; ++mask) {
        auto mb = make_factor_main_blocks(false, mask);
        auto db = make_factor_block_blocks(false, mask);
        if (mb.size() != layout.main_blocks.size() ||
            db.size() != layout.block_blocks.size()) {
            std::cerr << "maskshard factor block count mismatch mask=" << mask << '\n';
            std::exit(122);
        }

        for (uint32_t bid = 0; bid < shard.main_nblocks; ++bid) {
            const FBlock& f = mb[bid];
            const StorageBlock& a = layout.main_blocks[bid];
            const Code want = shard.main_block_off[size_t(mask) * shard.main_nblocks + bid];
            if (f.off != want || f.stride != a.cols) {
                std::cerr << "maskshard main local-rank mismatch mask=" << mask
                          << " bid=" << bid
                          << " off=" << f.off << '/' << want
                          << " stride=" << f.stride << '/' << a.cols << '\n';
                std::exit(123);
            }
        }
        for (uint32_t bid = 0; bid < shard.block_nblocks; ++bid) {
            const FBlock& f = db[bid];
            const StorageBlock& a = layout.block_blocks[bid];
            const Code want = shard.block_block_off[size_t(mask) * shard.block_nblocks + bid];
            if (f.off != want || f.stride != a.cols) {
                std::cerr << "maskshard block local-rank mismatch mask=" << mask
                          << " bid=" << bid
                          << " off=" << f.off << '/' << want
                          << " stride=" << f.stride << '/' << a.cols << '\n';
                std::exit(124);
            }
        }
        if ((!mb.empty() && mb.back().end != shard.main_group_size[mask]) ||
            (!db.empty() && db.back().end != shard.block_group_size[mask])) {
            std::cerr << "maskshard group end mismatch mask=" << mask << '\n';
            std::exit(125);
        }
    }
}

static std::vector<uint32_t> build_high_route_table(const StorageFactorHost& storage) {
    constexpr int H = HIGH_LUT_K;
    const uint32_t mask_mask = (1u << H) - 1u;
    std::vector<uint32_t> route(storage.high_all_codes.size(), 0xffffffffu);
    for (int he = 0; he <= H + 1; ++he) {
        const uint32_t a = storage.high_all_off[he];
        const uint32_t b = storage.high_all_off[he + 1];
        for (uint32_t r = 0; r < b - a; ++r) {
            const uint32_t code = storage.high_all_codes[a + r];
            const uint32_t packed = storage.high_packed_rank[code];
            if (packed == 0xffffffffu) {
                std::cerr << "maskshard route missing packed rank\n";
                std::exit(126);
            }
            const uint32_t mask = seg_occ(code, H);
            const uint32_t mask_rank = packed & mask_mask;
            if (mask >= (1u << H) || mask_rank >= (1u << H)) {
                std::cerr << "maskshard route encoding overflow\n";
                std::exit(127);
            }
            route[a + r] = mask | (mask_rank << H);
        }
    }
    return route;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (n + 1 != TARGET_W || n < 2 || TARGET_W > MAXW) {
        std::cerr << "maskshard plan specialized for n=" << TARGET_W - 1 << '\n';
        return 1;
    }
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) {
        std::cerr << "maskshard requires LOW+HIGH=W-1\n";
        return 1;
    }

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    MaskShardLayout shard = build_high_mask_shard_layout(storage, layout, ngpu);
    verify_low_window_direct_layout(storage, layout, shard);
    auto high_route = build_high_route_table(storage);

    report_high_mask_shard_layout(shard);
    Code max_group = 0;
    for (uint32_t mask = 0; mask < shard.masks; ++mask)
        max_group = std::max(max_group,
            shard.main_group_size[mask] + shard.block_group_size[mask]);

    std::cout << "backend=b300-factorized-maskshard-plan"
              << " n=" << n
              << " gpus=" << ngpu
              << " states=" << (layout.main_size + layout.block_size)
              << " authoritative_gib="
              << double((layout.main_size + layout.block_size) * sizeof(Count)) / double(1ULL << 30)
              << " max_high_mask_group_gib="
              << double(max_group * sizeof(Count)) / double(1ULL << 30)
              << " high_route_mib="
              << double(high_route.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " low_window_direct_rank=1"
              << '\n';

    for (int d = 0; d < ngpu; ++d) {
        std::cout << "gpu=" << d
                  << " main_gib=" << double(shard.gpu_main[d] * sizeof(Count)) / double(1ULL << 30)
                  << " block_gib=" << double(shard.gpu_block[d] * sizeof(Count)) / double(1ULL << 30)
                  << " total_gib=" << double((shard.gpu_main[d] + shard.gpu_block[d]) * sizeof(Count)) / double(1ULL << 30)
                  << '\n';
    }
    return 0;
}
