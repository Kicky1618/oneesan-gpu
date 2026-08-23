#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../gridfp/ramstream32_factorized_storage.hpp"
#include "../gridfp/ramstream32_highdesc.cuh"
#include "maskshard_layout.hpp"
#include "maskshard_lowlocal.cuh"

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

static size_t factor_device_bytes(
    const StorageFactorHost& storage, const FactorTablesHost& factor
) {
    size_t words = 0;
    words += storage.low_all_codes.size();
    words += factor.low_mask_codes.size();
    words += factor.low_mask_off.size();
    words += storage.low_packed_rank.size();
    words += storage.high_all_codes.size();
    words += factor.high_mask_codes.size();
    words += factor.high_mask_off.size();
    words += storage.high_packed_rank.size();
    return words * sizeof(uint32_t);
}

static size_t maskshard_meta_bytes(
    const MaskShardLayout& shard, const std::vector<uint32_t>& high_route
) {
    return shard.owner.size() * sizeof(uint8_t)
        + (shard.main_base.size() + shard.block_base.size()
           + shard.main_block_off.size() + shard.block_block_off.size()) * sizeof(Code)
        + high_route.size() * sizeof(uint32_t);
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    const double usable_gib = argc > 3 ? std::atof(argv[3]) : 268.59;
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
    HighDescHost high_desc = build_high_descriptors(storage, layout);

    report_high_mask_shard_layout(shard);

    Code max_low_group = 0;
    for (uint32_t mask = 0; mask < shard.masks; ++mask)
        max_low_group = std::max(max_low_group,
            shard.main_group_size[mask] + shard.block_group_size[mask]);

    Code max_high_group = 0;
    constexpr uint32_t LOW_MASKS = 1u << LOW_LUT_K;
    for (uint32_t mask = 0; mask < LOW_MASKS; ++mask) {
        const auto mb = make_factor_main_blocks(true, mask);
        const auto db = make_factor_block_blocks(true, mask);
        const Code mn = mb.empty() ? 0 : mb.back().end;
        const Code dn = db.empty() ? 0 : db.back().end;
        max_high_group = std::max(max_high_group, mn + dn);
    }

    const size_t low_scratch_bytes = size_t(max_low_group) * sizeof(Count);
    const size_t high_scratch_bytes = size_t(max_high_group) * 2 * sizeof(Count);
    const size_t scratch_bytes = std::max(low_scratch_bytes, high_scratch_bytes);
    const size_t factor_bytes = factor_device_bytes(storage, G_FACTOR);
    const size_t meta_bytes = maskshard_meta_bytes(shard, high_route);
    const size_t highdesc_bytes =
        (high_desc.main_desc.size() + high_desc.block_desc.size()) * sizeof(uint32_t);

    Code max_auth_states = 0;
    for (int d = 0; d < ngpu; ++d)
        max_auth_states = std::max(max_auth_states, shard.gpu_main[d] + shard.gpu_block[d]);
    const long double max_auth_bytes = static_cast<long double>(max_auth_states) * sizeof(Count);
    const long double peak_v01_bytes = max_auth_bytes + scratch_bytes + factor_bytes + meta_bytes;
    const long double peak_v02_bytes = peak_v01_bytes + highdesc_bytes;
    const long double gib = static_cast<long double>(1ULL << 30);
    const double peak_v01_gib = double(peak_v01_bytes / gib);
    const double peak_v02_gib = double(peak_v02_bytes / gib);

    std::cout << "backend=b300-factorized-maskshard-plan"
              << " n=" << n
              << " gpus=" << ngpu
              << " states=" << (layout.main_size + layout.block_size)
              << " authoritative_gib="
              << double(static_cast<long double>(layout.main_size + layout.block_size) * sizeof(Count) / gib)
              << " max_authoritative_gpu_gib=" << double(max_auth_bytes / gib)
              << " max_high_mask_group_gib=" << double(static_cast<long double>(low_scratch_bytes) / gib)
              << " low_alt_scratch_gib=" << double(static_cast<long double>(low_scratch_bytes) / gib)
              << " high_pingpong_scratch_gib=" << double(static_cast<long double>(high_scratch_bytes) / gib)
              << " shared_scratch_peak_gib=" << double(static_cast<long double>(scratch_bytes) / gib)
              << " factor_tables_mib=" << double(factor_bytes) / double(1ULL << 20)
              << " maskshard_meta_mib=" << double(meta_bytes) / double(1ULL << 20)
              << " high_route_mib="
              << double(high_route.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " highdesc_mib_per_gpu=" << double(highdesc_bytes) / double(1ULL << 20)
              << " v01_peak_gib=" << peak_v01_gib
              << " v02_highdesc_peak_gib=" << peak_v02_gib
              << " usable_gib=" << usable_gib
              << " v02_headroom_gib=" << (usable_gib - peak_v02_gib)
              << " low_window_direct_rank=1"
              << " low_window_copyback=" << (LOW_LUT_K & 1)
              << '\n';

    if (peak_v02_gib >= usable_gib) {
        std::cerr << "maskshard HIGH-descriptor plan exceeds requested usable HBM: peak="
                  << peak_v02_gib << " usable=" << usable_gib << '\n';
        return 128;
    }

    for (int d = 0; d < ngpu; ++d) {
        std::cout << "gpu=" << d
                  << " main_gib=" << double(static_cast<long double>(shard.gpu_main[d]) * sizeof(Count) / gib)
                  << " block_gib=" << double(static_cast<long double>(shard.gpu_block[d]) * sizeof(Count) / gib)
                  << " total_gib="
                  << double(static_cast<long double>(shard.gpu_main[d] + shard.gpu_block[d]) * sizeof(Count) / gib)
                  << '\n';
    }
    return 0;
}
