#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

// Host-side layout for authoritative HBM sharded by HIGH occupancy mask.
//
// This header is included after oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu
// and ramstream32_factorized_storage.hpp, so Code, FBlock, StorageFactorHost and
// StorageLayout are already defined.
//
// Invariant: for one HIGH occupancy mask, main and blocked states are packed in
// exactly the same factor-block order used by a LOW-active factorized group.
// Therefore LOW-window local ranks are authoritative ranks within the owning
// GPU arena: no canonical rank/unrank and no gather/scatter are required.

struct MaskShardLayout {
    int ngpu = 0;
    uint32_t masks = 0;
    uint32_t main_nblocks = 0;
    uint32_t block_nblocks = 0;

    std::vector<uint8_t> owner;
    std::vector<Code> main_group_size;
    std::vector<Code> block_group_size;
    std::vector<Code> main_base;
    std::vector<Code> block_base;

    // Offsets are relative to the beginning of one mask group.
    std::vector<Code> main_block_off;   // [mask * main_nblocks + bid]
    std::vector<Code> block_block_off;  // [mask * block_nblocks + bid]

    std::array<Code, 8> gpu_main{};
    std::array<Code, 8> gpu_block{};
};

static uint32_t maskshard_mask_count(
    const std::vector<uint32_t>& mask_begin,
    const std::array<uint32_t, MAXW + 2>& all_off,
    uint32_t mask, uint32_t nmasks, int h
) {
    constexpr int S = StorageFactorHost::S;
    const uint32_t a = mask_begin[size_t(mask) * S + h];
    const uint32_t b = (mask + 1 < nmasks)
        ? mask_begin[size_t(mask + 1) * S + h]
        : all_off[h + 1] - all_off[h];
    return b - a;
}

static MaskShardLayout build_high_mask_shard_layout(
    const StorageFactorHost& storage,
    const StorageLayout& layout,
    int ngpu
) {
    constexpr int H = HIGH_LUT_K;
    const uint32_t nmasks = 1u << H;
    if (ngpu < 1 || ngpu > 8) {
        std::cerr << "maskshard requires 1..8 GPUs\n";
        std::exit(120);
    }

    MaskShardLayout s;
    s.ngpu = ngpu;
    s.masks = nmasks;
    s.main_nblocks = uint32_t(layout.main_blocks.size());
    s.block_nblocks = uint32_t(layout.block_blocks.size());
    s.owner.resize(nmasks);
    s.main_group_size.resize(nmasks);
    s.block_group_size.resize(nmasks);
    s.main_base.resize(nmasks);
    s.block_base.resize(nmasks);
    s.main_block_off.resize(size_t(nmasks) * s.main_nblocks);
    s.block_block_off.resize(size_t(nmasks) * s.block_nblocks);

    for (uint32_t mask = 0; mask < nmasks; ++mask) {
        Code off = 0;
        for (uint32_t bid = 0; bid < s.main_nblocks; ++bid) {
            s.main_block_off[size_t(mask) * s.main_nblocks + bid] = off;
            const StorageBlock& b = layout.main_blocks[bid];
            if (!b.valid || !b.cols) continue;
            const uint32_t rows = maskshard_mask_count(
                storage.high_mask_begin, storage.high_all_off,
                mask, nmasks, b.he);
            off += Code(rows) * b.cols;
        }
        s.main_group_size[mask] = off;

        off = 0;
        for (uint32_t bid = 0; bid < s.block_nblocks; ++bid) {
            s.block_block_off[size_t(mask) * s.block_nblocks + bid] = off;
            const StorageBlock& b = layout.block_blocks[bid];
            if (!b.valid || !b.cols) continue;
            const uint32_t rows = maskshard_mask_count(
                storage.high_mask_begin, storage.high_all_off,
                mask, nmasks, b.he);
            off += Code(rows) * b.cols;
        }
        s.block_group_size[mask] = off;
    }

    // Longest-processing-time greedy balancing by total authoritative count.
    // The n=27 distribution is extremely flat, but keep the planner generic.
    std::vector<uint32_t> order(nmasks);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const Code wa = s.main_group_size[a] + s.block_group_size[a];
        const Code wb = s.main_group_size[b] + s.block_group_size[b];
        return wa != wb ? wa > wb : a < b;
    });

    std::array<Code, 8> load{};
    for (uint32_t mask : order) {
        int d = 0;
        for (int q = 1; q < ngpu; ++q)
            if (load[q] < load[d]) d = q;
        s.owner[mask] = uint8_t(d);
        load[d] += s.main_group_size[mask] + s.block_group_size[mask];
    }

    // Pack mask groups independently in the main and blocked arenas while
    // preserving the common owner chosen above.
    for (uint32_t mask = 0; mask < nmasks; ++mask) {
        const int d = s.owner[mask];
        s.main_base[mask] = s.gpu_main[d];
        s.block_base[mask] = s.gpu_block[d];
        s.gpu_main[d] += s.main_group_size[mask];
        s.gpu_block[d] += s.block_group_size[mask];
    }

    Code total_main = 0, total_block = 0;
    for (int d = 0; d < ngpu; ++d) {
        total_main += s.gpu_main[d];
        total_block += s.gpu_block[d];
    }
    if (total_main != layout.main_size || total_block != layout.block_size) {
        std::cerr << "maskshard authoritative size mismatch main="
                  << total_main << '/' << layout.main_size
                  << " block=" << total_block << '/' << layout.block_size << '\n';
        std::exit(121);
    }
    return s;
}

static void report_high_mask_shard_layout(const MaskShardLayout& s) {
    Code mn = ~Code(0), mx = 0;
    for (int d = 0; d < s.ngpu; ++d) {
        const Code z = s.gpu_main[d] + s.gpu_block[d];
        mn = std::min(mn, z);
        mx = std::max(mx, z);
    }
    std::cerr << "maskshard high masks=" << s.masks
              << " gpu_min_gib=" << double(mn * sizeof(Count)) / double(1ULL << 30)
              << " gpu_max_gib=" << double(mx * sizeof(Count)) / double(1ULL << 30)
              << " imbalance=" << (mx ? double(mx - mn) / (double(mx + mn) * 0.5) : 0.0)
              << '\n';
}
