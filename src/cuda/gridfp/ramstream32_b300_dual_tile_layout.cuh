#pragma once

#include "ramstream32_b300_precomputed_w28_partition.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

// Alternating HBM layout.
//
// Every StorageBlock is split into an 8x8 grid of ownership tiles:
//   tile(hi, lo) = HIGH-owner hi rows x LOW-owner lo columns.
// The directed tile stream (hi,lo) has one stable element order independent of
// the active orientation.  In HIGH orientation it lives on GPU hi; in LOW
// orientation it lives on GPU lo.  GPU g reserves, for every peer p, a slot of
// max(stream(g,p), stream(p,g)) elements, so changing orientation needs no full
// second authoritative buffer: pair (g,p) only swaps the contents of its two
// slots.  The diagonal stream never moves.
struct B300DualTileHost {
    int ngpu = 0;
    B300DirectMaskShardHost high;

    std::vector<uint8_t> low_mask_owner;
    std::vector<uint8_t> low_owner;
    std::vector<uint32_t> low_local;
    std::array<std::vector<uint32_t>, MAXGPU> owned_low_cols;
    std::array<std::array<uint32_t, MAXW + 2>, MAXGPU> owned_low_off{};

    std::array<std::array<uint32_t, MAXW + 2>, MAXGPU> high_count{};
    std::array<std::array<uint32_t, MAXW + 2>, MAXGPU> low_count{};

    // Flattened as ((hi * ngpu + lo) * nblocks + bid).
    std::vector<Code> pair_main_off;
    std::vector<Code> pair_block_off;
    std::array<std::array<Code, MAXGPU>, MAXGPU> pair_main_size{};
    std::array<std::array<Code, MAXGPU>, MAXGPU> pair_block_size{};

    // Physical slot bases on each owning GPU.  In HIGH orientation GPU g/slot p
    // contains stream(g,p); in LOW orientation it contains stream(p,g).
    std::array<std::array<Code, MAXGPU>, MAXGPU> main_slot_base{};
    std::array<std::array<Code, MAXGPU>, MAXGPU> block_slot_base{};
    std::array<std::array<Code, MAXGPU>, MAXGPU> main_slot_cap{};
    std::array<std::array<Code, MAXGPU>, MAXGPU> block_slot_cap{};
    std::array<Code, MAXGPU> main_count{};
    std::array<Code, MAXGPU> block_count{};
};

static std::vector<uint8_t> b300_dual_low_mask_lpt(
    const StorageFactorHost& storage, const StorageLayout& layout, int ngpu
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t NM = 1u << LOW_LUT_K;
    std::vector<unsigned long long> weight(NM, 0);
    auto add = [&](const StorageBlock& b) {
        if (!b.valid || !b.rows || !b.cols) return;
        for (uint32_t mask = 0; mask < NM; ++mask) {
            size_t ix = size_t(mask) * S + b.hs;
            uint32_t a = G_FACTOR.low_mask_off[ix];
            uint32_t e = G_FACTOR.low_mask_off[ix + 1];
            weight[mask] += (unsigned long long)(e - a) * b.rows * sizeof(Count);
        }
    };
    for (const auto& b : layout.main_blocks) add(b);
    for (const auto& b : layout.block_blocks) add(b);

    std::vector<uint32_t> order(NM);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return weight[a] != weight[b] ? weight[a] > weight[b] : a < b;
    });
    std::array<unsigned long long, MAXGPU> bytes{};
    std::vector<uint8_t> owner(NM, 0);
    for (uint32_t mask : order) {
        int g = 0;
        for (int q = 1; q < ngpu; ++q) if (bytes[q] < bytes[g]) g = q;
        owner[mask] = uint8_t(g);
        bytes[g] += weight[mask];
    }
    return owner;
}

static B300DualTileHost build_b300_dual_tile_layout(
    const StorageFactorHost& storage, const StorageLayout& layout, int ngpu
) {
    if (ngpu < 1 || ngpu > MAXGPU) std::exit(510);
    B300DualTileHost z;
    z.ngpu = ngpu;
    z.high = build_b300_direct_mask_shards_w28_precomputed(storage, layout, ngpu);
    z.low_mask_owner = b300_dual_low_mask_lpt(storage, layout, ngpu);
    z.low_owner.resize(storage.low_all_codes.size());
    z.low_local.resize(storage.low_all_codes.size());

    // HIGH counts come directly from the precomputed HIGH-mask shard.
    for (int g = 0; g < ngpu; ++g)
        for (int h = 0; h <= MAXW; ++h)
            z.high_count[g][h] = z.high.owned_off[g][h + 1] - z.high.owned_off[g][h];

    // Build LOW all-rank -> owner/local-rank and per-GPU owned column lists.
    for (int h = 0; h <= MAXW; ++h) {
        std::array<uint32_t, MAXGPU> local{};
        uint32_t n = storage.low_all_off[h + 1] - storage.low_all_off[h];
        for (uint32_t lr = 0; lr < n; ++lr) {
            uint32_t ai = storage.low_all_off[h] + lr;
            uint32_t mask = seg_occ(storage.low_all_codes[ai], LOW_LUT_K);
            int g = z.low_mask_owner[mask];
            z.low_owner[ai] = uint8_t(g);
            z.low_local[ai] = local[g]++;
        }
        for (int g = 0; g < ngpu; ++g) z.low_count[g][h] = local[g];
    }
    for (int g = 0; g < ngpu; ++g) {
        for (int h = 0; h <= MAXW; ++h) {
            z.owned_low_off[g][h] = uint32_t(z.owned_low_cols[g].size());
            uint32_t n = storage.low_all_off[h + 1] - storage.low_all_off[h];
            for (uint32_t lr = 0; lr < n; ++lr) {
                uint32_t ai = storage.low_all_off[h] + lr;
                if (z.low_owner[ai] == g) z.owned_low_cols[g].push_back(lr);
            }
        }
        z.owned_low_off[g][MAXW + 1] = uint32_t(z.owned_low_cols[g].size());
    }

    const int mn = int(layout.main_blocks.size());
    const int bn = int(layout.block_blocks.size());
    z.pair_main_off.resize(size_t(ngpu) * ngpu * mn);
    z.pair_block_off.resize(size_t(ngpu) * ngpu * bn);
    auto mix = [=](int hi, int lo, int bid) {
        return (size_t(hi) * ngpu + lo) * mn + bid;
    };
    auto bix = [=](int hi, int lo, int bid) {
        return (size_t(hi) * ngpu + lo) * bn + bid;
    };

    for (int hi = 0; hi < ngpu; ++hi) for (int lo = 0; lo < ngpu; ++lo) {
        Code off = 0;
        for (int bid = 0; bid < mn; ++bid) {
            z.pair_main_off[mix(hi, lo, bid)] = off;
            const auto& b = layout.main_blocks[bid];
            off += Code(z.high_count[hi][b.he]) * z.low_count[lo][b.hs];
        }
        z.pair_main_size[hi][lo] = off;
        off = 0;
        for (int bid = 0; bid < bn; ++bid) {
            z.pair_block_off[bix(hi, lo, bid)] = off;
            const auto& b = layout.block_blocks[bid];
            off += Code(z.high_count[hi][b.he]) * z.low_count[lo][b.hs];
        }
        z.pair_block_size[hi][lo] = off;
    }

    for (int g = 0; g < ngpu; ++g) {
        Code moff = 0, boff = 0;
        for (int p = 0; p < ngpu; ++p) {
            z.main_slot_base[g][p] = moff;
            z.block_slot_base[g][p] = boff;
            z.main_slot_cap[g][p] = std::max(z.pair_main_size[g][p], z.pair_main_size[p][g]);
            z.block_slot_cap[g][p] = std::max(z.pair_block_size[g][p], z.pair_block_size[p][g]);
            moff += z.main_slot_cap[g][p];
            boff += z.block_slot_cap[g][p];
        }
        z.main_count[g] = moff;
        z.block_count[g] = boff;
    }
    return z;
}

__constant__ uint8_t* D_DT_HIGH_OWNER;
__constant__ uint32_t* D_DT_HIGH_LOCAL;
__constant__ uint8_t* D_DT_LOW_OWNER;
__constant__ uint32_t* D_DT_LOW_LOCAL;
__constant__ uint32_t* D_DT_HIGH_OWNED_ROWS;
__constant__ uint32_t* D_DT_LOW_OWNED_COLS;
__constant__ uint32_t D_DT_HIGH_OWNED_OFF[MAXW + 2];
__constant__ uint32_t D_DT_LOW_OWNED_OFF[MAXW + 2];
__constant__ uint32_t D_DT_HIGH_COUNT[MAXGPU][MAXW + 2];
__constant__ uint32_t D_DT_LOW_COUNT[MAXGPU][MAXW + 2];
__constant__ Code D_DT_MAIN_SLOT_BASE[MAXGPU][MAXGPU];
__constant__ Code D_DT_BLOCK_SLOT_BASE[MAXGPU][MAXGPU];
__constant__ Code* D_DT_PAIR_MAIN_OFF;
__constant__ Code* D_DT_PAIR_BLOCK_OFF;
__constant__ int D_DT_MAIN_NBLOCKS;
__constant__ int D_DT_BLOCK_NBLOCKS;

struct B300DualTileDeviceTables {
    uint8_t *high_owner = nullptr, *low_owner = nullptr;
    uint32_t *high_local = nullptr, *low_local = nullptr;
    uint32_t *high_rows = nullptr, *low_cols = nullptr;
    Code *pair_main_off = nullptr, *pair_block_off = nullptr;

    template<class T>
    static void upload(T** p, const std::vector<T>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(p, v.size() * sizeof(T)), what);
        ck(cudaMemcpy(*p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(const B300DualTileHost& z, const StorageLayout& layout, int g) {
        upload(&high_owner, z.high.high_owner, "dual high owner");
        upload(&high_local, z.high.high_local, "dual high local");
        upload(&low_owner, z.low_owner, "dual low owner");
        upload(&low_local, z.low_local, "dual low local");
        upload(&high_rows, z.high.owned_rows[g], "dual owned high rows");
        upload(&low_cols, z.owned_low_cols[g], "dual owned low cols");
        upload(&pair_main_off, z.pair_main_off, "dual pair main off");
        upload(&pair_block_off, z.pair_block_off, "dual pair block off");

        ck(cudaMemcpyToSymbol(D_DT_HIGH_OWNER, &high_owner, sizeof(high_owner)), "dual high owner ptr");
        ck(cudaMemcpyToSymbol(D_DT_HIGH_LOCAL, &high_local, sizeof(high_local)), "dual high local ptr");
        ck(cudaMemcpyToSymbol(D_DT_LOW_OWNER, &low_owner, sizeof(low_owner)), "dual low owner ptr");
        ck(cudaMemcpyToSymbol(D_DT_LOW_LOCAL, &low_local, sizeof(low_local)), "dual low local ptr");
        ck(cudaMemcpyToSymbol(D_DT_HIGH_OWNED_ROWS, &high_rows, sizeof(high_rows)), "dual high rows ptr");
        ck(cudaMemcpyToSymbol(D_DT_LOW_OWNED_COLS, &low_cols, sizeof(low_cols)), "dual low cols ptr");
        ck(cudaMemcpyToSymbol(D_DT_HIGH_OWNED_OFF, z.high.owned_off[g].data(),
                              sizeof(uint32_t) * (MAXW + 2)), "dual high rows off");
        ck(cudaMemcpyToSymbol(D_DT_LOW_OWNED_OFF, z.owned_low_off[g].data(),
                              sizeof(uint32_t) * (MAXW + 2)), "dual low cols off");
        ck(cudaMemcpyToSymbol(D_DT_HIGH_COUNT, z.high_count.data(), sizeof(z.high_count)), "dual high count");
        ck(cudaMemcpyToSymbol(D_DT_LOW_COUNT, z.low_count.data(), sizeof(z.low_count)), "dual low count");
        ck(cudaMemcpyToSymbol(D_DT_MAIN_SLOT_BASE, z.main_slot_base.data(), sizeof(z.main_slot_base)), "dual main slots");
        ck(cudaMemcpyToSymbol(D_DT_BLOCK_SLOT_BASE, z.block_slot_base.data(), sizeof(z.block_slot_base)), "dual block slots");
        int mn = int(layout.main_blocks.size()), bn = int(layout.block_blocks.size());
        ck(cudaMemcpyToSymbol(D_DT_MAIN_NBLOCKS, &mn, sizeof(mn)), "dual main nblocks");
        ck(cudaMemcpyToSymbol(D_DT_BLOCK_NBLOCKS, &bn, sizeof(bn)), "dual block nblocks");
        ck(cudaMemcpyToSymbol(D_DT_PAIR_MAIN_OFF, &pair_main_off, sizeof(pair_main_off)), "dual pair main ptr");
        ck(cudaMemcpyToSymbol(D_DT_PAIR_BLOCK_OFF, &pair_block_off, sizeof(pair_block_off)), "dual pair block ptr");
    }

    void release() {
        if (high_owner) cudaFree(high_owner);
        if (high_local) cudaFree(high_local);
        if (low_owner) cudaFree(low_owner);
        if (low_local) cudaFree(low_local);
        if (high_rows) cudaFree(high_rows);
        if (low_cols) cudaFree(low_cols);
        if (pair_main_off) cudaFree(pair_main_off);
        if (pair_block_off) cudaFree(pair_block_off);
        high_owner = low_owner = nullptr;
        high_local = low_local = high_rows = low_cols = nullptr;
        pair_main_off = pair_block_off = nullptr;
    }
};

__device__ __forceinline__ Code b300_dt_pair_main_off(int hi, int lo, uint32_t bid) {
    return D_DT_PAIR_MAIN_OFF[(size_t(hi) * D_NGPU + lo) * D_DT_MAIN_NBLOCKS + bid];
}
__device__ __forceinline__ Code b300_dt_pair_block_off(int hi, int lo, uint32_t bid) {
    return D_DT_PAIR_BLOCK_OFF[(size_t(hi) * D_NGPU + lo) * D_DT_BLOCK_NBLOCKS + bid];
}

__device__ __forceinline__ Count* b300_dt_main_high(uint32_t bid, uint32_t hr, uint32_t lr) {
    StorageBlock b = D_DR_MAIN_BLOCKS[bid];
    uint32_t hai = D_F_HIGH_ALL_OFF[b.he] + hr, lai = D_F_LOW_ALL_OFF[b.hs] + lr;
    int hi = D_DT_HIGH_OWNER[hai], lo = D_DT_LOW_OWNER[lai];
    uint32_t h = D_DT_HIGH_LOCAL[hai], l = D_DT_LOW_LOCAL[lai];
    Code k = b300_dt_pair_main_off(hi, lo, bid)
           + Code(h) * D_DT_LOW_COUNT[lo][b.hs] + l;
    return D_MAIN_PTR[hi] + D_DT_MAIN_SLOT_BASE[hi][lo] + k;
}
__device__ __forceinline__ Count* b300_dt_block_high(uint32_t bid, uint32_t hr, uint32_t lr) {
    StorageBlock b = D_DR_BLOCK_BLOCKS[bid];
    uint32_t hai = D_F_HIGH_ALL_OFF[b.he] + hr, lai = D_F_LOW_ALL_OFF[b.hs] + lr;
    int hi = D_DT_HIGH_OWNER[hai], lo = D_DT_LOW_OWNER[lai];
    uint32_t h = D_DT_HIGH_LOCAL[hai], l = D_DT_LOW_LOCAL[lai];
    Code k = b300_dt_pair_block_off(hi, lo, bid)
           + Code(h) * D_DT_LOW_COUNT[lo][b.hs] + l;
    return D_BLOCK_PTR[hi] + D_DT_BLOCK_SLOT_BASE[hi][lo] + k;
}
__device__ __forceinline__ Count* b300_dt_main_low(uint32_t bid, uint32_t hr, uint32_t lr) {
    StorageBlock b = D_DR_MAIN_BLOCKS[bid];
    uint32_t hai = D_F_HIGH_ALL_OFF[b.he] + hr, lai = D_F_LOW_ALL_OFF[b.hs] + lr;
    int hi = D_DT_HIGH_OWNER[hai], lo = D_DT_LOW_OWNER[lai];
    uint32_t h = D_DT_HIGH_LOCAL[hai], l = D_DT_LOW_LOCAL[lai];
    Code k = b300_dt_pair_main_off(hi, lo, bid)
           + Code(h) * D_DT_LOW_COUNT[lo][b.hs] + l;
    return D_MAIN_PTR[lo] + D_DT_MAIN_SLOT_BASE[lo][hi] + k;
}
__device__ __forceinline__ Count* b300_dt_block_low(uint32_t bid, uint32_t hr, uint32_t lr) {
    StorageBlock b = D_DR_BLOCK_BLOCKS[bid];
    uint32_t hai = D_F_HIGH_ALL_OFF[b.he] + hr, lai = D_F_LOW_ALL_OFF[b.hs] + lr;
    int hi = D_DT_HIGH_OWNER[hai], lo = D_DT_LOW_OWNER[lai];
    uint32_t h = D_DT_HIGH_LOCAL[hai], l = D_DT_LOW_LOCAL[lai];
    Code k = b300_dt_pair_block_off(hi, lo, bid)
           + Code(h) * D_DT_LOW_COUNT[lo][b.hs] + l;
    return D_BLOCK_PTR[lo] + D_DT_BLOCK_SLOT_BASE[lo][hi] + k;
}
