#pragma once

#include "ramstream32_b300_rowkernels.cuh"

// HIGH-window groups fix a LOW occupancy mask.  Their LOW-class widths are
// usually tiny (median/average far below a full CUDA block), so assigning a
// 256-thread block to one HIGH row wastes most lanes.  Pack 8 independent HIGH
// rows into the 8 warps of a 256-thread block.  A warp loops over its LOW class
// in steps of 32; even the largest n=27 class (~1000 columns) stays manageable.
static constexpr uint32_t B300_HIGH_WARPS_PER_BLOCK = 8;
static_assert(B300_HIGH_WARPS_PER_BLOCK * 32 == 256);

__constant__ uint32_t D_HW_MAIN_ROW_OFF[65];
__constant__ uint32_t D_HW_BLOCK_ROW_OFF[33];
__constant__ uint32_t D_HW_MAIN_ROWS;
__constant__ uint32_t D_HW_BLOCK_ROWS;

static inline std::pair<uint32_t,uint32_t> install_high_warp_row_prefixes(
    const std::vector<FBlock>& mb, const std::vector<FBlock>& db
) {
    std::array<uint32_t,65> mo{};
    std::array<uint32_t,33> bo{};
    uint64_t mr = 0, br = 0;
    for (size_t i = 0; i < mb.size(); ++i) {
        mo[i] = uint32_t(mr);
        mr += mb[i].stride ? (mb[i].end - mb[i].off) / mb[i].stride : 0;
        if (mr > 0xffffffffULL) std::exit(340);
    }
    mo[mb.size()] = uint32_t(mr);
    for (size_t i = 0; i < db.size(); ++i) {
        bo[i] = uint32_t(br);
        br += db[i].stride ? (db[i].end - db[i].off) / db[i].stride : 0;
        if (br > 0xffffffffULL) std::exit(341);
    }
    bo[db.size()] = uint32_t(br);
    uint32_t mru = uint32_t(mr), bru = uint32_t(br);
    ck(cudaMemcpyToSymbol(D_HW_MAIN_ROW_OFF, mo.data(), sizeof(mo)), "high-warp main rows");
    ck(cudaMemcpyToSymbol(D_HW_BLOCK_ROW_OFF, bo.data(), sizeof(bo)), "high-warp block rows");
    ck(cudaMemcpyToSymbol(D_HW_MAIN_ROWS, &mru, sizeof(mru)), "high-warp main total");
    ck(cudaMemcpyToSymbol(D_HW_BLOCK_ROWS, &bru, sizeof(bru)), "high-warp block total");
    return {mru, bru};
}

__device__ __forceinline__ int high_warp_find_block(
    uint32_t row, const uint32_t* off, int n
) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (row < off[m + 1]) hi = m;
        else lo = m + 1;
    }
    return lo;
}

struct HighWarpRowPos {
    FBlock x;
    int bid;
    uint32_t hr;
};

__device__ __forceinline__ HighWarpRowPos high_warp_main_pos(uint32_t row) {
    int bid = high_warp_find_block(row, D_HW_MAIN_ROW_OFF, D_F_MAIN_NBLOCKS);
    FBlock x = D_F_MAIN_BLOCKS[bid];
    return {x, bid, row - D_HW_MAIN_ROW_OFF[bid]};
}
__device__ __forceinline__ HighWarpRowPos high_warp_block_pos(uint32_t row) {
    int bid = high_warp_find_block(row, D_HW_BLOCK_ROW_OFF, D_F_BLOCK_NBLOCKS);
    FBlock x = D_F_BLOCK_BLOCKS[bid];
    return {x, bid, row - D_HW_BLOCK_ROW_OFF[bid]};
}

__device__ __forceinline__ uint32_t high_warp_row_id() {
    return uint32_t(blockIdx.x) * B300_HIGH_WARPS_PER_BLOCK + (threadIdx.x >> 5);
}
__device__ __forceinline__ uint32_t high_warp_lane() {
    return threadIdx.x & 31u;
}

__global__ void high_warp_gather_main_kernel(Count* out) {
    uint32_t row = high_warp_row_id();
    if (row >= D_HW_MAIN_ROWS) return;
    uint32_t lane = high_warp_lane();
    HighWarpRowPos z = high_warp_main_pos(row);
    Code gbase = compact_main_canonical_row_base(z.x, z.hr);
    Code lbase = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = lane; lr < z.x.stride; lr += 32) {
        uint32_t lar = compact_low_all_rank(z.x, lr);
        out[lbase + lr] = global_load_main(gbase + lar);
    }
}

__global__ void high_warp_scatter_main_kernel(const Count* in) {
    uint32_t row = high_warp_row_id();
    if (row >= D_HW_MAIN_ROWS) return;
    uint32_t lane = high_warp_lane();
    HighWarpRowPos z = high_warp_main_pos(row);
    Code gbase = compact_main_canonical_row_base(z.x, z.hr);
    Code lbase = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = lane; lr < z.x.stride; lr += 32) {
        uint32_t lar = compact_low_all_rank(z.x, lr);
        global_store_main(gbase + lar, in[lbase + lr]);
    }
}

__global__ void high_warp_scatter_block_kernel(const Count* in) {
    uint32_t row = high_warp_row_id();
    if (row >= D_HW_BLOCK_ROWS) return;
    uint32_t lane = high_warp_lane();
    HighWarpRowPos z = high_warp_block_pos(row);
    Code gbase = compact_block_canonical_row_base(z.x, z.hr);
    Code lbase = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = lane; lr < z.x.stride; lr += 32) {
        uint32_t lar = compact_low_all_rank(z.x, lr);
        global_store_block(gbase + lar, in[lbase + lr]);
    }
}

__global__ void high_warp_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t row = high_warp_row_id();
    if (row >= D_HW_MAIN_ROWS) return;
    uint32_t lane = high_warp_lane();
    HighWarpRowPos z = high_warp_main_pos(row);
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint64_t ow = D_HIGH_ORBIT[size_t(pi) * D_HIGH_ORBIT_MAIN_TOTAL
                              + D_HIGH_ORBIT_MAIN_BASE[z.bid] + z.hr];
    uint32_t kind = high_orbit_kind(ow);
    if (kind < HIGH_ORBIT_NN || kind > HIGH_ORBIT_NL) return;
    FBlock jy = D_F_MAIN_BLOCKS[high_orbit_jblock(ow)];
    FBlock dy = D_F_BLOCK_BLOCKS[high_orbit_dblock(ow)];
    Code ib = z.x.off + Code(z.hr) * z.x.stride;
    Code jb = jy.off + Code(high_orbit_jhr(ow)) * jy.stride;
    Code db = dy.off + Code(high_orbit_dhr(ow)) * dy.stride;
    for (uint32_t lr = lane; lr < z.x.stride; lr += 32) {
        Count c = mainv[ib + lr];
        Count d = blockv[db + lr];
        if (kind == HIGH_ORBIT_NN) {
            mainv[jb + lr] = high_orbit_add(mainv[jb + lr], c);
            mainv[ib + lr] = high_orbit_add(c, d);
            blockv[db + lr] = 0;
        } else {
            Count cc = mainv[jb + lr];
            mainv[ib + lr] = high_orbit_add(high_orbit_add(c, cc), d);
            blockv[db + lr] = c;
        }
    }
}

__global__ void high_warp_closure_kernel(Count* mainv, Count* blockv, int p) {
    constexpr int S = MAXW + 2;
    uint32_t row = high_warp_row_id();
    if (row >= D_HW_MAIN_ROWS) return;
    uint32_t lane = high_warp_lane();
    HighWarpRowPos z = high_warp_main_pos(row);
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint64_t ow = D_HIGH_ORBIT[size_t(pi) * D_HIGH_ORBIT_MAIN_TOTAL
                              + D_HIGH_ORBIT_MAIN_BASE[z.bid] + z.hr];
    if (high_orbit_kind(ow) != HIGH_ORBIT_CLOSURE) return;
    uint32_t desc = D_HIGHDESC_MAIN[size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                                   + D_HIGHDESC_MAIN_BASE[z.bid] + z.hr];
    uint32_t kind = highdesc_kind(desc);
    FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
    Code ib = z.x.off + Code(z.hr) * z.x.stride;
    Code db = y.off + Code(highdesc_rank(desc)) * y.stride;
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t lr = lane; lr < z.x.stride; lr += 32) {
            Count c = mainv[ib + lr];
            if (c) atomic_add_mod(blockv + db + lr, c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + z.x.hs];
        for (uint32_t lr = lane; lr < z.x.stride; lr += 32) {
            Count c = mainv[ib + lr];
            if (!c) continue;
            uint32_t lc = D_F_LOW_MASK_CODES[a + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = bidesc_low_mask_rank(lc2, y.hs);
            if (lr2 != 0xffffffffu) atomic_add_mod(blockv + db + lr2, c);
        }
    } else {
        asm("trap;");
    }
}

static inline uint32_t high_warp_blocks(uint32_t rows) {
    return (rows + B300_HIGH_WARPS_PER_BLOCK - 1) / B300_HIGH_WARPS_PER_BLOCK;
}
