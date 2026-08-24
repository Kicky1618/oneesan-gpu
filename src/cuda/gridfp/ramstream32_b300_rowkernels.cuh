#pragma once

#include "ramstream32_b300_compact_io.cuh"
#include "ramstream32_high_orbit.cuh"
#include "ramstream32_low_orbit_device.cuh"

#include <array>

// Tile the LOW-column dimension in chunks of 1024 states.  HIGH-window
// occupancy classes are <= about 1000 columns at n=27, so they remain one
// block per HIGH row.  LOW-window rows can contain hundreds of thousands of
// LOW states, so they are automatically split into many blocks and keep all
// SMs busy.
static constexpr uint32_t CF_COL_TILE = 1024;

__constant__ uint32_t D_CF_MAIN_TILE_OFF[65];
__constant__ uint32_t D_CF_BLOCK_TILE_OFF[33];
__constant__ uint32_t D_CF_MAIN_TILES;
__constant__ uint32_t D_CF_BLOCK_TILES;

static inline uint32_t compact_tiles_per_row(uint32_t stride) {
    return stride ? (stride + CF_COL_TILE - 1) / CF_COL_TILE : 0;
}

static inline std::pair<uint32_t,uint32_t> install_compact_tile_prefixes(
    const std::vector<FBlock>& mb, const std::vector<FBlock>& db
) {
    std::array<uint32_t,65> mo{};
    std::array<uint32_t,33> bo{};
    uint64_t mt = 0, bt = 0;
    for (size_t i = 0; i < mb.size(); ++i) {
        mo[i] = uint32_t(mt);
        uint64_t rows = mb[i].stride ? (mb[i].end - mb[i].off) / mb[i].stride : 0;
        mt += rows * compact_tiles_per_row(mb[i].stride);
        if (mt > 0xffffffffULL) std::exit(290);
    }
    mo[mb.size()] = uint32_t(mt);
    for (size_t i = 0; i < db.size(); ++i) {
        bo[i] = uint32_t(bt);
        uint64_t rows = db[i].stride ? (db[i].end - db[i].off) / db[i].stride : 0;
        bt += rows * compact_tiles_per_row(db[i].stride);
        if (bt > 0xffffffffULL) std::exit(291);
    }
    bo[db.size()] = uint32_t(bt);
    uint32_t mtu = uint32_t(mt), btu = uint32_t(bt);
    ck(cudaMemcpyToSymbol(D_CF_MAIN_TILE_OFF, mo.data(), sizeof(mo)), "tile main prefix");
    ck(cudaMemcpyToSymbol(D_CF_BLOCK_TILE_OFF, bo.data(), sizeof(bo)), "tile block prefix");
    ck(cudaMemcpyToSymbol(D_CF_MAIN_TILES, &mtu, sizeof(mtu)), "tile main total");
    ck(cudaMemcpyToSymbol(D_CF_BLOCK_TILES, &btu, sizeof(btu)), "tile block total");
    return {mtu, btu};
}

__device__ __forceinline__ int compact_find_tile_block(
    uint32_t tile, const uint32_t* off, int n
) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (tile < off[m + 1]) hi = m;
        else lo = m + 1;
    }
    return lo;
}

struct CompactTilePos {
    FBlock x;
    int bid;
    uint32_t hr;
    uint32_t col0;
    uint32_t col1;
};

__device__ __forceinline__ CompactTilePos compact_main_tile_pos(uint32_t tile) {
    int bid = compact_find_tile_block(tile, D_CF_MAIN_TILE_OFF, D_F_MAIN_NBLOCKS);
    FBlock x = D_F_MAIN_BLOCKS[bid];
    uint32_t tpr = (x.stride + CF_COL_TILE - 1) / CF_COL_TILE;
    uint32_t local = tile - D_CF_MAIN_TILE_OFF[bid];
    uint32_t hr = local / tpr;
    uint32_t ti = local - hr * tpr;
    uint32_t c0 = ti * CF_COL_TILE;
    uint32_t c1 = c0 + CF_COL_TILE;
    if (c1 > x.stride) c1 = x.stride;
    return {x, bid, hr, c0, c1};
}

__device__ __forceinline__ CompactTilePos compact_block_tile_pos(uint32_t tile) {
    int bid = compact_find_tile_block(tile, D_CF_BLOCK_TILE_OFF, D_F_BLOCK_NBLOCKS);
    FBlock x = D_F_BLOCK_BLOCKS[bid];
    uint32_t tpr = (x.stride + CF_COL_TILE - 1) / CF_COL_TILE;
    uint32_t local = tile - D_CF_BLOCK_TILE_OFF[bid];
    uint32_t hr = local / tpr;
    uint32_t ti = local - hr * tpr;
    uint32_t c0 = ti * CF_COL_TILE;
    uint32_t c1 = c0 + CF_COL_TILE;
    if (c1 > x.stride) c1 = x.stride;
    return {x, bid, hr, c0, c1};
}

__device__ __forceinline__ Code compact_main_canonical_row_base(
    FBlock x, uint32_t hr
) {
    constexpr int S = MAXW + 2;
    uint32_t har;
    if (D_F_FIX_LOW) {
        har = hr;
    } else {
        uint32_t a = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];
        har = D_CF_HIGH_MASK_ALL_RANK[a + hr];
    }
    Code rank = D_F_HIGH_MAIN_BASE[D_F_HIGH_ALL_OFF[x.he] + har];
    MateValue c = MateValue(x.c);
    if (c > N) rank += D_FULL_DP[LOW_LUT_K][x.he];
    if (c > R && x.he > 0) rank += D_FULL_DP[LOW_LUT_K][x.he - 1];
    return rank;
}

__device__ __forceinline__ Code compact_block_canonical_row_base(
    FBlock x, uint32_t hr
) {
    constexpr int S = MAXW + 2;
    uint32_t har;
    if (D_F_FIX_LOW) {
        har = hr;
    } else {
        uint32_t a = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];
        har = D_CF_HIGH_MASK_ALL_RANK[a + hr];
    }
    return D_F_HIGH_BLOCK_BASE[D_F_HIGH_ALL_OFF[x.he] + har];
}

__device__ __forceinline__ uint32_t compact_low_all_rank(FBlock x, uint32_t lr) {
    if (!D_F_FIX_LOW) return lr;
    constexpr int S = MAXW + 2;
    uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
    return D_CF_LOW_MASK_ALL_RANK[a + lr];
}

__global__ void compact_tile_gather_main_kernel(Count* out) {
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_MAIN_TILES) return;
    CompactTilePos z = compact_main_tile_pos(tile);
    Code gbase = compact_main_canonical_row_base(z.x, z.hr);
    Code lbase = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
        uint32_t lar = compact_low_all_rank(z.x, lr);
        out[lbase + lr] = global_load_main(gbase + lar);
    }
}

__global__ void compact_tile_gather_block_kernel(Count* out) {
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_BLOCK_TILES) return;
    CompactTilePos z = compact_block_tile_pos(tile);
    Code gbase = compact_block_canonical_row_base(z.x, z.hr);
    Code lbase = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
        uint32_t lar = compact_low_all_rank(z.x, lr);
        out[lbase + lr] = global_load_block(gbase + lar);
    }
}

__global__ void compact_tile_scatter_main_kernel(const Count* in) {
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_MAIN_TILES) return;
    CompactTilePos z = compact_main_tile_pos(tile);
    Code gbase = compact_main_canonical_row_base(z.x, z.hr);
    Code lbase = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
        uint32_t lar = compact_low_all_rank(z.x, lr);
        global_store_main(gbase + lar, in[lbase + lr]);
    }
}

__global__ void compact_tile_scatter_block_kernel(const Count* in) {
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_BLOCK_TILES) return;
    CompactTilePos z = compact_block_tile_pos(tile);
    Code gbase = compact_block_canonical_row_base(z.x, z.hr);
    Code lbase = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
        uint32_t lar = compact_low_all_rank(z.x, lr);
        global_store_block(gbase + lar, in[lbase + lr]);
    }
}

// HIGH window: orbit metadata depends only on the HIGH row.  With the n=27
// 14/13 split every LOW occupancy class fits in one 1024-column tile, so each
// orbit/descriptor is normally loaded once for the whole LOW class.
__global__ void compact_tile_high_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_MAIN_TILES) return;
    CompactTilePos z = compact_main_tile_pos(tile);
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
    for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
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

__global__ void compact_tile_high_closure_kernel(Count* mainv, Count* blockv, int p) {
    constexpr int S = MAXW + 2;
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_MAIN_TILES) return;
    CompactTilePos z = compact_main_tile_pos(tile);
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
        for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
            Count c = mainv[ib + lr];
            if (c) atomic_add_mod(blockv + db + lr, c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + z.x.hs];
        for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
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

// LOW window: metadata depends on the LOW all-rank, so each state still loads
// one orbit/descriptor word.  The expensive FBlock lookup and row division are
// amortized over a 1024-state tile, and huge LOW rows expose enough CUDA blocks
// for full occupancy.
__global__ void compact_tile_low_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_MAIN_TILES) return;
    CompactTilePos z = compact_main_tile_pos(tile);
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    Code ib = z.x.off + Code(z.hr) * z.x.stride;
    for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
        uint64_t ow = D_LOW_ORBIT[size_t(pi) * D_LOW_ORBIT_MAIN_TOTAL
                                 + D_LOW_ORBIT_MAIN_BASE[z.bid] + lr];
        uint32_t kind = low_orbit_kind_dev(ow);
        if (kind < CPU_ORBIT_NN || kind > CPU_ORBIT_NL) continue;
        FBlock jy = D_F_MAIN_BLOCKS[low_orbit_jblock_dev(ow)];
        FBlock dy = D_F_BLOCK_BLOCKS[low_orbit_dblock_dev(ow)];
        Code j = jy.off + Code(z.hr) * jy.stride + low_orbit_jlr_dev(ow);
        Code d = dy.off + Code(z.hr) * dy.stride + low_orbit_dlr_dev(ow);
        Count c = mainv[ib + lr], oldd = blockv[d];
        if (kind == CPU_ORBIT_NN) {
            mainv[j] = low_orbit_add(mainv[j], c);
            mainv[ib + lr] = low_orbit_add(c, oldd);
            blockv[d] = 0;
        } else {
            Count cc = mainv[j];
            Count all = low_orbit_add(low_orbit_add(c, cc), oldd);
            if (p == 1) {
                mainv[ib + lr] = all;
                mainv[j] = low_orbit_add(c, cc);
                blockv[d] = 0;
            } else {
                mainv[ib + lr] = all;
                blockv[d] = c;
            }
        }
    }
}

__global__ void compact_tile_low_closure_kernel(Count* mainv, Count* blockv, int p) {
    constexpr int S = MAXW + 2;
    uint32_t tile = blockIdx.x;
    if (tile >= D_CF_MAIN_TILES) return;
    CompactTilePos z = compact_main_tile_pos(tile);
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    Code ib = z.x.off + Code(z.hr) * z.x.stride;
    uint32_t ha = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + z.x.he];
    uint32_t hc = D_F_HIGH_MASK_CODES[ha + z.hr];
    for (uint32_t lr = z.col0 + threadIdx.x; lr < z.col1; lr += blockDim.x) {
        uint64_t ow = D_LOW_ORBIT[size_t(pi) * D_LOW_ORBIT_MAIN_TOTAL
                                 + D_LOW_ORBIT_MAIN_BASE[z.bid] + lr];
        if (low_orbit_kind_dev(ow) != CPU_ORBIT_CLOSURE) continue;
        Count c = mainv[ib + lr];
        if (!c) continue;
        uint32_t desc = D_LOWDESC_MAIN[size_t(pi) * D_LOWDESC_MAIN_TOTAL
                                      + D_LOWDESC_MAIN_BASE[z.bid] + lr];
        uint32_t kind = lowdesc_kind(desc);
        if (kind == LOWDESC_MAIN) {
            FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            atomic_add_mod(mainv + y.off + Code(z.hr) * y.stride + lowdesc_lr(desc), c);
        } else if (kind == LOWDESC_BLOCK) {
            FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
            atomic_add_mod(blockv + y.off + Code(z.hr) * y.stride + lowdesc_lr(desc), c);
        } else if (kind == LOWDESC_CROSS) {
            uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            if (p == 1) {
                FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
                uint32_t hr2 = bidesc_high_mask_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(mainv + y.off + Code(hr2) * y.stride + lowdesc_lr(desc), c);
            } else {
                FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
                uint32_t hr2 = bidesc_high_mask_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(blockv + y.off + Code(hr2) * y.stride + lowdesc_lr(desc), c);
            }
        } else {
            asm("trap;");
        }
    }
}
