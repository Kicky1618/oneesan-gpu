#pragma once

#include "ramstream32_b300_sparse_actions.cuh"

static constexpr uint32_t B300_SPARSE_HIGH_WARPS = 8;
static_assert(B300_SPARSE_HIGH_WARPS * 32 == 256);

__device__ __forceinline__ uint32_t b300_sparse_high_relop() {
    return uint32_t(blockIdx.x) * B300_SPARSE_HIGH_WARPS + (threadIdx.x >> 5);
}
__device__ __forceinline__ uint32_t b300_sparse_high_lane() {
    return threadIdx.x & 31u;
}

__global__ void b300_sparse_highwarp_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_ORBIT_OFF[pi];
    uint32_t b = D_BS_HIGH_ORBIT_OFF[pi + 1];
    uint32_t q = a + b300_sparse_high_relop();
    if (q >= b) return;
    uint32_t lane = b300_sparse_high_lane();
    B300SparseOrbitOp op = D_BS_HIGH_ORBIT[q];
    FBlock x = D_F_MAIN_BLOCKS[b300_sparse_sblock(op)];
    if (!x.stride) return;
    FBlock jy = D_F_MAIN_BLOCKS[b300_sparse_jblock(op)];
    FBlock dy = D_F_BLOCK_BLOCKS[b300_sparse_dblock(op)];
    uint32_t hr = b300_sparse_src(op);
    Code ib = x.off + Code(hr) * x.stride;
    Code jb = jy.off + Code(b300_sparse_jrank(op)) * jy.stride;
    Code db = dy.off + Code(b300_sparse_drank(op)) * dy.stride;
    uint32_t kind = b300_sparse_kind(op);
    for (uint32_t lr = lane; lr < x.stride; lr += 32) {
        Count c = mainv[ib + lr];
        Count d = blockv[db + lr];
        if (kind == HIGH_ORBIT_NN) {
            mainv[jb + lr] = b300_sparse_add(mainv[jb + lr], c);
            mainv[ib + lr] = b300_sparse_add(c, d);
            blockv[db + lr] = 0;
        } else {
            Count cc = mainv[jb + lr];
            mainv[ib + lr] = b300_sparse_add(b300_sparse_add(c, cc), d);
            blockv[db + lr] = c;
        }
    }
}

__global__ void b300_sparse_highwarp_closure_kernel(Count* mainv, Count* blockv, int p) {
    constexpr int S = MAXW + 2;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_CLOSURE_OFF[pi];
    uint32_t b = D_BS_HIGH_CLOSURE_OFF[pi + 1];
    uint32_t q = a + b300_sparse_high_relop();
    if (q >= b) return;
    uint32_t lane = b300_sparse_high_lane();
    uint64_t op = D_BS_HIGH_CLOSURE[q];
    uint32_t sbid = b300_sparse_closure_sblock(op);
    uint32_t hr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    FBlock x = D_F_MAIN_BLOCKS[sbid];
    if (!x.stride) return;
    FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
    Code ib = x.off + Code(hr) * x.stride;
    Code db = y.off + Code(highdesc_rank(desc)) * y.stride;
    uint32_t kind = highdesc_kind(desc);
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t lr = lane; lr < x.stride; lr += 32) {
            Count c = mainv[ib + lr];
            if (c) atomic_add_mod(blockv + db + lr, c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        uint32_t low0 = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
        for (uint32_t lr = lane; lr < x.stride; lr += 32) {
            Count c = mainv[ib + lr];
            if (!c) continue;
            uint32_t lc = D_F_LOW_MASK_CODES[low0 + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = bidesc_low_mask_rank(lc2, y.hs);
            if (lr2 != 0xffffffffu) atomic_add_mod(blockv + db + lr2, c);
        }
    }
}

static inline uint32_t b300_sparse_high_orbit_blocks(
    const B300SparseActionsHost& s, int p
) {
    uint32_t n = b300_sparse_high_orbit_count(s, p);
    return (n + B300_SPARSE_HIGH_WARPS - 1) / B300_SPARSE_HIGH_WARPS;
}
static inline uint32_t b300_sparse_high_closure_blocks(
    const B300SparseActionsHost& s, int p
) {
    uint32_t n = b300_sparse_high_closure_count(s, p);
    return (n + B300_SPARSE_HIGH_WARPS - 1) / B300_SPARSE_HIGH_WARPS;
}
