#pragma once

#include "ramstream32_b300_dual_tile_layout.cuh"
#include "ramstream32_b300_sparse_actions.cuh"

// HIGH window runs in LOW-owner orientation.  Every GPU executes the same
// sparse HIGH action stream, but only across LOW columns owned by that GPU.
// HIGH transitions preserve LOW occupancy (including CROSS), so every source,
// partner and destination pointer is local to D_DR_SELF.
__global__ void b300_dt_high_orbit_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_ORBIT_OFF[pi], e = D_BS_HIGH_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    B300SparseOrbitOp op = D_BS_HIGH_ORBIT[q];
    uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
    uint32_t hr = b300_sparse_src(op), jhr = b300_sparse_jrank(op), dhr = b300_sparse_drank(op);
    uint32_t kind = b300_sparse_kind(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t ca = D_DT_LOW_OWNED_OFF[x.hs], ce = D_DT_LOW_OWNED_OFF[x.hs + 1];
    for (uint32_t k = ca + threadIdx.x; k < ce; k += blockDim.x) {
        uint32_t lr = D_DT_LOW_OWNED_COLS[k];
        Count* ip = b300_dt_main_low(sb, hr, lr);
        Count* jp = b300_dt_main_low(jb, jhr, lr);
        Count* dp = b300_dt_block_low(db, dhr, lr);
        Count c = *ip, d = *dp;
        if (kind == HIGH_ORBIT_NN) {
            *jp = b300_sparse_add(*jp, c);
            *ip = b300_sparse_add(c, d);
            *dp = 0;
        } else {
            Count cc = *jp;
            *ip = b300_sparse_add(b300_sparse_add(c, cc), d);
            *dp = c;
        }
    }
}

__global__ void b300_dt_high_closure_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_CLOSURE_OFF[pi], e = D_BS_HIGH_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    uint64_t op = D_BS_HIGH_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op), hr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    uint32_t db = highdesc_block(desc), dhr = highdesc_rank(desc);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb], y = D_DR_BLOCK_BLOCKS[db];
    uint32_t kind = highdesc_kind(desc);
    uint32_t ca = D_DT_LOW_OWNED_OFF[x.hs], ce = D_DT_LOW_OWNED_OFF[x.hs + 1];
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t k = ca + threadIdx.x; k < ce; k += blockDim.x) {
            uint32_t lr = D_DT_LOW_OWNED_COLS[k];
            Count c = *b300_dt_main_low(sb, hr, lr);
            if (c) atomic_add_mod(b300_dt_block_low(db, dhr, lr), c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        for (uint32_t k = ca + threadIdx.x; k < ce; k += blockDim.x) {
            uint32_t lr = D_DT_LOW_OWNED_COLS[k];
            Count c = *b300_dt_main_low(sb, hr, lr);
            if (!c) continue;
            uint32_t lc = D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs] + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = b300_direct_low_all_rank(lc2, y.hs);
            if (lr2 != 0xffffffffu)
                atomic_add_mod(b300_dt_block_low(db, dhr, lr2), c);
        }
    }
}

// LOW window runs in HIGH-owner orientation.  Every GPU executes the same LOW
// sparse stream across only its HIGH rows.  LOW CROSS preserves HIGH occupancy,
// so all destination updates are local too.
__global__ void b300_dt_low_orbit_kernel(int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_ORBIT_OFF[pi], e = D_BS_LOW_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    B300SparseOrbitOp op = D_BS_LOW_ORBIT[q];
    uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
    uint32_t lr = b300_sparse_src(op), jlr = b300_sparse_jrank(op), dlr = b300_sparse_drank(op);
    uint32_t kind = b300_sparse_kind(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t ra = D_DT_HIGH_OWNED_OFF[x.he], re = D_DT_HIGH_OWNED_OFF[x.he + 1];
    for (uint32_t k = ra + threadIdx.x; k < re; k += blockDim.x) {
        uint32_t hr = D_DT_HIGH_OWNED_ROWS[k];
        Count* ip = b300_dt_main_high(sb, hr, lr);
        Count* jp = b300_dt_main_high(jb, hr, jlr);
        Count* dp = b300_dt_block_high(db, hr, dlr);
        Count c = *ip, d = *dp;
        if (kind == CPU_ORBIT_NN) {
            *jp = b300_sparse_add(*jp, c);
            *ip = b300_sparse_add(c, d);
            *dp = 0;
        } else {
            Count cc = *jp;
            Count all = b300_sparse_add(b300_sparse_add(c, cc), d);
            if (p == 1) {
                *ip = all;
                *jp = b300_sparse_add(c, cc);
                *dp = 0;
            } else {
                *ip = all;
                *dp = c;
            }
        }
    }
}

__global__ void b300_dt_low_closure_kernel(int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_CLOSURE_OFF[pi], e = D_BS_LOW_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    uint64_t op = D_BS_LOW_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op), lr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t kind = lowdesc_kind(desc);
    uint32_t ra = D_DT_HIGH_OWNED_OFF[x.he], re = D_DT_HIGH_OWNED_OFF[x.he + 1];
    for (uint32_t k = ra + threadIdx.x; k < re; k += blockDim.x) {
        uint32_t hr = D_DT_HIGH_OWNED_ROWS[k];
        Count c = *b300_dt_main_high(sb, hr, lr);
        if (!c) continue;
        if (kind == LOWDESC_MAIN) {
            atomic_add_mod(b300_dt_main_high(lowdesc_block(desc), hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_BLOCK) {
            atomic_add_mod(b300_dt_block_high(lowdesc_block(desc), hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_CROSS) {
            uint32_t hc = D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he] + hr];
            uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            if (p == 1) {
                uint32_t mb = lowdesc_block(desc);
                StorageBlock y = D_DR_MAIN_BLOCKS[mb];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(b300_dt_main_high(mb, hr2, lowdesc_lr(desc)), c);
            } else {
                uint32_t db = lowdesc_block(desc);
                StorageBlock y = D_DR_BLOCK_BLOCKS[db];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu)
                    atomic_add_mod(b300_dt_block_high(db, hr2, lowdesc_lr(desc)), c);
            }
        }
    }
}
