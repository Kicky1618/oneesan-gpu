#pragma once

// CUDA 12.8+ scoped atomics.  Device-scope atomicCAS is insufficient when
// several GPUs may concurrently update the same peer allocation.  We keep the
// overwhelmingly common local updates on the cheaper existing atomic_add_mod
// and use system scope only when the destination owner differs from D_DR_SELF.
__device__ __forceinline__ void b300_atomic_add_mod_system(Count* p, Count v) {
    if (!v) return;
    Count mod = D_MOD;
    Count old = __nv_atomic_load_n(p, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
    for (;;) {
        Count neu = (old >= mod - v) ? old - (mod - v) : old + v;
        Count expected = old;
        if (__nv_atomic_compare_exchange_n(
                p, &expected, neu, false,
                __NV_ATOMIC_RELAXED, __NV_ATOMIC_RELAXED,
                __NV_THREAD_SCOPE_SYSTEM)) return;
        old = expected;
    }
}

__device__ __forceinline__ void b300_row_scoped_add_block(
    uint32_t bid, uint32_t hr, uint32_t lr, Count v
) {
    Count* p = b300_direct_block_ptr(bid, hr, lr);
    if (int(hr % uint32_t(D_NGPU)) == D_DR_SELF) atomic_add_mod(p, v);
    else b300_atomic_add_mod_system(p, v);
}
__device__ __forceinline__ void b300_row_scoped_add_main(
    uint32_t bid, uint32_t hr, uint32_t lr, Count v
) {
    Count* p = b300_direct_main_ptr(bid, hr, lr);
    if (int(hr % uint32_t(D_NGPU)) == D_DR_SELF) atomic_add_mod(p, v);
    else b300_atomic_add_mod_system(p, v);
}

__global__ void b300_direct_high_closure_scoped_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_CLOSURE_OFF[pi], e = D_BS_HIGH_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    uint64_t op = D_BS_HIGH_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op);
    uint32_t hr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    uint32_t db = highdesc_block(desc), dhr = highdesc_rank(desc);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    StorageBlock y = D_DR_BLOCK_BLOCKS[db];
    uint32_t kind = highdesc_kind(desc);
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_direct_main_ptr(sb, hr, lr);
            if (c) b300_row_scoped_add_block(db, dhr, lr, c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_direct_main_ptr(sb, hr, lr);
            if (!c) continue;
            uint32_t lc = D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs] + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = b300_direct_low_all_rank(lc2, y.hs);
            if (lr2 != 0xffffffffu) b300_row_scoped_add_block(db, dhr, lr2, c);
        }
    }
}

__global__ void b300_direct_low_closure_scoped_kernel(int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_CLOSURE_OFF[pi], e = D_BS_LOW_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    uint64_t op = D_BS_LOW_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op);
    uint32_t lr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    uint32_t kind = lowdesc_kind(desc);
    uint32_t first = uint32_t(D_DR_SELF) + threadIdx.x * uint32_t(D_NGPU);
    uint32_t step = blockDim.x * uint32_t(D_NGPU);
    for (uint32_t hr = first; hr < x.rows; hr += step) {
        Count c = *b300_direct_main_ptr(sb, hr, lr);
        if (!c) continue;
        if (kind == LOWDESC_MAIN) {
            atomic_add_mod(b300_direct_main_ptr(lowdesc_block(desc), hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_BLOCK) {
            atomic_add_mod(b300_direct_block_ptr(lowdesc_block(desc), hr, lowdesc_lr(desc)), c);
        } else if (kind == LOWDESC_CROSS) {
            uint32_t hc = D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he] + hr];
            uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            if (p == 1) {
                uint32_t mb = lowdesc_block(desc);
                StorageBlock y = D_DR_MAIN_BLOCKS[mb];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu) b300_row_scoped_add_main(mb, hr2, lowdesc_lr(desc), c);
            } else {
                uint32_t db = lowdesc_block(desc);
                StorageBlock y = D_DR_BLOCK_BLOCKS[db];
                uint32_t hr2 = b300_direct_high_all_rank(hc2, y.he);
                if (hr2 != 0xffffffffu) b300_row_scoped_add_block(db, hr2, lowdesc_lr(desc), c);
            }
        }
    }
}

__device__ __forceinline__ int b300_mask_owner_for_block_row(
    const StorageBlock& b, uint32_t hr
) {
    return int(D_DM_HIGH_OWNER[D_F_HIGH_ALL_OFF[b.he] + hr]);
}
__device__ __forceinline__ void b300_mask_scoped_add_block(
    uint32_t bid, uint32_t hr, uint32_t lr, Count v
) {
    StorageBlock b = D_DR_BLOCK_BLOCKS[bid];
    Count* p = b300_mask_block_ptr(bid, hr, lr);
    if (b300_mask_owner_for_block_row(b, hr) == D_DR_SELF) atomic_add_mod(p, v);
    else b300_atomic_add_mod_system(p, v);
}

__global__ void b300_mask_high_closure_scoped_kernel(int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_CLOSURE_OFF[pi], e = D_BS_HIGH_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= e) return;
    uint64_t op = D_BS_HIGH_CLOSURE[q];
    uint32_t sb = b300_sparse_closure_sblock(op), hr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    uint32_t db = highdesc_block(desc), dhr = highdesc_rank(desc);
    StorageBlock x = D_DR_MAIN_BLOCKS[sb];
    StorageBlock y = D_DR_BLOCK_BLOCKS[db];
    uint32_t kind = highdesc_kind(desc);
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_mask_main_ptr(sb, hr, lr);
            if (c) b300_mask_scoped_add_block(db, dhr, lr, c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        for (uint32_t lr = threadIdx.x; lr < x.cols; lr += blockDim.x) {
            Count c = *b300_mask_main_ptr(sb, hr, lr);
            if (!c) continue;
            uint32_t lc = D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs] + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            uint32_t lr2 = b300_direct_low_all_rank(lc2, y.hs);
            if (lr2 != 0xffffffffu) b300_mask_scoped_add_block(db, dhr, lr2, c);
        }
    }
}
