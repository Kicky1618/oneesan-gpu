#pragma once

// Compact device-side support for bidesc.  The descriptor fast paths do not
// need the 4^L / 4^H dense packed-rank tables.  Boundary-crossing transitions
// preserve occupancy, so after flipping the one inactive endpoint we recover
// its local rank by binary search inside the already sorted occupancy class.

__device__ __forceinline__ uint32_t bidesc_high_mask_rank(uint32_t code, int h) {
    constexpr int S = MAXW + 2;
    uint32_t a = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + h];
    uint32_t b = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + h + 1];
    uint32_t lo = a, hi = b;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t v = D_F_HIGH_MASK_CODES[mid];
        if (v < code) lo = mid + 1;
        else hi = mid;
    }
    return (lo < b && D_F_HIGH_MASK_CODES[lo] == code) ? lo - a : 0xffffffffu;
}

__device__ __forceinline__ uint32_t bidesc_low_mask_rank(uint32_t code, int h) {
    constexpr int S = MAXW + 2;
    uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + h];
    uint32_t b = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + h + 1];
    uint32_t lo = a, hi = b;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t v = D_F_LOW_MASK_CODES[mid];
        if (v < code) lo = mid + 1;
        else hi = mid;
    }
    return (lo < b && D_F_LOW_MASK_CODES[lo] == code) ? lo - a : 0xffffffffu;
}

__global__ void main_group_lowdesc_compact_kernel(
    const Count* in, Code n, Count* out_main, Count* out_block, int p
) {
    constexpr int S = MAXW + 2;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (; i < n; i += step) {
        Count c = in[i];
        if (!c) continue;
        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint32_t desc = D_LOWDESC_MAIN[size_t(pi) * D_LOWDESC_MAIN_TOTAL
                                        + D_LOWDESC_MAIN_BASE[bid] + lr];
        uint32_t kind = lowdesc_kind(desc);

        if (kind == LOWDESC_MAIN) {
            FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            atomic_add_mod(out_main + y.off + Code(hr) * y.stride + lowdesc_lr(desc), c);
        } else if (kind == LOWDESC_BLOCK) {
            FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
            atomic_add_mod(out_block + y.off + Code(hr) * y.stride + lowdesc_lr(desc), c);
        } else if (kind == LOWDESC_CROSS) {
            uint32_t a = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];
            uint32_t hc = D_F_HIGH_MASK_CODES[a + hr];
            uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            if (p == 1) {
                FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
                uint32_t hr2 = bidesc_high_mask_rank(hc2, y.he);
                if (hr2 == 0xffffffffu) continue;
                atomic_add_mod(out_main + y.off + Code(hr2) * y.stride + lowdesc_lr(desc), c);
            } else {
                FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
                uint32_t hr2 = bidesc_high_mask_rank(hc2, y.he);
                if (hr2 == 0xffffffffu) continue;
                atomic_add_mod(out_block + y.off + Code(hr2) * y.stride + lowdesc_lr(desc), c);
            }
        }
    }
}

__global__ void main_group_highdesc_compact_kernel(
    const Count* in, Code n, Count* out_main, Count* out_block, int p
) {
    constexpr int S = MAXW + 2;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (; i < n; i += step) {
        Count c = in[i];
        if (!c) continue;
        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint32_t desc = D_HIGHDESC_MAIN[size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                                         + D_HIGHDESC_MAIN_BASE[bid] + hr];
        uint32_t kind = highdesc_kind(desc);

        if (kind == HIGHDESC_MAIN) {
            FBlock y = D_F_MAIN_BLOCKS[highdesc_block(desc)];
            atomic_add_mod(out_main + y.off + Code(highdesc_rank(desc)) * y.stride + lr, c);
        } else if (kind == HIGHDESC_BLOCK) {
            FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            atomic_add_mod(out_block + y.off + Code(highdesc_rank(desc)) * y.stride + lr, c);
        } else if (kind == HIGHDESC_CROSS) {
            uint32_t a = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
            uint32_t lc = D_F_LOW_MASK_CODES[a + lr];
            uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(desc));
            if (lc2 == 0xffffffffu) continue;
            FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
            uint32_t lr2 = bidesc_low_mask_rank(lc2, y.hs);
            if (lr2 == 0xffffffffu) continue;
            atomic_add_mod(out_block + y.off + Code(highdesc_rank(desc)) * y.stride + lr2, c);
        }
    }
}

struct BidescMaskDeviceTables {
    uint32_t *low_mask = nullptr, *low_off = nullptr;
    uint32_t *high_mask = nullptr, *high_off = nullptr;

    static void copy_u32(uint32_t** dst, const std::vector<uint32_t>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(dst, v.size() * sizeof(uint32_t)), what);
        ck(cudaMemcpy(*dst, v.data(), v.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), what);
    }

    void install(const FactorTablesHost& base) {
        copy_u32(&low_mask, base.low_mask_codes, "bidesc low mask");
        copy_u32(&low_off, base.low_mask_off, "bidesc low off");
        copy_u32(&high_mask, base.high_mask_codes, "bidesc high mask");
        copy_u32(&high_off, base.high_mask_off, "bidesc high off");
        ck(cudaMemcpyToSymbol(D_F_LOW_MASK_CODES, &low_mask, sizeof(low_mask)), "bidesc low mask ptr");
        ck(cudaMemcpyToSymbol(D_F_LOW_MASK_OFF, &low_off, sizeof(low_off)), "bidesc low off ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_CODES, &high_mask, sizeof(high_mask)), "bidesc high mask ptr");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_OFF, &high_off, sizeof(high_off)), "bidesc high off ptr");
    }

    size_t bytes() const {
        return (G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
              + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size())
            * sizeof(uint32_t);
    }

    void release() {
        if (low_mask) cudaFree(low_mask);
        if (low_off) cudaFree(low_off);
        if (high_mask) cudaFree(high_mask);
        if (high_off) cudaFree(high_off);
        low_mask = low_off = high_mask = high_off = nullptr;
    }
};
