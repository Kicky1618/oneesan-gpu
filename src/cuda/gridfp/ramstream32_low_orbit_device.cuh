#pragma once

#include "ramstream32_cpu_low_inplace.hpp"
#include "ramstream32_bidesc_compact.cuh"

// Device-side LOW-window analogue of ramstream32_high_orbit.cuh.  The orbit
// table is indexed by LOW all-rank and is independent of exact HIGH topology;
// boundary-changing LL/RR closures use the compact LOW descriptor plus the
// 4-bit HIGH matching depth.
__constant__ uint64_t* D_LOW_ORBIT;
__constant__ uint32_t D_LOW_ORBIT_MAIN_BASE[64];
__constant__ uint32_t D_LOW_ORBIT_MAIN_TOTAL;

struct LowOrbitDeviceTables {
    uint64_t* rec = nullptr;

    void install(const LowOrbitHost& o) {
        if (!o.rec.empty()) {
            ck(cudaMalloc(&rec, o.rec.size() * sizeof(uint64_t)), "low orbit alloc");
            ck(cudaMemcpy(rec, o.rec.data(), o.rec.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "low orbit copy");
        }
        ck(cudaMemcpyToSymbol(D_LOW_ORBIT, &rec, sizeof(rec)), "low orbit ptr");
        ck(cudaMemcpyToSymbol(D_LOW_ORBIT_MAIN_BASE, o.main_base.data(),
                              sizeof(uint32_t) * o.main_base.size()), "low orbit bases");
        ck(cudaMemcpyToSymbol(D_LOW_ORBIT_MAIN_TOTAL, &o.main_total,
                              sizeof(o.main_total)), "low orbit total");
    }

    void release() {
        if (rec) cudaFree(rec);
        rec = nullptr;
    }
};

__device__ __forceinline__ uint32_t low_orbit_kind_dev(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_KIND_SHIFT) & 7u);
}
__device__ __forceinline__ uint32_t low_orbit_jlr_dev(uint64_t x) {
    return uint32_t(x & CPU_ORBIT_LR_MASK);
}
__device__ __forceinline__ uint32_t low_orbit_jblock_dev(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_JBLOCK_SHIFT) & CPU_ORBIT_BLOCK_MASK);
}
__device__ __forceinline__ uint32_t low_orbit_dlr_dev(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_DLR_SHIFT) & CPU_ORBIT_LR_MASK);
}
__device__ __forceinline__ uint32_t low_orbit_dblock_dev(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_DBLOCK_SHIFT) & CPU_ORBIT_BLOCK_MASK);
}

__global__ void main_group_low_orbit_inplace_kernel(
    Count* mainv, Code n, Count* blockv, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (; i < n; i += step) {
        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint64_t ow = D_LOW_ORBIT[size_t(pi) * D_LOW_ORBIT_MAIN_TOTAL
                                 + D_LOW_ORBIT_MAIN_BASE[bid] + lr];
        uint32_t kind = low_orbit_kind_dev(ow);
        if (kind < CPU_ORBIT_NN || kind > CPU_ORBIT_NL) continue;

        FBlock jy = D_F_MAIN_BLOCKS[low_orbit_jblock_dev(ow)];
        FBlock dy = D_F_BLOCK_BLOCKS[low_orbit_dblock_dev(ow)];
        Code j = jy.off + Code(hr) * jy.stride + low_orbit_jlr_dev(ow);
        Code dj = dy.off + Code(hr) * dy.stride + low_orbit_dlr_dev(ow);
        Count c = mainv[i];
        Count d = blockv[dj];

        if (kind == CPU_ORBIT_NN) {
            mainv[j] = high_orbit_add(mainv[j], c);
            mainv[i] = high_orbit_add(c, d);
            blockv[dj] = 0;
        } else {
            Count cc = mainv[j];
            Count all = high_orbit_add(high_orbit_add(c, cc), d);
            if (p == 1) {
                mainv[i] = all;
                mainv[j] = high_orbit_add(c, cc);
                blockv[dj] = 0;
            } else {
                mainv[i] = all;
                blockv[dj] = c;
                // RN/LN partner keeps its identity value in mainv[j].
            }
        }
    }
}

__global__ void main_group_low_closure_inplace_kernel(
    Count* mainv, Code n, Count* blockv, int p
) {
    constexpr int S = MAXW + 2;
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code step = Code(gridDim.x) * blockDim.x;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (; i < n; i += step) {
        int bid = f_find_main(i);
        FBlock x = D_F_MAIN_BLOCKS[bid];
        Code r = i - x.off;
        uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
        uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
        uint64_t ow = D_LOW_ORBIT[size_t(pi) * D_LOW_ORBIT_MAIN_TOTAL
                                 + D_LOW_ORBIT_MAIN_BASE[bid] + lr];
        if (low_orbit_kind_dev(ow) != CPU_ORBIT_CLOSURE) continue;
        Count c = mainv[i];
        if (!c) continue;

        uint32_t desc = D_LOWDESC_MAIN[size_t(pi) * D_LOWDESC_MAIN_TOTAL
                                      + D_LOWDESC_MAIN_BASE[bid] + lr];
        uint32_t kind = lowdesc_kind(desc);
        if (kind == LOWDESC_MAIN) {
            FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            atomic_add_mod(mainv + y.off + Code(hr) * y.stride + lowdesc_lr(desc), c);
        } else if (kind == LOWDESC_BLOCK) {
            FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
            atomic_add_mod(blockv + y.off + Code(hr) * y.stride + lowdesc_lr(desc), c);
        } else if (kind == LOWDESC_CROSS) {
            uint32_t a = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];
            uint32_t hc = D_F_HIGH_MASK_CODES[a + hr];
            uint32_t hc2 = lowdesc_flip_high(hc, lowdesc_depth(desc));
            if (hc2 == 0xffffffffu) continue;
            if (p == 1) {
                FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
                uint32_t hr2 = bidesc_high_mask_rank(hc2, y.he);
                if (hr2 == 0xffffffffu) continue;
                atomic_add_mod(mainv + y.off + Code(hr2) * y.stride + lowdesc_lr(desc), c);
            } else {
                FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
                uint32_t hr2 = bidesc_high_mask_rank(hc2, y.he);
                if (hr2 == 0xffffffffu) continue;
                atomic_add_mod(blockv + y.off + Code(hr2) * y.stride + lowdesc_lr(desc), c);
            }
        } else {
            asm("trap;");
        }
    }
}
