#pragma once

#include "ramstream32_high_orbit.cuh"
#include "ramstream32_cpu_low_inplace.hpp"
#include "ramstream32_bidesc_compact.cuh"

#include <cstdint>
#include <iostream>
#include <vector>

// Flat action streams for the B300 resident backend.  Dense orbit/descriptor
// tables inspect every active local topology at every edge.  Only N* orbit
// owners and valid closure states do work, so encode those states directly.
//
// Orbit op = 12 bytes:
//   a[19:0]   source active rank
//   a[22:20]  kind
//   a[28:23]  partner-main block
//   b[19:0]   partner-main active rank
//   b[25:20]  dropped-block block
//   c[19:0]   dropped-block active rank
//   c[25:20]  source-main block
//
// Closure op = 8 bytes:
//   bits  0..19 source active rank
//   bits 20..51 packed LOW/HIGH descriptor
//   bits 52..57 source-main block
//
// Streams are flat per edge position, so runtime launches only two kernels per
// edge, not one kernel per factor block.
struct B300SparseOrbitOp {
    uint32_t a = 0, b = 0, c = 0;
};
static_assert(sizeof(B300SparseOrbitOp) == 12);

static constexpr uint32_t B300_SPARSE_RANK_MASK = (1u << 20) - 1u;
static constexpr uint32_t B300_SPARSE_BLOCK_MASK = 0x3fu;

__host__ __device__ static inline uint32_t b300_sparse_src(const B300SparseOrbitOp& z) {
    return z.a & B300_SPARSE_RANK_MASK;
}
__host__ __device__ static inline uint32_t b300_sparse_kind(const B300SparseOrbitOp& z) {
    return (z.a >> 20) & 7u;
}
__host__ __device__ static inline uint32_t b300_sparse_jblock(const B300SparseOrbitOp& z) {
    return (z.a >> 23) & B300_SPARSE_BLOCK_MASK;
}
__host__ __device__ static inline uint32_t b300_sparse_jrank(const B300SparseOrbitOp& z) {
    return z.b & B300_SPARSE_RANK_MASK;
}
__host__ __device__ static inline uint32_t b300_sparse_dblock(const B300SparseOrbitOp& z) {
    return (z.b >> 20) & B300_SPARSE_BLOCK_MASK;
}
__host__ __device__ static inline uint32_t b300_sparse_drank(const B300SparseOrbitOp& z) {
    return z.c & B300_SPARSE_RANK_MASK;
}
__host__ __device__ static inline uint32_t b300_sparse_sblock(const B300SparseOrbitOp& z) {
    return (z.c >> 20) & B300_SPARSE_BLOCK_MASK;
}

static inline B300SparseOrbitOp b300_sparse_orbit_pack(
    uint32_t sblock, uint32_t src, uint32_t kind,
    uint32_t jblock, uint32_t jrank,
    uint32_t dblock, uint32_t drank
) {
    if (src > B300_SPARSE_RANK_MASK || jrank > B300_SPARSE_RANK_MASK
        || drank > B300_SPARSE_RANK_MASK || kind > 7
        || sblock > B300_SPARSE_BLOCK_MASK || jblock > B300_SPARSE_BLOCK_MASK
        || dblock > B300_SPARSE_BLOCK_MASK) {
        std::cerr << "B300 sparse orbit encoding overflow\n";
        std::exit(360);
    }
    B300SparseOrbitOp z;
    z.a = src | (kind << 20) | (jblock << 23);
    z.b = jrank | (dblock << 20);
    z.c = drank | (sblock << 20);
    return z;
}

__host__ __device__ static inline uint64_t b300_sparse_closure_pack(
    uint32_t sblock, uint32_t src, uint32_t desc
) {
#ifndef __CUDA_ARCH__
    if (src > B300_SPARSE_RANK_MASK || sblock > B300_SPARSE_BLOCK_MASK) {
        std::cerr << "B300 sparse closure encoding overflow\n";
        std::exit(361);
    }
#endif
    return uint64_t(src) | (uint64_t(desc) << 20) | (uint64_t(sblock) << 52);
}
__host__ __device__ static inline uint32_t b300_sparse_closure_src(uint64_t z) {
    return uint32_t(z & B300_SPARSE_RANK_MASK);
}
__host__ __device__ static inline uint32_t b300_sparse_closure_desc(uint64_t z) {
    return uint32_t((z >> 20) & 0xffffffffull);
}
__host__ __device__ static inline uint32_t b300_sparse_closure_sblock(uint64_t z) {
    return uint32_t((z >> 52) & B300_SPARSE_BLOCK_MASK);
}

struct B300SparseActionsHost {
    std::vector<B300SparseOrbitOp> low_orbit;
    std::vector<uint64_t> low_closure;
    std::vector<uint32_t> low_orbit_off;
    std::vector<uint32_t> low_closure_off;

    std::vector<B300SparseOrbitOp> high_orbit;
    std::vector<uint64_t> high_closure;
    std::vector<uint32_t> high_orbit_off;
    std::vector<uint32_t> high_closure_off;

    uint64_t bytes() const {
        return uint64_t(low_orbit.size() + high_orbit.size()) * sizeof(B300SparseOrbitOp)
             + uint64_t(low_closure.size() + high_closure.size()) * sizeof(uint64_t)
             + uint64_t(low_orbit_off.size() + low_closure_off.size()
                      + high_orbit_off.size() + high_closure_off.size()) * sizeof(uint32_t);
    }
};

static inline uint32_t b300_host_low_kind(uint32_t x) {
    return x >> LOWDESC_KIND_SHIFT;
}
static inline uint32_t b300_host_high_kind(uint32_t x) {
    return x >> HIGHDESC_KIND_SHIFT;
}

static B300SparseActionsHost build_b300_sparse_actions(
    const StorageLayout& layout,
    const LowDescHost& lowdesc, const LowOrbitHost& loworbit,
    const HighDescHost& highdesc, const HighOrbitHost& highorbit
) {
    B300SparseActionsHost s;
    s.low_orbit_off.resize(LOW_LUT_K + 1);
    s.low_closure_off.resize(LOW_LUT_K + 1);
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        s.low_orbit_off[pi] = uint32_t(s.low_orbit.size());
        s.low_closure_off[pi] = uint32_t(s.low_closure.size());
        for (uint32_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            uint32_t cols = layout.main_blocks[bid].cols;
            for (uint32_t lr = 0; lr < cols; ++lr) {
                uint64_t ow = loworbit.rec[
                    size_t(pi) * loworbit.main_total + loworbit.main_base[bid] + lr];
                uint32_t k = cpu_orbit_kind(ow);
                if (k >= CPU_ORBIT_NN && k <= CPU_ORBIT_NL) {
                    s.low_orbit.push_back(b300_sparse_orbit_pack(
                        bid, lr, k,
                        cpu_orbit_jblock(ow), cpu_orbit_jlr(ow),
                        cpu_orbit_dblock(ow), cpu_orbit_dlr(ow)));
                } else if (k == CPU_ORBIT_CLOSURE) {
                    uint32_t dw = lowdesc.main_desc[
                        size_t(pi) * lowdesc.main_total + lowdesc.main_base[bid] + lr];
                    if (b300_host_low_kind(dw) != LOWDESC_INVALID)
                        s.low_closure.push_back(b300_sparse_closure_pack(bid, lr, dw));
                }
            }
        }
    }
    s.low_orbit_off[LOW_LUT_K] = uint32_t(s.low_orbit.size());
    s.low_closure_off[LOW_LUT_K] = uint32_t(s.low_closure.size());

    s.high_orbit_off.resize(HIGH_LUT_K + 1);
    s.high_closure_off.resize(HIGH_LUT_K + 1);
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        s.high_orbit_off[pi] = uint32_t(s.high_orbit.size());
        s.high_closure_off[pi] = uint32_t(s.high_closure.size());
        for (uint32_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            uint32_t rows = layout.main_blocks[bid].rows;
            for (uint32_t hr = 0; hr < rows; ++hr) {
                uint64_t ow = highorbit.rec[
                    size_t(pi) * highorbit.main_total + highorbit.main_base[bid] + hr];
                uint32_t k = high_orbit_kind(ow);
                if (k >= HIGH_ORBIT_NN && k <= HIGH_ORBIT_NL) {
                    s.high_orbit.push_back(b300_sparse_orbit_pack(
                        bid, hr, k,
                        high_orbit_jblock(ow), high_orbit_jhr(ow),
                        high_orbit_dblock(ow), high_orbit_dhr(ow)));
                } else if (k == HIGH_ORBIT_CLOSURE) {
                    uint32_t dw = highdesc.main_desc[
                        size_t(pi) * highdesc.main_total + highdesc.main_base[bid] + hr];
                    if (b300_host_high_kind(dw) != HIGHDESC_INVALID)
                        s.high_closure.push_back(b300_sparse_closure_pack(bid, hr, dw));
                }
            }
        }
    }
    s.high_orbit_off[HIGH_LUT_K] = uint32_t(s.high_orbit.size());
    s.high_closure_off[HIGH_LUT_K] = uint32_t(s.high_closure.size());

    std::cerr << "b300_sparse_actions"
              << " low_orbit=" << s.low_orbit.size()
              << " low_closure=" << s.low_closure.size()
              << " high_orbit=" << s.high_orbit.size()
              << " high_closure=" << s.high_closure.size()
              << " total_mib=" << double(s.bytes()) / double(1 << 20)
              << '\n';
    return s;
}

__constant__ B300SparseOrbitOp* D_BS_LOW_ORBIT;
__constant__ uint64_t* D_BS_LOW_CLOSURE;
__constant__ uint32_t D_BS_LOW_ORBIT_OFF[MAXW + 1];
__constant__ uint32_t D_BS_LOW_CLOSURE_OFF[MAXW + 1];
__constant__ B300SparseOrbitOp* D_BS_HIGH_ORBIT;
__constant__ uint64_t* D_BS_HIGH_CLOSURE;
__constant__ uint32_t D_BS_HIGH_ORBIT_OFF[MAXW + 1];
__constant__ uint32_t D_BS_HIGH_CLOSURE_OFF[MAXW + 1];

struct B300SparseActionsDeviceTables {
    B300SparseOrbitOp* low_orbit = nullptr;
    uint64_t* low_closure = nullptr;
    B300SparseOrbitOp* high_orbit = nullptr;
    uint64_t* high_closure = nullptr;

    template<class T>
    static void upload(T** p, const std::vector<T>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(p, v.size() * sizeof(T)), what);
        ck(cudaMemcpy(*p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(const B300SparseActionsHost& s) {
        upload(&low_orbit, s.low_orbit, "B300 sparse low orbit");
        upload(&low_closure, s.low_closure, "B300 sparse low closure");
        upload(&high_orbit, s.high_orbit, "B300 sparse high orbit");
        upload(&high_closure, s.high_closure, "B300 sparse high closure");
        ck(cudaMemcpyToSymbol(D_BS_LOW_ORBIT, &low_orbit, sizeof(low_orbit)), "B300 sparse low orbit ptr");
        ck(cudaMemcpyToSymbol(D_BS_LOW_CLOSURE, &low_closure, sizeof(low_closure)), "B300 sparse low closure ptr");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_ORBIT, &high_orbit, sizeof(high_orbit)), "B300 sparse high orbit ptr");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_CLOSURE, &high_closure, sizeof(high_closure)), "B300 sparse high closure ptr");

        std::array<uint32_t, MAXW + 1> lo{}, lc{}, ho{}, hc{};
        std::copy(s.low_orbit_off.begin(), s.low_orbit_off.end(), lo.begin());
        std::copy(s.low_closure_off.begin(), s.low_closure_off.end(), lc.begin());
        std::copy(s.high_orbit_off.begin(), s.high_orbit_off.end(), ho.begin());
        std::copy(s.high_closure_off.begin(), s.high_closure_off.end(), hc.begin());
        ck(cudaMemcpyToSymbol(D_BS_LOW_ORBIT_OFF, lo.data(), sizeof(lo)), "B300 sparse low orbit off");
        ck(cudaMemcpyToSymbol(D_BS_LOW_CLOSURE_OFF, lc.data(), sizeof(lc)), "B300 sparse low closure off");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_ORBIT_OFF, ho.data(), sizeof(ho)), "B300 sparse high orbit off");
        ck(cudaMemcpyToSymbol(D_BS_HIGH_CLOSURE_OFF, hc.data(), sizeof(hc)), "B300 sparse high closure off");
    }

    void release() {
        if (low_orbit) cudaFree(low_orbit);
        if (low_closure) cudaFree(low_closure);
        if (high_orbit) cudaFree(high_orbit);
        if (high_closure) cudaFree(high_closure);
        low_orbit = high_orbit = nullptr;
        low_closure = high_closure = nullptr;
    }
};

__device__ __forceinline__ Count b300_sparse_add(Count a, Count b) {
    if (!b) return a;
    Count mod = D_MOD;
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

// One block owns one HIGH sparse action and spans its LOW columns.
__global__ void b300_sparse_high_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_ORBIT_OFF[pi];
    uint32_t b = D_BS_HIGH_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    B300SparseOrbitOp op = D_BS_HIGH_ORBIT[q];
    FBlock x = D_F_MAIN_BLOCKS[b300_sparse_sblock(op)];
    FBlock jy = D_F_MAIN_BLOCKS[b300_sparse_jblock(op)];
    FBlock dy = D_F_BLOCK_BLOCKS[b300_sparse_dblock(op)];
    uint32_t hr = b300_sparse_src(op);
    Code ib = x.off + Code(hr) * x.stride;
    Code jb = jy.off + Code(b300_sparse_jrank(op)) * jy.stride;
    Code db = dy.off + Code(b300_sparse_drank(op)) * dy.stride;
    uint32_t kind = b300_sparse_kind(op);
    for (uint32_t lr = threadIdx.x; lr < x.stride; lr += blockDim.x) {
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

__global__ void b300_sparse_high_closure_kernel(Count* mainv, Count* blockv, int p) {
    constexpr int S = MAXW + 2;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t a = D_BS_HIGH_CLOSURE_OFF[pi];
    uint32_t b = D_BS_HIGH_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    uint64_t op = D_BS_HIGH_CLOSURE[q];
    uint32_t sbid = b300_sparse_closure_sblock(op);
    uint32_t hr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    FBlock x = D_F_MAIN_BLOCKS[sbid];
    FBlock y = D_F_BLOCK_BLOCKS[highdesc_block(desc)];
    Code ib = x.off + Code(hr) * x.stride;
    Code db = y.off + Code(highdesc_rank(desc)) * y.stride;
    uint32_t kind = highdesc_kind(desc);
    if (kind == HIGHDESC_BLOCK) {
        for (uint32_t lr = threadIdx.x; lr < x.stride; lr += blockDim.x) {
            Count c = mainv[ib + lr];
            if (c) atomic_add_mod(blockv + db + lr, c);
        }
    } else if (kind == HIGHDESC_CROSS) {
        uint32_t low0 = D_F_LOW_MASK_OFF[size_t(D_F_MASK) * S + x.hs];
        for (uint32_t lr = threadIdx.x; lr < x.stride; lr += blockDim.x) {
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

// One block owns one LOW sparse action and spans all HIGH rows of the fixed
// HIGH occupancy group.  No FBlock search or dense metadata lookup remains.
__global__ void b300_sparse_low_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_ORBIT_OFF[pi];
    uint32_t b = D_BS_LOW_ORBIT_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    B300SparseOrbitOp op = D_BS_LOW_ORBIT[q];
    FBlock x = D_F_MAIN_BLOCKS[b300_sparse_sblock(op)];
    FBlock jy = D_F_MAIN_BLOCKS[b300_sparse_jblock(op)];
    FBlock dy = D_F_BLOCK_BLOCKS[b300_sparse_dblock(op)];
    uint32_t lr = b300_sparse_src(op);
    uint32_t jlr = b300_sparse_jrank(op);
    uint32_t dlr = b300_sparse_drank(op);
    uint32_t kind = b300_sparse_kind(op);
    Code rows = x.stride ? (x.end - x.off) / x.stride : 0;
    for (Code hr = threadIdx.x; hr < rows; hr += blockDim.x) {
        Count* ip = mainv + x.off + hr * x.stride + lr;
        Count* jp = mainv + jy.off + hr * jy.stride + jlr;
        Count* dp = blockv + dy.off + hr * dy.stride + dlr;
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

__global__ void b300_sparse_low_closure_kernel(Count* mainv, Count* blockv, int p) {
    constexpr int S = MAXW + 2;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t a = D_BS_LOW_CLOSURE_OFF[pi];
    uint32_t b = D_BS_LOW_CLOSURE_OFF[pi + 1];
    uint32_t q = a + blockIdx.x;
    if (q >= b) return;
    uint64_t op = D_BS_LOW_CLOSURE[q];
    uint32_t sbid = b300_sparse_closure_sblock(op);
    uint32_t lr = b300_sparse_closure_src(op);
    uint32_t desc = b300_sparse_closure_desc(op);
    FBlock x = D_F_MAIN_BLOCKS[sbid];
    uint32_t kind = lowdesc_kind(desc);
    Code rows = x.stride ? (x.end - x.off) / x.stride : 0;
    uint32_t high0 = D_F_HIGH_MASK_OFF[size_t(D_F_MASK) * S + x.he];

    if (kind == LOWDESC_MAIN) {
        FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
        for (Code hr = threadIdx.x; hr < rows; hr += blockDim.x) {
            Count c = mainv[x.off + hr * x.stride + lr];
            if (c) atomic_add_mod(mainv + y.off + hr * y.stride + lowdesc_lr(desc), c);
        }
    } else if (kind == LOWDESC_BLOCK) {
        FBlock y = D_F_BLOCK_BLOCKS[lowdesc_block(desc)];
        for (Code hr = threadIdx.x; hr < rows; hr += blockDim.x) {
            Count c = mainv[x.off + hr * x.stride + lr];
            if (c) atomic_add_mod(blockv + y.off + hr * y.stride + lowdesc_lr(desc), c);
        }
    } else if (kind == LOWDESC_CROSS) {
        for (Code hr = threadIdx.x; hr < rows; hr += blockDim.x) {
            Count c = mainv[x.off + hr * x.stride + lr];
            if (!c) continue;
            uint32_t hc = D_F_HIGH_MASK_CODES[high0 + uint32_t(hr)];
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
        }
    }
}

static inline uint32_t b300_sparse_low_orbit_count(const B300SparseActionsHost& s, int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    return s.low_orbit_off[pi + 1] - s.low_orbit_off[pi];
}
static inline uint32_t b300_sparse_low_closure_count(const B300SparseActionsHost& s, int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    return s.low_closure_off[pi + 1] - s.low_closure_off[pi];
}
static inline uint32_t b300_sparse_high_orbit_count(const B300SparseActionsHost& s, int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    return s.high_orbit_off[pi + 1] - s.high_orbit_off[pi];
}
static inline uint32_t b300_sparse_high_closure_count(const B300SparseActionsHost& s, int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    return s.high_closure_off[pi + 1] - s.high_closure_off[pi];
}
