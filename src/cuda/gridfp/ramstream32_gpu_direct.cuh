#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

// GPU zero-scratch direct executor for the occupancy-major authoritative layout.
//
// This header is intentionally layered on the existing RAM-stream metadata
// builders.  Include it after ramstream32_cpu_low_sparse.hpp and
// ramstream32_cpu_high_direct.hpp.  Runtime GPU work needs no MateID, group
// rank/unrank tables, occupancy-mask tables, identity copy, or blocked clear.
//
// LOW uses the dense per-LOW-rank orbit table so lanes traverse consecutive LOW
// columns. HIGH uses the compact per-HIGH-rank streams and lets lanes traverse
// consecutive LOW columns. Closure writes retain modular atomics in this first
// correctness-oriented version; orbit writes are conflict-free plain stores.

static constexpr uint32_t GPU_DIRECT_INVALID_RANK = 0xffffffffu;
static constexpr int GPU_DIRECT_MAX_MAIN_BLOCKS = 64;
static constexpr int GPU_DIRECT_MAX_BLOCK_BLOCKS = 32;

struct GpuDirectBlock {
    Code off;
    uint32_t rows;
    uint32_t cols;
    uint8_t he;
    uint8_t hs;
    uint8_t c;
    uint8_t valid;
};

static GpuDirectBlock gpu_direct_block(const StorageBlock& x) {
    return {x.off, x.rows, x.cols, x.he, x.hs, x.c, x.valid};
}

struct GpuDirectCrossHost {
    // LOW-window CROSS: [(depth-1) * high_pitch + global HIGH-all-code index]
    // -> destination HIGH all-rank inside the destination factor-block height.
    std::vector<uint32_t> high_rank;
    uint32_t high_pitch = 0;

    // HIGH-window CROSS: [(depth-1) * low_pitch + global LOW-all-code index]
    // -> destination LOW all-rank inside the destination factor-block height.
    std::vector<uint32_t> low_rank;
    uint32_t low_pitch = 0;
};

static GpuDirectCrossHost build_gpu_direct_cross(const StorageFactorHost& storage) {
    GpuDirectCrossHost out;

    out.high_pitch = uint32_t(storage.high_all_codes.size());
    out.high_rank.assign(
        size_t(HIGH_LUT_K) * out.high_pitch, GPU_DIRECT_INVALID_RANK);
    for (uint32_t depth = 1; depth <= uint32_t(HIGH_LUT_K); ++depth) {
        uint32_t* dst = out.high_rank.data() + size_t(depth - 1) * out.high_pitch;
        for (uint32_t i = 0; i < out.high_pitch; ++i) {
            uint32_t hc2 = cpu_low_flip_high(storage.high_all_codes[i], depth);
            if (hc2 == 0xffffffffu) continue;
            uint32_t packed = storage.high_packed_rank[hc2];
            if (packed == 0xffffffffu) continue;
            dst[i] = packed >> HIGH_LUT_K;
        }
    }

    out.low_pitch = uint32_t(storage.low_all_codes.size());
    out.low_rank.assign(
        size_t(LOW_LUT_K) * out.low_pitch, GPU_DIRECT_INVALID_RANK);
    for (uint32_t depth = 1; depth <= uint32_t(LOW_LUT_K); ++depth) {
        uint32_t* dst = out.low_rank.data() + size_t(depth - 1) * out.low_pitch;
        for (uint32_t i = 0; i < out.low_pitch; ++i) {
            uint32_t lc2 = cpu_high_flip_low(storage.low_all_codes[i], depth);
            if (lc2 == 0xffffffffu) continue;
            uint32_t packed = storage.low_packed_rank[lc2];
            if (packed == 0xffffffffu) continue;
            dst[i] = packed >> LOW_LUT_K;
        }
    }

    std::cerr << "gpu_direct_cross high_mib="
              << double(out.high_rank.size() * sizeof(uint32_t)) / (1 << 20)
              << " low_mib="
              << double(out.low_rank.size() * sizeof(uint32_t)) / (1 << 20)
              << '\n';
    return out;
}

__constant__ GpuDirectBlock D_GD_MAIN_BLOCKS[GPU_DIRECT_MAX_MAIN_BLOCKS];
__constant__ GpuDirectBlock D_GD_BLOCK_BLOCKS[GPU_DIRECT_MAX_BLOCK_BLOCKS];
__constant__ uint32_t D_GD_MAIN_NBLOCKS;
__constant__ uint32_t D_GD_BLOCK_NBLOCKS;
__constant__ uint32_t D_GD_HIGH_ALL_OFF[MAXW + 2];
__constant__ uint32_t D_GD_LOW_ALL_OFF[MAXW + 2];

__constant__ uint64_t* D_GD_LOW_ORBIT;
__constant__ uint32_t D_GD_LOW_ORBIT_MAIN_BASE[GPU_DIRECT_MAX_MAIN_BLOCKS];
__constant__ uint32_t D_GD_LOW_ORBIT_MAIN_TOTAL;

__constant__ CpuHighOrbitOp* D_GD_HIGH_NN_OPS;
__constant__ CpuHighOrbitOp* D_GD_HIGH_NRNL_OPS;
__constant__ CpuHighClosureOp* D_GD_HIGH_BLOCK_CLOSURE_OPS;
__constant__ CpuHighClosureOp* D_GD_HIGH_CROSS_CLOSURE_OPS;
__constant__ uint32_t* D_GD_HIGH_NN_OFF;
__constant__ uint32_t* D_GD_HIGH_NRNL_OFF;
__constant__ uint32_t* D_GD_HIGH_BLOCK_CLOSURE_OFF;
__constant__ uint32_t* D_GD_HIGH_CROSS_CLOSURE_OFF;
__constant__ uint32_t D_GD_HIGH_PITCH;

__constant__ uint32_t* D_GD_HIGH_CROSS_RANK;
__constant__ uint32_t D_GD_HIGH_CROSS_PITCH;
__constant__ uint32_t* D_GD_LOW_CROSS_RANK;
__constant__ uint32_t D_GD_LOW_CROSS_PITCH;

__device__ __forceinline__ Count gpu_direct_add(Count a, Count b) {
    if (!b) return a;
    Count mod = D_MOD;
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

__device__ __forceinline__ uint32_t gpu_direct_low_orbit_kind(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_KIND_SHIFT) & 7u);
}
__device__ __forceinline__ uint32_t gpu_direct_low_orbit_jlr(uint64_t x) {
    return uint32_t(x & CPU_ORBIT_LR_MASK);
}
__device__ __forceinline__ uint32_t gpu_direct_low_orbit_jblock(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_JBLOCK_SHIFT) & CPU_ORBIT_BLOCK_MASK);
}
__device__ __forceinline__ uint32_t gpu_direct_low_orbit_dlr(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_DLR_SHIFT) & CPU_ORBIT_LR_MASK);
}
__device__ __forceinline__ uint32_t gpu_direct_low_orbit_dblock(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_DBLOCK_SHIFT) & CPU_ORBIT_BLOCK_MASK);
}

__device__ __forceinline__ uint32_t gpu_direct_high_src(CpuHighOrbitOp x) {
    return uint32_t(x & CPU_HIGH_ORBIT_RANK_MASK);
}
__device__ __forceinline__ uint32_t gpu_direct_high_partner(CpuHighOrbitOp x) {
    return uint32_t((x >> CPU_HIGH_ORBIT_PARTNER_SHIFT) & CPU_HIGH_ORBIT_RANK_MASK);
}
__device__ __forceinline__ uint32_t gpu_direct_high_drop(CpuHighOrbitOp x) {
    return uint32_t((x >> CPU_HIGH_ORBIT_DROP_SHIFT) & CPU_HIGH_ORBIT_RANK_MASK);
}

__device__ __forceinline__ uint32_t gpu_direct_high_partner_block(
    uint32_t source_bid, const GpuDirectBlock& source, int p, bool nn_stream
) {
    if (p != LOW_LUT_K + 1) return source_bid;
    uint32_t center = nn_stream ? uint32_t(R) : uint32_t(N);
    int he = int(source.hs) + (center == uint32_t(R) ? 1 : 0);
    return uint32_t(3 * he + int(center));
}

__global__ void gpu_direct_low_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_GD_MAIN_NBLOCKS) return;
    GpuDirectBlock x = D_GD_MAIN_BLOCKS[bid];
    if (!x.valid || !x.rows || !x.cols) return;

    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t lr0 = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t lr_step = uint32_t(gridDim.x) * blockDim.x;

    for (uint32_t lr = lr0; lr < x.cols; lr += lr_step) {
        uint64_t ow = D_GD_LOW_ORBIT[
            size_t(pi) * D_GD_LOW_ORBIT_MAIN_TOTAL
            + D_GD_LOW_ORBIT_MAIN_BASE[bid] + lr];
        uint32_t kind = gpu_direct_low_orbit_kind(ow);
        if (kind < CPU_ORBIT_NN || kind > CPU_ORBIT_NL) continue;

        uint32_t jbid = gpu_direct_low_orbit_jblock(ow);
        uint32_t dbid = gpu_direct_low_orbit_dblock(ow);
        GpuDirectBlock y = D_GD_MAIN_BLOCKS[jbid];
        GpuDirectBlock d = D_GD_BLOCK_BLOCKS[dbid];
        uint32_t jlr = gpu_direct_low_orbit_jlr(ow);
        uint32_t dlr = gpu_direct_low_orbit_dlr(ow);

        for (uint32_t hr = blockIdx.y; hr < x.rows; hr += gridDim.y) {
            Count* ip = mainv + x.off + Code(hr) * x.cols + lr;
            Count* jp = mainv + y.off + Code(hr) * y.cols + jlr;
            Count* dp = blockv + d.off + Code(hr) * d.cols + dlr;
            Count c = *ip;
            Count old_d = *dp;

            if (kind == CPU_ORBIT_NN) {
                *jp = gpu_direct_add(*jp, c);
                *ip = gpu_direct_add(c, old_d);
                *dp = 0;
            } else {
                Count cc = *jp;
                Count all = gpu_direct_add(gpu_direct_add(c, cc), old_d);
                if (p == 1) {
                    *ip = all;
                    *jp = gpu_direct_add(c, cc);
                    *dp = 0;
                } else {
                    *ip = all;
                    *dp = c;
                }
            }
        }
    }
}

__global__ void gpu_direct_low_closure_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_GD_MAIN_NBLOCKS) return;
    GpuDirectBlock x = D_GD_MAIN_BLOCKS[bid];
    if (!x.valid || !x.rows || !x.cols) return;

    uint32_t pi = uint32_t(LOW_LUT_K - p);
    uint32_t lr0 = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t lr_step = uint32_t(gridDim.x) * blockDim.x;

    for (uint32_t lr = lr0; lr < x.cols; lr += lr_step) {
        uint64_t ow = D_GD_LOW_ORBIT[
            size_t(pi) * D_GD_LOW_ORBIT_MAIN_TOTAL
            + D_GD_LOW_ORBIT_MAIN_BASE[bid] + lr];
        if (gpu_direct_low_orbit_kind(ow) != CPU_ORBIT_CLOSURE) continue;

        uint32_t desc = D_LOWDESC_MAIN[
            size_t(pi) * D_LOWDESC_MAIN_TOTAL + D_LOWDESC_MAIN_BASE[bid] + lr];
        uint32_t kind = lowdesc_kind(desc);
        if (kind == LOWDESC_INVALID) continue;

        for (uint32_t hr = blockIdx.y; hr < x.rows; hr += gridDim.y) {
            Count c = mainv[x.off + Code(hr) * x.cols + lr];
            if (!c) continue;

            if (kind == LOWDESC_MAIN) {
                GpuDirectBlock y = D_GD_MAIN_BLOCKS[lowdesc_block(desc)];
                Code j = y.off + Code(hr) * y.cols + lowdesc_lr(desc);
                atomic_add_mod(mainv + j, c);
            } else if (kind == LOWDESC_BLOCK) {
                GpuDirectBlock y = D_GD_BLOCK_BLOCKS[lowdesc_block(desc)];
                Code j = y.off + Code(hr) * y.cols + lowdesc_lr(desc);
                atomic_add_mod(blockv + j, c);
            } else if (kind == LOWDESC_CROSS) {
                uint32_t depth = lowdesc_depth(desc);
                if (!depth || depth > uint32_t(HIGH_LUT_K)) continue;
                uint32_t gi = D_GD_HIGH_ALL_OFF[x.he] + hr;
                uint32_t hr2 = D_GD_HIGH_CROSS_RANK[
                    size_t(depth - 1) * D_GD_HIGH_CROSS_PITCH + gi];
                if (hr2 == GPU_DIRECT_INVALID_RANK) continue;
                if (p == 1) {
                    GpuDirectBlock y = D_GD_MAIN_BLOCKS[lowdesc_block(desc)];
                    Code j = y.off + Code(hr2) * y.cols + lowdesc_lr(desc);
                    atomic_add_mod(mainv + j, c);
                } else {
                    GpuDirectBlock y = D_GD_BLOCK_BLOCKS[lowdesc_block(desc)];
                    Code j = y.off + Code(hr2) * y.cols + lowdesc_lr(desc);
                    atomic_add_mod(blockv + j, c);
                }
            }
        }
    }
}

__global__ void gpu_direct_high_orbit_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_GD_MAIN_NBLOCKS) return;
    GpuDirectBlock x = D_GD_MAIN_BLOCKS[bid];
    if (!x.valid || !x.rows || !x.cols) return;

    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    size_t oi = size_t(pi) * D_GD_HIGH_PITCH + bid;
    uint32_t na = D_GD_HIGH_NN_OFF[oi], nb = D_GD_HIGH_NN_OFF[oi + 1];
    uint32_t ra = D_GD_HIGH_NRNL_OFF[oi], rb = D_GD_HIGH_NRNL_OFF[oi + 1];
    uint32_t nn_count = nb - na;
    uint32_t nrnl_count = rb - ra;
    uint32_t total = nn_count + nrnl_count;
    if (!total) return;

    uint32_t lr0 = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t lr_step = uint32_t(gridDim.x) * blockDim.x;
    uint32_t drop_bid = uint32_t(x.hs);
    GpuDirectBlock d = D_GD_BLOCK_BLOCKS[drop_bid];

    for (uint32_t k = blockIdx.y; k < total; k += gridDim.y) {
        bool nn = k < nn_count;
        CpuHighOrbitOp op = nn
            ? D_GD_HIGH_NN_OPS[na + k]
            : D_GD_HIGH_NRNL_OPS[ra + (k - nn_count)];
        uint32_t partner_bid = gpu_direct_high_partner_block(bid, x, p, nn);
        GpuDirectBlock y = D_GD_MAIN_BLOCKS[partner_bid];
        uint32_t shr = gpu_direct_high_src(op);
        uint32_t jhr = gpu_direct_high_partner(op);
        uint32_t dhr = gpu_direct_high_drop(op);

        for (uint32_t lr = lr0; lr < x.cols; lr += lr_step) {
            Count* ip = mainv + x.off + Code(shr) * x.cols + lr;
            Count* jp = mainv + y.off + Code(jhr) * y.cols + lr;
            Count* dp = blockv + d.off + Code(dhr) * d.cols + lr;
            Count c = *ip;
            Count old_d = *dp;
            if (nn) {
                *jp = gpu_direct_add(*jp, c);
                *ip = gpu_direct_add(c, old_d);
                *dp = 0;
            } else {
                Count cc = *jp;
                *ip = gpu_direct_add(gpu_direct_add(c, cc), old_d);
                *dp = c;
            }
        }
    }
}

__global__ void gpu_direct_high_closure_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t bid = blockIdx.z;
    if (bid >= D_GD_MAIN_NBLOCKS) return;
    GpuDirectBlock x = D_GD_MAIN_BLOCKS[bid];
    if (!x.valid || !x.rows || !x.cols) return;

    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    size_t oi = size_t(pi) * D_GD_HIGH_PITCH + bid;
    uint32_t ba = D_GD_HIGH_BLOCK_CLOSURE_OFF[oi];
    uint32_t bb = D_GD_HIGH_BLOCK_CLOSURE_OFF[oi + 1];
    uint32_t ca = D_GD_HIGH_CROSS_CLOSURE_OFF[oi];
    uint32_t cb = D_GD_HIGH_CROSS_CLOSURE_OFF[oi + 1];
    uint32_t block_count = bb - ba;
    uint32_t cross_count = cb - ca;
    uint32_t total = block_count + cross_count;
    if (!total) return;

    uint32_t lr0 = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t lr_step = uint32_t(gridDim.x) * blockDim.x;

    for (uint32_t k = blockIdx.y; k < total; k += gridDim.y) {
        bool cross = k >= block_count;
        CpuHighClosureOp op = cross
            ? D_GD_HIGH_CROSS_CLOSURE_OPS[ca + (k - block_count)]
            : D_GD_HIGH_BLOCK_CLOSURE_OPS[ba + k];
        uint32_t dbid = highdesc_block(op.desc);
        GpuDirectBlock y = D_GD_BLOCK_BLOCKS[dbid];
        uint32_t dhr = highdesc_rank(op.desc);
        uint32_t depth = cross ? highdesc_depth(op.desc) : 0;

        for (uint32_t lr = lr0; lr < x.cols; lr += lr_step) {
            Count c = mainv[x.off + Code(op.src_hr) * x.cols + lr];
            if (!c) continue;
            uint32_t lr2 = lr;
            if (cross) {
                if (!depth || depth > uint32_t(LOW_LUT_K)) continue;
                uint32_t gi = D_GD_LOW_ALL_OFF[x.hs] + lr;
                lr2 = D_GD_LOW_CROSS_RANK[
                    size_t(depth - 1) * D_GD_LOW_CROSS_PITCH + gi];
                if (lr2 == GPU_DIRECT_INVALID_RANK) continue;
            }
            Code j = y.off + Code(dhr) * y.cols + lr2;
            atomic_add_mod(blockv + j, c);
        }
    }
}

struct GpuDirectDeviceTables {
    LowDescDeviceTables lowdesc;
    uint64_t* low_orbit = nullptr;
    CpuHighOrbitOp* high_nn = nullptr;
    CpuHighOrbitOp* high_nrnl = nullptr;
    CpuHighClosureOp* high_block_closure = nullptr;
    CpuHighClosureOp* high_cross_closure = nullptr;
    uint32_t* high_nn_off = nullptr;
    uint32_t* high_nrnl_off = nullptr;
    uint32_t* high_block_closure_off = nullptr;
    uint32_t* high_cross_closure_off = nullptr;
    uint32_t* high_cross_rank = nullptr;
    uint32_t* low_cross_rank = nullptr;

    template<class T>
    static void copy_vec(T*& dst, const std::vector<T>& src, const char* what) {
        if (src.empty()) return;
        ck(cudaMalloc(&dst, src.size() * sizeof(T)), what);
        ck(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(
        const StorageFactorHost& storage, const StorageLayout& layout,
        const LowDescHost& lowdesc_h, const LowOrbitHost& loworbit_h,
        const CpuHighDirectHost& highdirect_h, const GpuDirectCrossHost& cross_h
    ) {
        if (layout.main_blocks.size() > GPU_DIRECT_MAX_MAIN_BLOCKS
            || layout.block_blocks.size() > GPU_DIRECT_MAX_BLOCK_BLOCKS) {
            std::cerr << "gpu direct block table overflow\n";
            std::exit(140);
        }

        std::vector<GpuDirectBlock> mb(layout.main_blocks.size());
        std::vector<GpuDirectBlock> db(layout.block_blocks.size());
        for (size_t i = 0; i < mb.size(); ++i) mb[i] = gpu_direct_block(layout.main_blocks[i]);
        for (size_t i = 0; i < db.size(); ++i) db[i] = gpu_direct_block(layout.block_blocks[i]);
        uint32_t mn = uint32_t(mb.size()), dn = uint32_t(db.size());
        ck(cudaMemcpyToSymbol(D_GD_MAIN_BLOCKS, mb.data(), mb.size() * sizeof(GpuDirectBlock)),
           "gpu direct main blocks");
        ck(cudaMemcpyToSymbol(D_GD_BLOCK_BLOCKS, db.data(), db.size() * sizeof(GpuDirectBlock)),
           "gpu direct block blocks");
        ck(cudaMemcpyToSymbol(D_GD_MAIN_NBLOCKS, &mn, sizeof(mn)), "gpu direct main nblocks");
        ck(cudaMemcpyToSymbol(D_GD_BLOCK_NBLOCKS, &dn, sizeof(dn)), "gpu direct block nblocks");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_ALL_OFF, storage.high_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "gpu direct high all off");
        ck(cudaMemcpyToSymbol(D_GD_LOW_ALL_OFF, storage.low_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "gpu direct low all off");

        lowdesc.install(lowdesc_h);
        copy_vec(low_orbit, loworbit_h.rec, "gpu direct low orbit");
        ck(cudaMemcpyToSymbol(D_GD_LOW_ORBIT, &low_orbit, sizeof(low_orbit)),
           "gpu direct low orbit ptr");
        ck(cudaMemcpyToSymbol(D_GD_LOW_ORBIT_MAIN_BASE, loworbit_h.main_base.data(),
                              sizeof(uint32_t) * loworbit_h.main_base.size()),
           "gpu direct low orbit base");
        ck(cudaMemcpyToSymbol(D_GD_LOW_ORBIT_MAIN_TOTAL, &loworbit_h.main_total,
                              sizeof(loworbit_h.main_total)), "gpu direct low orbit total");

        copy_vec(high_nn, highdirect_h.orbit_ops.nn, "gpu direct high nn");
        copy_vec(high_nrnl, highdirect_h.orbit_ops.nrnl, "gpu direct high nrnl");
        copy_vec(high_block_closure, highdirect_h.closure_ops.block,
                 "gpu direct high block closure");
        copy_vec(high_cross_closure, highdirect_h.closure_ops.cross,
                 "gpu direct high cross closure");
        copy_vec(high_nn_off, highdirect_h.orbit_off.nn, "gpu direct high nn off");
        copy_vec(high_nrnl_off, highdirect_h.orbit_off.nrnl, "gpu direct high nrnl off");
        copy_vec(high_block_closure_off, highdirect_h.closure_off.block,
                 "gpu direct high block closure off");
        copy_vec(high_cross_closure_off, highdirect_h.closure_off.cross,
                 "gpu direct high cross closure off");
        uint32_t hpitch = highdirect_h.nblocks + 1;
        ck(cudaMemcpyToSymbol(D_GD_HIGH_NN_OPS, &high_nn, sizeof(high_nn)), "gpu direct high nn ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_NRNL_OPS, &high_nrnl, sizeof(high_nrnl)),
           "gpu direct high nrnl ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_BLOCK_CLOSURE_OPS, &high_block_closure,
                              sizeof(high_block_closure)), "gpu direct high block closure ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_CROSS_CLOSURE_OPS, &high_cross_closure,
                              sizeof(high_cross_closure)), "gpu direct high cross closure ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_NN_OFF, &high_nn_off, sizeof(high_nn_off)),
           "gpu direct high nn off ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_NRNL_OFF, &high_nrnl_off, sizeof(high_nrnl_off)),
           "gpu direct high nrnl off ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_BLOCK_CLOSURE_OFF, &high_block_closure_off,
                              sizeof(high_block_closure_off)), "gpu direct high block closure off ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_CROSS_CLOSURE_OFF, &high_cross_closure_off,
                              sizeof(high_cross_closure_off)), "gpu direct high cross closure off ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_PITCH, &hpitch, sizeof(hpitch)), "gpu direct high pitch");

        copy_vec(high_cross_rank, cross_h.high_rank, "gpu direct high cross rank");
        copy_vec(low_cross_rank, cross_h.low_rank, "gpu direct low cross rank");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_CROSS_RANK, &high_cross_rank, sizeof(high_cross_rank)),
           "gpu direct high cross rank ptr");
        ck(cudaMemcpyToSymbol(D_GD_HIGH_CROSS_PITCH, &cross_h.high_pitch,
                              sizeof(cross_h.high_pitch)), "gpu direct high cross pitch");
        ck(cudaMemcpyToSymbol(D_GD_LOW_CROSS_RANK, &low_cross_rank, sizeof(low_cross_rank)),
           "gpu direct low cross rank ptr");
        ck(cudaMemcpyToSymbol(D_GD_LOW_CROSS_PITCH, &cross_h.low_pitch,
                              sizeof(cross_h.low_pitch)), "gpu direct low cross pitch");
    }

    void release() {
        lowdesc.release();
        if (low_orbit) cudaFree(low_orbit);
        if (high_nn) cudaFree(high_nn);
        if (high_nrnl) cudaFree(high_nrnl);
        if (high_block_closure) cudaFree(high_block_closure);
        if (high_cross_closure) cudaFree(high_cross_closure);
        if (high_nn_off) cudaFree(high_nn_off);
        if (high_nrnl_off) cudaFree(high_nrnl_off);
        if (high_block_closure_off) cudaFree(high_block_closure_off);
        if (high_cross_closure_off) cudaFree(high_cross_closure_off);
        if (high_cross_rank) cudaFree(high_cross_rank);
        if (low_cross_rank) cudaFree(low_cross_rank);
        low_orbit = nullptr;
        high_nn = high_nrnl = nullptr;
        high_block_closure = high_cross_closure = nullptr;
        high_nn_off = high_nrnl_off = nullptr;
        high_block_closure_off = high_cross_closure_off = nullptr;
        high_cross_rank = low_cross_rank = nullptr;
    }
};

static void gpu_direct_run_low(
    Count* mainv, Count* blockv, const StorageLayout& layout,
    int threads = 256, int grid_x = 16, int grid_y = 8
) {
    dim3 block(threads);
    dim3 grid(grid_x, grid_y, unsigned(layout.main_blocks.size()));
    for (int p = LOW_LUT_K; p >= 1; --p) {
        gpu_direct_low_orbit_kernel<<<grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct low orbit");
        gpu_direct_low_closure_kernel<<<grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct low closure");
    }
    ck(cudaDeviceSynchronize(), "gpu direct low sync");
}

static void gpu_direct_run_high(
    Count* mainv, Count* blockv, const StorageLayout& layout,
    int threads = 256, int grid_x = 16, int grid_y = 8
) {
    dim3 block(threads);
    dim3 grid(grid_x, grid_y, unsigned(layout.main_blocks.size()));
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        gpu_direct_high_orbit_kernel<<<grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct high orbit");
        gpu_direct_high_closure_kernel<<<grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct high closure");
    }
    ck(cudaDeviceSynchronize(), "gpu direct high sync");
}
