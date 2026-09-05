#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

// Destination-oriented closure backend layered on ramstream32_gpu_direct.cuh.
//
// Orbit updates remain source-oriented and conflict-free. Ordinary closure
// updates are inverted once on the host into a compact destination CSR. A GPU
// thread owns one destination cell, reads all of its source cells (the exact
// n=27 half-code graph has maximum indegree 7), performs modular reduction in
// registers, and stores once. CROSS closures remain source-oriented in this
// version; LOW CROSS is compacted into an 8-byte stream while HIGH reuses the
// existing compact CpuHighClosureOp stream.

static constexpr uint32_t GPU_DIRECT_GATHER_RANK_MASK = (1u << 20) - 1u;
static constexpr int GPU_DIRECT_GATHER_SRC_BLOCK_SHIFT = 20;

static inline uint32_t gpu_direct_gather_src_pack(uint32_t bid, uint32_t rank) {
    if (bid >= 64 || rank > GPU_DIRECT_GATHER_RANK_MASK) {
        std::cerr << "gpu direct gather source encoding overflow block=" << bid
                  << " rank=" << rank << '\n';
        std::exit(150);
    }
    return (bid << GPU_DIRECT_GATHER_SRC_BLOCK_SHIFT) | rank;
}

struct GpuDirectGatherDst {
    uint32_t dst_rank = 0;
    uint32_t edge_begin = 0;
    uint32_t edge_count = 0;
};
static_assert(sizeof(GpuDirectGatherDst) == 12);

struct GpuDirectLowCrossOp {
    uint32_t src_lr = 0;
    uint32_t desc = 0;
};
static_assert(sizeof(GpuDirectLowCrossOp) == 8);

struct GpuDirectGatherHost {
    // LOW ordinary closure: destination records are grouped by
    // [pi][destination block]. p==1 targets main; p>1 targets blocked.
    std::vector<GpuDirectGatherDst> low_dst;
    std::vector<uint32_t> low_src;
    std::vector<uint32_t> low_off;
    uint32_t low_pitch = GPU_DIRECT_MAX_MAIN_BLOCKS + 1;

    // LOW CROSS closure: source records grouped by [pi][source main block].
    std::vector<GpuDirectLowCrossOp> low_cross;
    std::vector<uint32_t> low_cross_off;
    uint32_t low_cross_pitch = GPU_DIRECT_MAX_MAIN_BLOCKS + 1;

    // HIGH ordinary closure: destination records grouped by
    // [pi][destination blocked block].
    std::vector<GpuDirectGatherDst> high_dst;
    std::vector<uint32_t> high_src;
    std::vector<uint32_t> high_off;
    uint32_t high_pitch = GPU_DIRECT_MAX_BLOCK_BLOCKS + 1;

    uint32_t low_max_indegree = 0;
    uint32_t high_max_indegree = 0;

    size_t bytes() const {
        return low_dst.size() * sizeof(GpuDirectGatherDst)
            + low_src.size() * sizeof(uint32_t)
            + low_off.size() * sizeof(uint32_t)
            + low_cross.size() * sizeof(GpuDirectLowCrossOp)
            + low_cross_off.size() * sizeof(uint32_t)
            + high_dst.size() * sizeof(GpuDirectGatherDst)
            + high_src.size() * sizeof(uint32_t)
            + high_off.size() * sizeof(uint32_t);
    }
};

static GpuDirectGatherHost build_gpu_direct_gather(
    const StorageLayout& layout, const LowDescHost& lowdesc,
    const LowOrbitHost& loworbit, const CpuHighDirectHost& highdirect
) {
    GpuDirectGatherHost out;
    out.low_off.assign(size_t(LOW_LUT_K) * out.low_pitch, 0);
    out.low_cross_off.assign(size_t(LOW_LUT_K) * out.low_cross_pitch, 0);
    out.high_off.assign(size_t(HIGH_LUT_K) * out.high_pitch, 0);

    // LOW ordinary/CROSS closures.
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        bool target_main = p == 1;
        uint32_t target_blocks = target_main
            ? uint32_t(layout.main_blocks.size())
            : uint32_t(layout.block_blocks.size());

        using Edge = std::pair<uint32_t, uint32_t>; // (dst rank, packed source)
        std::vector<std::vector<Edge>> by_dst(target_blocks);
        std::vector<std::vector<GpuDirectLowCrossOp>> cross_by_src(layout.main_blocks.size());
        std::unordered_set<uint32_t> ordinary_sources;
        std::unordered_set<uint32_t> ordinary_destinations;

        for (uint32_t sbid = 0; sbid < uint32_t(layout.main_blocks.size()); ++sbid) {
            const StorageBlock& sb = layout.main_blocks[sbid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                uint64_t ow = loworbit.rec[
                    size_t(pi) * loworbit.main_total + loworbit.main_base[sbid] + lr];
                if (cpu_orbit_kind(ow) != CPU_ORBIT_CLOSURE) continue;
                uint32_t desc = lowdesc.main_desc[
                    size_t(pi) * lowdesc.main_total + lowdesc.main_base[sbid] + lr];
                uint32_t kind = cpu_low_kind(desc);
                if (kind == LOWDESC_INVALID) continue;
                if (kind == LOWDESC_CROSS) {
                    cross_by_src[sbid].push_back({lr, desc});
                    continue;
                }

                uint32_t expected = target_main ? LOWDESC_MAIN : LOWDESC_BLOCK;
                if (kind != expected) {
                    std::cerr << "gpu direct gather LOW ordinary kind mismatch p=" << p
                              << " source_block=" << sbid << " lr=" << lr
                              << " kind=" << kind << " expected=" << expected << '\n';
                    std::exit(151);
                }
                uint32_t dbid = cpu_low_block(desc);
                uint32_t dlr = cpu_low_lr(desc);
                if (dbid >= target_blocks) {
                    std::cerr << "gpu direct gather LOW destination block overflow\n";
                    std::exit(152);
                }
                const StorageBlock& db = target_main
                    ? layout.main_blocks[dbid] : layout.block_blocks[dbid];
                if (dlr >= db.cols || db.rows != sb.rows || db.he != sb.he) {
                    std::cerr << "gpu direct gather LOW destination shape mismatch p=" << p
                              << " src=" << sbid << ':' << lr
                              << " dst=" << dbid << ':' << dlr << '\n';
                    std::exit(153);
                }
                uint32_t src = gpu_direct_gather_src_pack(sbid, lr);
                by_dst[dbid].push_back({dlr, src});
                if (target_main) {
                    ordinary_sources.insert(src);
                    ordinary_destinations.insert(gpu_direct_gather_src_pack(dbid, dlr));
                }
            }
        }

        // p==1 writes main in-place. Closure sources and ordinary closure
        // destinations must be disjoint so destination-gather cannot overwrite
        // a value another destination still has to read.
        if (target_main) {
            for (uint32_t d : ordinary_destinations) {
                if (ordinary_sources.count(d)) {
                    std::cerr << "gpu direct gather LOW p=1 source/destination alias\n";
                    std::exit(154);
                }
            }
        }

        for (uint32_t dbid = 0; dbid < target_blocks; ++dbid) {
            size_t oi = size_t(pi) * out.low_pitch + dbid;
            out.low_off[oi] = uint32_t(out.low_dst.size());
            auto& es = by_dst[dbid];
            std::sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });
            size_t i = 0;
            while (i < es.size()) {
                size_t j = i + 1;
                while (j < es.size() && es[j].first == es[i].first) ++j;
                uint32_t begin = uint32_t(out.low_src.size());
                for (size_t k = i; k < j; ++k) out.low_src.push_back(es[k].second);
                uint32_t count = uint32_t(j - i);
                out.low_max_indegree = std::max(out.low_max_indegree, count);
                out.low_dst.push_back({es[i].first, begin, count});
                i = j;
            }
        }
        out.low_off[size_t(pi) * out.low_pitch + target_blocks]
            = uint32_t(out.low_dst.size());

        for (uint32_t sbid = 0; sbid < uint32_t(layout.main_blocks.size()); ++sbid) {
            size_t oi = size_t(pi) * out.low_cross_pitch + sbid;
            out.low_cross_off[oi] = uint32_t(out.low_cross.size());
            auto& xs = cross_by_src[sbid];
            out.low_cross.insert(out.low_cross.end(), xs.begin(), xs.end());
        }
        out.low_cross_off[size_t(pi) * out.low_cross_pitch + layout.main_blocks.size()]
            = uint32_t(out.low_cross.size());
    }

    // HIGH ordinary blocked closures.
    for (uint32_t pi = 0; pi < uint32_t(HIGH_LUT_K); ++pi) {
        using Edge = std::pair<uint32_t, uint32_t>; // (dst rank, packed source)
        std::vector<std::vector<Edge>> by_dst(layout.block_blocks.size());
        for (uint32_t sbid = 0; sbid < highdirect.nblocks; ++sbid) {
            auto [a, b] = cpu_high_direct_range(
                highdirect.closure_off.block, highdirect.nblocks, pi, sbid);
            const StorageBlock& sb = layout.main_blocks[sbid];
            for (uint32_t q = a; q < b; ++q) {
                const CpuHighClosureOp& op = highdirect.closure_ops.block[q];
                uint32_t dbid = cpu_high_desc_block(op.desc);
                uint32_t dhr = cpu_high_desc_rank(op.desc);
                if (dbid >= layout.block_blocks.size()) {
                    std::cerr << "gpu direct gather HIGH destination block overflow\n";
                    std::exit(155);
                }
                const StorageBlock& db = layout.block_blocks[dbid];
                if (dhr >= db.rows || db.cols != sb.cols || db.hs != sb.hs) {
                    std::cerr << "gpu direct gather HIGH destination shape mismatch pi=" << pi
                              << " src=" << sbid << ':' << op.src_hr
                              << " dst=" << dbid << ':' << dhr << '\n';
                    std::exit(156);
                }
                by_dst[dbid].push_back({dhr,
                    gpu_direct_gather_src_pack(sbid, op.src_hr)});
            }
        }

        uint32_t target_blocks = uint32_t(layout.block_blocks.size());
        for (uint32_t dbid = 0; dbid < target_blocks; ++dbid) {
            size_t oi = size_t(pi) * out.high_pitch + dbid;
            out.high_off[oi] = uint32_t(out.high_dst.size());
            auto& es = by_dst[dbid];
            std::sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });
            size_t i = 0;
            while (i < es.size()) {
                size_t j = i + 1;
                while (j < es.size() && es[j].first == es[i].first) ++j;
                uint32_t begin = uint32_t(out.high_src.size());
                for (size_t k = i; k < j; ++k) out.high_src.push_back(es[k].second);
                uint32_t count = uint32_t(j - i);
                out.high_max_indegree = std::max(out.high_max_indegree, count);
                out.high_dst.push_back({es[i].first, begin, count});
                i = j;
            }
        }
        out.high_off[size_t(pi) * out.high_pitch + target_blocks]
            = uint32_t(out.high_dst.size());
    }

    std::cerr << "gpu_direct_gather"
              << " low_dst=" << out.low_dst.size()
              << " low_edges=" << out.low_src.size()
              << " low_cross=" << out.low_cross.size()
              << " low_max_indegree=" << out.low_max_indegree
              << " high_dst=" << out.high_dst.size()
              << " high_edges=" << out.high_src.size()
              << " high_max_indegree=" << out.high_max_indegree
              << " mib=" << double(out.bytes()) / double(1 << 20)
              << '\n';
    return out;
}

__constant__ GpuDirectGatherDst* D_GDG_LOW_DST;
__constant__ uint32_t* D_GDG_LOW_SRC;
__constant__ uint32_t* D_GDG_LOW_OFF;
__constant__ uint32_t D_GDG_LOW_PITCH;
__constant__ GpuDirectLowCrossOp* D_GDG_LOW_CROSS;
__constant__ uint32_t* D_GDG_LOW_CROSS_OFF;
__constant__ uint32_t D_GDG_LOW_CROSS_PITCH;

__constant__ GpuDirectGatherDst* D_GDG_HIGH_DST;
__constant__ uint32_t* D_GDG_HIGH_SRC;
__constant__ uint32_t* D_GDG_HIGH_OFF;
__constant__ uint32_t D_GDG_HIGH_PITCH;

__device__ __forceinline__ uint32_t gpu_direct_gather_src_block(uint32_t x) {
    return x >> GPU_DIRECT_GATHER_SRC_BLOCK_SHIFT;
}
__device__ __forceinline__ uint32_t gpu_direct_gather_src_rank(uint32_t x) {
    return x & GPU_DIRECT_GATHER_RANK_MASK;
}

__global__ void gpu_direct_low_local_gather_kernel(
    Count* mainv, Count* blockv, int p
) {
    uint32_t dbid = blockIdx.z;
    bool target_main = p == 1;
    uint32_t nblocks = target_main ? D_GD_MAIN_NBLOCKS : D_GD_BLOCK_NBLOCKS;
    if (dbid >= nblocks) return;
    GpuDirectBlock dstb = target_main ? D_GD_MAIN_BLOCKS[dbid] : D_GD_BLOCK_BLOCKS[dbid];
    if (!dstb.valid || !dstb.rows || !dstb.cols) return;

    uint32_t pi = uint32_t(LOW_LUT_K - p);
    size_t oi = size_t(pi) * D_GDG_LOW_PITCH + dbid;
    uint32_t a = D_GDG_LOW_OFF[oi], b = D_GDG_LOW_OFF[oi + 1];
    uint32_t q0 = a + uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t qstep = uint32_t(gridDim.x) * blockDim.x;

    for (uint32_t q = q0; q < b; q += qstep) {
        GpuDirectGatherDst rec = D_GDG_LOW_DST[q];
        for (uint32_t hr = blockIdx.y; hr < dstb.rows; hr += gridDim.y) {
            Count* dp = (target_main ? mainv : blockv)
                + dstb.off + Code(hr) * dstb.cols + rec.dst_rank;
            Count sum = *dp;
            uint32_t eend = rec.edge_begin + rec.edge_count;
            for (uint32_t e = rec.edge_begin; e < eend; ++e) {
                uint32_t loc = D_GDG_LOW_SRC[e];
                GpuDirectBlock srcb = D_GD_MAIN_BLOCKS[gpu_direct_gather_src_block(loc)];
                uint32_t lr = gpu_direct_gather_src_rank(loc);
                sum = gpu_direct_add(sum,
                    mainv[srcb.off + Code(hr) * srcb.cols + lr]);
            }
            *dp = sum;
        }
    }
}

__global__ void gpu_direct_low_cross_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t sbid = blockIdx.z;
    if (sbid >= D_GD_MAIN_NBLOCKS) return;
    GpuDirectBlock srcb = D_GD_MAIN_BLOCKS[sbid];
    if (!srcb.valid || !srcb.rows || !srcb.cols) return;

    uint32_t pi = uint32_t(LOW_LUT_K - p);
    size_t oi = size_t(pi) * D_GDG_LOW_CROSS_PITCH + sbid;
    uint32_t a = D_GDG_LOW_CROSS_OFF[oi], b = D_GDG_LOW_CROSS_OFF[oi + 1];
    for (uint32_t q = a + blockIdx.y; q < b; q += gridDim.y) {
        GpuDirectLowCrossOp op = D_GDG_LOW_CROSS[q];
        uint32_t depth = lowdesc_depth(op.desc);
        if (!depth || depth > uint32_t(HIGH_LUT_K)) continue;
        uint32_t dbid = lowdesc_block(op.desc);
        uint32_t dlr = lowdesc_lr(op.desc);
        bool target_main = p == 1;
        GpuDirectBlock dstb = target_main ? D_GD_MAIN_BLOCKS[dbid] : D_GD_BLOCK_BLOCKS[dbid];

        for (uint32_t hr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
             hr < srcb.rows; hr += uint32_t(gridDim.x) * blockDim.x) {
            Count c = mainv[srcb.off + Code(hr) * srcb.cols + op.src_lr];
            if (!c) continue;
            uint32_t gi = D_GD_HIGH_ALL_OFF[srcb.he] + hr;
            uint32_t hr2 = D_GD_HIGH_CROSS_RANK[
                size_t(depth - 1) * D_GD_HIGH_CROSS_PITCH + gi];
            if (hr2 == GPU_DIRECT_INVALID_RANK) continue;
            Code j = dstb.off + Code(hr2) * dstb.cols + dlr;
            atomic_add_mod((target_main ? mainv : blockv) + j, c);
        }
    }
}

__global__ void gpu_direct_high_local_gather_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t dbid = blockIdx.z;
    if (dbid >= D_GD_BLOCK_NBLOCKS) return;
    GpuDirectBlock dstb = D_GD_BLOCK_BLOCKS[dbid];
    if (!dstb.valid || !dstb.rows || !dstb.cols) return;

    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    size_t oi = size_t(pi) * D_GDG_HIGH_PITCH + dbid;
    uint32_t a = D_GDG_HIGH_OFF[oi], b = D_GDG_HIGH_OFF[oi + 1];
    for (uint32_t q = a + blockIdx.y; q < b; q += gridDim.y) {
        GpuDirectGatherDst rec = D_GDG_HIGH_DST[q];
        for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
             lr < dstb.cols; lr += uint32_t(gridDim.x) * blockDim.x) {
            Count* dp = blockv + dstb.off + Code(rec.dst_rank) * dstb.cols + lr;
            Count sum = *dp;
            uint32_t eend = rec.edge_begin + rec.edge_count;
            for (uint32_t e = rec.edge_begin; e < eend; ++e) {
                uint32_t loc = D_GDG_HIGH_SRC[e];
                GpuDirectBlock srcb = D_GD_MAIN_BLOCKS[gpu_direct_gather_src_block(loc)];
                uint32_t hr = gpu_direct_gather_src_rank(loc);
                sum = gpu_direct_add(sum,
                    mainv[srcb.off + Code(hr) * srcb.cols + lr]);
            }
            *dp = sum;
        }
    }
}

__global__ void gpu_direct_high_cross_only_kernel(Count* mainv, Count* blockv, int p) {
    uint32_t sbid = blockIdx.z;
    if (sbid >= D_GD_MAIN_NBLOCKS) return;
    GpuDirectBlock srcb = D_GD_MAIN_BLOCKS[sbid];
    if (!srcb.valid || !srcb.rows || !srcb.cols) return;

    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    size_t oi = size_t(pi) * D_GD_HIGH_PITCH + sbid;
    uint32_t a = D_GD_HIGH_CROSS_CLOSURE_OFF[oi];
    uint32_t b = D_GD_HIGH_CROSS_CLOSURE_OFF[oi + 1];
    for (uint32_t q = a + blockIdx.y; q < b; q += gridDim.y) {
        CpuHighClosureOp op = D_GD_HIGH_CROSS_CLOSURE_OPS[q];
        uint32_t dbid = highdesc_block(op.desc);
        uint32_t dhr = highdesc_rank(op.desc);
        uint32_t depth = highdesc_depth(op.desc);
        if (!depth || depth > uint32_t(LOW_LUT_K)) continue;
        GpuDirectBlock dstb = D_GD_BLOCK_BLOCKS[dbid];

        for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
             lr < srcb.cols; lr += uint32_t(gridDim.x) * blockDim.x) {
            Count c = mainv[srcb.off + Code(op.src_hr) * srcb.cols + lr];
            if (!c) continue;
            uint32_t gi = D_GD_LOW_ALL_OFF[srcb.hs] + lr;
            uint32_t lr2 = D_GD_LOW_CROSS_RANK[
                size_t(depth - 1) * D_GD_LOW_CROSS_PITCH + gi];
            if (lr2 == GPU_DIRECT_INVALID_RANK) continue;
            Code j = dstb.off + Code(dhr) * dstb.cols + lr2;
            atomic_add_mod(blockv + j, c);
        }
    }
}

struct GpuDirectGatherDeviceTables {
    GpuDirectGatherDst* low_dst = nullptr;
    uint32_t* low_src = nullptr;
    uint32_t* low_off = nullptr;
    GpuDirectLowCrossOp* low_cross = nullptr;
    uint32_t* low_cross_off = nullptr;
    GpuDirectGatherDst* high_dst = nullptr;
    uint32_t* high_src = nullptr;
    uint32_t* high_off = nullptr;

    template<class T>
    static void copy_vec(T*& dst, const std::vector<T>& src, const char* what) {
        if (src.empty()) return;
        ck(cudaMalloc(&dst, src.size() * sizeof(T)), what);
        ck(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(const GpuDirectGatherHost& h) {
        copy_vec(low_dst, h.low_dst, "gpu direct gather low dst");
        copy_vec(low_src, h.low_src, "gpu direct gather low src");
        copy_vec(low_off, h.low_off, "gpu direct gather low off");
        copy_vec(low_cross, h.low_cross, "gpu direct gather low cross");
        copy_vec(low_cross_off, h.low_cross_off, "gpu direct gather low cross off");
        copy_vec(high_dst, h.high_dst, "gpu direct gather high dst");
        copy_vec(high_src, h.high_src, "gpu direct gather high src");
        copy_vec(high_off, h.high_off, "gpu direct gather high off");

        ck(cudaMemcpyToSymbol(D_GDG_LOW_DST, &low_dst, sizeof(low_dst)), "gpu direct gather low dst ptr");
        ck(cudaMemcpyToSymbol(D_GDG_LOW_SRC, &low_src, sizeof(low_src)), "gpu direct gather low src ptr");
        ck(cudaMemcpyToSymbol(D_GDG_LOW_OFF, &low_off, sizeof(low_off)), "gpu direct gather low off ptr");
        ck(cudaMemcpyToSymbol(D_GDG_LOW_PITCH, &h.low_pitch, sizeof(h.low_pitch)), "gpu direct gather low pitch");
        ck(cudaMemcpyToSymbol(D_GDG_LOW_CROSS, &low_cross, sizeof(low_cross)), "gpu direct gather low cross ptr");
        ck(cudaMemcpyToSymbol(D_GDG_LOW_CROSS_OFF, &low_cross_off, sizeof(low_cross_off)), "gpu direct gather low cross off ptr");
        ck(cudaMemcpyToSymbol(D_GDG_LOW_CROSS_PITCH, &h.low_cross_pitch, sizeof(h.low_cross_pitch)), "gpu direct gather low cross pitch");
        ck(cudaMemcpyToSymbol(D_GDG_HIGH_DST, &high_dst, sizeof(high_dst)), "gpu direct gather high dst ptr");
        ck(cudaMemcpyToSymbol(D_GDG_HIGH_SRC, &high_src, sizeof(high_src)), "gpu direct gather high src ptr");
        ck(cudaMemcpyToSymbol(D_GDG_HIGH_OFF, &high_off, sizeof(high_off)), "gpu direct gather high off ptr");
        ck(cudaMemcpyToSymbol(D_GDG_HIGH_PITCH, &h.high_pitch, sizeof(h.high_pitch)), "gpu direct gather high pitch");
    }

    void release() {
        if (low_dst) cudaFree(low_dst);
        if (low_src) cudaFree(low_src);
        if (low_off) cudaFree(low_off);
        if (low_cross) cudaFree(low_cross);
        if (low_cross_off) cudaFree(low_cross_off);
        if (high_dst) cudaFree(high_dst);
        if (high_src) cudaFree(high_src);
        if (high_off) cudaFree(high_off);
        low_dst = nullptr; low_src = nullptr; low_off = nullptr;
        low_cross = nullptr; low_cross_off = nullptr;
        high_dst = nullptr; high_src = nullptr; high_off = nullptr;
    }
};

// Release metadata made redundant by destination-gather execution. The old
// constant pointers become stale but are never dereferenced by gather runners.
static void gpu_direct_gather_drop_redundant(GpuDirectDeviceTables& base) {
    if (base.lowdesc.main_desc) cudaFree(base.lowdesc.main_desc);
    if (base.lowdesc.block_desc) cudaFree(base.lowdesc.block_desc);
    base.lowdesc.main_desc = base.lowdesc.block_desc = nullptr;
    if (base.high_block_closure) cudaFree(base.high_block_closure);
    if (base.high_block_closure_off) cudaFree(base.high_block_closure_off);
    base.high_block_closure = nullptr;
    base.high_block_closure_off = nullptr;
}

static void gpu_direct_run_low_gather(
    Count* mainv, Count* blockv, const StorageLayout& layout,
    int threads = 256, int grid_x = 16, int grid_y = 8
) {
    dim3 block(threads);
    for (int p = LOW_LUT_K; p >= 1; --p) {
        dim3 orbit_grid(grid_x, grid_y, unsigned(layout.main_blocks.size()));
        gpu_direct_low_orbit_kernel<<<orbit_grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct gather low orbit");

        unsigned target_blocks = p == 1
            ? unsigned(layout.main_blocks.size()) : unsigned(layout.block_blocks.size());
        dim3 local_grid(grid_x, grid_y, target_blocks);
        gpu_direct_low_local_gather_kernel<<<local_grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct gather low local closure");

        dim3 cross_grid(grid_x, grid_y, unsigned(layout.main_blocks.size()));
        gpu_direct_low_cross_kernel<<<cross_grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct gather low cross closure");
    }
    ck(cudaDeviceSynchronize(), "gpu direct gather low sync");
}

static void gpu_direct_run_high_gather(
    Count* mainv, Count* blockv, const StorageLayout& layout,
    int threads = 256, int grid_x = 16, int grid_y = 8
) {
    dim3 block(threads);
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        dim3 orbit_grid(grid_x, grid_y, unsigned(layout.main_blocks.size()));
        gpu_direct_high_orbit_kernel<<<orbit_grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct gather high orbit");

        dim3 local_grid(grid_x, grid_y, unsigned(layout.block_blocks.size()));
        gpu_direct_high_local_gather_kernel<<<local_grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct gather high local closure");

        dim3 cross_grid(grid_x, grid_y, unsigned(layout.main_blocks.size()));
        gpu_direct_high_cross_only_kernel<<<cross_grid, block>>>(mainv, blockv, p);
        ck(cudaGetLastError(), "gpu direct gather high cross closure");
    }
    ck(cudaDeviceSynchronize(), "gpu direct gather high sync");
}
