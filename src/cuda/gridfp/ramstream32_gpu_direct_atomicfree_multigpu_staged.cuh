#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Bulk-P2P staging layer for the block-sharded atomic-free backend.
// Remote MAIN source blocks are copied into a fixed local mirror address space
// before destination-owned gather kernels consume them. The mirror reserves
// layout.main_size Counts per GPU, but each phase transfers only source blocks
// referenced by that GPU's destination blocks.

__constant__ Count* D_GDMS_STAGE_MAIN;
__constant__ Code D_GDMS_MAIN_GLOBAL_OFF[GPU_DIRECT_MAX_MAIN_BLOCKS];

struct GdmsPhaseDeps {
    std::array<std::vector<std::uint8_t>, GDM_MAX_GPU> source;
    std::array<std::vector<std::uint8_t>, GDM_MAX_GPU> cross_refresh;
    std::array<unsigned long long, GDM_MAX_GPU> bytes{};
    std::array<unsigned long long, GDM_MAX_GPU> refresh_bytes{};
};

struct GdmsStagePlan {
    std::vector<GdmsPhaseDeps> high;
    std::vector<GdmsPhaseDeps> low;
    unsigned long long copy_bytes_per_row = 0;
    unsigned long long max_device_phase_bytes = 0;
};

static inline std::uint32_t gdms_src_block(std::uint32_t packed) {
    return packed >> GPU_DIRECT_GATHER_SRC_BLOCK_SHIFT;
}
static inline std::uint32_t gdms_cross_block(std::uint32_t packed) {
    return (packed >> GPU_DIRECT_CROSS_OP_BLOCK_SHIFT) & GPU_DIRECT_CROSS_OP_BLOCK_MASK;
}

static GdmsStagePlan build_gdms_stage_plan(
    const StorageLayout& layout,
    const GdmShardHost& shard,
    const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,
    int ngpu
) {
    GdmsStagePlan out;
    out.high.resize(HIGH_LUT_K);
    out.low.resize(LOW_LUT_K);

    auto block_bytes = [&](std::uint32_t sbid) -> unsigned long long {
        const auto& b = layout.main_blocks[sbid];
        return static_cast<unsigned long long>(b.rows) * b.cols * sizeof(Count);
    };
    auto finish = [&](GdmsPhaseDeps& ph,
                      const std::array<std::array<bool, GPU_DIRECT_MAX_MAIN_BLOCKS>, GDM_MAX_GPU>& need,
                      const std::array<std::array<bool, GPU_DIRECT_MAX_MAIN_BLOCKS>, GDM_MAX_GPU>& refresh) {
        for (int d=0; d<ngpu; ++d) {
            for (std::uint32_t sbid=0; sbid<layout.main_blocks.size(); ++sbid) {
                if (need[d][sbid]) {
                    ph.source[d].push_back(std::uint8_t(sbid));
                    ph.bytes[d] += block_bytes(sbid);
                }
                if (refresh[d][sbid]) {
                    ph.cross_refresh[d].push_back(std::uint8_t(sbid));
                    ph.refresh_bytes[d] += block_bytes(sbid);
                }
            }
            out.copy_bytes_per_row += ph.bytes[d] + ph.refresh_bytes[d];
            out.max_device_phase_bytes = std::max(
                out.max_device_phase_bytes, ph.bytes[d] + ph.refresh_bytes[d]);
        }
    };

    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        std::uint32_t pi = std::uint32_t((TARGET_W - 1) - p);
        auto& ph = out.high[pi];
        std::array<std::array<bool, GPU_DIRECT_MAX_MAIN_BLOCKS>, GDM_MAX_GPU> need{};
        std::array<std::array<bool, GPU_DIRECT_MAX_MAIN_BLOCKS>, GDM_MAX_GPU> refresh{};
        for (std::uint32_t dbid=0; dbid<layout.block_blocks.size(); ++dbid) {
            int d = shard.block_blocks[dbid].owner;
            std::size_t oi = std::size_t(pi) * ordinary.high_pitch + dbid;
            std::uint32_t a = ordinary.high_off[oi], b = ordinary.high_off[oi+1];
            for (std::uint32_t q=a; q<b; ++q) {
                const auto& rec = ordinary.high_dst[q];
                for (std::uint32_t e=rec.edge_begin; e<rec.edge_begin+rec.edge_count; ++e) {
                    std::uint32_t sbid = gdms_src_block(ordinary.high_src[e]);
                    if (shard.main_blocks[sbid].owner != d) need[d][sbid] = true;
                }
            }
            oi = std::size_t(pi) * cross.high_pitch + dbid;
            a = cross.high_off[oi]; b = cross.high_off[oi+1];
            for (std::uint32_t q=a; q<b; ++q) {
                const auto& rec = cross.high_dst[q];
                for (std::uint32_t e=rec.edge_begin; e<rec.edge_begin+rec.edge_count; ++e) {
                    std::uint32_t sbid = gdms_cross_block(cross.high_op[e]);
                    if (shard.main_blocks[sbid].owner != d) need[d][sbid] = true;
                }
            }
        }
        finish(ph, need, refresh);
    }

    for (int p = LOW_LUT_K; p >= 1; --p) {
        std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
        auto& ph = out.low[pi];
        std::array<std::array<bool, GPU_DIRECT_MAX_MAIN_BLOCKS>, GDM_MAX_GPU> need{};
        std::array<std::array<bool, GPU_DIRECT_MAX_MAIN_BLOCKS>, GDM_MAX_GPU> refresh{};
        bool target_main = p == 1;
        std::uint32_t nblocks = target_main
            ? std::uint32_t(layout.main_blocks.size())
            : std::uint32_t(layout.block_blocks.size());
        for (std::uint32_t dbid=0; dbid<nblocks; ++dbid) {
            int d = target_main ? shard.main_blocks[dbid].owner : shard.block_blocks[dbid].owner;
            std::size_t oi = std::size_t(pi) * ordinary.low_pitch + dbid;
            std::uint32_t a = ordinary.low_off[oi], b = ordinary.low_off[oi+1];
            for (std::uint32_t q=a; q<b; ++q) {
                const auto& rec = ordinary.low_dst[q];
                for (std::uint32_t e=rec.edge_begin; e<rec.edge_begin+rec.edge_count; ++e) {
                    std::uint32_t sbid = gdms_src_block(ordinary.low_src[e]);
                    if (shard.main_blocks[sbid].owner != d) need[d][sbid] = true;
                }
            }
            oi = std::size_t(pi) * cross.low_pitch + dbid;
            a = cross.low_off[oi]; b = cross.low_off[oi+1];
            for (std::uint32_t q=a; q<b; ++q) {
                const auto& rec = cross.low_dst[q];
                for (std::uint32_t e=rec.edge_begin; e<rec.edge_begin+rec.edge_count; ++e) {
                    std::uint32_t sbid = gdms_cross_block(cross.low_op[e]);
                    if (shard.main_blocks[sbid].owner != d) {
                        need[d][sbid] = true;
                        if (p == 1) refresh[d][sbid] = true;
                    }
                }
            }
        }
        finish(ph, need, refresh);
    }
    return out;
}

struct GdmsStagePipeline {
    int ngpu = 0;
    int slots = 0;
    Code stage_elems = 0;
    std::array<cudaStream_t, GDM_MAX_GPU> stream{};
    std::array<Count*, GDM_MAX_GPU> stage{};
    std::vector<std::array<cudaEvent_t, GDM_MAX_GPU>> event;

    void init(int n, const StorageLayout& layout) {
        ngpu = n;
        slots = 2 * (LOW_LUT_K + HIGH_LUT_K) + 1;
        stage_elems = layout.main_size;
        event.resize(std::size_t(slots));
        std::array<Code, GPU_DIRECT_MAX_MAIN_BLOCKS> global_off{};
        for (std::size_t i=0; i<layout.main_blocks.size(); ++i)
            global_off[i] = layout.main_blocks[i].off;
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdms set device");
            ck(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking), "gdms stream");
            if (stage_elems)
                ck(cudaMalloc(&stage[d], std::size_t(stage_elems) * sizeof(Count)), "gdms stage");
            ck(cudaMemcpyToSymbol(D_GDMS_STAGE_MAIN, &stage[d], sizeof(stage[d])), "gdms stage ptr");
            ck(cudaMemcpyToSymbol(D_GDMS_MAIN_GLOBAL_OFF, global_off.data(),
                                  sizeof(Code) * global_off.size()), "gdms global off");
        }
        for (int s=0; s<slots; ++s) {
            for (int d=0; d<ngpu; ++d) {
                ck(cudaSetDevice(d), "gdms event device");
                ck(cudaEventCreateWithFlags(&event[std::size_t(s)][d], cudaEventDisableTiming),
                   "gdms event create");
            }
        }
    }

    void fence(int slot) {
        if (slot < 0 || slot >= slots) std::exit(180);
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdms record device");
            ck(cudaEventRecord(event[std::size_t(slot)][d], stream[d]), "gdms record");
        }
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdms wait device");
            for (int src=0; src<ngpu; ++src) if (src != d)
                ck(cudaStreamWaitEvent(stream[d], event[std::size_t(slot)][src], 0), "gdms wait");
        }
    }

    void sync_row() {
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdms sync device");
            ck(cudaStreamSynchronize(stream[d]), "gdms row sync");
        }
    }

    void destroy() {
        for (int d=0; d<ngpu; ++d) {
            ck(cudaSetDevice(d), "gdms destroy device");
            for (int s=0; s<slots; ++s)
                if (event[std::size_t(s)][d]) cudaEventDestroy(event[std::size_t(s)][d]);
            if (stream[d]) cudaStreamDestroy(stream[d]);
            if (stage[d]) cudaFree(stage[d]);
            stream[d] = nullptr; stage[d] = nullptr;
        }
        event.clear();
    }
};

static void gdms_stage_sources(
    GdmsStagePipeline& pipe,
    const std::array<std::vector<std::uint8_t>, GDM_MAX_GPU>& sources,
    const StorageLayout& layout,
    const GdmShardHost& shard,
    Count* const* main_ptr
) {
    for (int d=0; d<pipe.ngpu; ++d) {
        ck(cudaSetDevice(d), "gdms copy device");
        for (std::uint8_t sbid8 : sources[d]) {
            std::uint32_t sbid = sbid8;
            const auto& logical = layout.main_blocks[sbid];
            const auto& physical = shard.main_blocks[sbid];
            int srcdev = physical.owner;
            if (srcdev == d) continue;
            std::size_t bytes = std::size_t(logical.rows) * logical.cols * sizeof(Count);
            if (!bytes) continue;
            Count* dst = pipe.stage[d] + logical.off;
            const Count* src = main_ptr[srcdev] + physical.off;
            ck(cudaMemcpyPeerAsync(dst, d, src, srcdev, bytes, pipe.stream[d]), "gdms bulk p2p");
        }
    }
}

__device__ __forceinline__ Count gdms_main_load(std::uint32_t sbid, Code i) {
    GdmBlock b = D_GDM_MAIN_BLOCKS[sbid];
    if (b.owner == D_GDM_DEVICE) return *gdm_main_ptr(b, i);
    return D_GDMS_STAGE_MAIN[D_GDMS_MAIN_GLOBAL_OFF[sbid] + i];
}

__global__ void gdms_low_local_gather_kernel(int p) {
    std::uint32_t dbid=blockIdx.z; bool target_main=p==1;
    std::uint32_t nblocks=target_main?D_GD_MAIN_NBLOCKS:D_GD_BLOCK_NBLOCKS;
    if (dbid>=nblocks) return;
    GdmBlock dstb=target_main?D_GDM_MAIN_BLOCKS[dbid]:D_GDM_BLOCK_BLOCKS[dbid];
    if (!dstb.valid || dstb.owner!=D_GDM_DEVICE || !dstb.rows || !dstb.cols) return;
    std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);
    std::size_t oi=std::size_t(pi)*D_GDG_LOW_PITCH+dbid;
    std::uint32_t a=D_GDG_LOW_OFF[oi],b=D_GDG_LOW_OFF[oi+1];
    std::uint32_t q0=a+std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
    std::uint32_t qstep=std::uint32_t(gridDim.x)*blockDim.x;
    for (std::uint32_t q=q0;q<b;q+=qstep) {
        GpuDirectGatherDst rec=D_GDG_LOW_DST[q];
        for (std::uint32_t hr=blockIdx.y;hr<dstb.rows;hr+=gridDim.y) {
            Count* dp=target_main?gdm_main_ptr(dstb,Code(hr)*dstb.cols+rec.dst_rank)
                                 :gdm_block_ptr(dstb,Code(hr)*dstb.cols+rec.dst_rank);
            Count sum=*dp;
            for (std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                std::uint32_t loc=D_GDG_LOW_SRC[e], sbid=gpu_direct_gather_src_block(loc);
                GdmBlock srcb=D_GDM_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdms_main_load(
                    sbid,Code(hr)*srcb.cols+gpu_direct_gather_src_rank(loc)));
            }
            *dp=sum;
        }
    }
}

__global__ void gdms_high_local_gather_kernel(int p) {
    std::uint32_t dbid=blockIdx.z; if (dbid>=D_GD_BLOCK_NBLOCKS) return;
    GdmBlock dstb=D_GDM_BLOCK_BLOCKS[dbid];
    if (!dstb.valid || dstb.owner!=D_GDM_DEVICE || !dstb.rows || !dstb.cols) return;
    std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
    std::size_t oi=std::size_t(pi)*D_GDG_HIGH_PITCH+dbid;
    std::uint32_t a=D_GDG_HIGH_OFF[oi],b=D_GDG_HIGH_OFF[oi+1];
    for (std::uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y) {
        GpuDirectGatherDst rec=D_GDG_HIGH_DST[q];
        for (std::uint32_t lr=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
             lr<dstb.cols;lr+=std::uint32_t(gridDim.x)*blockDim.x) {
            Count* dp=gdm_block_ptr(dstb,Code(rec.dst_rank)*dstb.cols+lr);
            Count sum=*dp;
            for (std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                std::uint32_t loc=D_GDG_HIGH_SRC[e], sbid=gpu_direct_gather_src_block(loc);
                GdmBlock srcb=D_GDM_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdms_main_load(
                    sbid,Code(gpu_direct_gather_src_rank(loc))*srcb.cols+lr));
            }
            *dp=sum;
        }
    }
}

__device__ __forceinline__ Count gdms_sum_high_preimages(
    std::uint32_t dest_code,std::uint32_t depth,std::uint32_t source_he,
    std::uint32_t source_bid,std::uint32_t source_lr
) {
    Count sum=0; int s=int(depth);
#pragma unroll
    for (int pos=0;pos<HIGH_LUT_K;++pos) {
        std::uint32_t v=(dest_code>>(2*pos))&3u;
        if (v==std::uint32_t(::L)) { if (s==1) break; --s; }
        else if (v==std::uint32_t(R)) {
            if (s==1) {
                std::uint32_t z=3u<<(2*pos);
                std::uint32_t src_code=(dest_code&~z)|(std::uint32_t(::L)<<(2*pos));
                std::uint32_t gi=D_GDX_HIGH_DIRECT[gdx_ternary_key<HIGH_LUT_K>(src_code)];
                std::uint32_t a=D_GD_HIGH_ALL_OFF[source_he],b=D_GD_HIGH_ALL_OFF[source_he+1];
                if (gi>=a && gi<b) {
                    GdmBlock sb=D_GDM_MAIN_BLOCKS[source_bid];
                    sum=gpu_direct_add(sum,gdms_main_load(
                        source_bid,Code(gi-a)*sb.cols+source_lr));
                }
            }
            ++s;
        }
    }
    return sum;
}

__device__ __forceinline__ Count gdms_sum_low_preimages(
    std::uint32_t dest_code,std::uint32_t depth,std::uint32_t source_hs,
    std::uint32_t source_bid,std::uint32_t source_hr
) {
    Count sum=0; int s=int(depth);
#pragma unroll
    for (int pos=LOW_LUT_K-1;pos>=0;--pos) {
        std::uint32_t v=(dest_code>>(2*pos))&3u;
        if (v==std::uint32_t(R)) { if (s==1) break; --s; }
        else if (v==std::uint32_t(::L)) {
            if (s==1) {
                std::uint32_t z=3u<<(2*pos);
                std::uint32_t src_code=(dest_code&~z)|(std::uint32_t(R)<<(2*pos));
                std::uint32_t gi=D_GDX_LOW_DIRECT[gdx_ternary_key<LOW_LUT_K>(src_code)];
                std::uint32_t a=D_GD_LOW_ALL_OFF[source_hs],b=D_GD_LOW_ALL_OFF[source_hs+1];
                if (gi>=a && gi<b) {
                    GdmBlock sb=D_GDM_MAIN_BLOCKS[source_bid];
                    sum=gpu_direct_add(sum,gdms_main_load(
                        source_bid,Code(source_hr)*sb.cols+(gi-a)));
                }
            }
            ++s;
        }
    }
    return sum;
}

__global__ void gdms_low_cross_gather_kernel(int p) {
    std::uint32_t dbid=blockIdx.z; bool target_main=p==1;
    std::uint32_t nblocks=target_main?D_GD_MAIN_NBLOCKS:D_GD_BLOCK_NBLOCKS;
    if (dbid>=nblocks) return;
    GdmBlock db=target_main?D_GDM_MAIN_BLOCKS[dbid]:D_GDM_BLOCK_BLOCKS[dbid];
    if (!db.valid || db.owner!=D_GDM_DEVICE || !db.rows || !db.cols) return;
    std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);
    std::size_t oi=std::size_t(pi)*D_GDX_LOW_PITCH+dbid;
    std::uint32_t a=D_GDX_LOW_OFF[oi],b=D_GDX_LOW_OFF[oi+1];
    for (std::uint32_t q=a+std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
         q<b;q+=std::uint32_t(gridDim.x)*blockDim.x) {
        GpuDirectGatherDst rec=D_GDX_LOW_DST[q];
        for (std::uint32_t dhr=blockIdx.y;dhr<db.rows;dhr+=gridDim.y) {
            std::uint32_t dest_code=D_GDX_HIGH_CODES[D_GD_HIGH_ALL_OFF[db.he]+dhr];
            Count* dp=target_main?gdm_main_ptr(db,Code(dhr)*db.cols+rec.dst_rank)
                                 :gdm_block_ptr(db,Code(dhr)*db.cols+rec.dst_rank);
            Count sum=*dp;
            for (std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                std::uint32_t op=D_GDX_LOW_OP[e];
                std::uint32_t sbid=gdx_op_block(op),slr=gdx_op_rank(op),depth=gdx_op_depth(op);
                GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdms_sum_high_preimages(
                    dest_code,depth,sb.he,sbid,slr));
            }
            *dp=sum;
        }
    }
}

__global__ void gdms_high_cross_gather_kernel(int p) {
    std::uint32_t dbid=blockIdx.z; if (dbid>=D_GD_BLOCK_NBLOCKS) return;
    GdmBlock db=D_GDM_BLOCK_BLOCKS[dbid];
    if (!db.valid || db.owner!=D_GDM_DEVICE || !db.rows || !db.cols) return;
    std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
    std::size_t oi=std::size_t(pi)*D_GDX_HIGH_PITCH+dbid;
    std::uint32_t a=D_GDX_HIGH_OFF[oi],b=D_GDX_HIGH_OFF[oi+1];
    for (std::uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y) {
        GpuDirectGatherDst rec=D_GDX_HIGH_DST[q];
        for (std::uint32_t dlr=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
             dlr<db.cols;dlr+=std::uint32_t(gridDim.x)*blockDim.x) {
            std::uint32_t dest_code=D_GDX_LOW_CODES[D_GD_LOW_ALL_OFF[db.hs]+dlr];
            Count* dp=gdm_block_ptr(db,Code(rec.dst_rank)*db.cols+dlr);
            Count sum=*dp;
            for (std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                std::uint32_t op=D_GDX_HIGH_OP[e];
                std::uint32_t sbid=gdx_op_block(op),shr=gdx_op_rank(op),depth=gdx_op_depth(op);
                GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdms_sum_low_preimages(
                    dest_code,depth,sb.hs,sbid,shr));
            }
            *dp=sum;
        }
    }
}

static void gdms_enqueue_high(
    GdmsStagePipeline& pipe, const GdmsStagePlan& plan,
    const StorageLayout& layout, const GdmShardHost& shard,
    Count* const* main_ptr, int threads, int grid_x, int grid_y, int& slot
) {
    dim3 block(threads);
    for (int p=TARGET_W-1;p>=LOW_LUT_K+1;--p) {
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdms high orbit device");
            dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));
            gdm_high_orbit_kernel<<<g,block,0,pipe.stream[d]>>>(p);
            ck(cudaGetLastError(),"gdms high orbit");
        }
        pipe.fence(slot++);
        gdms_stage_sources(pipe,plan.high[pi].source,layout,shard,main_ptr);
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdms high gather device");
            dim3 g(grid_x,grid_y,unsigned(layout.block_blocks.size()));
            gdms_high_local_gather_kernel<<<g,block,0,pipe.stream[d]>>>(p);
            gdms_high_cross_gather_kernel<<<g,block,0,pipe.stream[d]>>>(p);
            ck(cudaGetLastError(),"gdms high gather");
        }
        pipe.fence(slot++);
    }
}

static void gdms_enqueue_low(
    GdmsStagePipeline& pipe, const GdmsStagePlan& plan,
    const StorageLayout& layout, const GdmShardHost& shard,
    Count* const* main_ptr, int threads, int grid_x, int grid_y, int& slot
) {
    dim3 block(threads);
    for (int p=LOW_LUT_K;p>=1;--p) {
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdms low orbit device");
            dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));
            gdm_low_orbit_kernel<<<g,block,0,pipe.stream[d]>>>(p);
            ck(cudaGetLastError(),"gdms low orbit");
        }
        pipe.fence(slot++);
        gdms_stage_sources(pipe,plan.low[pi].source,layout,shard,main_ptr);
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdms low local device");
            dim3 g(grid_x,grid_y,nt);
            gdms_low_local_gather_kernel<<<g,block,0,pipe.stream[d]>>>(p);
            ck(cudaGetLastError(),"gdms low local");
        }
        if (p==1) {
            pipe.fence(slot++);
            gdms_stage_sources(pipe,plan.low[pi].cross_refresh,layout,shard,main_ptr);
        }
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdms low cross device");
            dim3 g(grid_x,grid_y,nt);
            gdms_low_cross_gather_kernel<<<g,block,0,pipe.stream[d]>>>(p);
            ck(cudaGetLastError(),"gdms low cross");
        }
        pipe.fence(slot++);
    }
}
