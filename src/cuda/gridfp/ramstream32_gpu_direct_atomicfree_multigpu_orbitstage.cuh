#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

// v0.5: stage remote orbit READ operands in bulk. Orbit writes remain direct
// P2P stores for now. A global device-side snapshot barrier prevents an orbit
// kernel from modifying authoritative peer state before every GPU has finished
// copying the old partner/drop values it needs for that phase.

__constant__ Count* D_GDPO_STAGE_BLOCK;
__constant__ Code D_GDPO_BLOCK_GLOBAL_OFF[GPU_DIRECT_MAX_BLOCK_BLOCKS];

struct GdpoOrbitPhaseDeps {
    std::array<std::vector<std::uint8_t>,GDM_MAX_GPU> main_source;
    std::array<std::vector<std::uint8_t>,GDM_MAX_GPU> block_source;
    std::array<unsigned long long,GDM_MAX_GPU> bytes{};
};

struct GdpoOrbitPlan {
    std::vector<GdpoOrbitPhaseDeps> high;
    std::vector<GdpoOrbitPhaseDeps> low;
    unsigned long long copy_bytes_per_row = 0;
    unsigned long long max_device_phase_bytes = 0;
    unsigned long long block_copies_per_row = 0;
};

static GdpoOrbitPlan build_gdpo_orbit_plan(
    const StorageLayout& layout,
    const GdmShardHost& shard,
    const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect,
    int ngpu
) {
    GdpoOrbitPlan out;
    out.high.resize(HIGH_LUT_K);
    out.low.resize(LOW_LUT_K);

    auto finish = [&](GdpoOrbitPhaseDeps& ph,
                      const std::array<std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS>,GDM_MAX_GPU>& nm,
                      const std::array<std::array<bool,GPU_DIRECT_MAX_BLOCK_BLOCKS>,GDM_MAX_GPU>& nb) {
        for (int d=0; d<ngpu; ++d) {
            for (std::uint32_t bid=0; bid<layout.main_blocks.size(); ++bid) if (nm[d][bid]) {
                ph.main_source[d].push_back(std::uint8_t(bid));
                const auto& b=layout.main_blocks[bid];
                ph.bytes[d]+=static_cast<unsigned long long>(b.rows)*b.cols*sizeof(Count);
                ++out.block_copies_per_row;
            }
            for (std::uint32_t bid=0; bid<layout.block_blocks.size(); ++bid) if (nb[d][bid]) {
                ph.block_source[d].push_back(std::uint8_t(bid));
                const auto& b=layout.block_blocks[bid];
                ph.bytes[d]+=static_cast<unsigned long long>(b.rows)*b.cols*sizeof(Count);
                ++out.block_copies_per_row;
            }
            out.copy_bytes_per_row += ph.bytes[d];
            out.max_device_phase_bytes = std::max(out.max_device_phase_bytes,ph.bytes[d]);
        }
    };

    for (int p=TARGET_W-1; p>=LOW_LUT_K+1; --p) {
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
        auto& ph=out.high[pi];
        std::array<std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS>,GDM_MAX_GPU> nm{};
        std::array<std::array<bool,GPU_DIRECT_MAX_BLOCK_BLOCKS>,GDM_MAX_GPU> nb{};
        for (std::uint32_t bid=0; bid<layout.main_blocks.size(); ++bid) {
            const auto& sb=layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            int d=shard.main_blocks[bid].owner;
            auto [na,ne]=cpu_high_direct_range(highdirect.orbit_off.nn,highdirect.nblocks,pi,bid);
            auto [ra,re]=cpu_high_direct_range(highdirect.orbit_off.nrnl,highdirect.nblocks,pi,bid);
            if (na==ne && ra==re) continue;
            FBlock fb{}; fb.he=sb.he; fb.hs=sb.hs; fb.c=sb.c;
            if (na!=ne) {
                std::uint32_t jbid=cpu_high_orbit_partner_block(bid,fb,p,true);
                if (shard.main_blocks[jbid].owner!=d) nm[d][jbid]=true;
            }
            if (ra!=re) {
                std::uint32_t jbid=cpu_high_orbit_partner_block(bid,fb,p,false);
                if (shard.main_blocks[jbid].owner!=d) nm[d][jbid]=true;
            }
            std::uint32_t dbid=cpu_high_orbit_drop_block(fb);
            if (shard.block_blocks[dbid].owner!=d) nb[d][dbid]=true;
        }
        finish(ph,nm,nb);
    }

    for (int p=LOW_LUT_K; p>=1; --p) {
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);
        auto& ph=out.low[pi];
        std::array<std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS>,GDM_MAX_GPU> nm{};
        std::array<std::array<bool,GPU_DIRECT_MAX_BLOCK_BLOCKS>,GDM_MAX_GPU> nb{};
        for (std::uint32_t bid=0; bid<layout.main_blocks.size(); ++bid) {
            const auto& sb=layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            int d=shard.main_blocks[bid].owner;
            for (std::uint32_t lr=0; lr<sb.cols; ++lr) {
                std::uint64_t ow=loworbit.rec[
                    std::size_t(pi)*loworbit.main_total+loworbit.main_base[bid]+lr];
                std::uint32_t k=cpu_orbit_kind(ow);
                if (k<CPU_ORBIT_NN || k>CPU_ORBIT_NL) continue;
                std::uint32_t jbid=cpu_orbit_jblock(ow);
                std::uint32_t dbid=cpu_orbit_dblock(ow);
                if (shard.main_blocks[jbid].owner!=d) nm[d][jbid]=true;
                if (shard.block_blocks[dbid].owner!=d) nb[d][dbid]=true;
            }
        }
        finish(ph,nm,nb);
    }
    return out;
}

struct GdpoPipeline : GdmpStagePipeline {
    int orbit_phases = 0;
    Code block_stage_elems = 0;
    std::array<Count*,GDM_MAX_GPU> stage_block{};
    std::vector<std::array<cudaEvent_t,GDM_MAX_GPU>> orbit_ready;
    std::vector<std::array<std::array<cudaEvent_t,GDM_MAX_GPU>,GDM_MAX_GPU>> orbit_copy_done;

    void init(int n,const StorageLayout& layout) {
        GdmpStagePipeline::init(n,layout);
        orbit_phases=LOW_LUT_K+HIGH_LUT_K;
        block_stage_elems=layout.block_size;
        orbit_ready.resize(std::size_t(orbit_phases));
        orbit_copy_done.resize(std::size_t(orbit_phases));
        std::array<Code,GPU_DIRECT_MAX_BLOCK_BLOCKS> global_off{};
        for (std::size_t i=0;i<layout.block_blocks.size();++i)
            global_off[i]=layout.block_blocks[i].off;
        for (int d=0;d<ngpu;++d) {
            ck(cudaSetDevice(d),"gdpo block stage device");
            if (block_stage_elems)
                ck(cudaMalloc(&stage_block[d],std::size_t(block_stage_elems)*sizeof(Count)),"gdpo block stage");
            ck(cudaMemcpyToSymbol(D_GDPO_STAGE_BLOCK,&stage_block[d],sizeof(stage_block[d])),"gdpo block stage ptr");
            ck(cudaMemcpyToSymbol(D_GDPO_BLOCK_GLOBAL_OFF,global_off.data(),sizeof(Code)*global_off.size()),"gdpo block global off");
        }
        for (int k=0;k<orbit_phases;++k) for (int d=0;d<ngpu;++d) {
            ck(cudaSetDevice(d),"gdpo orbit event device");
            ck(cudaEventCreateWithFlags(&orbit_ready[std::size_t(k)][d],cudaEventDisableTiming),"gdpo orbit ready");
            for (int s=0;s<ngpu;++s) if (s!=d)
                ck(cudaEventCreateWithFlags(&orbit_copy_done[std::size_t(k)][d][s],cudaEventDisableTiming),"gdpo orbit copy done");
        }
    }

    void destroy() {
        for (int d=0;d<ngpu;++d) {
            ck(cudaSetDevice(d),"gdpo destroy device");
            for (int k=0;k<orbit_phases;++k) {
                if (orbit_ready[std::size_t(k)][d]) cudaEventDestroy(orbit_ready[std::size_t(k)][d]);
                for (int s=0;s<ngpu;++s) if (s!=d && orbit_copy_done[std::size_t(k)][d][s])
                    cudaEventDestroy(orbit_copy_done[std::size_t(k)][d][s]);
            }
            if (stage_block[d]) cudaFree(stage_block[d]);
            stage_block[d]=nullptr;
        }
        orbit_ready.clear(); orbit_copy_done.clear();
        GdmpStagePipeline::destroy();
    }
};

static void gdpo_stage_orbit(
    GdpoPipeline& pipe,int phase,int ready_slot,
    const GdpoOrbitPhaseDeps& deps,
    const StorageLayout& layout,const GdmShardHost& shard,
    Count* const* main_ptr,Count* const* block_ptr
) {
    if (phase<0 || phase>=pipe.orbit_phases) std::exit(184);
    for (int d=0;d<pipe.ngpu;++d) {
        std::array<bool,GDM_MAX_GPU> used{};
        ck(cudaSetDevice(d),"gdpo copy device");
        auto arm=[&](int s) {
            if (used[s]) return;
            cudaStream_t cs=pipe.copy[d][s];
            if (ready_slot>=0) {
                ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][s],0),"gdpo wait source");
                ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][d],0),"gdpo wait destination");
            }
            used[s]=true;
        };
        for (std::uint8_t bid8:deps.main_source[d]) {
            std::uint32_t bid=bid8; const auto& logical=layout.main_blocks[bid]; const auto& physical=shard.main_blocks[bid];
            int s=physical.owner; if (s==d) continue; arm(s);
            std::size_t bytes=std::size_t(logical.rows)*logical.cols*sizeof(Count); if (!bytes) continue;
            ck(cudaMemcpyPeerAsync(pipe.stage[d]+logical.off,d,main_ptr[s]+physical.off,s,bytes,pipe.copy[d][s]),"gdpo main stage");
        }
        for (std::uint8_t bid8:deps.block_source[d]) {
            std::uint32_t bid=bid8; const auto& logical=layout.block_blocks[bid]; const auto& physical=shard.block_blocks[bid];
            int s=physical.owner; if (s==d) continue; arm(s);
            std::size_t bytes=std::size_t(logical.rows)*logical.cols*sizeof(Count); if (!bytes) continue;
            ck(cudaMemcpyPeerAsync(pipe.stage_block[d]+logical.off,d,block_ptr[s]+physical.off,s,bytes,pipe.copy[d][s]),"gdpo block stage");
        }
        for (int s=0;s<pipe.ngpu;++s) if (s!=d && used[s]) {
            cudaEvent_t done=pipe.orbit_copy_done[std::size_t(phase)][d][s];
            ck(cudaEventRecord(done,pipe.copy[d][s]),"gdpo orbit copy record");
            ck(cudaStreamWaitEvent(pipe.compute[d],done,0),"gdpo compute wait orbit copy");
        }
    }
    // Every GPU must finish its complete snapshot before any GPU starts the
    // in-place orbit, because orbit writes may target peer authoritative state.
    for (int d=0;d<pipe.ngpu;++d) {
        ck(cudaSetDevice(d),"gdpo snapshot record device");
        ck(cudaEventRecord(pipe.orbit_ready[std::size_t(phase)][d],pipe.compute[d]),"gdpo snapshot record");
    }
    for (int d=0;d<pipe.ngpu;++d) {
        ck(cudaSetDevice(d),"gdpo snapshot wait device");
        for (int s=0;s<pipe.ngpu;++s) if (s!=d)
            ck(cudaStreamWaitEvent(pipe.compute[d],pipe.orbit_ready[std::size_t(phase)][s],0),"gdpo snapshot wait");
    }
}

__device__ __forceinline__ Count gdpo_main_read(std::uint32_t bid,Code i) {
    GdmBlock b=D_GDM_MAIN_BLOCKS[bid];
    if (b.owner==D_GDM_DEVICE) return *gdm_main_ptr(b,i);
    return D_GDMS_STAGE_MAIN[D_GDMS_MAIN_GLOBAL_OFF[bid]+i];
}

__device__ __forceinline__ Count gdpo_block_read(std::uint32_t bid,Code i) {
    GdmBlock b=D_GDM_BLOCK_BLOCKS[bid];
    if (b.owner==D_GDM_DEVICE) return *gdm_block_ptr(b,i);
    return D_GDPO_STAGE_BLOCK[D_GDPO_BLOCK_GLOBAL_OFF[bid]+i];
}

__global__ void gdpo_low_orbit_kernel(int p) {
    std::uint32_t bid=blockIdx.z; if (bid>=D_GD_MAIN_NBLOCKS) return;
    GdmBlock x=D_GDM_MAIN_BLOCKS[bid]; if (!x.valid || x.owner!=D_GDM_DEVICE || !x.rows || !x.cols) return;
    std::uint32_t pi=std::uint32_t(LOW_LUT_K-p),lr0=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
    std::uint32_t lr_step=std::uint32_t(gridDim.x)*blockDim.x;
    for (std::uint32_t lr=lr0;lr<x.cols;lr+=lr_step) {
        std::uint64_t ow=D_GD_LOW_ORBIT[std::size_t(pi)*D_GD_LOW_ORBIT_MAIN_TOTAL+D_GD_LOW_ORBIT_MAIN_BASE[bid]+lr];
        std::uint32_t kind=gpu_direct_low_orbit_kind(ow); if (kind<CPU_ORBIT_NN || kind>CPU_ORBIT_NL) continue;
        std::uint32_t jbid=gpu_direct_low_orbit_jblock(ow),dbid=gpu_direct_low_orbit_dblock(ow);
        GdmBlock y=D_GDM_MAIN_BLOCKS[jbid],d=D_GDM_BLOCK_BLOCKS[dbid];
        std::uint32_t jlr=gpu_direct_low_orbit_jlr(ow),dlr=gpu_direct_low_orbit_dlr(ow);
        for (std::uint32_t hr=blockIdx.y;hr<x.rows;hr+=gridDim.y) {
            Code xi=Code(hr)*x.cols+lr,ji=Code(hr)*y.cols+jlr,di=Code(hr)*d.cols+dlr;
            Count* ip=gdm_main_ptr(x,xi); Count* jp=gdm_main_ptr(y,ji); Count* dp=gdm_block_ptr(d,di);
            Count c=*ip,old_j=gdpo_main_read(jbid,ji),old_d=gdpo_block_read(dbid,di);
            if (kind==CPU_ORBIT_NN) { *jp=gpu_direct_add(old_j,c); *ip=gpu_direct_add(c,old_d); *dp=0; }
            else { Count all=gpu_direct_add(gpu_direct_add(c,old_j),old_d);
                if (p==1) { *ip=all; *jp=gpu_direct_add(c,old_j); *dp=0; }
                else { *ip=all; *dp=c; } }
        }
    }
}

__global__ void gdpo_high_orbit_kernel(int p) {
    std::uint32_t bid=blockIdx.z; if (bid>=D_GD_MAIN_NBLOCKS) return;
    GdmBlock x=D_GDM_MAIN_BLOCKS[bid]; if (!x.valid || x.owner!=D_GDM_DEVICE || !x.rows || !x.cols) return;
    std::uint32_t pi=std::uint32_t((TARGET_W-1)-p); std::size_t oi=std::size_t(pi)*D_GD_HIGH_PITCH+bid;
    std::uint32_t na=D_GD_HIGH_NN_OFF[oi],ne=D_GD_HIGH_NN_OFF[oi+1],ra=D_GD_HIGH_NRNL_OFF[oi],re=D_GD_HIGH_NRNL_OFF[oi+1];
    std::uint32_t nn_count=ne-na,nr_count=re-ra,total=nn_count+nr_count; if (!total) return;
    std::uint32_t lr0=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x,lr_step=std::uint32_t(gridDim.x)*blockDim.x;
    GdmBlock d=D_GDM_BLOCK_BLOCKS[std::uint32_t(x.hs)];
    for (std::uint32_t k=blockIdx.y;k<total;k+=gridDim.y) {
        bool nn=k<nn_count; CpuHighOrbitOp op=nn?D_GD_HIGH_NN_OPS[na+k]:D_GD_HIGH_NRNL_OPS[ra+(k-nn_count)];
        GpuDirectBlock xb{0,x.rows,x.cols,x.he,x.hs,x.c,x.valid};
        std::uint32_t jbid=gpu_direct_high_partner_block(bid,xb,p,nn); GdmBlock y=D_GDM_MAIN_BLOCKS[jbid];
        std::uint32_t dbid=std::uint32_t(x.hs),shr=gpu_direct_high_src(op),jhr=gpu_direct_high_partner(op),dhr=gpu_direct_high_drop(op);
        for (std::uint32_t lr=lr0;lr<x.cols;lr+=lr_step) {
            Code xi=Code(shr)*x.cols+lr,ji=Code(jhr)*y.cols+lr,di=Code(dhr)*d.cols+lr;
            Count* ip=gdm_main_ptr(x,xi); Count* jp=gdm_main_ptr(y,ji); Count* dp=gdm_block_ptr(d,di);
            Count c=*ip,old_j=gdpo_main_read(jbid,ji),old_d=gdpo_block_read(dbid,di);
            if (nn) { *jp=gpu_direct_add(old_j,c); *ip=gpu_direct_add(c,old_d); *dp=0; }
            else { *ip=gpu_direct_add(gpu_direct_add(c,old_j),old_d); *dp=c; }
        }
    }
}

static void gdpo_enqueue_row(
    GdpoPipeline& pipe,const GdpoOrbitPlan& orbit_plan,const GdmsStagePlan& gather_plan,
    const StorageLayout& layout,const GdmShardHost& shard,
    Count* const* main_ptr,Count* const* block_ptr,int threads,int grid_x,int grid_y,int& slot
) {
    dim3 block(threads); int phase=0; int ready_slot=-1;
    for (int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase) {
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
        gdpo_stage_orbit(pipe,phase,ready_slot,orbit_plan.high[pi],layout,shard,main_ptr,block_ptr);
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdpo high orbit device"); dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size())); gdpo_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdpo high orbit"); }
        int orbit_done=slot; pipe.fence(slot++);
        gdmp_stage_sources(pipe,orbit_done,gather_plan.high[pi].source,layout,shard,main_ptr);
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdpo high gather device"); dim3 g(grid_x,grid_y,unsigned(layout.block_blocks.size())); gdms_high_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); gdms_high_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdpo high gather"); }
        ready_slot=slot; pipe.fence(slot++);
    }
    for (int p=LOW_LUT_K;p>=1;--p,++phase) {
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);
        gdpo_stage_orbit(pipe,phase,ready_slot,orbit_plan.low[pi],layout,shard,main_ptr,block_ptr);
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdpo low orbit device"); dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size())); gdpo_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdpo low orbit"); }
        int orbit_done=slot; pipe.fence(slot++);
        gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdpo low local device"); dim3 g(grid_x,grid_y,nt); gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdpo low local"); }
        if (p==1) { int local_done=slot; pipe.fence(slot++); gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr); }
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdpo low cross device"); dim3 g(grid_x,grid_y,nt); gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdpo low cross"); }
        ready_slot=slot; pipe.fence(slot++);
    }
}
