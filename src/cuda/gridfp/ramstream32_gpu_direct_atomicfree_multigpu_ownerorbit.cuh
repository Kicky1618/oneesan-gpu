#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_orbitstage.cuh"

#include <array>
#include <cstdint>
#include <vector>

// v0.6: destination-owned orbit execution.
//
// All orbit inputs are snapshotted before a phase begins.  Each GPU then runs
// only source blocks whose orbit has at least one output cell owned by that GPU,
// and stores only to its local authoritative shard.  Therefore runtime scalar
// P2P reads and scalar P2P writes are both zero; peer traffic is bulk staging.

struct GdowOrbitPhase {
    GdpoOrbitPhaseDeps deps;
    std::array<std::uint64_t,GDM_MAX_GPU> active_source_mask{};
};

struct GdowOrbitPlan {
    std::vector<GdowOrbitPhase> high;
    std::vector<GdowOrbitPhase> low;
    unsigned long long copy_bytes_per_row = 0;
    unsigned long long max_device_phase_bytes = 0;
    unsigned long long block_copies_per_row = 0;
};

static GdowOrbitPlan build_gdow_orbit_plan(
    const StorageLayout& layout,
    const GdmShardHost& shard,
    const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect,
    int ngpu
) {
    GdowOrbitPlan out;
    out.high.resize(HIGH_LUT_K);
    out.low.resize(LOW_LUT_K);

    auto finish = [&](GdowOrbitPhase& ph,
                      const std::array<std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS>,GDM_MAX_GPU>& nm,
                      const std::array<std::array<bool,GPU_DIRECT_MAX_BLOCK_BLOCKS>,GDM_MAX_GPU>& nb) {
        for (int d=0; d<ngpu; ++d) {
            for (std::uint32_t bid=0; bid<layout.main_blocks.size(); ++bid) if (nm[d][bid]) {
                ph.deps.main_source[d].push_back(std::uint8_t(bid));
                const auto& b=layout.main_blocks[bid];
                ph.deps.bytes[d]+=static_cast<unsigned long long>(b.rows)*b.cols*sizeof(Count);
                ++out.block_copies_per_row;
            }
            for (std::uint32_t bid=0; bid<layout.block_blocks.size(); ++bid) if (nb[d][bid]) {
                ph.deps.block_source[d].push_back(std::uint8_t(bid));
                const auto& b=layout.block_blocks[bid];
                ph.deps.bytes[d]+=static_cast<unsigned long long>(b.rows)*b.cols*sizeof(Count);
                ++out.block_copies_per_row;
            }
            out.copy_bytes_per_row += ph.deps.bytes[d];
            out.max_device_phase_bytes = std::max(out.max_device_phase_bytes,ph.deps.bytes[d]);
        }
    };

    auto activate = [&](GdowOrbitPhase& ph,int d,std::uint32_t bid) {
        if (d<0 || d>=ngpu || bid>=64) return;
        ph.active_source_mask[d] |= (std::uint64_t(1) << bid);
    };

    for (int p=TARGET_W-1; p>=LOW_LUT_K+1; --p) {
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
        auto& ph=out.high[pi];
        std::array<std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS>,GDM_MAX_GPU> nm{};
        std::array<std::array<bool,GPU_DIRECT_MAX_BLOCK_BLOCKS>,GDM_MAX_GPU> nb{};

        for (std::uint32_t bid=0; bid<layout.main_blocks.size(); ++bid) {
            const auto& sb=layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            int ox=shard.main_blocks[bid].owner;
            std::uint32_t dbid=std::uint32_t(sb.hs);
            int od=shard.block_blocks[dbid].owner;
            FBlock fb{}; fb.he=sb.he; fb.hs=sb.hs; fb.c=sb.c;

            auto [na,ne]=cpu_high_direct_range(highdirect.orbit_off.nn,highdirect.nblocks,pi,bid);
            if (na!=ne) {
                std::uint32_t jbid=cpu_high_orbit_partner_block(bid,fb,p,true);
                int oy=shard.main_blocks[jbid].owner;
                activate(ph,ox,bid); activate(ph,oy,bid); activate(ph,od,bid);
                // x' = x + old_d
                if (od!=ox) nb[ox][dbid]=true;
                // y' = old_y + x
                if (ox!=oy) nm[oy][bid]=true;
                // d' = 0, no input required by od.
            }

            auto [ra,re]=cpu_high_direct_range(highdirect.orbit_off.nrnl,highdirect.nblocks,pi,bid);
            if (ra!=re) {
                std::uint32_t jbid=cpu_high_orbit_partner_block(bid,fb,p,false);
                int oy=shard.main_blocks[jbid].owner;
                activate(ph,ox,bid); activate(ph,od,bid);
                // x' = x + old_y + old_d
                if (oy!=ox) nm[ox][jbid]=true;
                if (od!=ox) nb[ox][dbid]=true;
                // d' = x
                if (ox!=od) nm[od][bid]=true;
            }
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
            int ox=shard.main_blocks[bid].owner;
            for (std::uint32_t lr=0; lr<sb.cols; ++lr) {
                std::uint64_t ow=loworbit.rec[
                    std::size_t(pi)*loworbit.main_total+loworbit.main_base[bid]+lr];
                std::uint32_t kind=cpu_orbit_kind(ow);
                if (kind<CPU_ORBIT_NN || kind>CPU_ORBIT_NL) continue;
                std::uint32_t jbid=cpu_orbit_jblock(ow),dbid=cpu_orbit_dblock(ow);
                int oy=shard.main_blocks[jbid].owner;
                int od=shard.block_blocks[dbid].owner;

                if (kind==CPU_ORBIT_NN) {
                    activate(ph,ox,bid); activate(ph,oy,bid); activate(ph,od,bid);
                    if (od!=ox) nb[ox][dbid]=true; // x' needs old_d
                    if (ox!=oy) nm[oy][bid]=true; // y' needs x
                } else {
                    activate(ph,ox,bid);
                    // x' always needs old_y and old_d.
                    if (oy!=ox) nm[ox][jbid]=true;
                    if (od!=ox) nb[ox][dbid]=true;
                    if (p==1) {
                        activate(ph,oy,bid); // y' = x + old_y
                        activate(ph,od,bid); // d' = 0
                        if (ox!=oy) nm[oy][bid]=true;
                    } else {
                        activate(ph,od,bid); // d' = x
                        if (ox!=od) nm[od][bid]=true;
                    }
                }
            }
        }
        finish(ph,nm,nb);
    }
    return out;
}

__global__ void gdow_low_orbit_kernel(int p,std::uint64_t active_mask) {
    std::uint32_t bid=blockIdx.z;
    if (bid>=D_GD_MAIN_NBLOCKS || bid>=64 || ((active_mask>>bid)&1u)==0u) return;
    GdmBlock x=D_GDM_MAIN_BLOCKS[bid];
    if (!x.valid || !x.rows || !x.cols) return;
    std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);
    std::uint32_t lr0=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
    std::uint32_t lr_step=std::uint32_t(gridDim.x)*blockDim.x;
    for (std::uint32_t lr=lr0;lr<x.cols;lr+=lr_step) {
        std::uint64_t ow=D_GD_LOW_ORBIT[
            std::size_t(pi)*D_GD_LOW_ORBIT_MAIN_TOTAL+D_GD_LOW_ORBIT_MAIN_BASE[bid]+lr];
        std::uint32_t kind=gpu_direct_low_orbit_kind(ow);
        if (kind<CPU_ORBIT_NN || kind>CPU_ORBIT_NL) continue;
        std::uint32_t jbid=gpu_direct_low_orbit_jblock(ow),dbid=gpu_direct_low_orbit_dblock(ow);
        GdmBlock y=D_GDM_MAIN_BLOCKS[jbid],d=D_GDM_BLOCK_BLOCKS[dbid];
        std::uint32_t jlr=gpu_direct_low_orbit_jlr(ow),dlr=gpu_direct_low_orbit_dlr(ow);
        bool wx=x.owner==D_GDM_DEVICE, wy=y.owner==D_GDM_DEVICE, wd=d.owner==D_GDM_DEVICE;
        if (!wx && !wy && !wd) continue;
        for (std::uint32_t hr=blockIdx.y;hr<x.rows;hr+=gridDim.y) {
            Code xi=Code(hr)*x.cols+lr,ji=Code(hr)*y.cols+jlr,di=Code(hr)*d.cols+dlr;
            Count c=gdpo_main_read(bid,xi);
            Count old_j=gdpo_main_read(jbid,ji);
            Count old_d=gdpo_block_read(dbid,di);
            if (kind==CPU_ORBIT_NN) {
                if (wx) *gdm_main_ptr(x,xi)=gpu_direct_add(c,old_d);
                if (wy) *gdm_main_ptr(y,ji)=gpu_direct_add(old_j,c);
                if (wd) *gdm_block_ptr(d,di)=0;
            } else if (p==1) {
                Count all=gpu_direct_add(gpu_direct_add(c,old_j),old_d);
                if (wx) *gdm_main_ptr(x,xi)=all;
                if (wy) *gdm_main_ptr(y,ji)=gpu_direct_add(c,old_j);
                if (wd) *gdm_block_ptr(d,di)=0;
            } else {
                if (wx) *gdm_main_ptr(x,xi)=gpu_direct_add(gpu_direct_add(c,old_j),old_d);
                if (wd) *gdm_block_ptr(d,di)=c;
            }
        }
    }
}

__global__ void gdow_high_orbit_kernel(int p,std::uint64_t active_mask) {
    std::uint32_t bid=blockIdx.z;
    if (bid>=D_GD_MAIN_NBLOCKS || bid>=64 || ((active_mask>>bid)&1u)==0u) return;
    GdmBlock x=D_GDM_MAIN_BLOCKS[bid];
    if (!x.valid || !x.rows || !x.cols) return;
    std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
    std::size_t oi=std::size_t(pi)*D_GD_HIGH_PITCH+bid;
    std::uint32_t na=D_GD_HIGH_NN_OFF[oi],ne=D_GD_HIGH_NN_OFF[oi+1];
    std::uint32_t ra=D_GD_HIGH_NRNL_OFF[oi],re=D_GD_HIGH_NRNL_OFF[oi+1];
    std::uint32_t nn_count=ne-na,nr_count=re-ra,total=nn_count+nr_count;
    if (!total) return;
    std::uint32_t lr0=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
    std::uint32_t lr_step=std::uint32_t(gridDim.x)*blockDim.x;
    std::uint32_t dbid=std::uint32_t(x.hs);
    GdmBlock d=D_GDM_BLOCK_BLOCKS[dbid];
    for (std::uint32_t k=blockIdx.y;k<total;k+=gridDim.y) {
        bool nn=k<nn_count;
        CpuHighOrbitOp op=nn?D_GD_HIGH_NN_OPS[na+k]:D_GD_HIGH_NRNL_OPS[ra+(k-nn_count)];
        GpuDirectBlock xb{0,x.rows,x.cols,x.he,x.hs,x.c,x.valid};
        std::uint32_t jbid=gpu_direct_high_partner_block(bid,xb,p,nn);
        GdmBlock y=D_GDM_MAIN_BLOCKS[jbid];
        bool wx=x.owner==D_GDM_DEVICE, wy=y.owner==D_GDM_DEVICE, wd=d.owner==D_GDM_DEVICE;
        if (nn) {
            if (!wx && !wy && !wd) continue;
        } else if (!wx && !wd) continue;
        std::uint32_t shr=gpu_direct_high_src(op),jhr=gpu_direct_high_partner(op),dhr=gpu_direct_high_drop(op);
        for (std::uint32_t lr=lr0;lr<x.cols;lr+=lr_step) {
            Code xi=Code(shr)*x.cols+lr,ji=Code(jhr)*y.cols+lr,di=Code(dhr)*d.cols+lr;
            Count c=gdpo_main_read(bid,xi);
            Count old_j=gdpo_main_read(jbid,ji);
            Count old_d=gdpo_block_read(dbid,di);
            if (nn) {
                if (wx) *gdm_main_ptr(x,xi)=gpu_direct_add(c,old_d);
                if (wy) *gdm_main_ptr(y,ji)=gpu_direct_add(old_j,c);
                if (wd) *gdm_block_ptr(d,di)=0;
            } else {
                if (wx) *gdm_main_ptr(x,xi)=gpu_direct_add(gpu_direct_add(c,old_j),old_d);
                if (wd) *gdm_block_ptr(d,di)=c;
            }
        }
    }
}

static void gdow_enqueue_row(
    GdpoPipeline& pipe,const GdowOrbitPlan& orbit_plan,const GdmsStagePlan& gather_plan,
    const StorageLayout& layout,const GdmShardHost& shard,
    Count* const* main_ptr,Count* const* block_ptr,
    int threads,int grid_x,int grid_y,int& slot
) {
    dim3 block(threads); int phase=0; int ready_slot=-1;
    for (int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase) {
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
        const auto& oph=orbit_plan.high[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdow high orbit device");
            dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));
            gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);
            ck(cudaGetLastError(),"gdow high orbit");
        }
        int orbit_done=slot; pipe.fence(slot++);
        gdmp_stage_sources(pipe,orbit_done,gather_plan.high[pi].source,layout,shard,main_ptr);
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdow high gather device");
            dim3 g(grid_x,grid_y,unsigned(layout.block_blocks.size()));
            gdms_high_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            gdms_high_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(),"gdow high gather");
        }
        ready_slot=slot; pipe.fence(slot++);
    }
    for (int p=LOW_LUT_K;p>=1;--p,++phase) {
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);
        const auto& oph=orbit_plan.low[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdow low orbit device");
            dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));
            gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);
            ck(cudaGetLastError(),"gdow low orbit");
        }
        int orbit_done=slot; pipe.fence(slot++);
        gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdow low local device");
            dim3 g(grid_x,grid_y,nt);
            gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(),"gdow low local");
        }
        if (p==1) {
            int local_done=slot; pipe.fence(slot++);
            gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr);
        }
        for (int d=0;d<pipe.ngpu;++d) {
            ck(cudaSetDevice(d),"gdow low cross device");
            dim3 g(grid_x,grid_y,nt);
            gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);
            ck(cudaGetLastError(),"gdow low cross");
        }
        ready_slot=slot; pipe.fence(slot++);
    }
}
