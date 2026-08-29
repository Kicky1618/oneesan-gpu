#pragma once

// Corrected destination-owned orbit kernels for v0.6.  The staging plan is
// role-minimal, so loads must be role-minimal too: a GPU that only owns d'=0
// must not touch unstaged x/y inputs.

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
            if (kind==CPU_ORBIT_NN) {
                Count c=0,old_j=0,old_d=0;
                if (wx||wy) c=gdpo_main_read(bid,xi);
                if (wy) old_j=gdpo_main_read(jbid,ji);
                if (wx) old_d=gdpo_block_read(dbid,di);
                if (wx) *gdm_main_ptr(x,xi)=gpu_direct_add(c,old_d);
                if (wy) *gdm_main_ptr(y,ji)=gpu_direct_add(old_j,c);
                if (wd) *gdm_block_ptr(d,di)=0;
            } else if (p==1) {
                Count c=0,old_j=0,old_d=0;
                if (wx||wy) c=gdpo_main_read(bid,xi);
                if (wx||wy) old_j=gdpo_main_read(jbid,ji);
                if (wx) old_d=gdpo_block_read(dbid,di);
                if (wx) *gdm_main_ptr(x,xi)=gpu_direct_add(gpu_direct_add(c,old_j),old_d);
                if (wy) *gdm_main_ptr(y,ji)=gpu_direct_add(c,old_j);
                if (wd) *gdm_block_ptr(d,di)=0;
            } else {
                Count c=0,old_j=0,old_d=0;
                if (wx||wd) c=gdpo_main_read(bid,xi);
                if (wx) {
                    old_j=gdpo_main_read(jbid,ji);
                    old_d=gdpo_block_read(dbid,di);
                    *gdm_main_ptr(x,xi)=gpu_direct_add(gpu_direct_add(c,old_j),old_d);
                }
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
        if (nn) { if (!wx && !wy && !wd) continue; }
        else if (!wx && !wd) continue;
        std::uint32_t shr=gpu_direct_high_src(op),jhr=gpu_direct_high_partner(op),dhr=gpu_direct_high_drop(op);
        for (std::uint32_t lr=lr0;lr<x.cols;lr+=lr_step) {
            Code xi=Code(shr)*x.cols+lr,ji=Code(jhr)*y.cols+lr,di=Code(dhr)*d.cols+lr;
            if (nn) {
                Count c=0,old_j=0,old_d=0;
                if (wx||wy) c=gdpo_main_read(bid,xi);
                if (wy) old_j=gdpo_main_read(jbid,ji);
                if (wx) old_d=gdpo_block_read(dbid,di);
                if (wx) *gdm_main_ptr(x,xi)=gpu_direct_add(c,old_d);
                if (wy) *gdm_main_ptr(y,ji)=gpu_direct_add(old_j,c);
                if (wd) *gdm_block_ptr(d,di)=0;
            } else {
                Count c=0,old_j=0,old_d=0;
                if (wx||wd) c=gdpo_main_read(bid,xi);
                if (wx) {
                    old_j=gdpo_main_read(jbid,ji);
                    old_d=gdpo_block_read(dbid,di);
                    *gdm_main_ptr(x,xi)=gpu_direct_add(gpu_direct_add(c,old_j),old_d);
                }
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
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p); const auto& oph=orbit_plan.high[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdow safe high orbit device"); dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size())); gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]); ck(cudaGetLastError(),"gdow safe high orbit"); }
        int orbit_done=slot; pipe.fence(slot++);
        gdmp_stage_sources(pipe,orbit_done,gather_plan.high[pi].source,layout,shard,main_ptr);
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdow safe high gather device"); dim3 g(grid_x,grid_y,unsigned(layout.block_blocks.size())); gdms_high_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); gdms_high_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdow safe high gather"); }
        ready_slot=slot; pipe.fence(slot++);
    }
    for (int p=LOW_LUT_K;p>=1;--p,++phase) {
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p); const auto& oph=orbit_plan.low[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdow safe low orbit device"); dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size())); gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]); ck(cudaGetLastError(),"gdow safe low orbit"); }
        int orbit_done=slot; pipe.fence(slot++);
        gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdow safe low local device"); dim3 g(grid_x,grid_y,nt); gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdow safe low local"); }
        if (p==1) { int local_done=slot; pipe.fence(slot++); gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr); }
        for (int d=0;d<pipe.ngpu;++d) { ck(cudaSetDevice(d),"gdow safe low cross device"); dim3 g(grid_x,grid_y,nt); gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p); ck(cudaGetLastError(),"gdow safe low cross"); }
        ready_slot=slot; pipe.fence(slot++);
    }
}
