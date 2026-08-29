#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_variable_wavefront.cuh"

// Avoid re-copying the variable dbid table every row.  The main program installs
// it once and calls refresh_meta only after an accepted online replan.
static void gdvw_enqueue_row_ready(
    GdvwPipeline&pipe,const GdowOrbitPlan&orbit_plan,const GdmsStagePlan&gather_plan,
    const GdvwPlan&wave_plan,const StorageLayout&layout,const GdmShardHost&shard,
    Count*const*main_ptr,Count*const*block_ptr,int threads,int grid_x,int grid_y,
    int&slot,bool measure
){
    dim3 block(threads);int phase=0,ready_slot=-1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase){
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);const auto&oph=orbit_plan.high[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw ready high orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdvw ready high orbit");}
        int orbit_done=slot;pipe.fence(slot++);
        gdvw_gather_high(pipe,wave_plan.high[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);
        ready_slot=slot;pipe.fence(slot++);
    }
    for(int p=LOW_LUT_K;p>=1;--p,++phase){
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);const auto&oph=orbit_plan.low[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw ready low orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdvw ready low orbit");}
        int orbit_done=slot;pipe.fence(slot++);
        if(p>1){
            gdvw_gather_low(pipe,wave_plan.low[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);
        }else{
            gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);
            unsigned nt=unsigned(layout.main_blocks.size());
            for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw ready p1 local device");dim3 g(grid_x,grid_y,nt);gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdvw ready p1 local");}
            int local_done=slot;pipe.fence(slot++);
            gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr);
            for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw ready p1 cross device");dim3 g(grid_x,grid_y,nt);gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdvw ready p1 cross");}
        }
        ready_slot=slot;pipe.fence(slot++);
    }
}

static double gdvw_replan_gain_pct(const GdvwReplan&r){
    return r.baseline_ms>0.0?100.0*(r.baseline_ms-r.candidate_ms)/r.baseline_ms:0.0;
}
static unsigned gdvw_wave_count_changes(const GdvwPlan&a,const GdvwPlan&b){
    unsigned n=0;for(std::size_t i=0;i<a.high.size()&&i<b.high.size();++i)n+=a.high[i].waves!=b.high[i].waves;
    for(int p=LOW_LUT_K;p>=2;--p){std::size_t i=std::size_t(LOW_LUT_K-p);if(i<a.low.size()&&i<b.low.size())n+=a.low[i].waves!=b.low[i].waves;}return n;
}
