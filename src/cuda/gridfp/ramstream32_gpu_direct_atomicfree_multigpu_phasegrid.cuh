#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_variable_wavefront_safe.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

// v1.4: phase-local gather grid selection. Wave count/placement still comes
// from v1.3. Gather grid_x/grid_y are selected independently per phase from
// {0.5x, 1x, 2x} neighborhoods around the user-provided base grid. The model
// uses phase-local measured work/ms, destination geometry, SM underfill, and the
// existing peer-copy pipeline model. Orbit and LOW p==1 retain the base grid.

struct GdpgShape { int x=1,y=1; };
static bool operator==(const GdpgShape&a,const GdpgShape&b){return a.x==b.x&&a.y==b.y;}
static bool operator!=(const GdpgShape&a,const GdpgShape&b){return !(a==b);}

struct GdpgGridPlan { std::vector<GdpgShape> high; std::vector<GdpgShape> low; };
struct GdpgPhaseCalibration {
    std::array<std::array<double,GDM_MAX_GPU>,GDVW_PHASES> work_per_ms{};
    std::array<bool,GDVW_PHASES> valid{};
};
struct GdpgJointReplan {
    GdvwPlan wave; GdpgGridPlan grid;
    double baseline_ms=0.0,candidate_ms=0.0;
    unsigned wave_count_changes=0,grid_shape_changes=0;
};

static GdpgGridPlan gdpg_initial_grid(int gx,int gy){GdpgGridPlan p;p.high.assign(HIGH_LUT_K,{gx,gy});p.low.assign(LOW_LUT_K,{gx,gy});return p;}
static std::vector<int> gdpg_axis_candidates(int base,int current,int cap){std::vector<int>v;auto add=[&](int x){x=std::max(1,std::min(cap,x));if(std::find(v.begin(),v.end(),x)==v.end())v.push_back(x);};add(base/2);add(base);add(base*2);add(current);std::sort(v.begin(),v.end());return v;}
static std::vector<GdpgShape> gdpg_shape_candidates(int base_x,int base_y,GdpgShape current){auto xs=gdpg_axis_candidates(base_x,current.x,128),ys=gdpg_axis_candidates(base_y,current.y,64);std::vector<GdpgShape>out;for(int x:xs)for(int y:ys)out.push_back({x,y});return out;}

static double gdpg_geometry_factor(
    bool high,std::uint32_t pi,int d,const GdvwPhase&ph,GdpgShape s,int threads,int sms,
    const StorageLayout&layout,const GdmShardHost&shard,
    const GpuDirectGatherHost&ordinary,const GpuDirectCrossGatherHost&cross
){
    double launched=0.0,useful=0.0;
    for(std::uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){
        const auto&b=layout.block_blocks[dbid];if(!b.valid||!b.rows||!b.cols||shard.block_blocks[dbid].owner!=d)continue;
        if(high){
            std::size_t oi=std::size_t(pi)*ordinary.high_pitch+dbid;std::uint32_t nl=ordinary.high_off[oi+1]-ordinary.high_off[oi];
            oi=std::size_t(pi)*cross.high_pitch+dbid;std::uint32_t nc=cross.high_off[oi+1]-cross.high_off[oi];
            auto add=[&](std::uint32_t nr){if(!nr)return;double xn=std::max(1.0,std::ceil(double(b.cols)/double(threads)));double yn=double(nr);launched+=double(s.x)*s.y;useful+=std::min(double(s.x),xn)*std::min(double(s.y),yn);};add(nl);add(nc);
        }else{
            std::size_t oi=std::size_t(pi)*ordinary.low_pitch+dbid;std::uint32_t nl=ordinary.low_off[oi+1]-ordinary.low_off[oi];
            oi=std::size_t(pi)*cross.low_pitch+dbid;std::uint32_t nc=cross.low_off[oi+1]-cross.low_off[oi];
            auto add=[&](std::uint32_t nr){if(!nr)return;double xn=std::max(1.0,std::ceil(double(nr)/double(threads)));double yn=double(b.rows);launched+=double(s.x)*s.y;useful+=std::min(double(s.x),xn)*std::min(double(s.y),yn);};add(nl);add(nc);
        }
    }
    double waste=useful>0.0?std::min(8.0,launched/useful):1.0;
    double under_sum=0.0,weight_sum=0.0,target=double(std::max(1,2*sms));
    for(int w=0;w<ph.waves;++w){unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;double ct=double(s.x)*s.y*nz;double pen=std::max(1.0,target/std::max(1.0,ct));double wt=std::max(1.0,double(ph.wave[w].work[d]));under_sum+=pen*wt;weight_sum+=wt;}
    double under=weight_sum>0.0?std::min(8.0,under_sum/weight_sum):1.0;
    return 0.65*waste+0.35*under;
}

static double gdpg_device_phase_score(
    bool high,std::uint32_t pi,int phase,int d,const GdvwPhase&candidate,const GdvwPhase&reference,
    GdpgShape shape,GdpgShape ref_shape,const GdpgPhaseCalibration&phase_model,
    const GdorOnlineModel&global_model,int threads,int sms,const StorageLayout&layout,
    const GdmShardHost&shard,const GpuDirectGatherHost&ordinary,const GpuDirectCrossGatherHost&cross
){
    double measured=phase_model.work_per_ms[phase][d];if(measured<=0.0)measured=global_model.compute.work_per_ms[d];if(measured<=0.0)return 1e300;
    double rf=gdpg_geometry_factor(high,pi,d,reference,ref_shape,threads,sms,layout,shard,ordinary,cross);
    double cf=gdpg_geometry_factor(high,pi,d,candidate,shape,threads,sms,layout,shard,ordinary,cross);
    double rate=measured*rf/std::max(1e-9,cf);
    auto item=high?gdaw_high_items(pi,d,layout,shard,ordinary,cross):gdaw_low_items(pi,d,layout,shard,ordinary,cross);
    auto bins=gdvw_bins_from_phase(candidate,d,item);
    return gdvw_score_bins(bins,layout,shard,global_model.topology,d,rate,gdvw_wave_group_ms());
}
static double gdpg_phase_score(
    bool high,std::uint32_t pi,int phase,const GdvwPhase&candidate,const GdvwPhase&reference,
    GdpgShape shape,GdpgShape ref_shape,const GdpgPhaseCalibration&phase_model,
    const GdorOnlineModel&global_model,int threads,int sms,const StorageLayout&layout,
    const GdmShardHost&shard,const GpuDirectGatherHost&ordinary,const GpuDirectCrossGatherHost&cross,int ngpu
){double z=0.0;for(int d=0;d<ngpu;++d)z=std::max(z,gdpg_device_phase_score(high,pi,phase,d,candidate,reference,shape,ref_shape,phase_model,global_model,threads,sms,layout,shard,ordinary,cross));return z;}

static GdpgPhaseCalibration gdpg_calibrate_phase(GdvwPipeline&pipe,const GdvwPlan&p,int ngpu){
    GdpgPhaseCalibration out;auto one=[&](int phase,const GdvwPhase&ph){for(int d=0;d<ngpu;++d){double work=0.0,ms=0.0;for(int w=0;w<ph.waves;++w){if(ph.wave[w].dbids[d].empty()||!ph.wave[w].work[d])continue;ck(cudaSetDevice(d),"gdpg phase elapsed device");float z=0.0f;ck(cudaEventElapsedTime(&z,pipe.compute_begin[phase][w][d],pipe.compute_end[phase][w][d]),"gdpg phase elapsed");if(z>0.0f){work+=double(ph.wave[w].work[d]);ms+=z;}}if(ms>0.0)out.work_per_ms[phase][d]=work/ms;}for(int d=0;d<ngpu;++d)if(out.work_per_ms[phase][d]>0.0){out.valid[phase]=true;break;}};
    for(int pi=0;pi<HIGH_LUT_K;++pi)one(pi,p.high[pi]);for(int q=LOW_LUT_K;q>=2;--q){int pi=LOW_LUT_K-q;one(HIGH_LUT_K+pi,p.low[pi]);}return out;
}
static void gdpg_update_phase_model(GdpgPhaseCalibration&model,const GdpgPhaseCalibration&sample,int ngpu,unsigned update_index){double a=update_index?0.35:0.65;for(int ph=0;ph<GDVW_PHASES;++ph){bool any=false;for(int d=0;d<ngpu;++d)if(sample.work_per_ms[ph][d]>0.0){double&x=model.work_per_ms[ph][d];x=x>0.0?(1.0-a)*x+a*sample.work_per_ms[ph][d]:sample.work_per_ms[ph][d];any=true;}model.valid[ph]=model.valid[ph]||any;}}

static GdpgJointReplan gdpg_joint_replan(
    const GdvwPlan&cur_wave,const GdpgGridPlan&cur_grid,const GdorOnlineModel&global_model,
    const GdpgPhaseCalibration&phase_model,int base_x,int base_y,int threads,int sms,
    const StorageLayout&layout,const GdmShardHost&shard,const GpuDirectGatherHost&ordinary,
    const GpuDirectCrossGatherHost&cross,const GdmsStagePlan&exact,int ngpu
){
    GdpgJointReplan out;out.wave=cur_wave;out.grid=cur_grid;if(!global_model.compute.valid||!global_model.topology.custom)return out;
    GdvwReplan wr=gdvw_replan(cur_wave,global_model,layout,shard,ordinary,cross,exact,ngpu);
    auto tune=[&](bool high,std::uint32_t pi,int phase,const GdvwPhase&old,const GdvwPhase&wave_cand,GdpgShape old_shape){
        double base=gdpg_phase_score(high,pi,phase,old,old,old_shape,old_shape,phase_model,global_model,threads,sms,layout,shard,ordinary,cross,ngpu);
        GdpgShape best_shape=old_shape;double best=base;
        for(auto s:gdpg_shape_candidates(base_x,base_y,old_shape)){double z=gdpg_phase_score(high,pi,phase,wave_cand,old,s,old_shape,phase_model,global_model,threads,sms,layout,shard,ordinary,cross,ngpu);if(z+1e-6<best){best=z;best_shape=s;}}
        out.baseline_ms+=base;double gain=base>0.0?100.0*(base-best)/base:0.0;
        if(gain>=GDOR_MIN_REPLAN_GAIN_PCT){out.candidate_ms+=best;if(wave_cand.waves!=old.waves)++out.wave_count_changes;if(best_shape!=old_shape)++out.grid_shape_changes;return std::pair<GdvwPhase,GdpgShape>{wave_cand,best_shape};}
        out.candidate_ms+=base;return std::pair<GdvwPhase,GdpgShape>{old,old_shape};
    };
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);auto r=tune(true,pi,int(pi),cur_wave.high[pi],wr.plan.high[pi],cur_grid.high[pi]);out.wave.high[pi]=std::move(r.first);out.grid.high[pi]=r.second;}
    for(int p=LOW_LUT_K;p>=2;--p){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);int ph=HIGH_LUT_K+int(pi);auto r=tune(false,pi,ph,cur_wave.low[pi],wr.plan.low[pi],cur_grid.low[pi]);out.wave.low[pi]=std::move(r.first);out.grid.low[pi]=r.second;}
    gdvw_recount(out.wave,exact,ngpu);return out;
}
static double gdpg_gain_pct(const GdpgJointReplan&r){return r.baseline_ms>0.0?100.0*(r.baseline_ms-r.candidate_ms)/r.baseline_ms:0.0;}

static void gdpg_enqueue_row_ready(
    GdvwPipeline&pipe,const GdowOrbitPlan&orbit_plan,const GdmsStagePlan&gather_plan,
    const GdvwPlan&wave_plan,const GdpgGridPlan&grid_plan,const StorageLayout&layout,
    const GdmShardHost&shard,Count*const*main_ptr,Count*const*block_ptr,int threads,
    int base_grid_x,int base_grid_y,int&slot,bool measure
){
    dim3 block(threads);int phase=0,ready_slot=-1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);const auto&oph=orbit_plan.high[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdpg high orbit device");dim3 g(base_grid_x,base_grid_y,unsigned(layout.main_blocks.size()));gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdpg high orbit");}int orbit_done=slot;pipe.fence(slot++);auto s=grid_plan.high[pi];gdvw_gather_high(pipe,wave_plan.high[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,s.x,s.y,measure);ready_slot=slot;pipe.fence(slot++);}
    for(int p=LOW_LUT_K;p>=1;--p,++phase){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);const auto&oph=orbit_plan.low[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdpg low orbit device");dim3 g(base_grid_x,base_grid_y,unsigned(layout.main_blocks.size()));gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdpg low orbit");}int orbit_done=slot;pipe.fence(slot++);if(p>1){auto s=grid_plan.low[pi];gdvw_gather_low(pipe,wave_plan.low[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,s.x,s.y,measure);}else{gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);unsigned nt=unsigned(layout.main_blocks.size());for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdpg p1 local device");dim3 g(base_grid_x,base_grid_y,nt);gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdpg p1 local");}int local_done=slot;pipe.fence(slot++);gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdpg p1 cross device");dim3 g(base_grid_x,base_grid_y,nt);gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdpg p1 cross");}}ready_slot=slot;pipe.fence(slot++);}
}
