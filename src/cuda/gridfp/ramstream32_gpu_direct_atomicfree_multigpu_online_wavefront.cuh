#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_adaptive_wavefront.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// v1.2: online adaptive wavefront.  Recalibrate on rows 1, 4 and 8.
// Gather compute rate and effective peer-copy GB/s are measured with CUDA
// events.  An EMA model absorbs clock/thermal/NVLink-contention changes.  A new
// wave layout is installed only when the exact-byte-preserving v1.1 planner
// predicts at least 0.5% lower pipeline time than the currently active layout.

static constexpr double GDOR_MIN_REPLAN_GAIN_PCT = 0.5;

static bool gdor_measure_row(int row0) {
    return row0==0 || row0==3 || row0==7;
}

struct GdorCopyCalibration {
    std::array<std::array<double,GDM_MAX_GPU>,GDM_MAX_GPU> gbps{};
    std::array<std::array<unsigned,GDM_MAX_GPU>,GDM_MAX_GPU> samples{};
    double min_gbps = 0.0;
    double max_gbps = 0.0;
    double mean_gbps = 0.0;
    bool valid = false;
};

struct GdorOnlineModel {
    GdawCalibration compute;
    GdtpTopology topology;
    unsigned updates = 0;
    double copy_min_gbps = 0.0;
    double copy_max_gbps = 0.0;
    double copy_mean_gbps = 0.0;
};

using GdorPeerEvent = std::array<std::array<cudaEvent_t,GDM_MAX_GPU>,GDM_MAX_GPU>;
struct GdorPipeline : GdawPipeline {
    std::vector<std::array<GdorPeerEvent,GDWF_WAVES>> copy_begin;
    std::vector<std::array<GdorPeerEvent,GDWF_WAVES>> copy_end;

    void init(int n,const StorageLayout& layout) {
        GdawPipeline::init(n,layout);
        copy_begin.resize(GDWF_PHASES);
        copy_end.resize(GDWF_PHASES);
        for(int ph=0;ph<GDWF_PHASES;++ph) for(int w=0;w<GDWF_WAVES;++w)
            for(int d=0;d<ngpu;++d) {
                ck(cudaSetDevice(d),"gdor timing device");
                for(int s=0;s<ngpu;++s) if(s!=d) {
                    ck(cudaEventCreate(&copy_begin[ph][w][d][s]),"gdor copy begin");
                    ck(cudaEventCreate(&copy_end[ph][w][d][s]),"gdor copy end");
                }
            }
    }

    void destroy() {
        for(int ph=0;ph<GDWF_PHASES;++ph) for(int w=0;w<GDWF_WAVES;++w)
            for(int d=0;d<ngpu;++d) {
                ck(cudaSetDevice(d),"gdor destroy timing device");
                for(int s=0;s<ngpu;++s) if(s!=d) {
                    if(copy_begin[ph][w][d][s]) cudaEventDestroy(copy_begin[ph][w][d][s]);
                    if(copy_end[ph][w][d][s]) cudaEventDestroy(copy_end[ph][w][d][s]);
                }
            }
        copy_begin.clear();copy_end.clear();
        GdawPipeline::destroy();
    }
};

static void gdor_stage_wave(
    GdorPipeline& pipe,int phase,int wave,int ready_slot,const GdwfWave& wp,
    const StorageLayout& layout,const GdmShardHost& shard,Count*const*main_ptr,bool measure
) {
    for(int d=0;d<pipe.ngpu;++d) {
        std::array<bool,GDM_MAX_GPU> used{};
        ck(cudaSetDevice(d),"gdor copy device");
        for(std::uint8_t id:wp.new_source[d]) {
            std::uint32_t b=id;
            const auto& logical=layout.main_blocks[b];
            const auto& physical=shard.main_blocks[b];
            int s=physical.owner;
            if(s==d) continue;
            std::size_t bytes=std::size_t(logical.rows)*logical.cols*sizeof(Count);
            if(!bytes) continue;
            cudaStream_t cs=pipe.copy[d][s];
            if(!used[s]) {
                ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][s],0),"gdor wait source");
                ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][d],0),"gdor wait dest");
                if(measure) ck(cudaEventRecord(pipe.copy_begin[phase][wave][d][s],cs),"gdor copy begin record");
                used[s]=true;
            }
            ck(cudaMemcpyPeerAsync(pipe.stage[d]+logical.off,d,main_ptr[s]+physical.off,s,bytes,cs),"gdor wave copy");
        }
        for(int s=0;s<pipe.ngpu;++s) if(used[s]) {
            cudaEvent_t done;
            if(measure) {
                done=pipe.copy_end[phase][wave][d][s];
                ck(cudaEventRecord(done,pipe.copy[d][s]),"gdor copy end record");
            } else {
                done=pipe.wave_done[phase][wave][d][s];
                ck(cudaEventRecord(done,pipe.copy[d][s]),"gdor wave done");
            }
            ck(cudaStreamWaitEvent(pipe.compute[d],done,0),"gdor compute wait copy");
        }
    }
}

static void gdor_gather_high(
    GdorPipeline& pipe,const GdwfPhase& ph,int phase,int ready_slot,const StorageLayout& layout,
    const GdmShardHost& shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y,bool measure
) {
    dim3 block(threads);
    for(int w=0;w<GDWF_WAVES;++w) {
        gdor_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr,measure);
        for(int d=0;d<pipe.ngpu;++d) {
            unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;
            ck(cudaSetDevice(d),"gdor high device");
            if(measure) ck(cudaEventRecord(pipe.compute_begin[phase][w][d],pipe.compute[d]),"gdor high begin");
            dim3 g(grid_x,grid_y,nz);
            gdwf_high_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);
            gdwf_high_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);
            ck(cudaGetLastError(),"gdor high gather");
            if(measure) ck(cudaEventRecord(pipe.compute_end[phase][w][d],pipe.compute[d]),"gdor high end");
        }
    }
}

static void gdor_gather_low(
    GdorPipeline& pipe,const GdwfPhase& ph,int phase,int ready_slot,const StorageLayout& layout,
    const GdmShardHost& shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y,bool measure
) {
    dim3 block(threads);
    for(int w=0;w<GDWF_WAVES;++w) {
        gdor_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr,measure);
        for(int d=0;d<pipe.ngpu;++d) {
            unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;
            ck(cudaSetDevice(d),"gdor low device");
            if(measure) ck(cudaEventRecord(pipe.compute_begin[phase][w][d],pipe.compute[d]),"gdor low begin");
            dim3 g(grid_x,grid_y,nz);
            gdwf_low_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);
            gdwf_low_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);
            ck(cudaGetLastError(),"gdor low gather");
            if(measure) ck(cudaEventRecord(pipe.compute_end[phase][w][d],pipe.compute[d]),"gdor low end");
        }
    }
}

static GdorCopyCalibration gdor_calibrate_copy(
    GdorPipeline& pipe,const GdwfPlan& plan,const StorageLayout& layout,
    const GdmShardHost& shard,int ngpu
) {
    GdorCopyCalibration out;
    double sum=0.0;unsigned count=0;
    auto add=[&](int phase,const GdwfPhase& ph) {
        for(int w=0;w<GDWF_WAVES;++w) for(int d=0;d<ngpu;++d) {
            std::array<unsigned long long,GDM_MAX_GPU> bytes{};
            for(std::uint8_t id:ph.wave[w].new_source[d]) {
                std::uint32_t b=id;int s=shard.main_blocks[b].owner;
                if(s!=d) bytes[s]+=gdpg_bytes(layout.main_blocks[b]);
            }
            for(int s=0;s<ngpu;++s) if(s!=d && bytes[s]) {
                ck(cudaSetDevice(d),"gdor copy elapsed device");
                float ms=0.0f;
                ck(cudaEventElapsedTime(&ms,pipe.copy_begin[phase][w][d][s],pipe.copy_end[phase][w][d][s]),"gdor copy elapsed");
                if(ms>0.0f) {
                    double g=double(bytes[s])/(double(ms)*1.0e6);
                    out.gbps[d][s]+=g;out.samples[d][s]++;
                }
            }
        }
    };
    for(int pi=0;pi<HIGH_LUT_K;++pi)add(pi,plan.high[pi]);
    for(int p=LOW_LUT_K;p>=2;--p){int pi=LOW_LUT_K-p;add(HIGH_LUT_K+pi,plan.low[pi]);}
    out.min_gbps=1e300;
    for(int d=0;d<ngpu;++d) for(int s=0;s<ngpu;++s) if(d!=s && out.samples[d][s]) {
        out.gbps[d][s]/=double(out.samples[d][s]);
        out.min_gbps=std::min(out.min_gbps,out.gbps[d][s]);
        out.max_gbps=std::max(out.max_gbps,out.gbps[d][s]);
        sum+=out.gbps[d][s];++count;
    }
    if(out.min_gbps==1e300)out.min_gbps=0.0;
    out.mean_gbps=count?sum/double(count):0.0;
    out.valid=count>0;
    return out;
}

static void gdor_refresh_compute_summary(GdawCalibration& c,int ngpu) {
    c.min_work_per_ms=1e300;c.max_work_per_ms=0.0;c.aggregate_work_per_ms=0.0;
    unsigned n=0;
    for(int d=0;d<ngpu;++d) if(c.work_per_ms[d]>0.0) {
        c.min_work_per_ms=std::min(c.min_work_per_ms,c.work_per_ms[d]);
        c.max_work_per_ms=std::max(c.max_work_per_ms,c.work_per_ms[d]);
        c.aggregate_work_per_ms+=c.work_per_ms[d];++n;
    }
    if(c.min_work_per_ms==1e300)c.min_work_per_ms=0.0;
    if(n)c.aggregate_work_per_ms/=double(n);
    c.valid=n>0;
}

static void gdor_update_model(
    GdorOnlineModel& m,const GdawCalibration& compute,const GdorCopyCalibration& copy,
    const GdtpTopology& initial,int ngpu
) {
    const double alpha=m.updates?0.35:0.65;
    if(m.updates==0) {
        m.topology=initial;
        if(!m.topology.custom) {
            double seed=copy.valid?copy.mean_gbps:1.0;
            for(int d=0;d<ngpu;++d)for(int s=0;s<ngpu;++s)
                m.topology.gbps[d][s]=d==s?1e300:seed;
        }
    }
    for(int d=0;d<ngpu;++d) if(compute.work_per_ms[d]>0.0) {
        double& x=m.compute.work_per_ms[d];
        x=x>0.0?(1.0-alpha)*x+alpha*compute.work_per_ms[d]:compute.work_per_ms[d];
    }
    gdor_refresh_compute_summary(m.compute,ngpu);
    if(copy.valid) {
        for(int d=0;d<ngpu;++d)for(int s=0;s<ngpu;++s)if(d!=s&&copy.samples[d][s]) {
            double& x=m.topology.gbps[d][s];
            x=x>0.0?(1.0-alpha)*x+alpha*copy.gbps[d][s]:copy.gbps[d][s];
        }
        m.topology.custom=true;
        m.copy_min_gbps=copy.min_gbps;m.copy_max_gbps=copy.max_gbps;m.copy_mean_gbps=copy.mean_gbps;
    }
    ++m.updates;
}

static bool gdor_accept(const GdawPlan& p,double* gain_pct=nullptr) {
    double gain=p.baseline_predicted_ms_per_row>0.0
        ?100.0*(p.baseline_predicted_ms_per_row-p.adaptive_predicted_ms_per_row)/p.baseline_predicted_ms_per_row:0.0;
    if(gain_pct)*gain_pct=gain;
    return p.selected && gain>=GDOR_MIN_REPLAN_GAIN_PCT;
}

static void gdor_enqueue_row(
    GdorPipeline& pipe,const GdowOrbitPlan& orbit_plan,const GdmsStagePlan& gather_plan,
    const GdwfPlan& wave_plan,const StorageLayout& layout,const GdmShardHost& shard,
    Count*const*main_ptr,Count*const*block_ptr,int threads,int grid_x,int grid_y,int& slot,bool measure
) {
    pipe.install_meta(wave_plan);dim3 block(threads);int phase=0,ready_slot=-1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase) {
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);const auto& oph=orbit_plan.high[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdor high orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdor high orbit");}
        int orbit_done=slot;pipe.fence(slot++);
        gdor_gather_high(pipe,wave_plan.high[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);
        ready_slot=slot;pipe.fence(slot++);
    }
    for(int p=LOW_LUT_K;p>=1;--p,++phase) {
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);const auto& oph=orbit_plan.low[pi];
        gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);
        for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdor low orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdor low orbit");}
        int orbit_done=slot;pipe.fence(slot++);
        if(p>1) {
            gdor_gather_low(pipe,wave_plan.low[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);
        } else {
            gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);
            unsigned nt=unsigned(layout.main_blocks.size());
            for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdor p1 local device");dim3 g(grid_x,grid_y,nt);gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdor p1 local");}
            int local_done=slot;pipe.fence(slot++);
            gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr);
            for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdor p1 cross device");dim3 g(grid_x,grid_y,nt);gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdor p1 cross");}
        }
        ready_slot=slot;pipe.fence(slot++);
    }
}
