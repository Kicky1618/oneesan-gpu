#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_wavefront.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

// v1.1: self-calibrating wavefront boundaries.
// Row 1 runs the v1.0 four-wave layout while CUDA events measure gather-only
// compute time. With a measured ONEESAN_P2P_GBPS matrix, rows 2..W use a
// local-search layout that minimizes a two-pipeline model: one serialized stream
// per source peer plus one compute stream per destination GPU. State ownership,
// phase order, source-block union, and copied byte count remain unchanged.

using GdawBins = std::array<std::vector<GdwfDest>,GDWF_WAVES>;

struct GdawCalibration {
    std::array<double,GDM_MAX_GPU> work_per_ms{};
    double min_work_per_ms = 0.0;
    double max_work_per_ms = 0.0;
    double aggregate_work_per_ms = 0.0;
    bool valid = false;
};

struct GdawPlan {
    GdwfPlan wave;
    double baseline_predicted_ms_per_row = 0.0;
    double adaptive_predicted_ms_per_row = 0.0;
    unsigned improved_device_phases = 0;
    bool selected = false;
};

static double gdaw_copy_ms(unsigned long long bytes,double gbps) {
    return (bytes && gbps>0.0) ? double(bytes)/(gbps*1.0e6) : 0.0;
}

static double gdaw_score_bins(
    const GdawBins& bins,const StorageLayout& layout,const GdmShardHost& shard,
    const GdtpTopology& topo,int d,double work_per_ms
) {
    if(work_per_ms<=0.0) return 1e300;
    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> staged{};
    std::array<double,GDM_MAX_GPU> peer_ready{};
    double compute_finish=0.0;
    for(int w=0;w<GDWF_WAVES;++w) {
        std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> need{};
        unsigned long long work=0;
        for(const auto& x:bins[w]) {
            work+=x.work;
            for(std::uint8_t s:x.source) if(!staged[s]) need[s]=true;
        }
        for(std::uint32_t s=0;s<layout.main_blocks.size();++s) if(need[s]) {
            staged[s]=true;
            int src=shard.main_blocks[s].owner;
            if(src==d) continue;
            peer_ready[src]+=gdaw_copy_ms(gdpg_bytes(layout.main_blocks[s]),topo.gbps[d][src]);
        }
        double ready=0.0;
        for(int s=0;s<GDM_MAX_GPU;++s) ready=std::max(ready,peer_ready[s]);
        compute_finish=std::max(compute_finish,ready)+double(work)/work_per_ms;
    }
    return compute_finish;
}

static GdawBins gdaw_best_order(
    const GdawBins& in,const StorageLayout& layout,const GdmShardHost& shard,
    const GdtpTopology& topo,int d,double work_per_ms,double* best_score=nullptr
) {
    std::array<int,GDWF_WAVES> p{}; std::iota(p.begin(),p.end(),0);
    GdawBins best=in; double bs=gdaw_score_bins(in,layout,shard,topo,d,work_per_ms);
    do {
        GdawBins c;
        for(int w=0;w<GDWF_WAVES;++w)c[w]=in[p[w]];
        double s=gdaw_score_bins(c,layout,shard,topo,d,work_per_ms);
        if(s+1e-9<bs){bs=s;best=std::move(c);}
    } while(std::next_permutation(p.begin(),p.end()));
    if(best_score)*best_score=bs;
    return best;
}

static GdawBins gdaw_bins_from_baseline(
    const GdwfPhase& base,int d,const std::vector<GdwfDest>& items
) {
    GdawBins bins;
    std::array<int,256> where{}; where.fill(-1);
    for(int w=0;w<GDWF_WAVES;++w) for(std::uint8_t b:base.wave[w].dbids[d]) where[b]=w;
    std::array<unsigned long long,GDWF_WAVES> load{};
    for(const auto& x:items) {
        int w=where[x.dbid];
        if(w<0){w=0;for(int k=1;k<GDWF_WAVES;++k)if(load[k]<load[w])w=k;}
        bins[w].push_back(x);load[w]+=x.work;
    }
    return bins;
}

static GdawBins gdaw_optimize_bins(
    GdawBins bins,const StorageLayout& layout,const GdmShardHost& shard,
    const GdtpTopology& topo,int d,double work_per_ms,double* baseline,double* optimized
) {
    const double original=gdaw_score_bins(bins,layout,shard,topo,d,work_per_ms);
    if(baseline)*baseline=original;
    double cur=0.0;bins=gdaw_best_order(bins,layout,shard,topo,d,work_per_ms,&cur);
    for(int sweep=0;sweep<8;++sweep) {
        GdawBins best=bins; double bs=cur;
        for(int from=0;from<GDWF_WAVES;++from) for(std::size_t i=0;i<bins[from].size();++i) {
            for(int to=0;to<GDWF_WAVES;++to) if(to!=from) {
                GdawBins c=bins;
                GdwfDest x=c[from][i];
                c[from].erase(c[from].begin()+static_cast<std::vector<GdwfDest>::difference_type>(i));
                c[to].push_back(std::move(x));
                double s=0.0;c=gdaw_best_order(c,layout,shard,topo,d,work_per_ms,&s);
                if(s+1e-6<bs){bs=s;best=std::move(c);}
            }
        }
        if(!(bs+1e-6<cur))break;
        bins=std::move(best);cur=bs;
    }
    if(optimized)*optimized=cur;
    return bins;
}

static void gdaw_emit_device(
    GdwfPhase& out,int d,const GdawBins& bins,const StorageLayout& layout
) {
    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> staged{};
    for(int w=0;w<GDWF_WAVES;++w) {
        std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> need{};
        for(const auto& x:bins[w]) {
            out.wave[w].dbids[d].push_back(x.dbid);
            out.wave[w].work[d]+=x.work;
            for(std::uint8_t s:x.source) need[s]=true;
        }
        for(std::uint32_t s=0;s<layout.main_blocks.size();++s) if(need[s]&&!staged[s]) {
            staged[s]=true;out.wave[w].new_source[d].push_back(std::uint8_t(s));
            out.wave[w].copy_bytes[d]+=gdpg_bytes(layout.main_blocks[s]);
        }
    }
}

static std::vector<GdwfDest> gdaw_high_items(
    std::uint32_t pi,int d,const StorageLayout& layout,const GdmShardHost& shard,
    const GpuDirectGatherHost& ordinary,const GpuDirectCrossGatherHost& cross
) {
    std::vector<GdwfDest> v;
    for(std::uint32_t b=0;b<layout.block_blocks.size();++b){const auto& sb=layout.block_blocks[b];if(!sb.valid||!sb.rows||!sb.cols||shard.block_blocks[b].owner!=d)continue;GdwfDest x;x.dbid=std::uint8_t(b);x.work=gdwf_high_work(pi,b,layout,ordinary,cross);x.source=gdwf_high_sources(pi,b,ordinary,cross,shard,d);v.push_back(std::move(x));}
    return v;
}
static std::vector<GdwfDest> gdaw_low_items(
    std::uint32_t pi,int d,const StorageLayout& layout,const GdmShardHost& shard,
    const GpuDirectGatherHost& ordinary,const GpuDirectCrossGatherHost& cross
) {
    std::vector<GdwfDest> v;
    for(std::uint32_t b=0;b<layout.block_blocks.size();++b){const auto& sb=layout.block_blocks[b];if(!sb.valid||!sb.rows||!sb.cols||shard.block_blocks[b].owner!=d)continue;GdwfDest x;x.dbid=std::uint8_t(b);x.work=gdwf_low_work(pi,b,layout,ordinary,cross);x.source=gdwf_low_sources(pi,b,ordinary,cross,shard,d);v.push_back(std::move(x));}
    return v;
}

static void gdaw_recount(GdwfPlan& p,const GdmsStagePlan& exact,int ngpu) {
    p.eligible_copy_bytes_per_row=0;p.first_wave_copy_bytes_per_row=0;p.overlap_candidate_bytes_per_row=0;p.serial_p1_copy_bytes_per_row=0;p.wave_launch_groups_per_row=0;
    auto one=[&](const GdwfPhase& ph){for(int w=0;w<GDWF_WAVES;++w){bool any=false;for(int d=0;d<ngpu;++d){p.eligible_copy_bytes_per_row+=ph.wave[w].copy_bytes[d];if(w==0)p.first_wave_copy_bytes_per_row+=ph.wave[w].copy_bytes[d];if(!ph.wave[w].dbids[d].empty())any=true;}if(any)++p.wave_launch_groups_per_row;}};
    for(const auto& ph:p.high)one(ph);for(int q=LOW_LUT_K;q>=2;--q)one(p.low[LOW_LUT_K-q]);
    p.overlap_candidate_bytes_per_row=p.eligible_copy_bytes_per_row-p.first_wave_copy_bytes_per_row;
    std::uint32_t p1=std::uint32_t(LOW_LUT_K-1);if(p1<exact.low.size())for(int d=0;d<ngpu;++d)p.serial_p1_copy_bytes_per_row+=exact.low[p1].bytes[d]+exact.low[p1].refresh_bytes[d];
    unsigned long long expected=0;for(const auto& ph:exact.high)for(int d=0;d<ngpu;++d)expected+=ph.bytes[d];for(int q=LOW_LUT_K;q>=2;--q){auto pi=std::uint32_t(LOW_LUT_K-q);for(int d=0;d<ngpu;++d)expected+=exact.low[pi].bytes[d];}
    if(expected!=p.eligible_copy_bytes_per_row){std::cerr<<"gdaw staged-byte mismatch "<<p.eligible_copy_bytes_per_row<<'/'<<expected<<'\n';std::exit(196);}
}

static GdawPlan build_gdaw_plan(
    const GdwfPlan& base,const GdawCalibration& cal,const GdtpTopology& topo,
    const StorageLayout& layout,const GdmShardHost& shard,
    const GpuDirectGatherHost& ordinary,const GpuDirectCrossGatherHost& cross,
    const GdmsStagePlan& exact,int ngpu
) {
    GdawPlan out;out.wave=base;
    if(!cal.valid||!topo.custom)return out;
    out.wave.high.assign(HIGH_LUT_K,GdwfPhase{});out.wave.low.assign(LOW_LUT_K,GdwfPhase{});
    double base_sum=0.0,new_sum=0.0;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);double phase_b=0.0,phase_n=0.0;for(int d=0;d<ngpu;++d){auto item=gdaw_high_items(pi,d,layout,shard,ordinary,cross);auto bins=gdaw_bins_from_baseline(base.high[pi],d,item);double b=0.0,n=0.0;auto opt=gdaw_optimize_bins(std::move(bins),layout,shard,topo,d,cal.work_per_ms[d],&b,&n);if(n+1e-6<b)++out.improved_device_phases;else{opt=gdaw_bins_from_baseline(base.high[pi],d,item);n=b;}gdaw_emit_device(out.wave.high[pi],d,opt,layout);phase_b=std::max(phase_b,b);phase_n=std::max(phase_n,n);}base_sum+=phase_b;new_sum+=phase_n;}
    for(int p=LOW_LUT_K;p>=2;--p){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);double phase_b=0.0,phase_n=0.0;for(int d=0;d<ngpu;++d){auto item=gdaw_low_items(pi,d,layout,shard,ordinary,cross);auto bins=gdaw_bins_from_baseline(base.low[pi],d,item);double b=0.0,n=0.0;auto opt=gdaw_optimize_bins(std::move(bins),layout,shard,topo,d,cal.work_per_ms[d],&b,&n);if(n+1e-6<b)++out.improved_device_phases;else{opt=gdaw_bins_from_baseline(base.low[pi],d,item);n=b;}gdaw_emit_device(out.wave.low[pi],d,opt,layout);phase_b=std::max(phase_b,b);phase_n=std::max(phase_n,n);}base_sum+=phase_b;new_sum+=phase_n;}
    out.baseline_predicted_ms_per_row=base_sum;out.adaptive_predicted_ms_per_row=new_sum;
    gdaw_recount(out.wave,exact,ngpu);
    out.selected=(out.improved_device_phases>0 && new_sum+1e-6<base_sum);
    if(!out.selected){out.wave=base;out.adaptive_predicted_ms_per_row=base_sum;gdaw_recount(out.wave,exact,ngpu);}
    return out;
}

using GdawTimingRow=std::array<std::array<cudaEvent_t,GDM_MAX_GPU>,GDWF_WAVES>;
struct GdawPipeline: GdwfPipeline {
    std::vector<GdawTimingRow> compute_begin,compute_end;
    void init(int n,const StorageLayout& layout){GdwfPipeline::init(n,layout);compute_begin.resize(GDWF_PHASES);compute_end.resize(GDWF_PHASES);for(int ph=0;ph<GDWF_PHASES;++ph)for(int w=0;w<GDWF_WAVES;++w)for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdaw timing device");ck(cudaEventCreate(&compute_begin[ph][w][d]),"gdaw timing begin");ck(cudaEventCreate(&compute_end[ph][w][d]),"gdaw timing end");}}
    void refresh_meta(const GdwfPlan& p){meta_installed=false;install_meta(p);}
    void destroy(){for(int ph=0;ph<GDWF_PHASES;++ph)for(int w=0;w<GDWF_WAVES;++w)for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdaw destroy timing device");if(compute_begin[ph][w][d])cudaEventDestroy(compute_begin[ph][w][d]);if(compute_end[ph][w][d])cudaEventDestroy(compute_end[ph][w][d]);}compute_begin.clear();compute_end.clear();GdwfPipeline::destroy();}
};

static void gdaw_gather_high(
    GdawPipeline& pipe,const GdwfPhase& ph,int phase,int ready_slot,const StorageLayout& layout,
    const GdmShardHost& shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y,bool measure
) {
    dim3 block(threads);for(int w=0;w<GDWF_WAVES;++w){gdwf_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;ck(cudaSetDevice(d),"gdaw high device");if(measure)ck(cudaEventRecord(pipe.compute_begin[phase][w][d],pipe.compute[d]),"gdaw high begin");dim3 g(grid_x,grid_y,nz);gdwf_high_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);gdwf_high_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);ck(cudaGetLastError(),"gdaw high gather");if(measure)ck(cudaEventRecord(pipe.compute_end[phase][w][d],pipe.compute[d]),"gdaw high end");}}
}
static void gdaw_gather_low(
    GdawPipeline& pipe,const GdwfPhase& ph,int phase,int ready_slot,const StorageLayout& layout,
    const GdmShardHost& shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y,bool measure
) {
    dim3 block(threads);for(int w=0;w<GDWF_WAVES;++w){gdwf_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;ck(cudaSetDevice(d),"gdaw low device");if(measure)ck(cudaEventRecord(pipe.compute_begin[phase][w][d],pipe.compute[d]),"gdaw low begin");dim3 g(grid_x,grid_y,nz);gdwf_low_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);gdwf_low_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);ck(cudaGetLastError(),"gdaw low gather");if(measure)ck(cudaEventRecord(pipe.compute_end[phase][w][d],pipe.compute[d]),"gdaw low end");}}
}

static GdawCalibration gdaw_calibrate(GdawPipeline& pipe,const GdwfPlan& p,int ngpu) {
    GdawCalibration out;std::array<double,GDM_MAX_GPU> work_sum{},ms_sum{};double total_work=0.0,total_ms=0.0;
    auto add=[&](int phase,const GdwfPhase& ph){for(int w=0;w<GDWF_WAVES;++w)for(int d=0;d<ngpu;++d){if(ph.wave[w].dbids[d].empty()||!ph.wave[w].work[d])continue;ck(cudaSetDevice(d),"gdaw elapsed device");float ms=0.0f;ck(cudaEventElapsedTime(&ms,pipe.compute_begin[phase][w][d],pipe.compute_end[phase][w][d]),"gdaw elapsed");if(ms>0.0f){work_sum[d]+=double(ph.wave[w].work[d]);ms_sum[d]+=ms;total_work+=double(ph.wave[w].work[d]);total_ms+=ms;}}};
    for(int pi=0;pi<HIGH_LUT_K;++pi)add(pi,p.high[pi]);
    for(int q=LOW_LUT_K;q>=2;--q){int pi=LOW_LUT_K-q,ph=HIGH_LUT_K+pi;add(ph,p.low[pi]);}
    out.aggregate_work_per_ms=total_ms>0.0?total_work/total_ms:0.0;
    out.min_work_per_ms=1e300;out.max_work_per_ms=0.0;
    for(int d=0;d<ngpu;++d){out.work_per_ms[d]=ms_sum[d]>0.0?work_sum[d]/ms_sum[d]:out.aggregate_work_per_ms;if(out.work_per_ms[d]>0.0){out.min_work_per_ms=std::min(out.min_work_per_ms,out.work_per_ms[d]);out.max_work_per_ms=std::max(out.max_work_per_ms,out.work_per_ms[d]);}}
    if(out.min_work_per_ms==1e300)out.min_work_per_ms=0.0;out.valid=out.aggregate_work_per_ms>0.0;return out;
}

static void gdaw_enqueue_row(
    GdawPipeline& pipe,const GdowOrbitPlan& orbit_plan,const GdmsStagePlan& gather_plan,
    const GdwfPlan& wave_plan,const StorageLayout& layout,const GdmShardHost& shard,
    Count*const*main_ptr,Count*const*block_ptr,int threads,int grid_x,int grid_y,int& slot,bool measure
) {
    pipe.install_meta(wave_plan);dim3 block(threads);int phase=0,ready_slot=-1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);const auto&oph=orbit_plan.high[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdaw high orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdaw high orbit");}int orbit_done=slot;pipe.fence(slot++);gdaw_gather_high(pipe,wave_plan.high[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);ready_slot=slot;pipe.fence(slot++);}
    for(int p=LOW_LUT_K;p>=1;--p,++phase){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);const auto&oph=orbit_plan.low[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdaw low orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdaw low orbit");}int orbit_done=slot;pipe.fence(slot++);if(p>1){gdaw_gather_low(pipe,wave_plan.low[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);}else{gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);unsigned nt=unsigned(layout.main_blocks.size());for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdaw p1 local device");dim3 g(grid_x,grid_y,nt);gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdaw p1 local");}int local_done=slot;pipe.fence(slot++);gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdaw p1 cross device");dim3 g(grid_x,grid_y,nt);gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdaw p1 cross");}}ready_slot=slot;pipe.fence(slot++);}
}
