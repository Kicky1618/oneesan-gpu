#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_online_wavefront.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

// v1.3: variable wave count (2..8) selected independently per phase after
// online calibration.  The exact source-block union and copied bytes are kept
// identical to the exact staging plan; only destination-to-wave assignment and
// wave order change.

static constexpr int GDVW_MIN_WAVES=2;
static constexpr int GDVW_MAX_WAVES=8;
static constexpr int GDVW_PHASES=LOW_LUT_K+HIGH_LUT_K;
static constexpr int GDVW_MAX_DB=GPU_DIRECT_MAX_BLOCK_BLOCKS;
static constexpr std::size_t GDVW_DBID_CAP=std::size_t(GDVW_PHASES)*GDVW_MAX_WAVES*GDVW_MAX_DB;
__constant__ const std::uint8_t* D_GDVW_DBID_PTR;

static double gdvw_wave_group_ms(){
    const char* e=std::getenv("ONEESAN_WAVE_GROUP_US");
    if(!e||!*e)return 0.008;
    char* end=nullptr;double us=std::strtod(e,&end);
    if(end==e||!std::isfinite(us)||us<0.0)return 0.008;
    return us/1000.0;
}

struct GdvwPhase{
    std::array<GdwfWave,GDVW_MAX_WAVES> wave;
    int waves=4;
};
struct GdvwPlan{
    std::vector<GdvwPhase> high;
    std::vector<GdvwPhase> low;
    unsigned long long eligible_copy_bytes_per_row=0;
    unsigned long long first_wave_copy_bytes_per_row=0;
    unsigned long long overlap_candidate_bytes_per_row=0;
    unsigned long long serial_p1_copy_bytes_per_row=0;
    unsigned long long wave_launch_groups_per_row=0;
    std::array<unsigned,GDVW_MAX_WAVES+1> phase_wave_hist{};
};

using GdvwBins=std::vector<std::vector<GdwfDest>>;

static GdvwPlan gdvw_from_fixed(const GdwfPlan& in,const GdmsStagePlan& exact,int ngpu){
    GdvwPlan out;out.high.resize(HIGH_LUT_K);out.low.resize(LOW_LUT_K);
    for(int pi=0;pi<HIGH_LUT_K;++pi){out.high[pi].waves=GDWF_WAVES;for(int w=0;w<GDWF_WAVES;++w)out.high[pi].wave[w]=in.high[pi].wave[w];}
    for(int p=LOW_LUT_K;p>=2;--p){int pi=LOW_LUT_K-p;out.low[pi].waves=GDWF_WAVES;for(int w=0;w<GDWF_WAVES;++w)out.low[pi].wave[w]=in.low[pi].wave[w];}
    std::uint32_t p1=std::uint32_t(LOW_LUT_K-1);if(p1<exact.low.size())for(int d=0;d<ngpu;++d)out.serial_p1_copy_bytes_per_row+=exact.low[p1].bytes[d]+exact.low[p1].refresh_bytes[d];
    return out;
}

static double gdvw_score_bins(const GdvwBins& bins,const StorageLayout& layout,const GdmShardHost& shard,const GdtpTopology& topo,int d,double work_per_ms,double launch_ms){
    if(work_per_ms<=0.0)return 1e300;
    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> staged{};
    std::array<double,GDM_MAX_GPU> peer_ready{};
    double compute_finish=0.0;
    for(const auto& bin:bins){
        if(bin.empty())continue;
        std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> need{};unsigned long long work=0;
        for(const auto&x:bin){work+=x.work;for(std::uint8_t s:x.source)if(!staged[s])need[s]=true;}
        for(std::uint32_t s=0;s<layout.main_blocks.size();++s)if(need[s]){staged[s]=true;int src=shard.main_blocks[s].owner;if(src!=d)peer_ready[src]+=gdaw_copy_ms(gdpg_bytes(layout.main_blocks[s]),topo.gbps[d][src]);}
        double ready=0.0;for(int s=0;s<GDM_MAX_GPU;++s)ready=std::max(ready,peer_ready[s]);
        compute_finish=std::max(compute_finish,ready)+double(work)/work_per_ms+launch_ms;
    }
    return compute_finish;
}

static double gdvw_bin_copy_ms(const std::vector<GdwfDest>& bin,const StorageLayout&layout,const GdmShardHost&shard,const GdtpTopology&topo,int d){
    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> seen{};std::array<unsigned long long,GDM_MAX_GPU> bytes{};
    for(const auto&x:bin)for(std::uint8_t s:x.source)seen[s]=true;
    for(std::uint32_t s=0;s<layout.main_blocks.size();++s)if(seen[s]){int src=shard.main_blocks[s].owner;if(src!=d)bytes[src]+=gdpg_bytes(layout.main_blocks[s]);}
    double m=0.0;for(int s=0;s<GDM_MAX_GPU;++s)if(s!=d)m=std::max(m,gdaw_copy_ms(bytes[s],topo.gbps[d][s]));return m;
}

static GdvwBins gdvw_order_bins(GdvwBins bins,const StorageLayout&layout,const GdmShardHost&shard,const GdtpTopology&topo,int d,double work_per_ms,double launch_ms,double*score=nullptr){
    std::stable_sort(bins.begin(),bins.end(),[&](const auto&a,const auto&b){bool ea=a.empty(),eb=b.empty();if(ea!=eb)return !ea;if(ea)return false;double ca=gdvw_bin_copy_ms(a,layout,shard,topo,d),cb=gdvw_bin_copy_ms(b,layout,shard,topo,d);if(ca!=cb)return ca<cb;unsigned long long wa=0,wb=0;for(auto&x:a)wa+=x.work;for(auto&x:b)wb+=x.work;return wa>wb;});
    double cur=gdvw_score_bins(bins,layout,shard,topo,d,work_per_ms,launch_ms);
    for(int sweep=0;sweep<3;++sweep){bool changed=false;for(std::size_t i=0;i<bins.size();++i)for(std::size_t j=i+1;j<bins.size();++j){auto c=bins;std::swap(c[i],c[j]);double s=gdvw_score_bins(c,layout,shard,topo,d,work_per_ms,launch_ms);if(s+1e-6<cur){bins=std::move(c);cur=s;changed=true;}}if(!changed)break;}
    if(score)*score=cur;return bins;
}

static GdvwBins gdvw_optimize_k(std::vector<GdwfDest> item,int k,const StorageLayout&layout,const GdmShardHost&shard,const GdtpTopology&topo,int d,double work_per_ms,double launch_ms,double*score){
    GdvwBins bins(std::size_t(k));std::array<unsigned long long,GDVW_MAX_WAVES> load{};
    std::sort(item.begin(),item.end(),[](const auto&a,const auto&b){return a.work>b.work;});
    for(auto&x:item){int w=0;for(int q=1;q<k;++q)if(load[q]<load[w])w=q;load[w]+=x.work;bins[w].push_back(std::move(x));}
    double cur=0.0;bins=gdvw_order_bins(std::move(bins),layout,shard,topo,d,work_per_ms,launch_ms,&cur);
    for(int sweep=0;sweep<4;++sweep){GdvwBins best=bins;double bs=cur;for(int from=0;from<k;++from)for(std::size_t i=0;i<bins[from].size();++i)for(int to=0;to<k;++to)if(to!=from){auto c=bins;GdwfDest x=c[from][i];c[from].erase(c[from].begin()+static_cast<std::vector<GdwfDest>::difference_type>(i));c[to].push_back(std::move(x));double s=0.0;c=gdvw_order_bins(std::move(c),layout,shard,topo,d,work_per_ms,launch_ms,&s);if(s+1e-6<bs){bs=s;best=std::move(c);}}if(!(bs+1e-6<cur))break;bins=std::move(best);cur=bs;}
    if(score)*score=cur;return bins;
}

static GdvwBins gdvw_bins_from_phase(const GdvwPhase&ph,int d,const std::vector<GdwfDest>&items){
    GdvwBins out(std::size_t(std::max(ph.waves,1)));std::array<int,256>where{};where.fill(-1);
    for(int w=0;w<ph.waves;++w)for(std::uint8_t b:ph.wave[w].dbids[d])where[b]=w;
    for(const auto&x:items){int w=where[x.dbid];if(w<0||w>=ph.waves)w=0;out[w].push_back(x);}return out;
}

static void gdvw_emit_device(GdvwPhase&out,int d,const GdvwBins&bins,const StorageLayout&layout){
    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS>staged{};
    for(std::size_t w=0;w<bins.size()&&w<GDVW_MAX_WAVES;++w){std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS>need{};for(const auto&x:bins[w]){out.wave[w].dbids[d].push_back(x.dbid);out.wave[w].work[d]+=x.work;for(std::uint8_t s:x.source)need[s]=true;}for(std::uint32_t s=0;s<layout.main_blocks.size();++s)if(need[s]&&!staged[s]){staged[s]=true;out.wave[w].new_source[d].push_back(std::uint8_t(s));out.wave[w].copy_bytes[d]+=gdpg_bytes(layout.main_blocks[s]);}}
}

static void gdvw_recount(GdvwPlan&p,const GdmsStagePlan&exact,int ngpu){
    p.eligible_copy_bytes_per_row=0;p.first_wave_copy_bytes_per_row=0;p.overlap_candidate_bytes_per_row=0;p.wave_launch_groups_per_row=0;p.phase_wave_hist.fill(0);
    auto one=[&](const GdvwPhase&ph){if(ph.waves>=0&&ph.waves<=GDVW_MAX_WAVES)p.phase_wave_hist[ph.waves]++;for(int w=0;w<ph.waves;++w){bool any=false;for(int d=0;d<ngpu;++d){p.eligible_copy_bytes_per_row+=ph.wave[w].copy_bytes[d];if(w==0)p.first_wave_copy_bytes_per_row+=ph.wave[w].copy_bytes[d];if(!ph.wave[w].dbids[d].empty())any=true;}if(any)++p.wave_launch_groups_per_row;}};
    for(const auto&ph:p.high)one(ph);for(int q=LOW_LUT_K;q>=2;--q)one(p.low[LOW_LUT_K-q]);
    p.overlap_candidate_bytes_per_row=p.eligible_copy_bytes_per_row-p.first_wave_copy_bytes_per_row;
    p.serial_p1_copy_bytes_per_row=0;std::uint32_t p1=std::uint32_t(LOW_LUT_K-1);if(p1<exact.low.size())for(int d=0;d<ngpu;++d)p.serial_p1_copy_bytes_per_row+=exact.low[p1].bytes[d]+exact.low[p1].refresh_bytes[d];
    unsigned long long expected=0;for(const auto&ph:exact.high)for(int d=0;d<ngpu;++d)expected+=ph.bytes[d];for(int q=LOW_LUT_K;q>=2;--q){auto pi=std::uint32_t(LOW_LUT_K-q);for(int d=0;d<ngpu;++d)expected+=exact.low[pi].bytes[d];}
    if(expected!=p.eligible_copy_bytes_per_row){std::cerr<<"gdvw staged-byte mismatch "<<p.eligible_copy_bytes_per_row<<'/'<<expected<<'\n';std::exit(199);}
}

struct GdvwReplan{GdvwPlan plan;double baseline_ms=0.0;double candidate_ms=0.0;unsigned changed_phases=0;};

static GdvwReplan gdvw_replan(const GdvwPlan&cur,const GdorOnlineModel&model,const StorageLayout&layout,const GdmShardHost&shard,const GpuDirectGatherHost&ordinary,const GpuDirectCrossGatherHost&cross,const GdmsStagePlan&exact,int ngpu){
    GdvwReplan out;out.plan=cur;if(!model.compute.valid||!model.topology.custom)return out;double launch_ms=gdvw_wave_group_ms();
    auto tune=[&](const GdvwPhase&old,std::uint32_t pi,bool high){GdvwPhase chosen=old;double base_phase=0.0;for(int d=0;d<ngpu;++d){auto item=high?gdaw_high_items(pi,d,layout,shard,ordinary,cross):gdaw_low_items(pi,d,layout,shard,ordinary,cross);auto bins=gdvw_bins_from_phase(old,d,item);base_phase=std::max(base_phase,gdvw_score_bins(bins,layout,shard,model.topology,d,model.compute.work_per_ms[d],launch_ms));}
        double best_phase=base_phase;GdvwPhase best=old;
        for(int k=GDVW_MIN_WAVES;k<=GDVW_MAX_WAVES;++k){GdvwPhase cand;cand.waves=k;double ph=0.0;std::array<GdvwBins,GDM_MAX_GPU> bb;for(int d=0;d<ngpu;++d){auto item=high?gdaw_high_items(pi,d,layout,shard,ordinary,cross):gdaw_low_items(pi,d,layout,shard,ordinary,cross);double s=0.0;bb[d]=gdvw_optimize_k(std::move(item),k,layout,shard,model.topology,d,model.compute.work_per_ms[d],launch_ms,&s);ph=std::max(ph,s);}if(ph+1e-6<best_phase){best_phase=ph;best=GdvwPhase{};best.waves=k;for(int d=0;d<ngpu;++d)gdvw_emit_device(best,d,bb[d],layout);}}
        out.baseline_ms+=base_phase;out.candidate_ms+=best_phase;double gain=base_phase>0.0?100.0*(base_phase-best_phase)/base_phase:0.0;if(gain>=GDOR_MIN_REPLAN_GAIN_PCT){if(best.waves!=old.waves)++out.changed_phases;return best;}out.candidate_ms+=base_phase-best_phase;return old;};
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);out.plan.high[pi]=tune(cur.high[pi],pi,true);}
    for(int p=LOW_LUT_K;p>=2;--p){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);out.plan.low[pi]=tune(cur.low[pi],pi,false);}
    gdvw_recount(out.plan,exact,ngpu);return out;
}

static __device__ __forceinline__ std::uint32_t gdvw_dbid(int phase,int wave){return D_GDVW_DBID_PTR[(std::size_t(phase)*GDVW_MAX_WAVES+wave)*GDVW_MAX_DB+blockIdx.z];}

__global__ void gdvw_high_local_kernel(int p,int phase,int wave){std::uint32_t dbid=gdvw_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock dstb=D_GDM_BLOCK_BLOCKS[dbid];if(!dstb.valid||dstb.owner!=D_GDM_DEVICE||!dstb.rows||!dstb.cols)return;std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);std::size_t oi=std::size_t(pi)*D_GDG_HIGH_PITCH+dbid;std::uint32_t a=D_GDG_HIGH_OFF[oi],b=D_GDG_HIGH_OFF[oi+1];for(std::uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){GpuDirectGatherDst rec=D_GDG_HIGH_DST[q];for(std::uint32_t lr=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<dstb.cols;lr+=std::uint32_t(gridDim.x)*blockDim.x){Count*dp=gdm_block_ptr(dstb,Code(rec.dst_rank)*dstb.cols+lr);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t loc=D_GDG_HIGH_SRC[e],sbid=gpu_direct_gather_src_block(loc);GdmBlock srcb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_main_load(sbid,Code(gpu_direct_gather_src_rank(loc))*srcb.cols+lr));}*dp=sum;}}}
__global__ void gdvw_low_local_kernel(int p,int phase,int wave){std::uint32_t dbid=gdvw_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock dstb=D_GDM_BLOCK_BLOCKS[dbid];if(!dstb.valid||dstb.owner!=D_GDM_DEVICE||!dstb.rows||!dstb.cols)return;std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);std::size_t oi=std::size_t(pi)*D_GDG_LOW_PITCH+dbid;std::uint32_t a=D_GDG_LOW_OFF[oi],b=D_GDG_LOW_OFF[oi+1],q0=a+std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x,qstep=std::uint32_t(gridDim.x)*blockDim.x;for(std::uint32_t q=q0;q<b;q+=qstep){GpuDirectGatherDst rec=D_GDG_LOW_DST[q];for(std::uint32_t hr=blockIdx.y;hr<dstb.rows;hr+=gridDim.y){Count*dp=gdm_block_ptr(dstb,Code(hr)*dstb.cols+rec.dst_rank);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t loc=D_GDG_LOW_SRC[e],sbid=gpu_direct_gather_src_block(loc);GdmBlock srcb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_main_load(sbid,Code(hr)*srcb.cols+gpu_direct_gather_src_rank(loc)));}*dp=sum;}}}
__global__ void gdvw_high_cross_kernel(int p,int phase,int wave){std::uint32_t dbid=gdvw_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock db=D_GDM_BLOCK_BLOCKS[dbid];if(!db.valid||db.owner!=D_GDM_DEVICE||!db.rows||!db.cols)return;std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);std::size_t oi=std::size_t(pi)*D_GDX_HIGH_PITCH+dbid;std::uint32_t a=D_GDX_HIGH_OFF[oi],b=D_GDX_HIGH_OFF[oi+1];for(std::uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){GpuDirectGatherDst rec=D_GDX_HIGH_DST[q];for(std::uint32_t dlr=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;dlr<db.cols;dlr+=std::uint32_t(gridDim.x)*blockDim.x){std::uint32_t dest_code=D_GDX_LOW_CODES[D_GD_LOW_ALL_OFF[db.hs]+dlr];Count*dp=gdm_block_ptr(db,Code(rec.dst_rank)*db.cols+dlr);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t op=D_GDX_HIGH_OP[e],sbid=gdx_op_block(op),shr=gdx_op_rank(op),depth=gdx_op_depth(op);GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_sum_low_preimages(dest_code,depth,sb.hs,sbid,shr));}*dp=sum;}}}
__global__ void gdvw_low_cross_kernel(int p,int phase,int wave){std::uint32_t dbid=gdvw_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock db=D_GDM_BLOCK_BLOCKS[dbid];if(!db.valid||db.owner!=D_GDM_DEVICE||!db.rows||!db.cols)return;std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);std::size_t oi=std::size_t(pi)*D_GDX_LOW_PITCH+dbid;std::uint32_t a=D_GDX_LOW_OFF[oi],b=D_GDX_LOW_OFF[oi+1];for(std::uint32_t q=a+std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=std::uint32_t(gridDim.x)*blockDim.x){GpuDirectGatherDst rec=D_GDX_LOW_DST[q];for(std::uint32_t dhr=blockIdx.y;dhr<db.rows;dhr+=gridDim.y){std::uint32_t dest_code=D_GDX_HIGH_CODES[D_GD_HIGH_ALL_OFF[db.he]+dhr];Count*dp=gdm_block_ptr(db,Code(dhr)*db.cols+rec.dst_rank);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t op=D_GDX_LOW_OP[e],sbid=gdx_op_block(op),slr=gdx_op_rank(op),depth=gdx_op_depth(op);GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_sum_high_preimages(dest_code,depth,sb.he,sbid,slr));}*dp=sum;}}}

using GdvwPeerEvents=std::array<std::array<cudaEvent_t,GDM_MAX_GPU>,GDM_MAX_GPU>;
using GdvwComputeEvents=std::array<cudaEvent_t,GDM_MAX_GPU>;
struct GdvwPipeline:GdpoPipeline{
    std::vector<std::array<GdvwPeerEvents,GDVW_MAX_WAVES>> copy_begin,copy_end;
    std::vector<std::array<GdvwComputeEvents,GDVW_MAX_WAVES>> compute_begin,compute_end;
    std::array<std::uint8_t*,GDM_MAX_GPU> dbid_meta{};
    bool meta_installed=false;
    void init(int n,const StorageLayout&layout){GdpoPipeline::init(n,layout);copy_begin.resize(GDVW_PHASES);copy_end.resize(GDVW_PHASES);compute_begin.resize(GDVW_PHASES);compute_end.resize(GDVW_PHASES);for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw init device");ck(cudaMalloc(&dbid_meta[d],GDVW_DBID_CAP),"gdvw dbid meta");const std::uint8_t*ptr=dbid_meta[d];ck(cudaMemcpyToSymbol(D_GDVW_DBID_PTR,&ptr,sizeof(ptr)),"gdvw dbid ptr");}for(int ph=0;ph<GDVW_PHASES;++ph)for(int w=0;w<GDVW_MAX_WAVES;++w)for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw event device");ck(cudaEventCreate(&compute_begin[ph][w][d]),"gdvw compute begin");ck(cudaEventCreate(&compute_end[ph][w][d]),"gdvw compute end");for(int s=0;s<ngpu;++s)if(s!=d){ck(cudaEventCreate(&copy_begin[ph][w][d][s]),"gdvw copy begin");ck(cudaEventCreate(&copy_end[ph][w][d][s]),"gdvw copy end");}}}
    void install_meta(const GdvwPlan&p){for(int d=0;d<ngpu;++d){std::vector<std::uint8_t>h(GDVW_DBID_CAP,0xff);for(int pi=0;pi<HIGH_LUT_K;++pi)for(int w=0;w<p.high[pi].waves;++w){const auto&v=p.high[pi].wave[w].dbids[d];for(std::size_t k=0;k<v.size()&&k<GDVW_MAX_DB;++k)h[(std::size_t(pi)*GDVW_MAX_WAVES+w)*GDVW_MAX_DB+k]=v[k];}for(int q=LOW_LUT_K;q>=2;--q){int pi=LOW_LUT_K-q,ph=HIGH_LUT_K+pi;for(int w=0;w<p.low[pi].waves;++w){const auto&v=p.low[pi].wave[w].dbids[d];for(std::size_t k=0;k<v.size()&&k<GDVW_MAX_DB;++k)h[(std::size_t(ph)*GDVW_MAX_WAVES+w)*GDVW_MAX_DB+k]=v[k];}}ck(cudaSetDevice(d),"gdvw meta device");ck(cudaMemcpy(dbid_meta[d],h.data(),h.size(),cudaMemcpyHostToDevice),"gdvw meta copy");}meta_installed=true;}
    void refresh_meta(const GdvwPlan&p){install_meta(p);}
    void destroy(){for(int ph=0;ph<GDVW_PHASES;++ph)for(int w=0;w<GDVW_MAX_WAVES;++w)for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw destroy event device");if(compute_begin[ph][w][d])cudaEventDestroy(compute_begin[ph][w][d]);if(compute_end[ph][w][d])cudaEventDestroy(compute_end[ph][w][d]);for(int s=0;s<ngpu;++s)if(s!=d){if(copy_begin[ph][w][d][s])cudaEventDestroy(copy_begin[ph][w][d][s]);if(copy_end[ph][w][d][s])cudaEventDestroy(copy_end[ph][w][d][s]);}}for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw destroy meta device");if(dbid_meta[d])cudaFree(dbid_meta[d]);dbid_meta[d]=nullptr;}copy_begin.clear();copy_end.clear();compute_begin.clear();compute_end.clear();GdpoPipeline::destroy();}
};

static void gdvw_stage_wave(GdvwPipeline&pipe,int phase,int wave,int ready_slot,const GdwfWave&wp,const StorageLayout&layout,const GdmShardHost&shard,Count*const*main_ptr,bool measure){for(int d=0;d<pipe.ngpu;++d){std::array<bool,GDM_MAX_GPU>used{};ck(cudaSetDevice(d),"gdvw copy device");for(std::uint8_t id:wp.new_source[d]){std::uint32_t b=id;const auto&logical=layout.main_blocks[b];const auto&physical=shard.main_blocks[b];int s=physical.owner;if(s==d)continue;std::size_t bytes=std::size_t(logical.rows)*logical.cols*sizeof(Count);if(!bytes)continue;cudaStream_t cs=pipe.copy[d][s];if(!used[s]){ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][s],0),"gdvw wait source");ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][d],0),"gdvw wait dest");if(measure)ck(cudaEventRecord(pipe.copy_begin[phase][wave][d][s],cs),"gdvw copy begin record");used[s]=true;}ck(cudaMemcpyPeerAsync(pipe.stage[d]+logical.off,d,main_ptr[s]+physical.off,s,bytes,cs),"gdvw copy");}for(int s=0;s<pipe.ngpu;++s)if(used[s]){cudaEvent_t done=pipe.copy_end[phase][wave][d][s];ck(cudaEventRecord(done,pipe.copy[d][s]),"gdvw copy end record");ck(cudaStreamWaitEvent(pipe.compute[d],done,0),"gdvw compute wait copy");}}}

static void gdvw_gather_high(GdvwPipeline&pipe,const GdvwPhase&ph,int phase,int ready_slot,const StorageLayout&layout,const GdmShardHost&shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y,bool measure){dim3 block(threads);for(int w=0;w<ph.waves;++w){gdvw_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr,measure);for(int d=0;d<pipe.ngpu;++d){unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;ck(cudaSetDevice(d),"gdvw high device");if(measure)ck(cudaEventRecord(pipe.compute_begin[phase][w][d],pipe.compute[d]),"gdvw high begin");dim3 g(grid_x,grid_y,nz);gdvw_high_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);gdvw_high_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);ck(cudaGetLastError(),"gdvw high gather");if(measure)ck(cudaEventRecord(pipe.compute_end[phase][w][d],pipe.compute[d]),"gdvw high end");}}}
static void gdvw_gather_low(GdvwPipeline&pipe,const GdvwPhase&ph,int phase,int ready_slot,const StorageLayout&layout,const GdmShardHost&shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y,bool measure){dim3 block(threads);for(int w=0;w<ph.waves;++w){gdvw_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr,measure);for(int d=0;d<pipe.ngpu;++d){unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;ck(cudaSetDevice(d),"gdvw low device");if(measure)ck(cudaEventRecord(pipe.compute_begin[phase][w][d],pipe.compute[d]),"gdvw low begin");dim3 g(grid_x,grid_y,nz);gdvw_low_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);gdvw_low_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);ck(cudaGetLastError(),"gdvw low gather");if(measure)ck(cudaEventRecord(pipe.compute_end[phase][w][d],pipe.compute[d]),"gdvw low end");}}}

static GdawCalibration gdvw_calibrate_compute(GdvwPipeline&pipe,const GdvwPlan&p,int ngpu){GdawCalibration out;std::array<double,GDM_MAX_GPU>work{},ms{};auto add=[&](int ph,const GdvwPhase&x){for(int w=0;w<x.waves;++w)for(int d=0;d<ngpu;++d)if(!x.wave[w].dbids[d].empty()&&x.wave[w].work[d]){ck(cudaSetDevice(d),"gdvw elapsed compute device");float z=0;ck(cudaEventElapsedTime(&z,pipe.compute_begin[ph][w][d],pipe.compute_end[ph][w][d]),"gdvw elapsed compute");if(z>0){work[d]+=double(x.wave[w].work[d]);ms[d]+=z;}}};for(int pi=0;pi<HIGH_LUT_K;++pi)add(pi,p.high[pi]);for(int q=LOW_LUT_K;q>=2;--q){int pi=LOW_LUT_K-q;add(HIGH_LUT_K+pi,p.low[pi]);}double sw=0,sm=0;out.min_work_per_ms=1e300;for(int d=0;d<ngpu;++d){if(ms[d]>0)out.work_per_ms[d]=work[d]/ms[d];sw+=work[d];sm+=ms[d];if(out.work_per_ms[d]>0){out.min_work_per_ms=std::min(out.min_work_per_ms,out.work_per_ms[d]);out.max_work_per_ms=std::max(out.max_work_per_ms,out.work_per_ms[d]);}}if(out.min_work_per_ms==1e300)out.min_work_per_ms=0;out.aggregate_work_per_ms=sm>0?sw/sm:0;out.valid=out.aggregate_work_per_ms>0;return out;}

static GdorCopyCalibration gdvw_calibrate_copy(GdvwPipeline&pipe,const GdvwPlan&p,const StorageLayout&layout,const GdmShardHost&shard,int ngpu){GdorCopyCalibration out;double sum=0;unsigned cnt=0;auto add=[&](int ph,const GdvwPhase&x){for(int w=0;w<x.waves;++w)for(int d=0;d<ngpu;++d){std::array<unsigned long long,GDM_MAX_GPU>bytes{};for(std::uint8_t id:x.wave[w].new_source[d]){std::uint32_t b=id;int s=shard.main_blocks[b].owner;if(s!=d)bytes[s]+=gdpg_bytes(layout.main_blocks[b]);}for(int s=0;s<ngpu;++s)if(s!=d&&bytes[s]){ck(cudaSetDevice(d),"gdvw elapsed copy device");float ms=0;ck(cudaEventElapsedTime(&ms,pipe.copy_begin[ph][w][d][s],pipe.copy_end[ph][w][d][s]),"gdvw elapsed copy");if(ms>0){double g=double(bytes[s])/(double(ms)*1e6);out.gbps[d][s]+=g;out.samples[d][s]++;}}}};for(int pi=0;pi<HIGH_LUT_K;++pi)add(pi,p.high[pi]);for(int q=LOW_LUT_K;q>=2;--q){int pi=LOW_LUT_K-q;add(HIGH_LUT_K+pi,p.low[pi]);}out.min_gbps=1e300;for(int d=0;d<ngpu;++d)for(int s=0;s<ngpu;++s)if(d!=s&&out.samples[d][s]){out.gbps[d][s]/=double(out.samples[d][s]);out.min_gbps=std::min(out.min_gbps,out.gbps[d][s]);out.max_gbps=std::max(out.max_gbps,out.gbps[d][s]);sum+=out.gbps[d][s];++cnt;}if(out.min_gbps==1e300)out.min_gbps=0;out.mean_gbps=cnt?sum/double(cnt):0;out.valid=cnt>0;return out;}

static void gdvw_enqueue_row(GdvwPipeline&pipe,const GdowOrbitPlan&orbit_plan,const GdmsStagePlan&gather_plan,const GdvwPlan&wave_plan,const StorageLayout&layout,const GdmShardHost&shard,Count*const*main_ptr,Count*const*block_ptr,int threads,int grid_x,int grid_y,int&slot,bool measure){pipe.install_meta(wave_plan);dim3 block(threads);int phase=0,ready_slot=-1;for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);const auto&oph=orbit_plan.high[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw high orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdvw high orbit");}int orbit_done=slot;pipe.fence(slot++);gdvw_gather_high(pipe,wave_plan.high[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);ready_slot=slot;pipe.fence(slot++);}for(int p=LOW_LUT_K;p>=1;--p,++phase){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);const auto&oph=orbit_plan.low[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw low orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdvw low orbit");}int orbit_done=slot;pipe.fence(slot++);if(p>1){gdvw_gather_low(pipe,wave_plan.low[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y,measure);}else{gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);unsigned nt=unsigned(layout.main_blocks.size());for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw p1 local device");dim3 g(grid_x,grid_y,nt);gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdvw p1 local");}int local_done=slot;pipe.fence(slot++);gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdvw p1 cross device");dim3 g(grid_x,grid_y,nt);gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdvw p1 cross");}}ready_slot=slot;pipe.fence(slot++);}}
