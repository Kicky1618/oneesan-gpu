#pragma once

#include "ramstream32_gpu_direct_atomicfree_multigpu_topomap.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

// v1.0: gather wavefront.  A naive k+1 prefetch is invalid because the next
// phase must observe the current phase's completed state.  Instead, keep exact
// phase ordering and split only the gather part of phases whose sources are
// immutable MAIN and destinations are BLOCKED (all HIGH and LOW p>1).  Remote
// source blocks are copied once, in four waves.  While compute consumes wave w,
// peer copy streams can stage the unique source blocks first needed by wave w+1.
// LOW p==1 remains the v0.9 full-snapshot path because it writes MAIN.

static constexpr int GDWF_WAVES = 4;
static constexpr int GDWF_PHASES = LOW_LUT_K + HIGH_LUT_K;
static constexpr int GDWF_MAX_DB = GPU_DIRECT_MAX_BLOCK_BLOCKS;
static constexpr int GDWF_DBID_CAP = GDWF_PHASES * GDWF_WAVES * GDWF_MAX_DB;

__constant__ std::uint8_t D_GDWF_DBID[GDWF_DBID_CAP];

struct GdwfWave {
    std::array<std::vector<std::uint8_t>,GDM_MAX_GPU> dbids;
    std::array<std::vector<std::uint8_t>,GDM_MAX_GPU> new_source;
    std::array<unsigned long long,GDM_MAX_GPU> work{};
    std::array<unsigned long long,GDM_MAX_GPU> copy_bytes{};
};
struct GdwfPhase { std::array<GdwfWave,GDWF_WAVES> wave; };
struct GdwfPlan {
    std::vector<GdwfPhase> high;
    std::vector<GdwfPhase> low;
    unsigned long long eligible_copy_bytes_per_row = 0;
    unsigned long long first_wave_copy_bytes_per_row = 0;
    unsigned long long overlap_candidate_bytes_per_row = 0;
    unsigned long long serial_p1_copy_bytes_per_row = 0;
    unsigned long long wave_launch_groups_per_row = 0;
};

struct GdwfDest {
    std::uint8_t dbid = 0;
    unsigned long long work = 0;
    std::vector<std::uint8_t> source;
};

static std::vector<std::uint8_t> gdwf_high_sources(
    std::uint32_t pi,std::uint32_t dbid,const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,const GdmShardHost& shard,int d
) {
    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> seen{};
    std::size_t oi=std::size_t(pi)*ordinary.high_pitch+dbid;
    for(std::uint32_t q=ordinary.high_off[oi];q<ordinary.high_off[oi+1];++q){
        const auto&r=ordinary.high_dst[q];
        for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e){
            std::uint32_t s=gdms_src_block(ordinary.high_src[e]);
            if(shard.main_blocks[s].owner!=d)seen[s]=true;
        }
    }
    oi=std::size_t(pi)*cross.high_pitch+dbid;
    for(std::uint32_t q=cross.high_off[oi];q<cross.high_off[oi+1];++q){
        const auto&r=cross.high_dst[q];
        for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e){
            std::uint32_t s=gdms_cross_block(cross.high_op[e]);
            if(shard.main_blocks[s].owner!=d)seen[s]=true;
        }
    }
    std::vector<std::uint8_t> out;
    for(std::uint32_t s=0;s<GPU_DIRECT_MAX_MAIN_BLOCKS;++s)if(seen[s])out.push_back(std::uint8_t(s));
    return out;
}

static std::vector<std::uint8_t> gdwf_low_sources(
    std::uint32_t pi,std::uint32_t dbid,const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,const GdmShardHost& shard,int d
) {
    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> seen{};
    std::size_t oi=std::size_t(pi)*ordinary.low_pitch+dbid;
    for(std::uint32_t q=ordinary.low_off[oi];q<ordinary.low_off[oi+1];++q){
        const auto&r=ordinary.low_dst[q];
        for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e){
            std::uint32_t s=gdms_src_block(ordinary.low_src[e]);
            if(shard.main_blocks[s].owner!=d)seen[s]=true;
        }
    }
    oi=std::size_t(pi)*cross.low_pitch+dbid;
    for(std::uint32_t q=cross.low_off[oi];q<cross.low_off[oi+1];++q){
        const auto&r=cross.low_dst[q];
        for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e){
            std::uint32_t s=gdms_cross_block(cross.low_op[e]);
            if(shard.main_blocks[s].owner!=d)seen[s]=true;
        }
    }
    std::vector<std::uint8_t> out;
    for(std::uint32_t s=0;s<GPU_DIRECT_MAX_MAIN_BLOCKS;++s)if(seen[s])out.push_back(std::uint8_t(s));
    return out;
}

static unsigned long long gdwf_high_work(
    std::uint32_t pi,std::uint32_t dbid,const StorageLayout& layout,
    const GpuDirectGatherHost& ordinary,const GpuDirectCrossGatherHost& cross
) {
    unsigned long long z=0; const auto& b=layout.block_blocks[dbid];
    std::size_t oi=std::size_t(pi)*ordinary.high_pitch+dbid;
    for(std::uint32_t q=ordinary.high_off[oi];q<ordinary.high_off[oi+1];++q)z+=static_cast<unsigned long long>(ordinary.high_dst[q].edge_count)*b.cols;
    oi=std::size_t(pi)*cross.high_pitch+dbid;
    for(std::uint32_t q=cross.high_off[oi];q<cross.high_off[oi+1];++q)z+=2ull*cross.high_dst[q].edge_count*b.cols;
    return z;
}
static unsigned long long gdwf_low_work(
    std::uint32_t pi,std::uint32_t dbid,const StorageLayout& layout,
    const GpuDirectGatherHost& ordinary,const GpuDirectCrossGatherHost& cross
) {
    unsigned long long z=0; const auto& b=layout.block_blocks[dbid];
    std::size_t oi=std::size_t(pi)*ordinary.low_pitch+dbid;
    for(std::uint32_t q=ordinary.low_off[oi];q<ordinary.low_off[oi+1];++q)z+=static_cast<unsigned long long>(ordinary.low_dst[q].edge_count)*b.rows;
    oi=std::size_t(pi)*cross.low_pitch+dbid;
    for(std::uint32_t q=cross.low_off[oi];q<cross.low_off[oi+1];++q)z+=2ull*cross.low_dst[q].edge_count*b.rows;
    return z;
}

static void gdwf_assign_device(
    GdwfPhase& ph,int d,std::vector<GdwfDest> item,const StorageLayout& layout
) {
    std::sort(item.begin(),item.end(),[](const GdwfDest&a,const GdwfDest&b){return a.work>b.work;});
    std::array<std::vector<GdwfDest>,GDWF_WAVES> bin;
    std::array<unsigned long long,GDWF_WAVES> work{};
    for(auto& x:item){int w=0;for(int k=1;k<GDWF_WAVES;++k)if(work[k]<work[w])w=k;work[w]+=x.work;bin[w].push_back(std::move(x));}

    std::array<unsigned long long,GDWF_WAVES> union_bytes{};
    for(int w=0;w<GDWF_WAVES;++w){std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> seen{};for(const auto&x:bin[w])for(auto s:x.source)seen[s]=true;for(std::uint32_t s=0;s<GPU_DIRECT_MAX_MAIN_BLOCKS;++s)if(seen[s]&&s<layout.main_blocks.size())union_bytes[w]+=gdpg_bytes(layout.main_blocks[s]);}
    std::array<int,GDWF_WAVES> order{};std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){bool ea=bin[a].empty(),eb=bin[b].empty();if(ea!=eb)return !ea;if(union_bytes[a]!=union_bytes[b])return union_bytes[a]<union_bytes[b];return work[a]>work[b];});

    std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> staged{};
    for(int nw=0;nw<GDWF_WAVES;++nw){int ow=order[nw];auto& dst=ph.wave[nw];std::array<bool,GPU_DIRECT_MAX_MAIN_BLOCKS> need{};for(const auto&x:bin[ow]){dst.dbids[d].push_back(x.dbid);dst.work[d]+=x.work;for(auto s:x.source)need[s]=true;}for(std::uint32_t s=0;s<GPU_DIRECT_MAX_MAIN_BLOCKS;++s)if(need[s]&&!staged[s]){staged[s]=true;dst.new_source[d].push_back(std::uint8_t(s));if(s<layout.main_blocks.size())dst.copy_bytes[d]+=gdpg_bytes(layout.main_blocks[s]);}}
}

static GdwfPlan build_gdwf_plan(
    const StorageLayout& layout,const GdmShardHost& shard,
    const GpuDirectGatherHost& ordinary,const GpuDirectCrossGatherHost& cross,
    const GdmsStagePlan& exact,int ngpu
) {
    GdwfPlan out;out.high.resize(HIGH_LUT_K);out.low.resize(LOW_LUT_K);
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);for(int d=0;d<ngpu;++d){std::vector<GdwfDest> item;for(std::uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){const auto&b=layout.block_blocks[dbid];if(!b.valid||!b.rows||!b.cols||shard.block_blocks[dbid].owner!=d)continue;GdwfDest x;x.dbid=std::uint8_t(dbid);x.work=gdwf_high_work(pi,dbid,layout,ordinary,cross);x.source=gdwf_high_sources(pi,dbid,ordinary,cross,shard,d);item.push_back(std::move(x));}gdwf_assign_device(out.high[pi],d,std::move(item),layout);}}
    for(int p=LOW_LUT_K;p>=2;--p){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);for(int d=0;d<ngpu;++d){std::vector<GdwfDest> item;for(std::uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){const auto&b=layout.block_blocks[dbid];if(!b.valid||!b.rows||!b.cols||shard.block_blocks[dbid].owner!=d)continue;GdwfDest x;x.dbid=std::uint8_t(dbid);x.work=gdwf_low_work(pi,dbid,layout,ordinary,cross);x.source=gdwf_low_sources(pi,dbid,ordinary,cross,shard,d);item.push_back(std::move(x));}gdwf_assign_device(out.low[pi],d,std::move(item),layout);}}

    auto tally=[&](const std::vector<GdwfPhase>& v){for(const auto&ph:v){for(int w=0;w<GDWF_WAVES;++w){bool any=false;for(int d=0;d<ngpu;++d){out.eligible_copy_bytes_per_row+=ph.wave[w].copy_bytes[d];if(w==0)out.first_wave_copy_bytes_per_row+=ph.wave[w].copy_bytes[d];if(!ph.wave[w].dbids[d].empty())any=true;}if(any)++out.wave_launch_groups_per_row;}}};
    tally(out.high);for(int p=LOW_LUT_K;p>=2;--p){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);for(int w=0;w<GDWF_WAVES;++w){bool any=false;for(int d=0;d<ngpu;++d){out.eligible_copy_bytes_per_row+=out.low[pi].wave[w].copy_bytes[d];if(w==0)out.first_wave_copy_bytes_per_row+=out.low[pi].wave[w].copy_bytes[d];if(!out.low[pi].wave[w].dbids[d].empty())any=true;}if(any)++out.wave_launch_groups_per_row;}}
    out.overlap_candidate_bytes_per_row=out.eligible_copy_bytes_per_row-out.first_wave_copy_bytes_per_row;
    std::uint32_t p1pi=std::uint32_t(LOW_LUT_K-1);if(p1pi<exact.low.size()){for(int d=0;d<ngpu;++d)out.serial_p1_copy_bytes_per_row+=exact.low[p1pi].bytes[d]+exact.low[p1pi].refresh_bytes[d];}
    unsigned long long expected=0;for(const auto&ph:exact.high)for(int d=0;d<ngpu;++d)expected+=ph.bytes[d];for(int p=LOW_LUT_K;p>=2;--p){auto pi=std::uint32_t(LOW_LUT_K-p);for(int d=0;d<ngpu;++d)expected+=exact.low[pi].bytes[d];}
    if(expected!=out.eligible_copy_bytes_per_row){std::cerr<<"gdwf staged-byte mismatch "<<out.eligible_copy_bytes_per_row<<'/'<<expected<<'\n';std::exit(195);}return out;
}

static __device__ __forceinline__ std::uint32_t gdwf_dbid(int phase,int wave){return D_GDWF_DBID[(phase*GDWF_WAVES+wave)*GDWF_MAX_DB+blockIdx.z];}

__global__ void gdwf_high_local_kernel(int p,int phase,int wave){
    std::uint32_t dbid=gdwf_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock dstb=D_GDM_BLOCK_BLOCKS[dbid];if(!dstb.valid||dstb.owner!=D_GDM_DEVICE||!dstb.rows||!dstb.cols)return;std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);std::size_t oi=std::size_t(pi)*D_GDG_HIGH_PITCH+dbid;std::uint32_t a=D_GDG_HIGH_OFF[oi],b=D_GDG_HIGH_OFF[oi+1];for(std::uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){GpuDirectGatherDst rec=D_GDG_HIGH_DST[q];for(std::uint32_t lr=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<dstb.cols;lr+=std::uint32_t(gridDim.x)*blockDim.x){Count*dp=gdm_block_ptr(dstb,Code(rec.dst_rank)*dstb.cols+lr);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t loc=D_GDG_HIGH_SRC[e],sbid=gpu_direct_gather_src_block(loc);GdmBlock srcb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_main_load(sbid,Code(gpu_direct_gather_src_rank(loc))*srcb.cols+lr));}*dp=sum;}}
}
__global__ void gdwf_low_local_kernel(int p,int phase,int wave){
    std::uint32_t dbid=gdwf_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock dstb=D_GDM_BLOCK_BLOCKS[dbid];if(!dstb.valid||dstb.owner!=D_GDM_DEVICE||!dstb.rows||!dstb.cols)return;std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);std::size_t oi=std::size_t(pi)*D_GDG_LOW_PITCH+dbid;std::uint32_t a=D_GDG_LOW_OFF[oi],b=D_GDG_LOW_OFF[oi+1],q0=a+std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x,qstep=std::uint32_t(gridDim.x)*blockDim.x;for(std::uint32_t q=q0;q<b;q+=qstep){GpuDirectGatherDst rec=D_GDG_LOW_DST[q];for(std::uint32_t hr=blockIdx.y;hr<dstb.rows;hr+=gridDim.y){Count*dp=gdm_block_ptr(dstb,Code(hr)*dstb.cols+rec.dst_rank);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t loc=D_GDG_LOW_SRC[e],sbid=gpu_direct_gather_src_block(loc);GdmBlock srcb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_main_load(sbid,Code(hr)*srcb.cols+gpu_direct_gather_src_rank(loc)));}*dp=sum;}}
}
__global__ void gdwf_high_cross_kernel(int p,int phase,int wave){
    std::uint32_t dbid=gdwf_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock db=D_GDM_BLOCK_BLOCKS[dbid];if(!db.valid||db.owner!=D_GDM_DEVICE||!db.rows||!db.cols)return;std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);std::size_t oi=std::size_t(pi)*D_GDX_HIGH_PITCH+dbid;std::uint32_t a=D_GDX_HIGH_OFF[oi],b=D_GDX_HIGH_OFF[oi+1];for(std::uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){GpuDirectGatherDst rec=D_GDX_HIGH_DST[q];for(std::uint32_t dlr=std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;dlr<db.cols;dlr+=std::uint32_t(gridDim.x)*blockDim.x){std::uint32_t dest_code=D_GDX_LOW_CODES[D_GD_LOW_ALL_OFF[db.hs]+dlr];Count*dp=gdm_block_ptr(db,Code(rec.dst_rank)*db.cols+dlr);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t op=D_GDX_HIGH_OP[e],sbid=gdx_op_block(op),shr=gdx_op_rank(op),depth=gdx_op_depth(op);GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_sum_low_preimages(dest_code,depth,sb.hs,sbid,shr));}*dp=sum;}}
}
__global__ void gdwf_low_cross_kernel(int p,int phase,int wave){
    std::uint32_t dbid=gdwf_dbid(phase,wave);if(dbid>=D_GD_BLOCK_NBLOCKS)return;GdmBlock db=D_GDM_BLOCK_BLOCKS[dbid];if(!db.valid||db.owner!=D_GDM_DEVICE||!db.rows||!db.cols)return;std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);std::size_t oi=std::size_t(pi)*D_GDX_LOW_PITCH+dbid;std::uint32_t a=D_GDX_LOW_OFF[oi],b=D_GDX_LOW_OFF[oi+1];for(std::uint32_t q=a+std::uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=std::uint32_t(gridDim.x)*blockDim.x){GpuDirectGatherDst rec=D_GDX_LOW_DST[q];for(std::uint32_t dhr=blockIdx.y;dhr<db.rows;dhr+=gridDim.y){std::uint32_t dest_code=D_GDX_HIGH_CODES[D_GD_HIGH_ALL_OFF[db.he]+dhr];Count*dp=gdm_block_ptr(db,Code(dhr)*db.cols+rec.dst_rank);Count sum=*dp;for(std::uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){std::uint32_t op=D_GDX_LOW_OP[e],sbid=gdx_op_block(op),slr=gdx_op_rank(op),depth=gdx_op_depth(op);GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdms_sum_high_preimages(dest_code,depth,sb.he,sbid,slr));}*dp=sum;}}
}

using GdwfPeerEvent=std::array<std::array<cudaEvent_t,GDM_MAX_GPU>,GDM_MAX_GPU>;
struct GdwfPipeline: GdpoPipeline {
    std::vector<std::array<GdwfPeerEvent,GDWF_WAVES>> wave_done;
    bool meta_installed=false;
    void init(int n,const StorageLayout& layout){GdpoPipeline::init(n,layout);wave_done.resize(GDWF_PHASES);for(int ph=0;ph<GDWF_PHASES;++ph)for(int w=0;w<GDWF_WAVES;++w)for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdwf event device");for(int s=0;s<ngpu;++s)if(s!=d)ck(cudaEventCreateWithFlags(&wave_done[ph][w][d][s],cudaEventDisableTiming),"gdwf wave event");}}
    void install_meta(const GdwfPlan& plan){if(meta_installed)return;for(int d=0;d<ngpu;++d){std::array<std::uint8_t,GDWF_DBID_CAP> h{};h.fill(0xff);for(int pi=0;pi<HIGH_LUT_K;++pi)for(int w=0;w<GDWF_WAVES;++w){const auto&v=plan.high[pi].wave[w].dbids[d];for(std::size_t k=0;k<v.size()&&k<GDWF_MAX_DB;++k)h[(pi*GDWF_WAVES+w)*GDWF_MAX_DB+k]=v[k];}for(int p=LOW_LUT_K;p>=2;--p){int pi=LOW_LUT_K-p,ph=HIGH_LUT_K+pi;for(int w=0;w<GDWF_WAVES;++w){const auto&v=plan.low[pi].wave[w].dbids[d];for(std::size_t k=0;k<v.size()&&k<GDWF_MAX_DB;++k)h[(ph*GDWF_WAVES+w)*GDWF_MAX_DB+k]=v[k];}}ck(cudaSetDevice(d),"gdwf meta device");ck(cudaMemcpyToSymbol(D_GDWF_DBID,h.data(),h.size()*sizeof(h[0])),"gdwf dbids");}meta_installed=true;}
    void destroy(){for(int ph=0;ph<GDWF_PHASES;++ph)for(int w=0;w<GDWF_WAVES;++w)for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdwf destroy event device");for(int s=0;s<ngpu;++s)if(s!=d&&wave_done[ph][w][d][s])cudaEventDestroy(wave_done[ph][w][d][s]);}wave_done.clear();GdpoPipeline::destroy();}
};

static void gdwf_stage_wave(
    GdwfPipeline& pipe,int phase,int wave,int ready_slot,const GdwfWave& wp,
    const StorageLayout& layout,const GdmShardHost& shard,Count*const*main_ptr
){
    for(int d=0;d<pipe.ngpu;++d){std::array<bool,GDM_MAX_GPU>used{};ck(cudaSetDevice(d),"gdwf copy device");for(std::uint8_t id:wp.new_source[d]){std::uint32_t b=id;const auto&logical=layout.main_blocks[b];const auto&physical=shard.main_blocks[b];int s=physical.owner;if(s==d)continue;cudaStream_t cs=pipe.copy[d][s];if(!used[s]){ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][s],0),"gdwf wait source");ck(cudaStreamWaitEvent(cs,pipe.fence_event[std::size_t(ready_slot)][d],0),"gdwf wait dest");used[s]=true;}std::size_t bytes=std::size_t(logical.rows)*logical.cols*sizeof(Count);if(bytes)ck(cudaMemcpyPeerAsync(pipe.stage[d]+logical.off,d,main_ptr[s]+physical.off,s,bytes,cs),"gdwf wave copy");}for(int s=0;s<pipe.ngpu;++s)if(used[s]){cudaEvent_t e=pipe.wave_done[phase][wave][d][s];ck(cudaEventRecord(e,pipe.copy[d][s]),"gdwf wave done");ck(cudaStreamWaitEvent(pipe.compute[d],e,0),"gdwf compute wait wave");}}
}

static void gdwf_gather_high(
    GdwfPipeline&pipe,const GdwfPhase&ph,int phase,int ready_slot,const StorageLayout&layout,
    const GdmShardHost&shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y
){
    dim3 block(threads);for(int w=0;w<GDWF_WAVES;++w){gdwf_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;ck(cudaSetDevice(d),"gdwf high device");dim3 g(grid_x,grid_y,nz);gdwf_high_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);gdwf_high_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);ck(cudaGetLastError(),"gdwf high gather");}}
}
static void gdwf_gather_low(
    GdwfPipeline&pipe,const GdwfPhase&ph,int phase,int ready_slot,const StorageLayout&layout,
    const GdmShardHost&shard,Count*const*main_ptr,int p,int threads,int grid_x,int grid_y
){
    dim3 block(threads);for(int w=0;w<GDWF_WAVES;++w){gdwf_stage_wave(pipe,phase,w,ready_slot,ph.wave[w],layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){unsigned nz=unsigned(ph.wave[w].dbids[d].size());if(!nz)continue;ck(cudaSetDevice(d),"gdwf low device");dim3 g(grid_x,grid_y,nz);gdwf_low_local_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);gdwf_low_cross_kernel<<<g,block,0,pipe.compute[d]>>>(p,phase,w);ck(cudaGetLastError(),"gdwf low gather");}}
}

static void gdwf_enqueue_row(
    GdwfPipeline&pipe,const GdowOrbitPlan&orbit_plan,const GdmsStagePlan&gather_plan,
    const GdwfPlan&wave_plan,const StorageLayout&layout,const GdmShardHost&shard,
    Count*const*main_ptr,Count*const*block_ptr,int threads,int grid_x,int grid_y,int&slot
){
    pipe.install_meta(wave_plan);dim3 block(threads);int phase=0,ready_slot=-1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p,++phase){std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);const auto&oph=orbit_plan.high[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdwf high orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_high_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdwf high orbit");}int orbit_done=slot;pipe.fence(slot++);gdwf_gather_high(pipe,wave_plan.high[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y);ready_slot=slot;pipe.fence(slot++);}
    for(int p=LOW_LUT_K;p>=1;--p,++phase){std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);const auto&oph=orbit_plan.low[pi];gdpo_stage_orbit(pipe,phase,ready_slot,oph.deps,layout,shard,main_ptr,block_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdwf low orbit device");dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size()));gdow_low_orbit_kernel<<<g,block,0,pipe.compute[d]>>>(p,oph.active_source_mask[d]);ck(cudaGetLastError(),"gdwf low orbit");}int orbit_done=slot;pipe.fence(slot++);if(p>1){gdwf_gather_low(pipe,wave_plan.low[pi],phase,orbit_done,layout,shard,main_ptr,p,threads,grid_x,grid_y);}else{gdmp_stage_sources(pipe,orbit_done,gather_plan.low[pi].source,layout,shard,main_ptr);unsigned nt=unsigned(layout.main_blocks.size());for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdwf p1 local device");dim3 g(grid_x,grid_y,nt);gdms_low_local_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdwf p1 local");}int local_done=slot;pipe.fence(slot++);gdmp_stage_sources(pipe,local_done,gather_plan.low[pi].cross_refresh,layout,shard,main_ptr);for(int d=0;d<pipe.ngpu;++d){ck(cudaSetDevice(d),"gdwf p1 cross device");dim3 g(grid_x,grid_y,nt);gdms_low_cross_gather_kernel<<<g,block,0,pipe.compute[d]>>>(p);ck(cudaGetLastError(),"gdwf p1 cross");}}ready_slot=slot;pipe.fence(slot++);}
}
