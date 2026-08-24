#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

// Fully atomic-free CROSS closure backend.
//
// A CROSS closure is a tensor product of one active-half closure operation and
// one inactive-half bracket repair. Expanding that product into full-state CSR
// would be enormous. Instead, group only the active-half operations by their
// destination and invert the inactive-half repair algorithmically. The inverse
// has at most seven source codes at W=28. A compact ternary-code -> global-rank
// table (3^K entries) ranks each generated inverse candidate, replacing the
// much larger depth x source forward-rank tables used by v0.1.

static constexpr uint32_t GPU_DIRECT_CROSS_OP_RANK_MASK = (1u << 20) - 1u;
static constexpr int GPU_DIRECT_CROSS_OP_BLOCK_SHIFT = 20;
static constexpr int GPU_DIRECT_CROSS_OP_DEPTH_SHIFT = 26;
static constexpr uint32_t GPU_DIRECT_CROSS_OP_BLOCK_MASK = 0x3fu;
static constexpr uint32_t GPU_DIRECT_CROSS_OP_DEPTH_MASK = 0x0fu;
static constexpr uint32_t GPU_DIRECT_CROSS_DIRECT_INVALID = 0xffffffffu;

using GpuDirectCrossGatherOp = uint32_t;
static inline GpuDirectCrossGatherOp gpu_direct_cross_op_pack(
    uint32_t sbid, uint32_t srank, uint32_t depth
) {
    if (sbid > GPU_DIRECT_CROSS_OP_BLOCK_MASK
        || srank > GPU_DIRECT_CROSS_OP_RANK_MASK
        || !depth || depth > GPU_DIRECT_CROSS_OP_DEPTH_MASK) {
        std::cerr << "gpu direct cross op encoding overflow block=" << sbid
                  << " rank=" << srank << " depth=" << depth << '\n';
        std::exit(160);
    }
    return srank | (sbid << GPU_DIRECT_CROSS_OP_BLOCK_SHIFT)
        | (depth << GPU_DIRECT_CROSS_OP_DEPTH_SHIFT);
}

static uint32_t gpu_direct_ternary_key_host(uint32_t code, int K) {
    uint32_t key = 0;
    for (int p = K - 1; p >= 0; --p) {
        uint32_t v = (code >> (2 * p)) & 3u;
        uint32_t d = v == uint32_t(R) ? 1u : (v == uint32_t(::L) ? 2u : 0u);
        key = key * 3u + d;
    }
    return key;
}
static uint32_t gpu_direct_pow3(int K) {
    uint32_t z = 1;
    for (int i=0;i<K;++i) z *= 3u;
    return z;
}

struct GpuDirectCrossGatherHost {
    std::vector<GpuDirectGatherDst> low_dst;
    std::vector<GpuDirectCrossGatherOp> low_op;
    std::vector<uint32_t> low_off;
    uint32_t low_pitch = GPU_DIRECT_MAX_MAIN_BLOCKS + 1;

    std::vector<GpuDirectGatherDst> high_dst;
    std::vector<GpuDirectCrossGatherOp> high_op;
    std::vector<uint32_t> high_off;
    uint32_t high_pitch = GPU_DIRECT_MAX_BLOCK_BLOCKS + 1;

    std::vector<uint32_t> high_codes;
    std::vector<uint32_t> low_codes;
    std::vector<uint32_t> high_direct;
    std::vector<uint32_t> low_direct;

    size_t bytes() const {
        return low_dst.size()*sizeof(GpuDirectGatherDst)
            + low_op.size()*sizeof(GpuDirectCrossGatherOp)
            + low_off.size()*sizeof(uint32_t)
            + high_dst.size()*sizeof(GpuDirectGatherDst)
            + high_op.size()*sizeof(GpuDirectCrossGatherOp)
            + high_off.size()*sizeof(uint32_t)
            + high_codes.size()*sizeof(uint32_t)
            + low_codes.size()*sizeof(uint32_t)
            + high_direct.size()*sizeof(uint32_t)
            + low_direct.size()*sizeof(uint32_t);
    }
};

static GpuDirectCrossGatherHost build_gpu_direct_cross_gather(
    const StorageFactorHost& storage, const StorageLayout& layout,
    const LowDescHost& lowdesc, const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect
) {
    GpuDirectCrossGatherHost out;
    out.low_off.assign(size_t(LOW_LUT_K)*out.low_pitch,0);
    out.high_off.assign(size_t(HIGH_LUT_K)*out.high_pitch,0);

    using Edge = std::pair<uint32_t,GpuDirectCrossGatherOp>;

    for (int p=LOW_LUT_K;p>=1;--p) {
        uint32_t pi=uint32_t(LOW_LUT_K-p);
        bool target_main=p==1;
        uint32_t target_blocks=target_main
            ? uint32_t(layout.main_blocks.size())
            : uint32_t(layout.block_blocks.size());
        std::vector<std::vector<Edge>> by_dst(target_blocks);
        for (uint32_t sbid=0;sbid<uint32_t(layout.main_blocks.size());++sbid) {
            const StorageBlock& sb=layout.main_blocks[sbid];
            if(!sb.valid||!sb.rows||!sb.cols) continue;
            for(uint32_t lr=0;lr<sb.cols;++lr) {
                uint64_t ow=loworbit.rec[size_t(pi)*loworbit.main_total
                    +loworbit.main_base[sbid]+lr];
                if(cpu_orbit_kind(ow)!=CPU_ORBIT_CLOSURE) continue;
                uint32_t desc=lowdesc.main_desc[size_t(pi)*lowdesc.main_total
                    +lowdesc.main_base[sbid]+lr];
                if(cpu_low_kind(desc)!=LOWDESC_CROSS) continue;
                uint32_t dbid=cpu_low_block(desc), dlr=cpu_low_lr(desc);
                uint32_t depth=cpu_low_depth(desc);
                if(dbid>=target_blocks) std::exit(161);
                const StorageBlock& db=target_main?layout.main_blocks[dbid]:layout.block_blocks[dbid];
                if(dlr>=db.cols) std::exit(162);
                by_dst[dbid].push_back({dlr,gpu_direct_cross_op_pack(sbid,lr,depth)});
            }
        }
        for(uint32_t dbid=0;dbid<target_blocks;++dbid) {
            out.low_off[size_t(pi)*out.low_pitch+dbid]=uint32_t(out.low_dst.size());
            auto& es=by_dst[dbid];
            std::sort(es.begin(),es.end(),[](const Edge&a,const Edge&b){
                if(a.first!=b.first)return a.first<b.first;return a.second<b.second;
            });
            size_t i=0;
            while(i<es.size()) {
                size_t j=i+1;while(j<es.size()&&es[j].first==es[i].first)++j;
                uint32_t begin=uint32_t(out.low_op.size());
                for(size_t k=i;k<j;++k)out.low_op.push_back(es[k].second);
                out.low_dst.push_back({es[i].first,begin,uint32_t(j-i)});
                i=j;
            }
        }
        out.low_off[size_t(pi)*out.low_pitch+target_blocks]=uint32_t(out.low_dst.size());
    }

    for(uint32_t pi=0;pi<uint32_t(HIGH_LUT_K);++pi) {
        uint32_t target_blocks=uint32_t(layout.block_blocks.size());
        std::vector<std::vector<Edge>> by_dst(target_blocks);
        for(uint32_t sbid=0;sbid<highdirect.nblocks;++sbid) {
            auto [a,b]=cpu_high_direct_range(
                highdirect.closure_off.cross,highdirect.nblocks,pi,sbid);
            for(uint32_t q=a;q<b;++q) {
                const CpuHighClosureOp& op=highdirect.closure_ops.cross[q];
                uint32_t dbid=cpu_high_desc_block(op.desc);
                uint32_t dhr=cpu_high_desc_rank(op.desc);
                uint32_t depth=cpu_high_desc_depth(op.desc);
                if(dbid>=target_blocks||dhr>=layout.block_blocks[dbid].rows) std::exit(163);
                by_dst[dbid].push_back({dhr,
                    gpu_direct_cross_op_pack(sbid,op.src_hr,depth)});
            }
        }
        for(uint32_t dbid=0;dbid<target_blocks;++dbid) {
            out.high_off[size_t(pi)*out.high_pitch+dbid]=uint32_t(out.high_dst.size());
            auto& es=by_dst[dbid];
            std::sort(es.begin(),es.end(),[](const Edge&a,const Edge&b){
                if(a.first!=b.first)return a.first<b.first;return a.second<b.second;
            });
            size_t i=0;
            while(i<es.size()) {
                size_t j=i+1;while(j<es.size()&&es[j].first==es[i].first)++j;
                uint32_t begin=uint32_t(out.high_op.size());
                for(size_t k=i;k<j;++k)out.high_op.push_back(es[k].second);
                out.high_dst.push_back({es[i].first,begin,uint32_t(j-i)});
                i=j;
            }
        }
        out.high_off[size_t(pi)*out.high_pitch+target_blocks]=uint32_t(out.high_dst.size());
    }

    out.high_codes=storage.high_all_codes;
    out.low_codes=storage.low_all_codes;
    out.high_direct.assign(gpu_direct_pow3(HIGH_LUT_K),GPU_DIRECT_CROSS_DIRECT_INVALID);
    out.low_direct.assign(gpu_direct_pow3(LOW_LUT_K),GPU_DIRECT_CROSS_DIRECT_INVALID);
    for(uint32_t i=0;i<uint32_t(out.high_codes.size());++i) {
        uint32_t key=gpu_direct_ternary_key_host(out.high_codes[i],HIGH_LUT_K);
        if(out.high_direct[key]!=GPU_DIRECT_CROSS_DIRECT_INVALID) std::exit(164);
        out.high_direct[key]=i;
    }
    for(uint32_t i=0;i<uint32_t(out.low_codes.size());++i) {
        uint32_t key=gpu_direct_ternary_key_host(out.low_codes[i],LOW_LUT_K);
        if(out.low_direct[key]!=GPU_DIRECT_CROSS_DIRECT_INVALID) std::exit(165);
        out.low_direct[key]=i;
    }

    std::cerr << "gpu_direct_cross_gather"
              << " low_dst="<<out.low_dst.size()<<" low_ops="<<out.low_op.size()
              << " high_dst="<<out.high_dst.size()<<" high_ops="<<out.high_op.size()
              << " high_direct_mib="<<double(out.high_direct.size()*4)/(1<<20)
              << " low_direct_mib="<<double(out.low_direct.size()*4)/(1<<20)
              << " total_mib="<<double(out.bytes())/(1<<20)<<'\n';
    return out;
}

__constant__ GpuDirectGatherDst* D_GDX_LOW_DST;
__constant__ GpuDirectCrossGatherOp* D_GDX_LOW_OP;
__constant__ uint32_t* D_GDX_LOW_OFF;
__constant__ uint32_t D_GDX_LOW_PITCH;
__constant__ GpuDirectGatherDst* D_GDX_HIGH_DST;
__constant__ GpuDirectCrossGatherOp* D_GDX_HIGH_OP;
__constant__ uint32_t* D_GDX_HIGH_OFF;
__constant__ uint32_t D_GDX_HIGH_PITCH;
__constant__ uint32_t* D_GDX_HIGH_CODES;
__constant__ uint32_t* D_GDX_LOW_CODES;
__constant__ uint32_t* D_GDX_HIGH_DIRECT;
__constant__ uint32_t* D_GDX_LOW_DIRECT;

__device__ __forceinline__ uint32_t gdx_op_rank(uint32_t x){return x&GPU_DIRECT_CROSS_OP_RANK_MASK;}
__device__ __forceinline__ uint32_t gdx_op_block(uint32_t x){return (x>>GPU_DIRECT_CROSS_OP_BLOCK_SHIFT)&GPU_DIRECT_CROSS_OP_BLOCK_MASK;}
__device__ __forceinline__ uint32_t gdx_op_depth(uint32_t x){return (x>>GPU_DIRECT_CROSS_OP_DEPTH_SHIFT)&GPU_DIRECT_CROSS_OP_DEPTH_MASK;}

template<int K>
__device__ __forceinline__ uint32_t gdx_ternary_key(uint32_t code){
    uint32_t key=0;
#pragma unroll
    for(int p=K-1;p>=0;--p){
        uint32_t v=(code>>(2*p))&3u;
        uint32_t d=v==uint32_t(R)?1u:(v==uint32_t(::L)?2u:0u);
        key=key*3u+d;
    }
    return key;
}

// Inverse of cpu_low_flip_high: destination has R at the position that was L
// in the source. Before that position the forward scan must remain positive.
__device__ __forceinline__ Count gdx_sum_high_preimages(
    const Count* mainv, uint32_t dest_code, uint32_t depth,
    uint32_t source_he, uint32_t source_bid, uint32_t source_lr, uint32_t hr_dummy
){
    (void)hr_dummy;
    Count sum=0; int s=int(depth);
#pragma unroll
    for(int pos=0;pos<HIGH_LUT_K;++pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(::L)){
            if(s==1) break;
            --s;
        }else if(v==uint32_t(R)){
            if(s==1){
                uint32_t z=3u<<(2*pos);
                uint32_t src_code=(dest_code&~z)|(uint32_t(::L)<<(2*pos));
                uint32_t gi=D_GDX_HIGH_DIRECT[gdx_ternary_key<HIGH_LUT_K>(src_code)];
                uint32_t a=D_GD_HIGH_ALL_OFF[source_he],b=D_GD_HIGH_ALL_OFF[source_he+1];
                if(gi>=a&&gi<b){
                    uint32_t shr=gi-a;
                    GpuDirectBlock sb=D_GD_MAIN_BLOCKS[source_bid];
                    sum=gpu_direct_add(sum,mainv[sb.off+Code(shr)*sb.cols+source_lr]);
                }
            }
            ++s;
        }
    }
    return sum;
}

// Inverse of cpu_high_flip_low: destination has L at the position that was R
// in the source. Scan from the center boundary toward position zero.
__device__ __forceinline__ Count gdx_sum_low_preimages(
    const Count* mainv, uint32_t dest_code, uint32_t depth,
    uint32_t source_hs, uint32_t source_bid, uint32_t source_hr, uint32_t lr_dummy
){
    (void)lr_dummy;
    Count sum=0; int s=int(depth);
#pragma unroll
    for(int pos=LOW_LUT_K-1;pos>=0;--pos){
        uint32_t v=(dest_code>>(2*pos))&3u;
        if(v==uint32_t(R)){
            if(s==1) break;
            --s;
        }else if(v==uint32_t(::L)){
            if(s==1){
                uint32_t z=3u<<(2*pos);
                uint32_t src_code=(dest_code&~z)|(uint32_t(R)<<(2*pos));
                uint32_t gi=D_GDX_LOW_DIRECT[gdx_ternary_key<LOW_LUT_K>(src_code)];
                uint32_t a=D_GD_LOW_ALL_OFF[source_hs],b=D_GD_LOW_ALL_OFF[source_hs+1];
                if(gi>=a&&gi<b){
                    uint32_t slr=gi-a;
                    GpuDirectBlock sb=D_GD_MAIN_BLOCKS[source_bid];
                    sum=gpu_direct_add(sum,mainv[sb.off+Code(source_hr)*sb.cols+slr]);
                }
            }
            ++s;
        }
    }
    return sum;
}

__global__ void gpu_direct_low_cross_gather_kernel(Count* mainv,Count* blockv,int p){
    uint32_t dbid=blockIdx.z;bool target_main=p==1;
    uint32_t nblocks=target_main?D_GD_MAIN_NBLOCKS:D_GD_BLOCK_NBLOCKS;
    if(dbid>=nblocks)return;
    GpuDirectBlock db=target_main?D_GD_MAIN_BLOCKS[dbid]:D_GD_BLOCK_BLOCKS[dbid];
    if(!db.valid||!db.rows||!db.cols)return;
    uint32_t pi=uint32_t(LOW_LUT_K-p);
    size_t oi=size_t(pi)*D_GDX_LOW_PITCH+dbid;
    uint32_t a=D_GDX_LOW_OFF[oi],b=D_GDX_LOW_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
        q<b;q+=uint32_t(gridDim.x)*blockDim.x){
        GpuDirectGatherDst rec=D_GDX_LOW_DST[q];
        for(uint32_t dhr=blockIdx.y;dhr<db.rows;dhr+=gridDim.y){
            uint32_t dest_code=D_GDX_HIGH_CODES[D_GD_HIGH_ALL_OFF[db.he]+dhr];
            Count* dp=(target_main?mainv:blockv)+db.off+Code(dhr)*db.cols+rec.dst_rank;
            Count sum=*dp;
            for(uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){
                uint32_t op=D_GDX_LOW_OP[e];
                uint32_t sbid=gdx_op_block(op),slr=gdx_op_rank(op),depth=gdx_op_depth(op);
                GpuDirectBlock sb=D_GD_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdx_sum_high_preimages(
                    mainv,dest_code,depth,sb.he,sbid,slr,dhr));
            }
            *dp=sum;
        }
    }
}

__global__ void gpu_direct_high_cross_gather_kernel(Count* mainv,Count* blockv,int p){
    uint32_t dbid=blockIdx.z;if(dbid>=D_GD_BLOCK_NBLOCKS)return;
    GpuDirectBlock db=D_GD_BLOCK_BLOCKS[dbid];if(!db.valid||!db.rows||!db.cols)return;
    uint32_t pi=uint32_t((TARGET_W-1)-p);
    size_t oi=size_t(pi)*D_GDX_HIGH_PITCH+dbid;
    uint32_t a=D_GDX_HIGH_OFF[oi],b=D_GDX_HIGH_OFF[oi+1];
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){
        GpuDirectGatherDst rec=D_GDX_HIGH_DST[q];
        for(uint32_t dlr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
            dlr<db.cols;dlr+=uint32_t(gridDim.x)*blockDim.x){
            uint32_t dest_code=D_GDX_LOW_CODES[D_GD_LOW_ALL_OFF[db.hs]+dlr];
            Count* dp=blockv+db.off+Code(rec.dst_rank)*db.cols+dlr;
            Count sum=*dp;
            for(uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e){
                uint32_t op=D_GDX_HIGH_OP[e];
                uint32_t sbid=gdx_op_block(op),shr=gdx_op_rank(op),depth=gdx_op_depth(op);
                GpuDirectBlock sb=D_GD_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdx_sum_low_preimages(
                    mainv,dest_code,depth,sb.hs,sbid,shr,dlr));
            }
            *dp=sum;
        }
    }
}

struct GpuDirectCrossGatherDeviceTables{
    GpuDirectGatherDst* low_dst=nullptr;GpuDirectCrossGatherOp* low_op=nullptr;uint32_t* low_off=nullptr;
    GpuDirectGatherDst* high_dst=nullptr;GpuDirectCrossGatherOp* high_op=nullptr;uint32_t* high_off=nullptr;
    uint32_t* high_codes=nullptr;uint32_t* low_codes=nullptr;uint32_t* high_direct=nullptr;uint32_t* low_direct=nullptr;
    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const GpuDirectCrossGatherHost&h){
        cp(low_dst,h.low_dst,"gdx low dst");cp(low_op,h.low_op,"gdx low op");cp(low_off,h.low_off,"gdx low off");
        cp(high_dst,h.high_dst,"gdx high dst");cp(high_op,h.high_op,"gdx high op");cp(high_off,h.high_off,"gdx high off");
        cp(high_codes,h.high_codes,"gdx high codes");cp(low_codes,h.low_codes,"gdx low codes");
        cp(high_direct,h.high_direct,"gdx high direct");cp(low_direct,h.low_direct,"gdx low direct");
        ck(cudaMemcpyToSymbol(D_GDX_LOW_DST,&low_dst,sizeof(low_dst)),"gdx low dst ptr");
        ck(cudaMemcpyToSymbol(D_GDX_LOW_OP,&low_op,sizeof(low_op)),"gdx low op ptr");
        ck(cudaMemcpyToSymbol(D_GDX_LOW_OFF,&low_off,sizeof(low_off)),"gdx low off ptr");
        ck(cudaMemcpyToSymbol(D_GDX_LOW_PITCH,&h.low_pitch,sizeof(h.low_pitch)),"gdx low pitch");
        ck(cudaMemcpyToSymbol(D_GDX_HIGH_DST,&high_dst,sizeof(high_dst)),"gdx high dst ptr");
        ck(cudaMemcpyToSymbol(D_GDX_HIGH_OP,&high_op,sizeof(high_op)),"gdx high op ptr");
        ck(cudaMemcpyToSymbol(D_GDX_HIGH_OFF,&high_off,sizeof(high_off)),"gdx high off ptr");
        ck(cudaMemcpyToSymbol(D_GDX_HIGH_PITCH,&h.high_pitch,sizeof(h.high_pitch)),"gdx high pitch");
        ck(cudaMemcpyToSymbol(D_GDX_HIGH_CODES,&high_codes,sizeof(high_codes)),"gdx high codes ptr");
        ck(cudaMemcpyToSymbol(D_GDX_LOW_CODES,&low_codes,sizeof(low_codes)),"gdx low codes ptr");
        ck(cudaMemcpyToSymbol(D_GDX_HIGH_DIRECT,&high_direct,sizeof(high_direct)),"gdx high direct ptr");
        ck(cudaMemcpyToSymbol(D_GDX_LOW_DIRECT,&low_direct,sizeof(low_direct)),"gdx low direct ptr");
    }
    void release(){
        if(low_dst)cudaFree(low_dst);if(low_op)cudaFree(low_op);if(low_off)cudaFree(low_off);
        if(high_dst)cudaFree(high_dst);if(high_op)cudaFree(high_op);if(high_off)cudaFree(high_off);
        if(high_codes)cudaFree(high_codes);if(low_codes)cudaFree(low_codes);if(high_direct)cudaFree(high_direct);if(low_direct)cudaFree(low_direct);
        low_dst=nullptr;low_op=nullptr;low_off=nullptr;high_dst=nullptr;high_op=nullptr;high_off=nullptr;
        high_codes=low_codes=high_direct=low_direct=nullptr;
    }
};

static void gpu_direct_cross_gather_drop_redundant(
    GpuDirectDeviceTables&base,GpuDirectGatherDeviceTables&ordinary){
    if(base.high_cross_closure)cudaFree(base.high_cross_closure);
    if(base.high_cross_closure_off)cudaFree(base.high_cross_closure_off);
    if(base.high_cross_rank)cudaFree(base.high_cross_rank);
    if(base.low_cross_rank)cudaFree(base.low_cross_rank);
    base.high_cross_closure=nullptr;base.high_cross_closure_off=nullptr;
    base.high_cross_rank=base.low_cross_rank=nullptr;
    if(ordinary.low_cross)cudaFree(ordinary.low_cross);
    if(ordinary.low_cross_off)cudaFree(ordinary.low_cross_off);
    ordinary.low_cross=nullptr;ordinary.low_cross_off=nullptr;
}

static void gpu_direct_run_low_atomicfree(
    Count*mainv,Count*blockv,const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads);
    for(int p=LOW_LUT_K;p>=1;--p){
        dim3 og(grid_x,grid_y,unsigned(layout.main_blocks.size()));gpu_direct_low_orbit_kernel<<<og,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdx low orbit");
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        dim3 lg(grid_x,grid_y,nt);gpu_direct_low_local_gather_kernel<<<lg,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdx low local");
        gpu_direct_low_cross_gather_kernel<<<lg,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdx low cross gather");
    }
    ck(cudaDeviceSynchronize(),"gdx low sync");
}
static void gpu_direct_run_high_atomicfree(
    Count*mainv,Count*blockv,const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads);
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        dim3 og(grid_x,grid_y,unsigned(layout.main_blocks.size()));gpu_direct_high_orbit_kernel<<<og,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdx high orbit");
        dim3 cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));gpu_direct_high_local_gather_kernel<<<cg,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdx high local");
        gpu_direct_high_cross_gather_kernel<<<cg,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdx high cross gather");
    }
    ck(cudaDeviceSynchronize(),"gdx high sync");
}
