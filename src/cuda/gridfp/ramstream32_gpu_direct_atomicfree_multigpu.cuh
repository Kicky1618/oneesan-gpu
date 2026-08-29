#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

// Correctness-first multi-GPU port of the zero-scratch atomic-free direct
// executor. Authoritative state is sharded by whole occupancy-major blocks.
// Destination-gather kernels execute only on the GPU owning the destination
// block and read source blocks through P2P. Orbit kernels execute only on the
// GPU owning their source block; the original orbit mapping is conflict-free,
// so its partner/drop stores may safely target peer memory.

static constexpr int GDM_MAX_GPU = 8;

struct GdmBlock {
    Code off = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint8_t he = 0;
    uint8_t hs = 0;
    uint8_t c = 0;
    uint8_t valid = 0;
    uint8_t owner = 0;
    uint8_t pad[3]{};
};

struct GdmShardHost {
    std::vector<GdmBlock> main_blocks;
    std::vector<GdmBlock> block_blocks;
    std::array<Code,GDM_MAX_GPU> main_elems{};
    std::array<Code,GDM_MAX_GPU> block_elems{};
    std::array<Code,GDM_MAX_GPU> total_elems{};
    Code max_elems = 0;
    Code min_elems = 0;
};

static GdmShardHost build_gdm_shards(const StorageLayout& layout, int ngpu) {
    if (ngpu < 1 || ngpu > GDM_MAX_GPU) std::exit(170);
    GdmShardHost out;
    out.main_blocks.resize(layout.main_blocks.size());
    out.block_blocks.resize(layout.block_blocks.size());
    struct Item { bool main; uint32_t bid; Code elems; };
    std::vector<Item> items;
    items.reserve(layout.main_blocks.size() + layout.block_blocks.size());
    for (uint32_t i=0;i<uint32_t(layout.main_blocks.size());++i) {
        const auto& b=layout.main_blocks[i]; Code n=Code(b.rows)*b.cols;
        if (b.valid && n) items.push_back({true,i,n});
    }
    for (uint32_t i=0;i<uint32_t(layout.block_blocks.size());++i) {
        const auto& b=layout.block_blocks[i]; Code n=Code(b.rows)*b.cols;
        if (b.valid && n) items.push_back({false,i,n});
    }
    std::sort(items.begin(),items.end(),[](const Item&a,const Item&b){return a.elems>b.elems;});
    std::array<Code,GDM_MAX_GPU> load{};
    for (const Item& it:items) {
        int owner=0;
        for (int d=1;d<ngpu;++d) if (load[d]<load[owner]) owner=d;
        const StorageBlock& sb=it.main?layout.main_blocks[it.bid]:layout.block_blocks[it.bid];
        GdmBlock db; db.rows=sb.rows; db.cols=sb.cols; db.he=sb.he; db.hs=sb.hs;
        db.c=sb.c; db.valid=sb.valid; db.owner=uint8_t(owner);
        if (it.main) {
            db.off=out.main_elems[owner]; out.main_elems[owner]+=it.elems; out.main_blocks[it.bid]=db;
        } else {
            db.off=out.block_elems[owner]; out.block_elems[owner]+=it.elems; out.block_blocks[it.bid]=db;
        }
        load[owner]+=it.elems;
    }
    for (int d=0;d<ngpu;++d) out.total_elems[d]=out.main_elems[d]+out.block_elems[d];
    out.max_elems=*std::max_element(out.total_elems.begin(),out.total_elems.begin()+ngpu);
    out.min_elems=*std::min_element(out.total_elems.begin(),out.total_elems.begin()+ngpu);
    Code expect_main=0,expect_block=0,got_main=0,got_block=0;
    for (const auto& b:layout.main_blocks) expect_main+=Code(b.rows)*b.cols;
    for (const auto& b:layout.block_blocks) expect_block+=Code(b.rows)*b.cols;
    for (int d=0;d<ngpu;++d) { got_main+=out.main_elems[d]; got_block+=out.block_elems[d]; }
    if (expect_main!=got_main || expect_block!=got_block) {
        std::cerr<<"gdm shard size mismatch main="<<got_main<<'/'<<expect_main
                 <<" block="<<got_block<<'/'<<expect_block<<'\n'; std::exit(171);
    }
    return out;
}

static std::pair<int,Code> gdm_locate_main(Code global_rank,const StorageLayout& layout,const GdmShardHost& shard) {
    for (uint32_t bid=0;bid<uint32_t(layout.main_blocks.size());++bid) {
        const auto& sb=layout.main_blocks[bid]; Code n=Code(sb.rows)*sb.cols;
        if (global_rank>=sb.off && global_rank<sb.off+n) {
            const auto& db=shard.main_blocks[bid]; return {int(db.owner),db.off+(global_rank-sb.off)};
        }
    }
    std::cerr<<"gdm main rank not covered: "<<global_rank<<'\n'; std::exit(172);
}

__constant__ GdmBlock D_GDM_MAIN_BLOCKS[GPU_DIRECT_MAX_MAIN_BLOCKS];
__constant__ GdmBlock D_GDM_BLOCK_BLOCKS[GPU_DIRECT_MAX_BLOCK_BLOCKS];
__constant__ Count* D_GDM_MAIN_PTR[GDM_MAX_GPU];
__constant__ Count* D_GDM_BLOCK_PTR[GDM_MAX_GPU];
__constant__ int D_GDM_NGPU;
__constant__ int D_GDM_DEVICE;

__device__ __forceinline__ Count* gdm_main_ptr(const GdmBlock& b, Code i) { return D_GDM_MAIN_PTR[b.owner]+b.off+i; }
__device__ __forceinline__ Count* gdm_block_ptr(const GdmBlock& b, Code i) { return D_GDM_BLOCK_PTR[b.owner]+b.off+i; }
__device__ __forceinline__ Count gdm_main_load(const GdmBlock& b, Code i) { return *gdm_main_ptr(b,i); }

static void gdm_install_shards(const GdmShardHost& h,Count* const* main_ptr,Count* const* block_ptr,int ngpu,int device) {
    if (h.main_blocks.size()>GPU_DIRECT_MAX_MAIN_BLOCKS || h.block_blocks.size()>GPU_DIRECT_MAX_BLOCK_BLOCKS) std::exit(173);
    std::array<Count*,GDM_MAX_GPU> mp{},bp{};
    for (int d=0;d<ngpu;++d) { mp[d]=main_ptr[d]; bp[d]=block_ptr[d]; }
    ck(cudaMemcpyToSymbol(D_GDM_MAIN_BLOCKS,h.main_blocks.data(),h.main_blocks.size()*sizeof(GdmBlock)),"gdm main blocks");
    ck(cudaMemcpyToSymbol(D_GDM_BLOCK_BLOCKS,h.block_blocks.data(),h.block_blocks.size()*sizeof(GdmBlock)),"gdm block blocks");
    ck(cudaMemcpyToSymbol(D_GDM_MAIN_PTR,mp.data(),sizeof(mp)),"gdm main ptrs");
    ck(cudaMemcpyToSymbol(D_GDM_BLOCK_PTR,bp.data(),sizeof(bp)),"gdm block ptrs");
    ck(cudaMemcpyToSymbol(D_GDM_NGPU,&ngpu,sizeof(ngpu)),"gdm ngpu");
    ck(cudaMemcpyToSymbol(D_GDM_DEVICE,&device,sizeof(device)),"gdm device");
}

__global__ void gdm_low_orbit_kernel(int p) {
    uint32_t bid=blockIdx.z; if (bid>=D_GD_MAIN_NBLOCKS) return;
    GdmBlock x=D_GDM_MAIN_BLOCKS[bid]; if (!x.valid || x.owner!=D_GDM_DEVICE || !x.rows || !x.cols) return;
    uint32_t pi=uint32_t(LOW_LUT_K-p),lr0=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
    uint32_t lr_step=uint32_t(gridDim.x)*blockDim.x;
    for (uint32_t lr=lr0;lr<x.cols;lr+=lr_step) {
        uint64_t ow=D_GD_LOW_ORBIT[size_t(pi)*D_GD_LOW_ORBIT_MAIN_TOTAL+D_GD_LOW_ORBIT_MAIN_BASE[bid]+lr];
        uint32_t kind=gpu_direct_low_orbit_kind(ow); if (kind<CPU_ORBIT_NN || kind>CPU_ORBIT_NL) continue;
        uint32_t jbid=gpu_direct_low_orbit_jblock(ow),dbid=gpu_direct_low_orbit_dblock(ow);
        GdmBlock y=D_GDM_MAIN_BLOCKS[jbid],d=D_GDM_BLOCK_BLOCKS[dbid];
        uint32_t jlr=gpu_direct_low_orbit_jlr(ow),dlr=gpu_direct_low_orbit_dlr(ow);
        for (uint32_t hr=blockIdx.y;hr<x.rows;hr+=gridDim.y) {
            Count* ip=gdm_main_ptr(x,Code(hr)*x.cols+lr); Count* jp=gdm_main_ptr(y,Code(hr)*y.cols+jlr);
            Count* dp=gdm_block_ptr(d,Code(hr)*d.cols+dlr); Count c=*ip,old_d=*dp;
            if (kind==CPU_ORBIT_NN) { *jp=gpu_direct_add(*jp,c); *ip=gpu_direct_add(c,old_d); *dp=0; }
            else { Count cc=*jp,all=gpu_direct_add(gpu_direct_add(c,cc),old_d);
                if (p==1) { *ip=all; *jp=gpu_direct_add(c,cc); *dp=0; } else { *ip=all; *dp=c; } }
        }
    }
}

__global__ void gdm_high_orbit_kernel(int p) {
    uint32_t bid=blockIdx.z; if (bid>=D_GD_MAIN_NBLOCKS) return;
    GdmBlock x=D_GDM_MAIN_BLOCKS[bid]; if (!x.valid || x.owner!=D_GDM_DEVICE || !x.rows || !x.cols) return;
    uint32_t pi=uint32_t((TARGET_W-1)-p); size_t oi=size_t(pi)*D_GD_HIGH_PITCH+bid;
    uint32_t na=D_GD_HIGH_NN_OFF[oi],nb=D_GD_HIGH_NN_OFF[oi+1],ra=D_GD_HIGH_NRNL_OFF[oi],rb=D_GD_HIGH_NRNL_OFF[oi+1];
    uint32_t nn_count=nb-na,nrnl_count=rb-ra,total=nn_count+nrnl_count; if (!total) return;
    uint32_t lr0=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x,lr_step=uint32_t(gridDim.x)*blockDim.x;
    GdmBlock d=D_GDM_BLOCK_BLOCKS[uint32_t(x.hs)];
    for (uint32_t k=blockIdx.y;k<total;k+=gridDim.y) {
        bool nn=k<nn_count; CpuHighOrbitOp op=nn?D_GD_HIGH_NN_OPS[na+k]:D_GD_HIGH_NRNL_OPS[ra+(k-nn_count)];
        GpuDirectBlock xb{0,x.rows,x.cols,x.he,x.hs,x.c,x.valid};
        GdmBlock y=D_GDM_MAIN_BLOCKS[gpu_direct_high_partner_block(bid,xb,p,nn)];
        uint32_t shr=gpu_direct_high_src(op),jhr=gpu_direct_high_partner(op),dhr=gpu_direct_high_drop(op);
        for (uint32_t lr=lr0;lr<x.cols;lr+=lr_step) {
            Count* ip=gdm_main_ptr(x,Code(shr)*x.cols+lr); Count* jp=gdm_main_ptr(y,Code(jhr)*y.cols+lr);
            Count* dp=gdm_block_ptr(d,Code(dhr)*d.cols+lr); Count c=*ip,old_d=*dp;
            if (nn) { *jp=gpu_direct_add(*jp,c); *ip=gpu_direct_add(c,old_d); *dp=0; }
            else { Count cc=*jp; *ip=gpu_direct_add(gpu_direct_add(c,cc),old_d); *dp=c; }
        }
    }
}

__global__ void gdm_low_local_gather_kernel(int p) {
    uint32_t dbid=blockIdx.z; bool target_main=p==1; uint32_t nblocks=target_main?D_GD_MAIN_NBLOCKS:D_GD_BLOCK_NBLOCKS;
    if (dbid>=nblocks) return; GdmBlock dstb=target_main?D_GDM_MAIN_BLOCKS[dbid]:D_GDM_BLOCK_BLOCKS[dbid];
    if (!dstb.valid || dstb.owner!=D_GDM_DEVICE || !dstb.rows || !dstb.cols) return;
    uint32_t pi=uint32_t(LOW_LUT_K-p); size_t oi=size_t(pi)*D_GDG_LOW_PITCH+dbid;
    uint32_t a=D_GDG_LOW_OFF[oi],b=D_GDG_LOW_OFF[oi+1],q0=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;
    uint32_t qstep=uint32_t(gridDim.x)*blockDim.x;
    for (uint32_t q=q0;q<b;q+=qstep) { GpuDirectGatherDst rec=D_GDG_LOW_DST[q];
        for (uint32_t hr=blockIdx.y;hr<dstb.rows;hr+=gridDim.y) {
            Count* dp=target_main?gdm_main_ptr(dstb,Code(hr)*dstb.cols+rec.dst_rank):gdm_block_ptr(dstb,Code(hr)*dstb.cols+rec.dst_rank);
            Count sum=*dp;
            for (uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                uint32_t loc=D_GDG_LOW_SRC[e]; GdmBlock srcb=D_GDM_MAIN_BLOCKS[gpu_direct_gather_src_block(loc)];
                sum=gpu_direct_add(sum,gdm_main_load(srcb,Code(hr)*srcb.cols+gpu_direct_gather_src_rank(loc)));
            }
            *dp=sum;
        }
    }
}

__global__ void gdm_high_local_gather_kernel(int p) {
    uint32_t dbid=blockIdx.z; if (dbid>=D_GD_BLOCK_NBLOCKS) return; GdmBlock dstb=D_GDM_BLOCK_BLOCKS[dbid];
    if (!dstb.valid || dstb.owner!=D_GDM_DEVICE || !dstb.rows || !dstb.cols) return;
    uint32_t pi=uint32_t((TARGET_W-1)-p); size_t oi=size_t(pi)*D_GDG_HIGH_PITCH+dbid;
    uint32_t a=D_GDG_HIGH_OFF[oi],b=D_GDG_HIGH_OFF[oi+1];
    for (uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y) { GpuDirectGatherDst rec=D_GDG_HIGH_DST[q];
        for (uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<dstb.cols;lr+=uint32_t(gridDim.x)*blockDim.x) {
            Count* dp=gdm_block_ptr(dstb,Code(rec.dst_rank)*dstb.cols+lr); Count sum=*dp;
            for (uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                uint32_t loc=D_GDG_HIGH_SRC[e]; GdmBlock srcb=D_GDM_MAIN_BLOCKS[gpu_direct_gather_src_block(loc)];
                sum=gpu_direct_add(sum,gdm_main_load(srcb,Code(gpu_direct_gather_src_rank(loc))*srcb.cols+lr));
            }
            *dp=sum;
        }
    }
}

__device__ __forceinline__ Count gdm_sum_high_preimages(uint32_t dest_code,uint32_t depth,uint32_t source_he,uint32_t source_bid,uint32_t source_lr) {
    Count sum=0; int s=int(depth);
#pragma unroll
    for (int pos=0;pos<HIGH_LUT_K;++pos) { uint32_t v=(dest_code>>(2*pos))&3u;
        if (v==uint32_t(::L)) { if (s==1) break; --s; }
        else if (v==uint32_t(R)) { if (s==1) { uint32_t z=3u<<(2*pos),src_code=(dest_code&~z)|(uint32_t(::L)<<(2*pos));
                uint32_t gi=D_GDX_HIGH_DIRECT[gdx_ternary_key<HIGH_LUT_K>(src_code)],a=D_GD_HIGH_ALL_OFF[source_he],b=D_GD_HIGH_ALL_OFF[source_he+1];
                if (gi>=a && gi<b) { GdmBlock sb=D_GDM_MAIN_BLOCKS[source_bid]; sum=gpu_direct_add(sum,gdm_main_load(sb,Code(gi-a)*sb.cols+source_lr)); } }
            ++s; }
    }
    return sum;
}

__device__ __forceinline__ Count gdm_sum_low_preimages(uint32_t dest_code,uint32_t depth,uint32_t source_hs,uint32_t source_bid,uint32_t source_hr) {
    Count sum=0; int s=int(depth);
#pragma unroll
    for (int pos=LOW_LUT_K-1;pos>=0;--pos) { uint32_t v=(dest_code>>(2*pos))&3u;
        if (v==uint32_t(R)) { if (s==1) break; --s; }
        else if (v==uint32_t(::L)) { if (s==1) { uint32_t z=3u<<(2*pos),src_code=(dest_code&~z)|(uint32_t(R)<<(2*pos));
                uint32_t gi=D_GDX_LOW_DIRECT[gdx_ternary_key<LOW_LUT_K>(src_code)],a=D_GD_LOW_ALL_OFF[source_hs],b=D_GD_LOW_ALL_OFF[source_hs+1];
                if (gi>=a && gi<b) { GdmBlock sb=D_GDM_MAIN_BLOCKS[source_bid]; sum=gpu_direct_add(sum,gdm_main_load(sb,Code(source_hr)*sb.cols+(gi-a))); } }
            ++s; }
    }
    return sum;
}

__global__ void gdm_low_cross_gather_kernel(int p) {
    uint32_t dbid=blockIdx.z; bool target_main=p==1; uint32_t nblocks=target_main?D_GD_MAIN_NBLOCKS:D_GD_BLOCK_NBLOCKS;
    if (dbid>=nblocks) return; GdmBlock db=target_main?D_GDM_MAIN_BLOCKS[dbid]:D_GDM_BLOCK_BLOCKS[dbid];
    if (!db.valid || db.owner!=D_GDM_DEVICE || !db.rows || !db.cols) return;
    uint32_t pi=uint32_t(LOW_LUT_K-p); size_t oi=size_t(pi)*D_GDX_LOW_PITCH+dbid; uint32_t a=D_GDX_LOW_OFF[oi],b=D_GDX_LOW_OFF[oi+1];
    for (uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x) {
        GpuDirectGatherDst rec=D_GDX_LOW_DST[q];
        for (uint32_t dhr=blockIdx.y;dhr<db.rows;dhr+=gridDim.y) {
            uint32_t dest_code=D_GDX_HIGH_CODES[D_GD_HIGH_ALL_OFF[db.he]+dhr];
            Count* dp=target_main?gdm_main_ptr(db,Code(dhr)*db.cols+rec.dst_rank):gdm_block_ptr(db,Code(dhr)*db.cols+rec.dst_rank);
            Count sum=*dp;
            for (uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                uint32_t op=D_GDX_LOW_OP[e],sbid=gdx_op_block(op),slr=gdx_op_rank(op),depth=gdx_op_depth(op); GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdm_sum_high_preimages(dest_code,depth,sb.he,sbid,slr));
            }
            *dp=sum;
        }
    }
}

__global__ void gdm_high_cross_gather_kernel(int p) {
    uint32_t dbid=blockIdx.z; if (dbid>=D_GD_BLOCK_NBLOCKS) return; GdmBlock db=D_GDM_BLOCK_BLOCKS[dbid];
    if (!db.valid || db.owner!=D_GDM_DEVICE || !db.rows || !db.cols) return;
    uint32_t pi=uint32_t((TARGET_W-1)-p); size_t oi=size_t(pi)*D_GDX_HIGH_PITCH+dbid; uint32_t a=D_GDX_HIGH_OFF[oi],b=D_GDX_HIGH_OFF[oi+1];
    for (uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y) { GpuDirectGatherDst rec=D_GDX_HIGH_DST[q];
        for (uint32_t dlr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;dlr<db.cols;dlr+=uint32_t(gridDim.x)*blockDim.x) {
            uint32_t dest_code=D_GDX_LOW_CODES[D_GD_LOW_ALL_OFF[db.hs]+dlr]; Count* dp=gdm_block_ptr(db,Code(rec.dst_rank)*db.cols+dlr); Count sum=*dp;
            for (uint32_t e=rec.edge_begin;e<rec.edge_begin+rec.edge_count;++e) {
                uint32_t op=D_GDX_HIGH_OP[e],sbid=gdx_op_block(op),shr=gdx_op_rank(op),depth=gdx_op_depth(op); GdmBlock sb=D_GDM_MAIN_BLOCKS[sbid];
                sum=gpu_direct_add(sum,gdm_sum_low_preimages(dest_code,depth,sb.hs,sbid,shr));
            }
            *dp=sum;
        }
    }
}

static void gdm_sync_all(int ngpu,const char* what) {
    for (int d=0;d<ngpu;++d) { ck(cudaSetDevice(d),"gdm set sync"); ck(cudaDeviceSynchronize(),what); }
}

static void gdm_run_low(int ngpu,const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8) {
    dim3 block(threads);
    for (int p=LOW_LUT_K;p>=1;--p) {
        for (int d=0;d<ngpu;++d) { ck(cudaSetDevice(d),"gdm low orbit device"); dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size())); gdm_low_orbit_kernel<<<g,block>>>(p); ck(cudaGetLastError(),"gdm low orbit"); }
        gdm_sync_all(ngpu,"gdm low orbit sync"); unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        for (int d=0;d<ngpu;++d) { ck(cudaSetDevice(d),"gdm low local device"); dim3 g(grid_x,grid_y,nt); gdm_low_local_gather_kernel<<<g,block>>>(p); ck(cudaGetLastError(),"gdm low local"); }
        gdm_sync_all(ngpu,"gdm low local sync");
        for (int d=0;d<ngpu;++d) { ck(cudaSetDevice(d),"gdm low cross device"); dim3 g(grid_x,grid_y,nt); gdm_low_cross_gather_kernel<<<g,block>>>(p); ck(cudaGetLastError(),"gdm low cross"); }
        gdm_sync_all(ngpu,"gdm low cross sync");
    }
}

static void gdm_run_high(int ngpu,const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8) {
    dim3 block(threads);
    for (int p=TARGET_W-1;p>=LOW_LUT_K+1;--p) {
        for (int d=0;d<ngpu;++d) { ck(cudaSetDevice(d),"gdm high orbit device"); dim3 g(grid_x,grid_y,unsigned(layout.main_blocks.size())); gdm_high_orbit_kernel<<<g,block>>>(p); ck(cudaGetLastError(),"gdm high orbit"); }
        gdm_sync_all(ngpu,"gdm high orbit sync");
        for (int d=0;d<ngpu;++d) { ck(cudaSetDevice(d),"gdm high local device"); dim3 g(grid_x,grid_y,unsigned(layout.block_blocks.size())); gdm_high_local_gather_kernel<<<g,block>>>(p); ck(cudaGetLastError(),"gdm high local"); }
        gdm_sync_all(ngpu,"gdm high local sync");
        for (int d=0;d<ngpu;++d) { ck(cudaSetDevice(d),"gdm high cross device"); dim3 g(grid_x,grid_y,unsigned(layout.block_blocks.size())); gdm_high_cross_gather_kernel<<<g,block>>>(p); ck(cudaGetLastError(),"gdm high cross"); }
        gdm_sync_all(ngpu,"gdm high cross sync");
    }
}
