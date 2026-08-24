#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

// Fuse ordinary and CROSS destination-gather records. The underlying source
// arrays stay factorized: ordinary edges use D_GDG_*_SRC, CROSS edges use
// D_GDX_*_OP plus the ternary inverse machinery. One thread owns a destination
// cell, accumulates both contribution classes, then performs one plain store.

struct GpuDirectFusedDst {
    uint32_t dst_rank = 0;
    uint32_t local_begin = 0;
    uint32_t cross_begin = 0;
    uint32_t counts = 0; // low16=ordinary count, high16=CROSS op count
};
static_assert(sizeof(GpuDirectFusedDst)==16);

static inline uint32_t gdf_count_pack(uint32_t local,uint32_t cross){
    if(local>0xffffu||cross>0xffffu){std::cerr<<"gpu direct fused count overflow\n";std::exit(170);}
    return local|(cross<<16);
}

struct GpuDirectFusedHost {
    std::vector<GpuDirectFusedDst> low_dst;
    std::vector<uint32_t> low_off;
    uint32_t low_pitch=GPU_DIRECT_MAX_MAIN_BLOCKS+1;
    std::vector<GpuDirectFusedDst> high_dst;
    std::vector<uint32_t> high_off;
    uint32_t high_pitch=GPU_DIRECT_MAX_BLOCK_BLOCKS+1;
    size_t bytes()const{return (low_dst.size()+high_dst.size())*sizeof(GpuDirectFusedDst)
        +(low_off.size()+high_off.size())*sizeof(uint32_t);}
};

static GpuDirectFusedHost build_gpu_direct_fused(
    const StorageLayout&layout,const GpuDirectGatherHost&ordinary,
    const GpuDirectCrossGatherHost&cross){
    GpuDirectFusedHost out;
    out.low_off.assign(size_t(LOW_LUT_K)*out.low_pitch,0);
    out.high_off.assign(size_t(HIGH_LUT_K)*out.high_pitch,0);

    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);
        uint32_t nt=p==1?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        for(uint32_t bid=0;bid<nt;++bid){
            out.low_off[size_t(pi)*out.low_pitch+bid]=uint32_t(out.low_dst.size());
            uint32_t oa=ordinary.low_off[size_t(pi)*ordinary.low_pitch+bid];
            uint32_t ob=ordinary.low_off[size_t(pi)*ordinary.low_pitch+bid+1];
            uint32_t xa=cross.low_off[size_t(pi)*cross.low_pitch+bid];
            uint32_t xb=cross.low_off[size_t(pi)*cross.low_pitch+bid+1];
            while(oa<ob||xa<xb){
                uint32_t orank=oa<ob?ordinary.low_dst[oa].dst_rank:0xffffffffu;
                uint32_t xrank=xa<xb?cross.low_dst[xa].dst_rank:0xffffffffu;
                uint32_t rank=orank<xrank?orank:xrank;
                if(rank==0xffffffffu)break;
                uint32_t lb=0,lc=0,cb=0,cc=0;
                if(orank==rank){auto&r=ordinary.low_dst[oa++];lb=r.edge_begin;lc=r.edge_count;}
                if(xrank==rank){auto&r=cross.low_dst[xa++];cb=r.edge_begin;cc=r.edge_count;}
                out.low_dst.push_back({rank,lb,cb,gdf_count_pack(lc,cc)});
            }
        }
        out.low_off[size_t(pi)*out.low_pitch+nt]=uint32_t(out.low_dst.size());
    }

    for(uint32_t pi=0;pi<uint32_t(HIGH_LUT_K);++pi){
        uint32_t nt=uint32_t(layout.block_blocks.size());
        for(uint32_t bid=0;bid<nt;++bid){
            out.high_off[size_t(pi)*out.high_pitch+bid]=uint32_t(out.high_dst.size());
            uint32_t oa=ordinary.high_off[size_t(pi)*ordinary.high_pitch+bid];
            uint32_t ob=ordinary.high_off[size_t(pi)*ordinary.high_pitch+bid+1];
            uint32_t xa=cross.high_off[size_t(pi)*cross.high_pitch+bid];
            uint32_t xb=cross.high_off[size_t(pi)*cross.high_pitch+bid+1];
            while(oa<ob||xa<xb){
                uint32_t orank=oa<ob?ordinary.high_dst[oa].dst_rank:0xffffffffu;
                uint32_t xrank=xa<xb?cross.high_dst[xa].dst_rank:0xffffffffu;
                uint32_t rank=orank<xrank?orank:xrank;
                if(rank==0xffffffffu)break;
                uint32_t lb=0,lc=0,cb=0,cc=0;
                if(orank==rank){auto&r=ordinary.high_dst[oa++];lb=r.edge_begin;lc=r.edge_count;}
                if(xrank==rank){auto&r=cross.high_dst[xa++];cb=r.edge_begin;cc=r.edge_count;}
                out.high_dst.push_back({rank,lb,cb,gdf_count_pack(lc,cc)});
            }
        }
        out.high_off[size_t(pi)*out.high_pitch+nt]=uint32_t(out.high_dst.size());
    }
    std::cerr<<"gpu_direct_fused low_dst="<<out.low_dst.size()<<" high_dst="<<out.high_dst.size()
             <<" mib="<<double(out.bytes())/double(1<<20)<<'\n';
    return out;
}

__constant__ GpuDirectFusedDst* D_GDF_LOW_DST;
__constant__ uint32_t* D_GDF_LOW_OFF;
__constant__ uint32_t D_GDF_LOW_PITCH;
__constant__ GpuDirectFusedDst* D_GDF_HIGH_DST;
__constant__ uint32_t* D_GDF_HIGH_OFF;
__constant__ uint32_t D_GDF_HIGH_PITCH;

__global__ void gpu_direct_low_fused_closure_kernel(Count*mainv,Count*blockv,int p){
    uint32_t dbid=blockIdx.z;bool target_main=p==1;
    uint32_t nt=target_main?D_GD_MAIN_NBLOCKS:D_GD_BLOCK_NBLOCKS;if(dbid>=nt)return;
    GpuDirectBlock db=target_main?D_GD_MAIN_BLOCKS[dbid]:D_GD_BLOCK_BLOCKS[dbid];if(!db.valid||!db.rows||!db.cols)return;
    uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_GDF_LOW_PITCH+dbid;
    uint32_t a=D_GDF_LOW_OFF[oi],b=D_GDF_LOW_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){
        GpuDirectFusedDst rec=D_GDF_LOW_DST[q];uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t dhr=blockIdx.y;dhr<db.rows;dhr+=gridDim.y){
            Count*dp=(target_main?mainv:blockv)+db.off+Code(dhr)*db.cols+rec.dst_rank;Count sum=*dp;
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t loc=D_GDG_LOW_SRC[e];GpuDirectBlock sb=D_GD_MAIN_BLOCKS[gpu_direct_gather_src_block(loc)];uint32_t slr=gpu_direct_gather_src_rank(loc);sum=gpu_direct_add(sum,mainv[sb.off+Code(dhr)*sb.cols+slr]);}
            if(cc){uint32_t dest_code=D_GDX_HIGH_CODES[D_GD_HIGH_ALL_OFF[db.he]+dhr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t op=D_GDX_LOW_OP[e];uint32_t sbid=gdx_op_block(op),slr=gdx_op_rank(op),depth=gdx_op_depth(op);GpuDirectBlock sb=D_GD_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdx_sum_high_preimages(mainv,dest_code,depth,sb.he,sbid,slr,dhr));}}
            *dp=sum;
        }
    }
}

__global__ void gpu_direct_high_fused_closure_kernel(Count*mainv,Count*blockv,int p){
    uint32_t dbid=blockIdx.z;if(dbid>=D_GD_BLOCK_NBLOCKS)return;GpuDirectBlock db=D_GD_BLOCK_BLOCKS[dbid];if(!db.valid||!db.rows||!db.cols)return;
    uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_GDF_HIGH_PITCH+dbid;
    uint32_t a=D_GDF_HIGH_OFF[oi],b=D_GDF_HIGH_OFF[oi+1];
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){
        GpuDirectFusedDst rec=D_GDF_HIGH_DST[q];uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
        for(uint32_t dlr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;dlr<db.cols;dlr+=uint32_t(gridDim.x)*blockDim.x){
            Count*dp=blockv+db.off+Code(rec.dst_rank)*db.cols+dlr;Count sum=*dp;
            for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t loc=D_GDG_HIGH_SRC[e];GpuDirectBlock sb=D_GD_MAIN_BLOCKS[gpu_direct_gather_src_block(loc)];uint32_t shr=gpu_direct_gather_src_rank(loc);sum=gpu_direct_add(sum,mainv[sb.off+Code(shr)*sb.cols+dlr]);}
            if(cc){uint32_t dest_code=D_GDX_LOW_CODES[D_GD_LOW_ALL_OFF[db.hs]+dlr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t op=D_GDX_HIGH_OP[e];uint32_t sbid=gdx_op_block(op),shr=gdx_op_rank(op),depth=gdx_op_depth(op);GpuDirectBlock sb=D_GD_MAIN_BLOCKS[sbid];sum=gpu_direct_add(sum,gdx_sum_low_preimages(mainv,dest_code,depth,sb.hs,sbid,shr,dlr));}}
            *dp=sum;
        }
    }
}

struct GpuDirectFusedDeviceTables{
    GpuDirectFusedDst*low_dst=nullptr;uint32_t*low_off=nullptr;GpuDirectFusedDst*high_dst=nullptr;uint32_t*high_off=nullptr;
    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const GpuDirectFusedHost&h){cp(low_dst,h.low_dst,"gdf low dst");cp(low_off,h.low_off,"gdf low off");cp(high_dst,h.high_dst,"gdf high dst");cp(high_off,h.high_off,"gdf high off");ck(cudaMemcpyToSymbol(D_GDF_LOW_DST,&low_dst,sizeof(low_dst)),"gdf low dst ptr");ck(cudaMemcpyToSymbol(D_GDF_LOW_OFF,&low_off,sizeof(low_off)),"gdf low off ptr");ck(cudaMemcpyToSymbol(D_GDF_LOW_PITCH,&h.low_pitch,sizeof(h.low_pitch)),"gdf low pitch");ck(cudaMemcpyToSymbol(D_GDF_HIGH_DST,&high_dst,sizeof(high_dst)),"gdf high dst ptr");ck(cudaMemcpyToSymbol(D_GDF_HIGH_OFF,&high_off,sizeof(high_off)),"gdf high off ptr");ck(cudaMemcpyToSymbol(D_GDF_HIGH_PITCH,&h.high_pitch,sizeof(h.high_pitch)),"gdf high pitch");}
    void release(){if(low_dst)cudaFree(low_dst);if(low_off)cudaFree(low_off);if(high_dst)cudaFree(high_dst);if(high_off)cudaFree(high_off);low_dst=nullptr;low_off=nullptr;high_dst=nullptr;high_off=nullptr;}
};

static void gpu_direct_fused_drop_destination_tables(GpuDirectGatherDeviceTables&o,GpuDirectCrossGatherDeviceTables&x){
    if(o.low_dst)cudaFree(o.low_dst);if(o.low_off)cudaFree(o.low_off);if(o.high_dst)cudaFree(o.high_dst);if(o.high_off)cudaFree(o.high_off);
    o.low_dst=nullptr;o.low_off=nullptr;o.high_dst=nullptr;o.high_off=nullptr;
    if(x.low_dst)cudaFree(x.low_dst);if(x.low_off)cudaFree(x.low_off);if(x.high_dst)cudaFree(x.high_dst);if(x.high_off)cudaFree(x.high_off);
    x.low_dst=nullptr;x.low_off=nullptr;x.high_dst=nullptr;x.high_off=nullptr;
}

static void gpu_direct_run_low_fused(Count*mainv,Count*blockv,const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),grid(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K;p>=1;--p){gpu_direct_low_orbit_kernel<<<grid,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdf low orbit");unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());dim3 cg(grid_x,grid_y,nt);gpu_direct_low_fused_closure_kernel<<<cg,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdf low closure");}
    ck(cudaDeviceSynchronize(),"gdf low sync");
}
static void gpu_direct_run_high_fused(Count*mainv,Count*blockv,const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),og(grid_x,grid_y,unsigned(layout.main_blocks.size())),cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){gpu_direct_high_orbit_kernel<<<og,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdf high orbit");gpu_direct_high_fused_closure_kernel<<<cg,block>>>(mainv,blockv,p);ck(cudaGetLastError(),"gdf high closure");}
    ck(cudaDeviceSynchronize(),"gdf high sync");
}
