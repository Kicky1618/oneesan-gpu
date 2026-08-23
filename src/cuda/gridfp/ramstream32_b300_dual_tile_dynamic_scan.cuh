#pragma once

#include "ramstream32_b300_dual_tile_pruned_plan.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

struct B300DualScanTask {
    Code off = 0;
    uint32_t n = 0;
    uint16_t bid = 0;
    uint8_t bit_index = 0;
    uint8_t pad = 0;
};
static_assert(sizeof(B300DualScanTask) == 16);

__global__ void b300_dt_scan_nonzero_kernel(
    const Count* data, const B300DualScanTask* tasks, uint32_t ntasks,
    const uint64_t* allowed, unsigned long long* active
) {
    uint32_t ti = blockIdx.x;
    if (ti >= ntasks) return;
    B300DualScanTask t = tasks[ti];
    unsigned long long bit = 1ull << t.bit_index;
    if (!(allowed[t.bid] & bit)) return;
    if (active[t.bid] & bit) return;

    for (uint32_t q = threadIdx.x; q < t.n; q += blockDim.x) {
        if (data[t.off + q] != 0) {
            atomicOr(active + t.bid, bit);
            return;
        }
    }
}

struct B300DualScanListHost {
    std::array<std::vector<B300DualScanTask>,MAXGPU> task;
};

static B300DualScanListHost b300_dt_build_scan_list(
    const B300DualTileHost&z,const StorageLayout&l,
    bool blocked,bool high_orientation,uint32_t chunk_elems
){
    if(!chunk_elems)std::exit(680);
    B300DualScanListHost out;
    const int nb=blocked?int(l.block_blocks.size()):int(l.main_blocks.size());
    const auto&bs=blocked?z.block_slot_base:z.main_slot_base;
    for(int hi=0;hi<z.ngpu;++hi)for(int lo=0;lo<z.ngpu;++lo){
        if(hi==lo)continue; // diagonal never crosses NVLink and needs no flag.
        int owner=high_orientation?hi:lo;
        int peer=high_orientation?lo:hi;
        uint8_t bit=uint8_t(hi*z.ngpu+lo);
        for(int bid=0;bid<nb;++bid){
            const StorageBlock&b=blocked?l.block_blocks[bid]:l.main_blocks[bid];
            Code n=b300_dt_pruned_seg_len(z,b,hi,lo);if(!n)continue;
            Code off=bs[owner][peer]+b300_dt_pruned_seg_off(z,blocked,nb,hi,lo,bid);
            for(Code q=0;q<n;q+=chunk_elems){
                Code take=std::min<Code>(chunk_elems,n-q);
                B300DualScanTask t;t.off=off+q;t.n=uint32_t(take);t.bid=uint16_t(bid);t.bit_index=bit;
                out.task[owner].push_back(t);
            }
        }
    }
    return out;
}

struct B300DualScanGpu {
    B300DualScanTask* task=nullptr;
    uint32_t ntask=0;
    uint64_t* allowed=nullptr;
    unsigned long long* active=nullptr;

    void install(const std::vector<B300DualScanTask>&v,int nblocks){
        ntask=uint32_t(v.size());
        if(ntask){
            ck(cudaMalloc(&task,size_t(ntask)*sizeof(B300DualScanTask)),"dual scan tasks");
            ck(cudaMemcpy(task,v.data(),size_t(ntask)*sizeof(B300DualScanTask),cudaMemcpyHostToDevice),"dual scan task upload");
        }
        ck(cudaMalloc(&allowed,size_t(nblocks)*sizeof(uint64_t)),"dual scan allowed");
        ck(cudaMalloc(&active,size_t(nblocks)*sizeof(unsigned long long)),"dual scan active");
    }
    void release(){
        if(active)cudaFree(active);if(allowed)cudaFree(allowed);if(task)cudaFree(task);
        active=nullptr;allowed=nullptr;task=nullptr;ntask=0;
    }
};

struct B300DualDynamicScanContext {
    int ngpu=0;
    uint32_t chunk_elems=0;
    B300DualScanListHost main_low_host,block_low_host,main_high_host;
    std::array<B300DualScanGpu,MAXGPU> main_low,block_low,main_high;
    bool ready=false;

    void init(const B300DualTileHost&z,const StorageLayout&l,uint32_t chunk){
        if(ready)return;
        ngpu=z.ngpu;chunk_elems=chunk;
        main_low_host=b300_dt_build_scan_list(z,l,false,false,chunk);
        block_low_host=b300_dt_build_scan_list(z,l,true,false,chunk);
        main_high_host=b300_dt_build_scan_list(z,l,false,true,chunk);
        for(int g=0;g<ngpu;++g){
            ck(cudaSetDevice(g),"dual scan install device");
            main_low[g].install(main_low_host.task[g],int(l.main_blocks.size()));
            block_low[g].install(block_low_host.task[g],int(l.block_blocks.size()));
            main_high[g].install(main_high_host.task[g],int(l.main_blocks.size()));
        }
        ready=true;
    }

    void release(){
        if(!ready)return;
        for(int g=0;g<ngpu;++g){
            ck(cudaSetDevice(g),"dual scan release device");
            main_low[g].release();block_low[g].release();main_high[g].release();
        }
        ready=false;
    }
};

static inline void b300_dt_scan_one_vector(
    const B300DualTileHost&z,Count**ptrs,
    std::array<B300DualScanGpu,MAXGPU>&scan,
    const std::array<uint64_t,64>&allowed_host,
    int nblocks,std::array<uint64_t,64>&active_host
){
    active_host.fill(0);
    for(int g=0;g<z.ngpu;++g){
        ck(cudaSetDevice(g),"dual scan vector device");
        ck(cudaMemcpy(scan[g].allowed,allowed_host.data(),size_t(nblocks)*sizeof(uint64_t),cudaMemcpyHostToDevice),"dual scan allowed upload");
        ck(cudaMemset(scan[g].active,0,size_t(nblocks)*sizeof(unsigned long long)),"dual scan zero active");
        if(scan[g].ntask){
            b300_dt_scan_nonzero_kernel<<<scan[g].ntask,256>>>(
                ptrs[g],scan[g].task,scan[g].ntask,scan[g].allowed,scan[g].active);
            ck(cudaGetLastError(),"dual scan launch");
        }
    }
    for(int g=0;g<z.ngpu;++g){
        ck(cudaSetDevice(g),"dual scan copy device");
        std::array<uint64_t,64> tmp{};
        ck(cudaMemcpy(tmp.data(),scan[g].active,size_t(nblocks)*sizeof(uint64_t),cudaMemcpyDeviceToHost),"dual scan active download");
        for(int b=0;b<nblocks;++b)active_host[b]|=tmp[b];
    }
}

static inline B300DualReachStage b300_dt_dynamic_scan_l2h(
    const B300DualTileHost&z,const StorageLayout&l,
    Count**mainp,Count**blockp,const B300DualReachStage&allowed,
    B300DualDynamicScanContext&ctx
){
    B300DualReachStage actual;
    std::array<uint64_t,64> ma{},ba{};
    b300_dt_scan_one_vector(z,mainp,ctx.main_low,allowed.main,int(l.main_blocks.size()),ma);
    // Repackage the 32-block allowed/output arrays through the same fixed-64 helper.
    std::array<uint64_t,64> block_allowed{};
    for(size_t i=0;i<allowed.block.size();++i)block_allowed[i]=allowed.block[i];
    b300_dt_scan_one_vector(z,blockp,ctx.block_low,block_allowed,int(l.block_blocks.size()),ba);
    for(size_t i=0;i<actual.main.size();++i)actual.main[i]=ma[i];
    for(size_t i=0;i<actual.block.size();++i)actual.block[i]=ba[i];
    return actual;
}

static inline B300DualReachStage b300_dt_dynamic_scan_h2l_main(
    const B300DualTileHost&z,const StorageLayout&l,
    Count**mainp,const B300DualReachStage&allowed,
    B300DualDynamicScanContext&ctx
){
    B300DualReachStage actual;
    std::array<uint64_t,64> ma{};
    b300_dt_scan_one_vector(z,mainp,ctx.main_high,allowed.main,int(l.main_blocks.size()),ma);
    for(size_t i=0;i<actual.main.size();++i)actual.main[i]=ma[i];
    return actual;
}
