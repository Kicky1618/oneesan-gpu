#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

// Raw in-place H-major <-> L-major transpose for fixed-capacity bucket slots.
//
// GPU g owns eight slots. Slot s has capacity max(|B[g,s]|,|B[s,g]|), so the
// paired slots (g,s) and (s,g) always have exactly the same byte size. Swapping
// those raw slots therefore changes ownership without repacking any state.
//
// Safety rule: both peer reads of a chunk complete before either source slot is
// overwritten. v0 uses one staging buffer per GPU and host stream barriers.

static constexpr int BUCKET_TRANSPOSE_MAX_GPU=8;

struct BucketTransposeSlot {
    uint64_t off_bytes=0;
    uint64_t capacity_bytes=0;
    uint64_t logical_bytes=0;
};
struct BucketTransposePlan {
    int ngpu=0;
    std::array<std::array<BucketTransposeSlot,BUCKET_TRANSPOSE_MAX_GPU>,BUCKET_TRANSPOSE_MAX_GPU> slot{};
    std::array<uint64_t,BUCKET_TRANSPOSE_MAX_GPU> gpu_bytes{};
};

static BucketTransposePlan build_bucket_transpose_plan(
    const BucketPhysicalLayoutHost& layout,int ngpu=BUCKET_TRANSPOSE_MAX_GPU
){
    if(ngpu<2||ngpu>BUCKET_TRANSPOSE_MAX_GPU||(ngpu&1)){
        std::cerr<<"bucket transpose requires even 2..8 GPUs\n";std::exit(200);
    }
    BucketTransposePlan p;p.ngpu=ngpu;
    for(int g=0;g<ngpu;++g){
        uint64_t off=0;
        for(int s=0;s<ngpu;++s){
            uint64_t cap=uint64_t(layout.slot_capacity[g][s])*sizeof(Count);
            uint64_t logical=uint64_t(layout.pair[g][s].size)*sizeof(Count);
            p.slot[g][s]={off,cap,logical};off+=cap;
            if(cap!=uint64_t(layout.slot_capacity[s][g])*sizeof(Count)){
                std::cerr<<"bucket transpose asymmetric slot capacity\n";std::exit(201);
            }
        }
        p.gpu_bytes[g]=off;
    }
    return p;
}

static std::vector<std::array<std::pair<int,int>,BUCKET_TRANSPOSE_MAX_GPU/2>>
bucket_transpose_rounds(int ngpu){
    if(ngpu<2||ngpu>BUCKET_TRANSPOSE_MAX_GPU||(ngpu&1))std::exit(202);
    std::vector<int> ring(ngpu);for(int i=0;i<ngpu;++i)ring[i]=i;
    std::vector<std::array<std::pair<int,int>,BUCKET_TRANSPOSE_MAX_GPU/2>> out;
    out.resize(size_t(ngpu-1));
    std::vector<std::vector<int>> seen(size_t(ngpu),std::vector<int>(size_t(ngpu),0));
    for(int r=0;r<ngpu-1;++r){
        for(int i=0;i<ngpu/2;++i){
            int a=ring[i],b=ring[ngpu-1-i];if(a>b)std::swap(a,b);
            out[size_t(r)][size_t(i)]={a,b};
            if(seen[a][b]++)std::exit(203);
        }
        int last=ring.back();for(int i=ngpu-1;i>=2;--i)ring[i]=ring[i-1];ring[1]=last;
    }
    int edges=0;for(int a=0;a<ngpu;++a)for(int b=a+1;b<ngpu;++b)edges+=seen[a][b];
    if(edges!=ngpu*(ngpu-1)/2)std::exit(204);
    return out;
}

struct BucketTransposeCtx {
    int ngpu=0;
    size_t chunk_bytes=0;
    std::array<uint8_t*,BUCKET_TRANSPOSE_MAX_GPU> base{};
    std::array<uint8_t*,BUCKET_TRANSPOSE_MAX_GPU> staging{};
    std::array<cudaStream_t,BUCKET_TRANSPOSE_MAX_GPU> stream{};
    std::vector<std::array<std::pair<int,int>,BUCKET_TRANSPOSE_MAX_GPU/2>> rounds;
    double peer_gib=0.0;
    double local_gib=0.0;
    uint64_t transposes=0;

    static void ckbt(cudaError_t e,const char* what){
        if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<'\n';std::exit(205);}
    }

    void init(const BucketTransposePlan& plan,const std::array<Count*,BUCKET_TRANSPOSE_MAX_GPU>& ptrs,size_t chunk){
        ngpu=plan.ngpu;chunk_bytes=chunk;rounds=bucket_transpose_rounds(ngpu);
        if(!chunk_bytes){std::cerr<<"bucket transpose zero chunk\n";std::exit(206);}
        for(int g=0;g<ngpu;++g){
            base[g]=reinterpret_cast<uint8_t*>(ptrs[g]);
            ckbt(cudaSetDevice(g),"bucket transpose set device init");
            for(int peer=0;peer<ngpu;++peer)if(peer!=g){
                int can=0;ckbt(cudaDeviceCanAccessPeer(&can,g,peer),"bucket transpose peer query");
                if(!can){std::cerr<<"bucket transpose no P2P "<<g<<"->"<<peer<<'\n';std::exit(207);}
                cudaError_t e=cudaDeviceEnablePeerAccess(peer,0);
                if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)ckbt(e,"bucket transpose enable peer");
                if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
            }
            ckbt(cudaMalloc(&staging[g],chunk_bytes),"bucket transpose staging alloc");
            ckbt(cudaStreamCreateWithFlags(&stream[g],cudaStreamNonBlocking),"bucket transpose stream");
        }
    }

    void transpose(const BucketTransposePlan& plan){
        if(plan.ngpu!=ngpu)std::exit(208);
        for(auto const& round:rounds){
            uint64_t max_cap=0;
            for(int i=0;i<ngpu/2;++i){auto[a,b]=round[size_t(i)];max_cap=std::max(max_cap,plan.slot[a][b].capacity_bytes);}
            for(uint64_t off=0;off<max_cap;off+=chunk_bytes){
                std::array<size_t,BUCKET_TRANSPOSE_MAX_GPU> nbytes{};
                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];uint64_t cap=plan.slot[a][b].capacity_bytes;
                    if(off>=cap)continue;size_t n=size_t(std::min<uint64_t>(chunk_bytes,cap-off));nbytes[a]=nbytes[b]=n;
                    ckbt(cudaSetDevice(a),"bucket transpose set A fetch");
                    ckbt(cudaMemcpyPeerAsync(staging[a],a,base[b]+plan.slot[b][a].off_bytes+off,b,n,stream[a]),"bucket transpose fetch B to A");
                    ckbt(cudaSetDevice(b),"bucket transpose set B fetch");
                    ckbt(cudaMemcpyPeerAsync(staging[b],b,base[a]+plan.slot[a][b].off_bytes+off,a,n,stream[b]),"bucket transpose fetch A to B");
                    peer_gib+=double(2*n)/double(1ULL<<30);
                }
                for(int g=0;g<ngpu;++g)if(nbytes[g]){ckbt(cudaSetDevice(g),"bucket transpose set fetch sync");ckbt(cudaStreamSynchronize(stream[g]),"bucket transpose fetch sync");}
                for(int i=0;i<ngpu/2;++i){
                    auto[a,b]=round[size_t(i)];size_t n=nbytes[a];if(!n)continue;
                    ckbt(cudaSetDevice(a),"bucket transpose set A commit");
                    ckbt(cudaMemcpyAsync(base[a]+plan.slot[a][b].off_bytes+off,staging[a],n,cudaMemcpyDeviceToDevice,stream[a]),"bucket transpose commit A");
                    ckbt(cudaSetDevice(b),"bucket transpose set B commit");
                    ckbt(cudaMemcpyAsync(base[b]+plan.slot[b][a].off_bytes+off,staging[b],n,cudaMemcpyDeviceToDevice,stream[b]),"bucket transpose commit B");
                    local_gib+=double(2*n)/double(1ULL<<30);
                }
                for(int g=0;g<ngpu;++g)if(nbytes[g]){ckbt(cudaSetDevice(g),"bucket transpose set commit sync");ckbt(cudaStreamSynchronize(stream[g]),"bucket transpose commit sync");}
            }
        }
        ++transposes;
    }

    void release(){
        for(int g=0;g<ngpu;++g){ckbt(cudaSetDevice(g),"bucket transpose set release");if(stream[g])cudaStreamDestroy(stream[g]);if(staging[g])cudaFree(staging[g]);stream[g]=nullptr;staging[g]=nullptr;base[g]=nullptr;}
        ngpu=0;chunk_bytes=0;rounds.clear();
    }
};

#ifndef BUCKET_TRANSPOSE_STAGING_MULTIPLIER
#define BUCKET_TRANSPOSE_STAGING_MULTIPLIER 1
#endif

// Selection hooks. Both variants preserve BucketTransposeCtx's public API.
// The pipeline owns two chunk-sized staging buffers per GPU; sync/events own
// one, so drivers can use BUCKET_TRANSPOSE_STAGING_MULTIPLIER for exact HBM
// preflight accounting.
#if defined(BUCKET_TRANSPOSE_USE_PIPELINE)
#undef BUCKET_TRANSPOSE_STAGING_MULTIPLIER
#define BUCKET_TRANSPOSE_STAGING_MULTIPLIER 2
#include "gridfp_bucket_transpose_pipeline.cuh"
#define BucketTransposeCtx BucketTransposePipelineCtx
#elif defined(BUCKET_TRANSPOSE_USE_EVENTS)
#include "gridfp_bucket_transpose_events.cuh"
#define BucketTransposeCtx BucketTransposeEventCtx
#endif
