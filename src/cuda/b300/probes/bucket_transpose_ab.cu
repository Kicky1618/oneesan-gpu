#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../../gridfp/oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../../gridfp/ramstream32_bucket_layout.hpp"
#include "../gridfp_bucket_transpose.cuh"
#include "../gridfp_bucket_transpose_events.cuh"
#include "../gridfp_bucket_transpose_pipeline.cuh"

static __global__ void bt_fill(uint32_t*p,uint64_t n,uint32_t v){
    uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    uint64_t step=uint64_t(gridDim.x)*blockDim.x;
    for(;i<n;i+=step)p[i]=v;
}

static void btc(cudaError_t e,const char*w){if(e!=cudaSuccess){std::cerr<<w<<": "<<cudaGetErrorString(e)<<'\n';std::exit(1);}}
static double secs(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}

static BucketTransposePlan make_plan(int ngpu,uint64_t slot_bytes){
    BucketTransposePlan p;p.ngpu=ngpu;
    for(int g=0;g<ngpu;++g){
        uint64_t off=0;
        for(int s=0;s<ngpu;++s){
            uint64_t cap=g==s?0:slot_bytes;
            p.slot[g][s]={off,cap,cap};off+=cap;
        }
        p.gpu_bytes[g]=off;
    }
    return p;
}

static void fill_all(const BucketTransposePlan&p,const std::array<Count*,8>&base,bool transposed){
    for(int g=0;g<p.ngpu;++g){
        btc(cudaSetDevice(g),"bt fill set device");auto*b=reinterpret_cast<uint8_t*>(base[g]);
        for(int s=0;s<p.ngpu;++s){
            uint64_t bytes=p.slot[g][s].capacity_bytes;if(!bytes)continue;
            uint32_t v=transposed?uint32_t(s*16+g):uint32_t(g*16+s);
            uint32_t*ptr=reinterpret_cast<uint32_t*>(b+p.slot[g][s].off_bytes);uint64_t n=bytes/sizeof(uint32_t);
            unsigned blocks=unsigned(std::min<uint64_t>(65535,(n+255)/256));
            bt_fill<<<blocks,256>>>(ptr,n,v);btc(cudaGetLastError(),"bt fill kernel");
        }
    }
    for(int g=0;g<p.ngpu;++g){btc(cudaSetDevice(g),"bt fill sync set");btc(cudaDeviceSynchronize(),"bt fill sync");}
}

static bool verify_all(const BucketTransposePlan&p,const std::array<Count*,8>&base,bool transposed){
    bool ok=true;
    for(int g=0;g<p.ngpu;++g){
        btc(cudaSetDevice(g),"bt verify set device");auto*b=reinterpret_cast<uint8_t*>(base[g]);
        for(int s=0;s<p.ngpu;++s){
            uint64_t bytes=p.slot[g][s].capacity_bytes;if(!bytes)continue;
            uint32_t want=transposed?uint32_t(s*16+g):uint32_t(g*16+s),first=0,last=0;uint8_t*ptr=b+p.slot[g][s].off_bytes;
            btc(cudaMemcpy(&first,ptr,sizeof(first),cudaMemcpyDeviceToHost),"bt verify first");
            btc(cudaMemcpy(&last,ptr+bytes-sizeof(last),sizeof(last),cudaMemcpyDeviceToHost),"bt verify last");
            if(first!=want||last!=want){std::cerr<<"FAIL transpose pattern gpu="<<g<<" slot="<<s<<" first="<<first<<" last="<<last<<" want="<<want<<'\n';ok=false;}
        }
    }
    return ok;
}

template<class Ctx>
static bool run_mode(const char*name,const BucketTransposePlan&p,const std::array<Count*,8>&base,size_t chunk_bytes,int repeats){
    fill_all(p,base,false);Ctx ctx;ctx.init(p,base,chunk_bytes);double total_s=0.0;
    for(int r=0;r<repeats;++r){
        auto t=std::chrono::steady_clock::now();ctx.transpose(p);double dt=secs(t);total_s+=dt;
        if(!verify_all(p,base,(r&1)==0)){ctx.release();return false;}
    }
    double avg=total_s/repeats,peer_per=ctx.peer_gib/repeats;
    std::cout<<std::setprecision(12)
             <<"transpose_mode="<<name<<" ngpu="<<p.ngpu<<" repeats="<<repeats
             <<" avg_s="<<avg<<" peer_gib_per_transpose="<<peer_per
             <<" local_gib_per_transpose="<<(ctx.local_gib/repeats)
             <<" peer_gib_s="<<(peer_per/avg)<<" chunk_mib="<<(chunk_bytes>>20)<<'\n';
    ctx.release();return true;
}

int main(int argc,char**argv){
    int ngpu=argc>1?std::atoi(argv[1]):8;
    size_t slot_mib=argc>2?size_t(std::strtoull(argv[2],nullptr,10)):256;
    size_t chunk_mib=argc>3?size_t(std::strtoull(argv[3],nullptr,10)):64;
    int repeats=argc>4?std::atoi(argv[4]):4;
    std::string mode=argc>5?argv[5]:"all";
    if(ngpu<2||ngpu>8||(ngpu&1)||!slot_mib||!chunk_mib||repeats<1){
        std::cerr<<"usage: [ngpu even 2..8] [slot_mib] [chunk_mib] [repeats] [sync|events|pipeline|both|all]\n";return 2;
    }
    int visible=0;btc(cudaGetDeviceCount(&visible),"bt device count");if(visible<ngpu){std::cerr<<"need "<<ngpu<<" GPUs, visible="<<visible<<'\n';return 3;}
    uint64_t slot_bytes=uint64_t(slot_mib)<<20;size_t chunk_bytes=chunk_mib<<20;BucketTransposePlan plan=make_plan(ngpu,slot_bytes);
    std::array<Count*,8>base{};for(int g=0;g<ngpu;++g){btc(cudaSetDevice(g),"bt alloc set");btc(cudaMalloc(&base[g],size_t(plan.gpu_bytes[g])),"bt base alloc");}

    bool ok=true;
    if(mode=="sync"||mode=="both"||mode=="all")ok=run_mode<BucketTransposeCtx>("sync",plan,base,chunk_bytes,repeats)&&ok;
    if(mode=="events"||mode=="both"||mode=="all")ok=run_mode<BucketTransposeEventCtx>("events",plan,base,chunk_bytes,repeats)&&ok;
    if(mode=="pipeline"||mode=="all")ok=run_mode<BucketTransposePipelineCtx>("pipeline",plan,base,chunk_bytes,repeats)&&ok;
    if(mode!="sync"&&mode!="events"&&mode!="pipeline"&&mode!="both"&&mode!="all"){std::cerr<<"invalid mode\n";ok=false;}

    for(int g=0;g<ngpu;++g){btc(cudaSetDevice(g),"bt free set");cudaFree(base[g]);}
    return ok?0:4;
}
