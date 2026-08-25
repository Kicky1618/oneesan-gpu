#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/ramstream32_cpu_low_sparse.hpp"
#include "../gridfp/ramstream32_cpu_high.hpp"
#include "../gridfp/ramstream32_cpu_high_direct.hpp"
#include "../gridfp/ramstream32_gpu_direct.cuh"
#include "../gridfp/ramstream32_gpu_direct_gather.cuh"
#include "../gridfp/ramstream32_gpu_direct_gather_cross.cuh"
#include "../gridfp/ramstream32_gpu_direct_fused.cuh"
#include "../gridfp/ramstream32_gpu_direct_fused_validate.hpp"
#include "../gridfp/ramstream32_bucket_layout.hpp"
#include "../gridfp/ramstream32_bucket_direct.hpp"
#include "../gridfp/ramstream32_bucket_fused.cuh"
#include "../gridfp/ramstream32_bucket_fused_async.cuh"
#include "../gridfp/ramstream32_reverse_desc.hpp"
#include "../gridfp/ramstream32_reverse_orbit.hpp"
#include "../gridfp/ramstream32_bucket_reverse_atomic.cuh"
#ifdef BUCKET_SNAKE_REVERSE_FUSED
#include "../gridfp/ramstream32_bucket_reverse_fused.cuh"
#if GPU_DIRECT_PM_ACCUM
#include "../gridfp/ramstream32_bucket_reverse_fused_pm.cuh"
#endif
#endif
#include "gridfp_bucket_transpose.cuh"

static bool bsn_has_arg(int argc,char**argv,const char*x){for(int i=1;i<argc;++i)if(std::strcmp(argv[i],x)==0)return true;return false;}
static double bsn_since(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
static size_t bsn_env_mib(const char*name,size_t def){const char*s=std::getenv(name);return s?size_t(std::strtoull(s,nullptr,10)):def;}

struct SnakeReverseHost {
    ReverseBucketAtomicHost atomic;
#ifdef BUCKET_SNAKE_REVERSE_FUSED
    ReverseBucketFusedHost fused;
#endif
    size_t bytes() const {
#ifdef BUCKET_SNAKE_REVERSE_FUSED
        return reverse_bucket_orbit_bytes(atomic)+fused.bytes();
#else
        return atomic.bytes();
#endif
    }
};

static SnakeReverseHost bsn_build_reverse(
    const StorageFactorHost&storage,const StorageLayout&layout,const BucketOwnerHost&owner,
    const ReverseLowDescHost&rlow,const ReverseHighDescHost&rhigh,
    const ReverseOrbitHost&rlo,const ReverseOrbitHost&rhi
){
    SnakeReverseHost z;
    z.atomic=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);
#ifdef BUCKET_SNAKE_REVERSE_FUSED
    z.fused=build_reverse_bucket_fused(layout,z.atomic);
#endif
    return z;
}

struct SnakeReverseDeviceTables {
#ifdef BUCKET_SNAKE_REVERSE_FUSED
    ReverseBucketFusedDeviceTables impl;
#else
    ReverseBucketAtomicDeviceTables impl;
#endif
    void install(const SnakeReverseHost&h){
#ifdef BUCKET_SNAKE_REVERSE_FUSED
        impl.install(h.atomic,h.fused);
#else
        impl.install(h.atomic);
#endif
    }
    void release(){impl.release();}
};

static void bsn_launch_reverse_low(const StorageLayout&layout,int threads,int gx,int gy){
#ifdef BUCKET_SNAKE_REVERSE_FUSED
#if GPU_DIRECT_PM_ACCUM
    bucket_launch_reverse_low_fused_pm(layout,threads,gx,gy);
#else
    bucket_launch_reverse_low_fused(layout,threads,gx,gy);
#endif
#else
    bucket_launch_reverse_low_atomic(layout,threads,gx,gy);
#endif
}
static void bsn_launch_reverse_high(const StorageLayout&layout,int threads,int gx,int gy){
#ifdef BUCKET_SNAKE_REVERSE_FUSED
#if GPU_DIRECT_PM_ACCUM
    bucket_launch_reverse_high_fused_pm(layout,threads,gx,gy);
#else
    bucket_launch_reverse_high_fused(layout,threads,gx,gy);
#endif
#else
    bucket_launch_reverse_high_atomic(layout,threads,gx,gy);
#endif
}

#ifdef BUCKET_SNAKE_REVERSE_FUSED
static constexpr const char* BSN_BACKEND_PLAN="gridfp-b300-bucket-snake-fused-v0.2-plan";
static constexpr const char* BSN_BACKEND="gridfp-b300-bucket-snake-fused-v0.2";
static constexpr int BSN_REVERSE_CLOSURE_ATOMIC=0;
#else
static constexpr const char* BSN_BACKEND_PLAN="gridfp-b300-bucket-snake-atomic-v0.1-plan";
static constexpr const char* BSN_BACKEND="gridfp-b300-bucket-snake-atomic-v0.1";
static constexpr int BSN_REVERSE_CLOSURE_ATOMIC=1;
#endif

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int threads=argc>3?std::atoi(argv[3]):256,gx=argc>4?std::atoi(argv[4]):16,gy=argc>5?std::atoi(argv[5]):8;
    size_t chunk_mib=argc>6?size_t(std::strtoull(argv[6],nullptr,10)):bsn_env_mib("BUCKET_TRANSPOSE_CHUNK_MIB",1024);bool plan_only=bsn_has_arg(argc,argv,"--plan-only");
    constexpr int NG=BUCKET_NGPU;int W=n+1;if(W!=TARGET_W||LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;if(threads<=0||threads>1024||gx<=0||gy<=0||!chunk_mib)return 2;

    auto prep0=std::chrono::steady_clock::now();build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);SnakeReverseHost reverse=bsn_build_reverse(storage,layout,owner,rlow,rhigh,rlo,rhi);
    BucketTransposePlan tplan=build_bucket_transpose_plan(phy,NG);double prepare_s=bsn_since(prep0);

    size_t view_bytes=2ull*NG*(layout.main_blocks.size()+layout.block_blocks.size())*sizeof(BucketPhysicalBlock);
    size_t forward_meta=borbit.bytes()+bfused.bytes()+view_bytes,reverse_meta=reverse.bytes();size_t metadata_bytes=forward_meta+reverse_meta;size_t chunk_bytes=chunk_mib<<20;
    uint64_t max_gpu=0,min_gpu=~uint64_t(0),peer_per_transpose=0,max_need=0;
    for(int g=0;g<NG;++g){max_gpu=std::max(max_gpu,tplan.gpu_bytes[g]);min_gpu=std::min(min_gpu,tplan.gpu_bytes[g]);max_need=std::max<uint64_t>(max_need,tplan.gpu_bytes[g]+metadata_bytes+chunk_bytes);for(int s=g+1;s<NG;++s)peer_per_transpose+=2ull*tplan.slot[g][s].capacity_bytes;}
    long double snake_peer_tib=static_cast<long double>(peer_per_transpose)*static_cast<long double>(W)/static_cast<long double>(1ULL<<40);
    std::cout<<std::setprecision(15)<<"backend="<<BSN_BACKEND_PLAN<<" n="<<n
             <<" states="<<(layout.main_size+layout.block_size)<<" authoritative_tib="<<double((layout.main_size+layout.block_size)*sizeof(Count))/double(1ULL<<40)
             <<" max_gpu_authoritative_gib="<<double(max_gpu)/double(1ULL<<30)<<" gpu_spread_mib="<<double(max_gpu-min_gpu)/double(1<<20)
             <<" forward_metadata_mib="<<double(forward_meta)/double(1<<20)<<" reverse_metadata_mib="<<double(reverse_meta)/double(1<<20)
             <<" metadata_mib_per_gpu="<<double(metadata_bytes)/double(1<<20)<<" transpose_chunk_mib="<<chunk_mib<<" max_device_need_gib="<<double(max_need)/double(1ULL<<30)
             <<" transposes_per_residue="<<W<<" standard_transposes="<<(2*W-1)<<" peer_gib_per_transpose="<<double(peer_per_transpose)/double(1ULL<<30)
             <<" snake_peer_tib_per_residue="<<double(snake_peer_tib)
             <<" reverse_closure_atomic="<<BSN_REVERSE_CLOSURE_ATOMIC<<" forward_closure_atomic=0 pm_accum="<<GPU_DIRECT_PM_ACCUM<<" prepare_s="<<prepare_s<<'\n';
    if(plan_only)return 0;

    int visible=0;ck(cudaGetDeviceCount(&visible),"snake device count");if(visible<NG){std::cerr<<"need 8 GPUs visible="<<visible<<'\n';return 3;}size_t reserve=bsn_env_mib("BUCKET_RESERVE_MIB",8192)<<20;
    for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake mem set");size_t freeb=0,totalb=0;ck(cudaMemGetInfo(&freeb,&totalb),"snake mem info");uint64_t need=tplan.gpu_bytes[g]+metadata_bytes+chunk_bytes+reserve;std::cerr<<"gpu"<<g<<" free_gib="<<double(freeb)/double(1ULL<<30)<<" need_with_reserve_gib="<<double(need)/double(1ULL<<30)<<'\n';if(need>freeb)return 4;}

    std::array<Count*,NG>base{};std::array<std::array<Count*,NG>,NG>slots{};std::array<BucketFusedDeviceTables,NG>ft;std::array<SnakeReverseDeviceTables,NG>rt;
    for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake alloc set");ck(cudaMalloc(&base[g],size_t(tplan.gpu_bytes[g])),"snake auth alloc");ck(cudaMemset(base[g],0,size_t(tplan.gpu_bytes[g])),"snake auth zero");uint8_t*bb=reinterpret_cast<uint8_t*>(base[g]);for(int s=0;s<NG;++s)slots[g][s]=reinterpret_cast<Count*>(bb+tplan.slot[g][s].off_bytes);ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"snake modulus");ft[g].install_metadata(layout,borbit,bfused);rt[g].install(reverse);ft[g].bind_owner(uint32_t(g),phy,slots[g]);}

    MateID init=MateID(R)<<(2*(W-1));BucketAddress ia=bucket_rank_main_host(init,storage,layout,owner,phy);Count one=1;ck(cudaSetDevice(ia.owner_l),"snake init set");ck(cudaMemcpy(reinterpret_cast<uint8_t*>(base[ia.owner_l])+tplan.slot[ia.owner_l][ia.owner_h].off_bytes+uint64_t(ia.off)*sizeof(Count),&one,sizeof(one),cudaMemcpyHostToDevice),"snake init");
    BucketTransposeCtx tx;tx.init(tplan,base,chunk_bytes);double fh=0,fl=0,rl=0,rh=0,ts=0;auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){
        if((row&1)==0){auto t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake fhigh set");bucket_launch_high_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);fh+=bsn_since(t);t=std::chrono::steady_clock::now();tx.transpose(tplan);ts+=bsn_since(t);t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake flow set");bucket_launch_low_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);fl+=bsn_since(t);}
        else{auto t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake rlow set");bsn_launch_reverse_low(layout,threads,gx,gy);}bucket_sync_devices(NG);rl+=bsn_since(t);t=std::chrono::steady_clock::now();tx.transpose(tplan);ts+=bsn_since(t);t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake rhigh set");bsn_launch_reverse_high(layout,threads,gx,gy);}bucket_sync_devices(NG);rh+=bsn_since(t);}
        std::cerr<<"row "<<(row+1)<<'/'<<W<<" fh="<<fh<<" fl="<<fl<<" rl="<<rl<<" rh="<<rh<<" transpose="<<ts<<'\n';
    }
    double wall=bsn_since(wall0);BucketAddress fa=bucket_rank_main_host(MateID(R),storage,layout,owner,phy);bool final_lmajor=(W%2)==0;int agpu=final_lmajor?int(fa.owner_l):int(fa.owner_h),aslot=final_lmajor?int(fa.owner_h):int(fa.owner_l);Count answer=0;ck(cudaSetDevice(agpu),"snake answer set");ck(cudaMemcpy(&answer,reinterpret_cast<uint8_t*>(base[agpu])+tplan.slot[agpu][aslot].off_bytes+uint64_t(fa.off)*sizeof(Count),sizeof(answer),cudaMemcpyDeviceToHost),"snake answer");
    std::cout<<"backend="<<BSN_BACKEND<<" n="<<n<<" residue="<<answer<<" modulus="<<mod<<" forward_high_s="<<fh<<" forward_low_s="<<fl<<" reverse_low_s="<<rl<<" reverse_high_s="<<rh<<" transpose_s="<<ts<<" wall_s="<<wall<<" transposes="<<tx.transposes<<" peer_gib="<<tx.peer_gib<<" final_layout="<<(final_lmajor?"L-major":"H-major")<<" reverse_closure_atomic="<<BSN_REVERSE_CLOSURE_ATOMIC<<" forward_closure_atomic=0 pm_accum="<<GPU_DIRECT_PM_ACCUM<<"\n";
    bool ok=tx.transposes==uint64_t(W);if(n==21&&mod==4294967291u&&answer!=998035516u){std::cerr<<"n21 snake residue mismatch got="<<answer<<" expected=998035516\n";ok=false;}
    tx.release();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake release set");rt[g].release();ft[g].release();cudaFree(base[g]);}return ok?0:5;
}
