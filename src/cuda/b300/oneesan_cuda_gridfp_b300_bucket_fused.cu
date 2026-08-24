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
#include "../gridfp/oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/ramstream32_cpu_low_sparse.hpp"
#include "../gridfp/ramstream32_cpu_high.hpp"
#include "../gridfp/ramstream32_cpu_high_direct.hpp"
#include "../gridfp/ramstream32_gpu_direct.cuh"
#include "../gridfp/ramstream32_gpu_direct_gather.cuh"
#include "../gridfp/ramstream32_gpu_direct_gather_cross.cuh"
#include "../gridfp/ramstream32_gpu_direct_fused.cuh"
#include "../gridfp/ramstream32_bucket_layout.hpp"
#include "../gridfp/ramstream32_bucket_direct.hpp"
#include "../gridfp/ramstream32_bucket_fused.cuh"
#include "../gridfp/ramstream32_bucket_fused_v2.cuh"
#include "../gridfp/ramstream32_bucket_fused_async.cuh"
#include "gridfp_bucket_transpose.cuh"

static bool bkb_has_arg(int argc,char**argv,const char*x){for(int i=1;i<argc;++i)if(std::strcmp(argv[i],x)==0)return true;return false;}
static double bkb_since(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
static size_t bkb_env_mib(const char*name,size_t def){const char*s=std::getenv(name);return s?size_t(std::strtoull(s,nullptr,10)):def;}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int threads=argc>3?std::atoi(argv[3]):256;
    int grid_x=argc>4?std::atoi(argv[4]):16;
    int grid_y=argc>5?std::atoi(argv[5]):8;
    size_t chunk_mib=argc>6?size_t(std::strtoull(argv[6],nullptr,10)):bkb_env_mib("BUCKET_TRANSPOSE_CHUNK_MIB",4096);
    bool plan_only=bkb_has_arg(argc,argv,"--plan-only");
    constexpr int NG=BUCKET_NGPU;
    int W=n+1;
    if(W!=TARGET_W||LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W){std::cerr<<"binary split/width mismatch\n";return 1;}
    if(threads<=0||threads>1024||grid_x<=0||grid_y<=0||!chunk_mib){std::cerr<<"invalid launch/transpose geometry\n";return 2;}

    auto prep0=std::chrono::steady_clock::now();
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);
    GpuDirectFusedHost fused=build_gpu_direct_fused(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);
    BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);
    BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    BucketTransposePlan tplan=build_bucket_transpose_plan(phy,NG);
    double prepare_s=bkb_since(prep0);

    size_t view_bytes=2ull*NG*(layout.main_blocks.size()+layout.block_blocks.size())*sizeof(BucketPhysicalBlock);
    size_t metadata_bytes=borbit.bytes()+bfused.bytes()+view_bytes;
    size_t chunk_bytes=chunk_mib<<20;
    uint64_t max_gpu_bytes=0,min_gpu_bytes=~uint64_t(0),max_need=0;
    for(int g=0;g<NG;++g){max_gpu_bytes=std::max(max_gpu_bytes,tplan.gpu_bytes[g]);min_gpu_bytes=std::min(min_gpu_bytes,tplan.gpu_bytes[g]);max_need=std::max<uint64_t>(max_need,tplan.gpu_bytes[g]+metadata_bytes+chunk_bytes);}
    std::cout<<std::setprecision(15)
        <<"backend=gridfp-b300-bucket-fused-v0.1-plan"
        <<" n="<<n
        <<" main_states="<<layout.main_size
        <<" blocked_states="<<layout.block_size
        <<" authoritative_tib="<<double((layout.main_size+layout.block_size)*sizeof(Count))/double(1ULL<<40)
        <<" max_gpu_authoritative_gib="<<double(max_gpu_bytes)/double(1ULL<<30)
        <<" gpu_authoritative_spread_mib="<<double(max_gpu_bytes-min_gpu_bytes)/double(1<<20)
        <<" metadata_mib_per_gpu="<<double(metadata_bytes)/double(1<<20)
        <<" transpose_chunk_mib="<<chunk_mib
        <<" max_device_need_gib="<<double(max_need)/double(1ULL<<30)
        <<" locator_bits="<<BUCKET_LOCATOR_BITS
        <<" transposes_per_residue="<<(2*W-1)
        <<" high_launches_per_row="<<(2*HIGH_LUT_K)
        <<" low_launches_per_row="<<(2*LOW_LUT_K)
        <<" closure_atomic=0"
        <<" group_scratch_bytes=0"
        <<" prepare_s="<<prepare_s<<'\n';
    if(plan_only)return 0;

    int visible=0;ck(cudaGetDeviceCount(&visible),"bucket b300 device count");if(visible<NG){std::cerr<<"need 8 CUDA devices, visible="<<visible<<'\n';return 3;}
    size_t reserve_bytes=bkb_env_mib("BUCKET_RESERVE_MIB",8192)<<20;
    for(int g=0;g<NG;++g){
        ck(cudaSetDevice(g),"bucket b300 set device memcheck");size_t freeb=0,totalb=0;ck(cudaMemGetInfo(&freeb,&totalb),"bucket b300 mem info");
        uint64_t need=tplan.gpu_bytes[g]+metadata_bytes+chunk_bytes+reserve_bytes;
        std::cerr<<"gpu"<<g<<" free_gib="<<double(freeb)/double(1ULL<<30)<<" need_with_reserve_gib="<<double(need)/double(1ULL<<30)<<'\n';
        if(need>freeb){std::cerr<<"insufficient HBM on gpu"<<g<<'\n';return 4;}
    }

    std::array<Count*,NG> base{};
    std::array<std::array<Count*,NG>,NG> slots{};
    std::array<BucketFusedDeviceTables,NG> tables;
    for(int g=0;g<NG;++g){
        ck(cudaSetDevice(g),"bucket b300 set alloc device");
        ck(cudaMalloc(&base[g],size_t(tplan.gpu_bytes[g])),"bucket b300 authoritative alloc");
        ck(cudaMemset(base[g],0,size_t(tplan.gpu_bytes[g])),"bucket b300 authoritative zero");
        uint8_t*bb=reinterpret_cast<uint8_t*>(base[g]);
        for(int s=0;s<NG;++s)slots[g][s]=reinterpret_cast<Count*>(bb+tplan.slot[g][s].off_bytes);
        ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"bucket b300 modulus");
        tables[g].install_metadata(layout,borbit,bfused);
        tables[g].bind_owner(uint32_t(g),phy,slots[g]);
    }

    // Start directly in L-major: B[a,b] lives on GPU b, slot a.
    MateID init=MateID(R)<<(2*(W-1));
    BucketAddress ia=bucket_rank_main_host(init,storage,layout,owner,phy);
    Count one=1;
    ck(cudaSetDevice(ia.owner_l),"bucket b300 init set device");
    ck(cudaMemcpy(reinterpret_cast<uint8_t*>(base[ia.owner_l])+tplan.slot[ia.owner_l][ia.owner_h].off_bytes+uint64_t(ia.off)*sizeof(Count),&one,sizeof(one),cudaMemcpyHostToDevice),"bucket b300 init state");

    BucketTransposeCtx tx;tx.init(tplan,base,chunk_bytes);
    double high_s=0.0,low_s=0.0,transpose_s=0.0;
    auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){
        auto t=std::chrono::steady_clock::now();
        for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"bucket b300 high set device");bucket_launch_high_fused(layout,threads,grid_x,grid_y);}
        bucket_sync_devices(NG);high_s+=bkb_since(t);

        t=std::chrono::steady_clock::now();tx.transpose(tplan);transpose_s+=bkb_since(t);

        t=std::chrono::steady_clock::now();
        for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"bucket b300 low set device");bucket_launch_low_fused_v2(layout,threads,grid_x,grid_y);}
        bucket_sync_devices(NG);low_s+=bkb_since(t);

        if(row+1<W){t=std::chrono::steady_clock::now();tx.transpose(tplan);transpose_s+=bkb_since(t);}
        std::cerr<<"row "<<(row+1)<<'/'<<W<<" high_s="<<high_s<<" low_s="<<low_s<<" transpose_s="<<transpose_s<<'\n';
    }
    double wall_s=bkb_since(wall0);

    // End in H-major: B[a,b] lives on GPU a, slot b.
    BucketAddress fa=bucket_rank_main_host(MateID(R),storage,layout,owner,phy);
    Count answer=0;ck(cudaSetDevice(fa.owner_h),"bucket b300 answer set device");
    ck(cudaMemcpy(&answer,reinterpret_cast<uint8_t*>(base[fa.owner_h])+tplan.slot[fa.owner_h][fa.owner_l].off_bytes+uint64_t(fa.off)*sizeof(Count),sizeof(answer),cudaMemcpyDeviceToHost),"bucket b300 answer");

    std::cout<<"backend=gridfp-b300-bucket-fused-v0.1"
             <<" n="<<n<<" residue="<<answer<<" modulus="<<mod
             <<" high_s="<<high_s<<" low_s="<<low_s<<" transpose_s="<<transpose_s
             <<" wall_s="<<wall_s<<" transposes="<<tx.transposes
             <<" peer_gib="<<tx.peer_gib<<" local_commit_gib="<<tx.local_gib
             <<" transpose_chunk_mib="<<chunk_mib
             <<" closure_atomic=0 group_scratch_bytes=0\n";

    bool ok=true;if(n==21&&mod==4294967291u&&answer!=998035516u){std::cerr<<"n21 residue mismatch got="<<answer<<" expected=998035516\n";ok=false;}
    tx.release();
    for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"bucket b300 release set device");tables[g].release();cudaFree(base[g]);base[g]=nullptr;}
    return ok?0:5;
}
