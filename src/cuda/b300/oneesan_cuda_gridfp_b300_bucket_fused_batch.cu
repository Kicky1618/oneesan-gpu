#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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
#include "gridfp_bucket_transpose.cuh"

static double bkb_batch_since(std::chrono::steady_clock::time_point t){
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
static size_t bkb_batch_env_mib(const char*name,size_t def){
    const char*s=std::getenv(name);return s?size_t(std::strtoull(s,nullptr,10)):def;
}

int main(int argc,char**argv){
    // Compatibility with scripts/solve/solve_b300_exact_batch.py:
    //   binary n target_mib max_window gpus p1 p2 ...
    if(argc<6){
        std::cerr<<"usage: "<<argv[0]<<" n target_mib max_window gpus modulus...\n";
        return 2;
    }
    int n=std::atoi(argv[1]);
    size_t legacy_target_mib=size_t(std::strtoull(argv[2],nullptr,10));
    int legacy_max_window=std::atoi(argv[3]);
    int requested_gpus=std::atoi(argv[4]);
    std::vector<Count> mods;
    for(int i=5;i<argc;++i){
        unsigned long v=std::strtoul(argv[i],nullptr,10);
        if(v<3||v>0xfffffffful){std::cerr<<"invalid modulus: "<<argv[i]<<'\n';return 2;}
        mods.push_back(Count(v));
    }

    constexpr int NG=BUCKET_NGPU;
    int W=n+1;
    if(W!=TARGET_W||LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W){std::cerr<<"binary split/width mismatch\n";return 1;}
    if(requested_gpus!=NG){std::cerr<<"bucket batch backend currently requires exactly 8 GPUs\n";return 2;}
    if(legacy_max_window<1){std::cerr<<"invalid max_window\n";return 2;}

#ifdef BUCKET_TRANSPOSE_USE_EVENTS
    constexpr size_t DEFAULT_CHUNK_MIB=1024;
    const char*transpose_mode="events";
#else
    constexpr size_t DEFAULT_CHUNK_MIB=4096;
    const char*transpose_mode="sync";
#endif
    size_t chunk_mib=bkb_batch_env_mib("BUCKET_TRANSPOSE_CHUNK_MIB",DEFAULT_CHUNK_MIB);
    size_t chunk_bytes=chunk_mib<<20;
    int threads=int(bkb_batch_env_mib("BUCKET_THREADS",256));
    int grid_x=int(bkb_batch_env_mib("BUCKET_GRID_X",16));
    int grid_y=int(bkb_batch_env_mib("BUCKET_GRID_Y",8));
    if(!chunk_mib||threads<=0||threads>1024||grid_x<=0||grid_y<=0)return 2;

    auto prep0=std::chrono::steady_clock::now();
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);
    HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);
    GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);
    BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);
    BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);
    BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    BucketTransposePlan tplan=build_bucket_transpose_plan(phy,NG);
    double prepare_s=bkb_batch_since(prep0);

    size_t view_bytes=2ull*NG*(layout.main_blocks.size()+layout.block_blocks.size())*sizeof(BucketPhysicalBlock);
    size_t metadata_bytes=borbit.bytes()+bfused.bytes()+view_bytes;
    uint64_t max_gpu_bytes=0,min_gpu_bytes=~uint64_t(0);
    for(int g=0;g<NG;++g){max_gpu_bytes=std::max(max_gpu_bytes,tplan.gpu_bytes[g]);min_gpu_bytes=std::min(min_gpu_bytes,tplan.gpu_bytes[g]);}

    std::cerr<<std::setprecision(15)
        <<"bucket batch prepare"
        <<" n="<<n
        <<" residues="<<mods.size()
        <<" transpose_mode="<<transpose_mode
        <<" transpose_chunk_mib="<<chunk_mib
        <<" legacy_target_mib_ignored="<<legacy_target_mib
        <<" legacy_max_window_ignored="<<legacy_max_window
        <<" max_gpu_authoritative_gib="<<double(max_gpu_bytes)/double(1ULL<<30)
        <<" gpu_authoritative_spread_mib="<<double(max_gpu_bytes-min_gpu_bytes)/double(1<<20)
        <<" metadata_mib_per_gpu="<<double(metadata_bytes)/double(1<<20)
        <<" prepare_s="<<prepare_s<<'\n';

    int visible=0;ck(cudaGetDeviceCount(&visible),"bucket batch device count");
    if(visible<NG){std::cerr<<"need 8 CUDA devices, visible="<<visible<<'\n';return 3;}
    size_t reserve_bytes=bkb_batch_env_mib("BUCKET_RESERVE_MIB",8192)<<20;
    for(int g=0;g<NG;++g){
        ck(cudaSetDevice(g),"bucket batch set memcheck device");
        size_t freeb=0,totalb=0;ck(cudaMemGetInfo(&freeb,&totalb),"bucket batch mem info");
        uint64_t need=tplan.gpu_bytes[g]+metadata_bytes+chunk_bytes+reserve_bytes;
        std::cerr<<"gpu"<<g<<" free_gib="<<double(freeb)/double(1ULL<<30)
                 <<" need_with_reserve_gib="<<double(need)/double(1ULL<<30)<<'\n';
        if(need>freeb){std::cerr<<"insufficient HBM on gpu"<<g<<'\n';return 4;}
    }

    std::array<Count*,NG> base{};
    std::array<std::array<Count*,NG>,NG> slots{};
    std::array<BucketFusedDeviceTables,NG> tables;
    for(int g=0;g<NG;++g){
        ck(cudaSetDevice(g),"bucket batch set alloc device");
        ck(cudaMalloc(&base[g],size_t(tplan.gpu_bytes[g])),"bucket batch authoritative alloc");
        uint8_t*bb=reinterpret_cast<uint8_t*>(base[g]);
        for(int s=0;s<NG;++s)slots[g][s]=reinterpret_cast<Count*>(bb+tplan.slot[g][s].off_bytes);
        tables[g].install_metadata(layout,borbit,bfused);
        tables[g].bind_owner(uint32_t(g),phy,slots[g]);
    }

    BucketTransposeCtx tx;tx.init(tplan,base,chunk_bytes);
    MateID init=MateID(R)<<(2*(W-1));
    BucketAddress ia=bucket_rank_main_host(init,storage,layout,owner,phy);
    BucketAddress fa=bucket_rank_main_host(MateID(R),storage,layout,owner,phy);

    bool all_ok=true;
    for(size_t mi=0;mi<mods.size();++mi){
        Count mod=mods[mi];
        auto reset0=std::chrono::steady_clock::now();
        for(int g=0;g<NG;++g){
            ck(cudaSetDevice(g),"bucket batch reset set device");
            ck(cudaMemsetAsync(base[g],0,size_t(tplan.gpu_bytes[g]),nullptr),"bucket batch authoritative zero");
            ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"bucket batch modulus");
        }
        bucket_sync_devices(NG);
        Count one=1;
        ck(cudaSetDevice(ia.owner_l),"bucket batch init set device");
        ck(cudaMemcpy(reinterpret_cast<uint8_t*>(base[ia.owner_l])
            +tplan.slot[ia.owner_l][ia.owner_h].off_bytes+uint64_t(ia.off)*sizeof(Count),
            &one,sizeof(one),cudaMemcpyHostToDevice),"bucket batch init state");
        double reset_s=bkb_batch_since(reset0);

        uint64_t tx0=tx.transposes;
        double peer0=tx.peer_gib,local0=tx.local_gib;
        double high_s=0.0,low_s=0.0,transpose_s=0.0;
        auto wall0=std::chrono::steady_clock::now();
        for(int row=0;row<W;++row){
            auto t=std::chrono::steady_clock::now();
            for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"bucket batch high set device");bucket_launch_high_fused(layout,threads,grid_x,grid_y);}
            bucket_sync_devices(NG);high_s+=bkb_batch_since(t);

            t=std::chrono::steady_clock::now();tx.transpose(tplan);transpose_s+=bkb_batch_since(t);

            t=std::chrono::steady_clock::now();
            for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"bucket batch low set device");bucket_launch_low_fused(layout,threads,grid_x,grid_y);}
            bucket_sync_devices(NG);low_s+=bkb_batch_since(t);

            if(row+1<W){t=std::chrono::steady_clock::now();tx.transpose(tplan);transpose_s+=bkb_batch_since(t);}
        }
        double wall_s=bkb_batch_since(wall0);

        Count answer=0;
        ck(cudaSetDevice(fa.owner_h),"bucket batch answer set device");
        ck(cudaMemcpy(&answer,reinterpret_cast<uint8_t*>(base[fa.owner_h])
            +tplan.slot[fa.owner_h][fa.owner_l].off_bytes+uint64_t(fa.off)*sizeof(Count),
            sizeof(answer),cudaMemcpyDeviceToHost),"bucket batch answer");

        uint64_t tx_count=tx.transposes-tx0;
        double peer_gib=tx.peer_gib-peer0,local_gib=tx.local_gib-local0;
        std::cout<<std::setprecision(15)
                 <<"backend=gridfp-b300-bucket-fused-batch-v0.1"
                 <<" residue="<<answer
                 <<" modulus="<<mod
                 <<" wall_s="<<wall_s
                 <<" reset_s="<<reset_s
                 <<" high_s="<<high_s
                 <<" low_s="<<low_s
                 <<" transpose_s="<<transpose_s
                 <<" transposes="<<tx_count
                 <<" peer_gib="<<peer_gib
                 <<" local_commit_gib="<<local_gib
                 <<" transpose_mode="<<transpose_mode
                 <<" transpose_chunk_mib="<<chunk_mib
                 <<" closure_atomic=0 group_scratch_bytes=0\n";
        std::cout.flush();

        if(tx_count!=uint64_t(2*W-1)){
            std::cerr<<"transpose count mismatch for modulus="<<mod<<" got="<<tx_count
                     <<" expected="<<(2*W-1)<<'\n';
            all_ok=false;break;
        }
        if(n==21&&mod==4294967291u&&answer!=998035516u){
            std::cerr<<"n21 residue mismatch got="<<answer<<" expected=998035516\n";
            all_ok=false;break;
        }
        std::cerr<<"completed residue "<<(mi+1)<<'/'<<mods.size()<<" modulus="<<mod
                 <<" wall_s="<<wall_s<<'\n';
    }

    tx.release();
    for(int g=0;g<NG;++g){
        ck(cudaSetDevice(g),"bucket batch release set device");
        tables[g].release();cudaFree(base[g]);base[g]=nullptr;
    }
    return all_ok?0:5;
}
