#ifndef BUCKET_SNAKE_REVERSE_FUSED
#define BUCKET_SNAKE_REVERSE_FUSED 1
#endif
#define main bsn_single_main_unused
#include "oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main

#include <vector>

int main(int argc,char**argv){
    if(argc<5){
        std::cerr<<"usage: "<<argv[0]<<" n target_mib max_window gpus p1 [p2 ...] [--plan-only]\n";
        return 1;
    }
    int n=std::atoi(argv[1]);
    (void)std::atoi(argv[2]); // legacy target_mib, intentionally ignored
    (void)std::atoi(argv[3]); // legacy max_window, intentionally ignored
    int ngpu=std::atoi(argv[4]);
    bool plan_only=bsn_has_arg(argc,argv,"--plan-only");
    std::vector<Count> mods;
    for(int i=5;i<argc;++i){
        if(std::strcmp(argv[i],"--plan-only")==0)continue;
        unsigned long x=std::strtoul(argv[i],nullptr,10);
        if(!x||x>0xfffffffful){std::cerr<<"invalid modulus "<<argv[i]<<'\n';return 2;}
        mods.push_back(Count(x));
    }
    constexpr int NG=BUCKET_NGPU;
    int W=n+1;
    if(W!=TARGET_W||LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W||ngpu!=NG)return 3;
    int threads=256,gx=16,gy=8;
    if(const char*s=std::getenv("BUCKET_THREADS"))threads=std::atoi(s);
    if(const char*s=std::getenv("BUCKET_GRID_X"))gx=std::atoi(s);
    if(const char*s=std::getenv("BUCKET_GRID_Y"))gy=std::atoi(s);
    size_t chunk_mib=bsn_env_mib("BUCKET_TRANSPOSE_CHUNK_MIB",1024);
    if(threads<=0||threads>1024||gx<=0||gy<=0||!chunk_mib)return 4;

    auto prep0=std::chrono::steady_clock::now();
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);
    GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);
    BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);
    BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);
    ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);
    SnakeReverseHost reverse=bsn_build_reverse(storage,layout,owner,rlow,rhigh,rlo,rhi);
    BucketTransposePlan tplan=build_bucket_transpose_plan(phy,NG);
    double prepare_s=bsn_since(prep0);

    size_t view_bytes=2ull*NG*(layout.main_blocks.size()+layout.block_blocks.size())*sizeof(BucketPhysicalBlock);
    size_t forward_meta=borbit.bytes()+bfused.bytes()+view_bytes;
    size_t reverse_meta=reverse.bytes();
    size_t metadata_bytes=forward_meta+reverse_meta;
    size_t chunk_bytes=chunk_mib<<20;
    uint64_t max_gpu=0,min_gpu=~uint64_t(0),peer_per_transpose=0,max_need=0;
    for(int g=0;g<NG;++g){
        max_gpu=std::max(max_gpu,tplan.gpu_bytes[g]);min_gpu=std::min(min_gpu,tplan.gpu_bytes[g]);
        max_need=std::max<uint64_t>(max_need,tplan.gpu_bytes[g]+metadata_bytes+chunk_bytes);
        for(int s=g+1;s<NG;++s)peer_per_transpose+=2ull*tplan.slot[g][s].capacity_bytes;
    }
    long double snake_peer_tib=static_cast<long double>(peer_per_transpose)*W/static_cast<long double>(1ULL<<40);
    std::cerr<<std::setprecision(15)
             <<"backend=gridfp-b300-bucket-snake-fused-batch-v0.1-plan n="<<n
             <<" states="<<(layout.main_size+layout.block_size)
             <<" max_gpu_authoritative_gib="<<double(max_gpu)/double(1ULL<<30)
             <<" gpu_spread_mib="<<double(max_gpu-min_gpu)/double(1<<20)
             <<" forward_metadata_mib="<<double(forward_meta)/double(1<<20)
             <<" reverse_metadata_mib="<<double(reverse_meta)/double(1<<20)
             <<" metadata_mib_per_gpu="<<double(metadata_bytes)/double(1<<20)
             <<" transpose_chunk_mib="<<chunk_mib
             <<" max_device_need_gib="<<double(max_need)/double(1ULL<<30)
             <<" transposes_per_residue="<<W<<" standard_transposes="<<(2*W-1)
             <<" peer_gib_per_transpose="<<double(peer_per_transpose)/double(1ULL<<30)
             <<" snake_peer_tib_per_residue="<<double(snake_peer_tib)
             <<" reverse_closure_atomic=0 forward_closure_atomic=0 prepare_s="<<prepare_s<<'\n';
    if(plan_only)return 0;
    if(mods.empty()){std::cerr<<"no moduli supplied\n";return 5;}

    int visible=0;ck(cudaGetDeviceCount(&visible),"snake batch device count");if(visible<NG){std::cerr<<"need 8 GPUs visible="<<visible<<'\n';return 6;}
    size_t reserve=bsn_env_mib("BUCKET_RESERVE_MIB",8192)<<20;
    for(int g=0;g<NG;++g){
        ck(cudaSetDevice(g),"snake batch mem set");size_t freeb=0,totalb=0;ck(cudaMemGetInfo(&freeb,&totalb),"snake batch mem info");
        uint64_t need=tplan.gpu_bytes[g]+metadata_bytes+chunk_bytes+reserve;
        std::cerr<<"gpu"<<g<<" free_gib="<<double(freeb)/double(1ULL<<30)<<" need_with_reserve_gib="<<double(need)/double(1ULL<<30)<<'\n';
        if(need>freeb)return 7;
    }

    std::array<Count*,NG>base{};std::array<std::array<Count*,NG>,NG>slots{};
    std::array<BucketFusedDeviceTables,NG>ft;std::array<SnakeReverseDeviceTables,NG>rt;
    for(int g=0;g<NG;++g){
        ck(cudaSetDevice(g),"snake batch alloc set");ck(cudaMalloc(&base[g],size_t(tplan.gpu_bytes[g])),"snake batch auth alloc");
        uint8_t*bb=reinterpret_cast<uint8_t*>(base[g]);for(int s=0;s<NG;++s)slots[g][s]=reinterpret_cast<Count*>(bb+tplan.slot[g][s].off_bytes);
        ft[g].install_metadata(layout,borbit,bfused);rt[g].install(reverse);ft[g].bind_owner(uint32_t(g),phy,slots[g]);
    }
    BucketTransposeCtx tx;tx.init(tplan,base,chunk_bytes);
    MateID init=MateID(R)<<(2*(W-1));BucketAddress ia=bucket_rank_main_host(init,storage,layout,owner,phy);
    BucketAddress fa=bucket_rank_main_host(MateID(R),storage,layout,owner,phy);
    bool final_lmajor=(W%2)==0;

    for(Count mod:mods){
        auto reset0=std::chrono::steady_clock::now();
        for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake batch reset set");ck(cudaMemset(base[g],0,size_t(tplan.gpu_bytes[g])),"snake batch zero");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"snake batch modulus");}
        bucket_sync_devices(NG);
        Count one=1;ck(cudaSetDevice(ia.owner_l),"snake batch init set");
        ck(cudaMemcpy(reinterpret_cast<uint8_t*>(base[ia.owner_l])+tplan.slot[ia.owner_l][ia.owner_h].off_bytes+uint64_t(ia.off)*sizeof(Count),&one,sizeof(one),cudaMemcpyHostToDevice),"snake batch init");
        double reset_s=bsn_since(reset0),fh=0,fl=0,rl=0,rh=0,ts=0;
        uint64_t t0count=tx.transposes;double p0=tx.peer_gib;
        auto wall0=std::chrono::steady_clock::now();
        for(int row=0;row<W;++row){
            if((row&1)==0){
                auto t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake batch fhigh set");bucket_launch_high_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);fh+=bsn_since(t);
                t=std::chrono::steady_clock::now();tx.transpose(tplan);ts+=bsn_since(t);
                t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake batch flow set");bucket_launch_low_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);fl+=bsn_since(t);
            }else{
                auto t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake batch rlow set");bsn_launch_reverse_low(layout,threads,gx,gy);}bucket_sync_devices(NG);rl+=bsn_since(t);
                t=std::chrono::steady_clock::now();tx.transpose(tplan);ts+=bsn_since(t);
                t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake batch rhigh set");bsn_launch_reverse_high(layout,threads,gx,gy);}bucket_sync_devices(NG);rh+=bsn_since(t);
            }
        }
        double wall=bsn_since(wall0);
        int agpu=final_lmajor?int(fa.owner_l):int(fa.owner_h),aslot=final_lmajor?int(fa.owner_h):int(fa.owner_l);Count answer=0;
        ck(cudaSetDevice(agpu),"snake batch answer set");ck(cudaMemcpy(&answer,reinterpret_cast<uint8_t*>(base[agpu])+tplan.slot[agpu][aslot].off_bytes+uint64_t(fa.off)*sizeof(Count),sizeof(answer),cudaMemcpyDeviceToHost),"snake batch answer");
        uint64_t td=tx.transposes-t0count;double pd=tx.peer_gib-p0;
        if(td!=uint64_t(W)){std::cerr<<"snake batch transpose count mismatch got="<<td<<" expected="<<W<<'\n';return 8;}
        if(n==21&&mod==4294967291u&&answer!=998035516u){std::cerr<<"n21 snake batch residue mismatch got="<<answer<<" expected=998035516\n";return 9;}
        std::cerr<<"snake_batch modulus="<<mod<<" reset_s="<<reset_s<<" forward_high_s="<<fh<<" forward_low_s="<<fl
                 <<" reverse_low_s="<<rl<<" reverse_high_s="<<rh<<" transpose_s="<<ts<<" transposes="<<td<<" peer_gib="<<pd<<'\n';
        std::cout<<"residue="<<answer<<" modulus="<<mod<<" wall_s="<<wall<<'\n';
        std::cout.flush();
    }

    tx.release();
    for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake batch release set");rt[g].release();ft[g].release();cudaFree(base[g]);}
    return 0;
}
