#ifndef BUCKET_SNAKE_REVERSE_FUSED
#define BUCKET_SNAKE_REVERSE_FUSED 1
#endif
#define main bsn_twopass_main_unused
#include "oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../gridfp/ramstream32_bucket_orbit_closure_preflight.cuh"

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int threads=argc>3?std::atoi(argv[3]):256,gx=argc>4?std::atoi(argv[4]):16,gy=argc>5?std::atoi(argv[5]):8;
    size_t chunk_mib=argc>6?size_t(std::strtoull(argv[6],nullptr,10)):bsn_env_mib("BUCKET_TRANSPOSE_CHUNK_MIB",1024);bool plan_only=bsn_has_arg(argc,argv,"--plan-only");
    constexpr int NG=BUCKET_NGPU;int W=n+1;if(W!=TARGET_W||LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;if(threads<=0||threads>1024||gx<=0||gy<=0||!chunk_mib)return 2;

    auto prep0=std::chrono::steady_clock::now();build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);BucketForwardOrbitClosureAttachHost fattach=build_bucket_forward_orbit_closure_attach(layout,borbit,bfused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);SnakeReverseHost reverse=bsn_build_reverse(storage,layout,owner,rlow,rhigh,rlo,rhi);BucketReverseOrbitClosureAttachHost rattach=build_bucket_reverse_orbit_closure_attach_checked(layout,borbit,bfused,reverse.atomic,reverse.fused);
    BucketTransposePlan tplan=build_bucket_transpose_plan(phy,NG);double prepare_s=bsn_since(prep0);

    size_t view_bytes=2ull*NG*(layout.main_blocks.size()+layout.block_blocks.size())*sizeof(BucketPhysicalBlock);
    size_t forward_meta=borbit.bytes()+bfused.bytes()+fattach.bytes()+view_bytes,reverse_meta=reverse.bytes()+rattach.bytes();size_t metadata_bytes=forward_meta+reverse_meta;
    size_t chunk_bytes=chunk_mib<<20,staging_bytes=chunk_bytes*size_t(BUCKET_TRANSPOSE_STAGING_MULTIPLIER);
    uint64_t max_gpu=0,min_gpu=~uint64_t(0),peer_per_transpose=0,max_need=0;
    for(int g=0;g<NG;++g){max_gpu=std::max(max_gpu,tplan.gpu_bytes[g]);min_gpu=std::min(min_gpu,tplan.gpu_bytes[g]);max_need=std::max<uint64_t>(max_need,tplan.gpu_bytes[g]+metadata_bytes+staging_bytes);for(int s=g+1;s<NG;++s)peer_per_transpose+=2ull*tplan.slot[g][s].capacity_bytes;}
    long double snake_peer_tib=static_cast<long double>(peer_per_transpose)*static_cast<long double>(W)/static_cast<long double>(1ULL<<40);
    std::cout<<std::setprecision(15)<<"backend=gridfp-b300-bucket-snake-onepass-v0.1-plan n="<<n
             <<" states="<<(layout.main_size+layout.block_size)<<" authoritative_tib="<<double((layout.main_size+layout.block_size)*sizeof(Count))/double(1ULL<<40)
             <<" max_gpu_authoritative_gib="<<double(max_gpu)/double(1ULL<<30)<<" gpu_spread_mib="<<double(max_gpu-min_gpu)/double(1<<20)
             <<" forward_attach_mib="<<double(fattach.bytes())/double(1<<20)<<" reverse_attach_mib="<<double(rattach.bytes())/double(1<<20)
             <<" forward_metadata_mib="<<double(forward_meta)/double(1<<20)<<" reverse_metadata_mib="<<double(reverse_meta)/double(1<<20)
             <<" metadata_mib_per_gpu="<<double(metadata_bytes)/double(1<<20)<<" transpose_chunk_mib="<<chunk_mib
             <<" transpose_staging_multiplier="<<BUCKET_TRANSPOSE_STAGING_MULTIPLIER<<" transpose_staging_mib="<<double(staging_bytes)/double(1<<20)
             <<" max_device_need_gib="<<double(max_need)/double(1ULL<<30)
             <<" transposes_per_residue="<<W<<" standard_transposes="<<(2*W-1)<<" peer_gib_per_transpose="<<double(peer_per_transpose)/double(1ULL<<30)
             <<" snake_peer_tib_per_residue="<<double(snake_peer_tib)<<" kernels_per_position=1 reverse_closure_atomic=0 forward_closure_atomic=0 pm_accum="<<GPU_DIRECT_PM_ACCUM<<" prepare_s="<<prepare_s<<'\n';
    if(plan_only)return 0;

    int visible=0;ck(cudaGetDeviceCount(&visible),"snake onepass device count");if(visible<NG){std::cerr<<"need 8 GPUs visible="<<visible<<'\n';return 3;}size_t reserve=bsn_env_mib("BUCKET_RESERVE_MIB",8192)<<20;
    for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake onepass mem set");size_t freeb=0,totalb=0;ck(cudaMemGetInfo(&freeb,&totalb),"snake onepass mem info");uint64_t need=tplan.gpu_bytes[g]+metadata_bytes+staging_bytes+reserve;std::cerr<<"gpu"<<g<<" free_gib="<<double(freeb)/double(1ULL<<30)<<" need_with_reserve_gib="<<double(need)/double(1ULL<<30)<<'\n';if(need>freeb)return 4;}

    std::array<Count*,NG>base{};std::array<std::array<Count*,NG>,NG>slots{};std::array<BucketFusedDeviceTables,NG>ft;std::array<SnakeReverseDeviceTables,NG>rt;std::array<BucketForwardOrbitClosureAttachDeviceTables,NG>fat;std::array<BucketReverseOrbitClosureAttachDeviceTables,NG>rat;
    for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake onepass alloc set");ck(cudaMalloc(&base[g],size_t(tplan.gpu_bytes[g])),"snake onepass auth alloc");ck(cudaMemset(base[g],0,size_t(tplan.gpu_bytes[g])),"snake onepass auth zero");uint8_t*bb=reinterpret_cast<uint8_t*>(base[g]);for(int s=0;s<NG;++s)slots[g][s]=reinterpret_cast<Count*>(bb+tplan.slot[g][s].off_bytes);ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"snake onepass modulus");ft[g].install_metadata(layout,borbit,bfused);rt[g].install(reverse);fat[g].install(fattach);rat[g].install(rattach);ft[g].bind_owner(uint32_t(g),phy,slots[g]);}

    MateID init=MateID(R)<<(2*(W-1));BucketAddress ia=bucket_rank_main_host(init,storage,layout,owner,phy);Count one=1;ck(cudaSetDevice(ia.owner_l),"snake onepass init set");ck(cudaMemcpy(reinterpret_cast<uint8_t*>(base[ia.owner_l])+tplan.slot[ia.owner_l][ia.owner_h].off_bytes+uint64_t(ia.off)*sizeof(Count),&one,sizeof(one),cudaMemcpyHostToDevice),"snake onepass init");
    BucketTransposeCtx tx;tx.init(tplan,base,chunk_bytes);double fh=0,fl=0,rl=0,rh=0,ts=0;auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){
        if((row&1)==0){auto t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake onepass fhigh set");bucket_launch_high_orbit_closure_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);fh+=bsn_since(t);t=std::chrono::steady_clock::now();tx.transpose(tplan);ts+=bsn_since(t);t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake onepass flow set");bucket_launch_low_orbit_closure_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);fl+=bsn_since(t);}
        else{auto t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake onepass rlow set");bucket_launch_reverse_low_orbit_closure_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);rl+=bsn_since(t);t=std::chrono::steady_clock::now();tx.transpose(tplan);ts+=bsn_since(t);t=std::chrono::steady_clock::now();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake onepass rhigh set");bucket_launch_reverse_high_orbit_closure_fused(layout,threads,gx,gy);}bucket_sync_devices(NG);rh+=bsn_since(t);}
        std::cerr<<"row "<<(row+1)<<'/'<<W<<" fh="<<fh<<" fl="<<fl<<" rl="<<rl<<" rh="<<rh<<" transpose="<<ts<<'\n';
    }
    double wall=bsn_since(wall0);BucketAddress fa=bucket_rank_main_host(MateID(R),storage,layout,owner,phy);bool final_lmajor=(W%2)==0;int agpu=final_lmajor?int(fa.owner_l):int(fa.owner_h),aslot=final_lmajor?int(fa.owner_h):int(fa.owner_l);Count answer=0;ck(cudaSetDevice(agpu),"snake onepass answer set");ck(cudaMemcpy(&answer,reinterpret_cast<uint8_t*>(base[agpu])+tplan.slot[agpu][aslot].off_bytes+uint64_t(fa.off)*sizeof(Count),sizeof(answer),cudaMemcpyDeviceToHost),"snake onepass answer");
    std::cout<<"backend=gridfp-b300-bucket-snake-onepass-v0.1 n="<<n<<" residue="<<answer<<" modulus="<<mod<<" forward_high_s="<<fh<<" forward_low_s="<<fl<<" reverse_low_s="<<rl<<" reverse_high_s="<<rh<<" transpose_s="<<ts<<" wall_s="<<wall<<" transposes="<<tx.transposes<<" peer_gib="<<tx.peer_gib<<" final_layout="<<(final_lmajor?"L-major":"H-major")<<" kernels_per_position=1 reverse_closure_atomic=0 forward_closure_atomic=0 pm_accum="<<GPU_DIRECT_PM_ACCUM<<"\n";
    bool ok=tx.transposes==uint64_t(W);if(n==21&&mod==4294967291u&&answer!=998035516u){std::cerr<<"n21 snake onepass residue mismatch got="<<answer<<" expected=998035516\n";ok=false;}
    tx.release();for(int g=0;g<NG;++g){ck(cudaSetDevice(g),"snake onepass release set");rat[g].release();fat[g].release();rt[g].release();ft[g].release();cudaFree(base[g]);}return ok?0:5;
}
