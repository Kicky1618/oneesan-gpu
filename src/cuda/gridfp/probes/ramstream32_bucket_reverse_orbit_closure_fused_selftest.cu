#define main bucket_reverse_two_pass_reference_main
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#undef main

#include "../ramstream32_bucket_orbit_closure_fused.cuh"

static void broc_test_low(
    BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,
    BucketFusedDeviceTables&dt
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[fixed][s];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"broc low H2D");}dt.bind_owner(fixed,phy,d);bucket_launch_reverse_low_orbit_closure_fused(layout,256,4,4);ck(cudaDeviceSynchronize(),"broc low sync");for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[fixed][s];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"broc low D2H");}bkft_free_slots(d);}
}
static void broc_test_high(
    BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,
    BucketFusedDeviceTables&dt
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[s][fixed];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"broc high H2D");}dt.bind_owner(fixed,phy,d);bucket_launch_reverse_high_orbit_closure_fused(layout,256,4,4);ck(cudaDeviceSynchronize(),"broc high sync");for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[s][fixed];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"broc high D2H");}bkft_free_slots(d);}
}

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W,L=LOW_LUT_K;static_assert(W<=12,"reverse one-pass bucket selftest intentionally uses small width");
    int visible=0;cudaError_t ce=cudaGetDeviceCount(&visible);if(ce!=cudaSuccess||visible<1){std::cout<<"bucket-reverse-orbit-closure-selftest SKIP no CUDA device\n";return 0;}ck(cudaSetDevice(0),"broc set device");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);BucketReverseOrbitClosureAttachHost attach=build_bucket_reverse_orbit_closure_attach(layout,rb,rf);
    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);std::mt19937_64 rng(0x1618c105eULL);std::vector<Count>im(ms.size()),ib(bs.size());for(auto&x:im)x=Count(rng()%mod);for(auto&x:ib)x=Count(rng()%mod);
    auto[lm,lb]=bra_reference(1,L,mod,ms,bs,mi,di,im,ib);auto[hm,hb]=bra_reference(L+1,W-1,mod,ms,bs,mi,di,im,ib);
    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"broc modulus");BucketFusedDeviceTables dt;dt.install_metadata(layout,borbit,bfused);ReverseBucketFusedDeviceTables rt;rt.install(rb,rf);BucketReverseOrbitClosureAttachDeviceTables at;at.install(attach);
    auto lg=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);broc_test_low(lg,layout,phy,dt);if(!bkft_compare("reverse-orbit-closure-low",lg,ms,bs,lm,lb,storage,layout,owner,phy))return 10;
    auto hg=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);broc_test_high(hg,layout,phy,dt);if(!bkft_compare("reverse-orbit-closure-high",hg,ms,bs,hm,hb,storage,layout,owner,phy))return 11;
    std::cout<<"bucket-reverse-orbit-closure-selftest OK W="<<W<<" attach_mib="<<double(attach.bytes())/double(1<<20)<<" kernels_per_position=1 closure_atomic=0 scratch_bytes=0 pm_accum="<<GPU_DIRECT_PM_ACCUM<<"\n";
    at.release();rt.release();dt.release();return 0;
}
