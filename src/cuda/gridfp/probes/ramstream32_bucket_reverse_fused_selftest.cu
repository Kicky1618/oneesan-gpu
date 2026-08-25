#define main bucket_reverse_atomic_reference_main
#include "ramstream32_bucket_reverse_atomic_selftest.cu"
#undef main

#include "../ramstream32_bucket_reverse_fused.cuh"
#include "../ramstream32_bucket_reverse_fused_validate.hpp"
#if GPU_DIRECT_PM_ACCUM
#include "../ramstream32_bucket_reverse_fused_pm.cuh"
#endif

static void brf_run_low(
    BucketHostGrid& g,const StorageLayout& layout,const BucketPhysicalLayoutHost& phy,
    BucketFusedDeviceTables& dt
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){
        std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[fixed][s];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"brf low H2D");}
        dt.bind_owner(fixed,phy,d);
#if GPU_DIRECT_PM_ACCUM
        bucket_launch_reverse_low_fused_pm(layout,256,4,4);
#else
        bucket_launch_reverse_low_fused(layout,256,4,4);
#endif
        ck(cudaDeviceSynchronize(),"brf low sync");
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[fixed][s];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"brf low D2H");}
        bkft_free_slots(d);
    }
}
static void brf_run_high(
    BucketHostGrid& g,const StorageLayout& layout,const BucketPhysicalLayoutHost& phy,
    BucketFusedDeviceTables& dt
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){
        std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[s][fixed];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"brf high H2D");}
        dt.bind_owner(fixed,phy,d);
#if GPU_DIRECT_PM_ACCUM
        bucket_launch_reverse_high_fused_pm(layout,256,4,4);
#else
        bucket_launch_reverse_high_fused(layout,256,4,4);
#endif
        ck(cudaDeviceSynchronize(),"brf high sync");
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[s][fixed];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"brf high D2H");}
        bkft_free_slots(d);
    }
}

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W,L=LOW_LUT_K;
    static_assert(W<=12,"reverse fused bucket selftest intentionally uses small width");
    int visible=0;cudaError_t ce=cudaGetDeviceCount(&visible);if(ce!=cudaSuccess||visible<1){std::cout<<"bucket-reverse-fused-selftest SKIP no CUDA device pm="<<GPU_DIRECT_PM_ACCUM<<"\n";return 0;}
    ck(cudaSetDevice(0),"brf set device");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rho=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rho);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);
    std::mt19937_64 rng(0x1618f05edULL);std::vector<Count>im(ms.size()),ib(bs.size());for(auto&x:im)x=Count(rng()%mod);for(auto&x:ib)x=Count(rng()%mod);
    auto[lm,lb]=bra_reference(1,L,mod,ms,bs,mi,di,im,ib);auto[hm,hb]=bra_reference(L+1,W-1,mod,ms,bs,mi,di,im,ib);
    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"brf modulus");BucketFusedDeviceTables dt;dt.install_metadata(layout,borbit,bfused);ReverseBucketFusedDeviceTables rt;rt.install(rb,rf);
    auto lg=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);brf_run_low(lg,layout,phy,dt);if(!bkft_compare("reverse-bucket-fused-low",lg,ms,bs,lm,lb,storage,layout,owner,phy))return 10;
    auto hg=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);brf_run_high(hg,layout,phy,dt);if(!bkft_compare("reverse-bucket-fused-high",hg,ms,bs,hm,hb,storage,layout,owner,phy))return 11;
    size_t resident=reverse_bucket_orbit_bytes(rb)+rf.bytes();
    std::cout<<"bucket-reverse-fused-selftest OK W="<<W<<" metadata_mib="<<double(resident)/(1<<20)
             <<" low_dst="<<rf.low_dst.size()<<" high_dst="<<rf.high_dst.size()
             <<" closure_atomic=0 pm="<<GPU_DIRECT_PM_ACCUM<<"\n";
    rt.release();dt.release();return 0;
}
