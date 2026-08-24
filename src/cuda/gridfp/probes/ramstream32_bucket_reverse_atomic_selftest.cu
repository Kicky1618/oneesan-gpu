#define main bucket_forward_reference_main
#include "ramstream32_bucket_fused_selftest.cu"
#undef main

#include "../ramstream32_reverse_desc.hpp"
#include "../ramstream32_reverse_orbit.hpp"
#include "../ramstream32_bucket_reverse_atomic.cuh"

static std::pair<std::vector<Count>,std::vector<Count>> bra_reference(
    int p0,int p1,Count mod,const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::unordered_map<MateID,size_t>&mi,const std::unordered_map<MateID,size_t>&di,
    const std::vector<Count>&im,const std::vector<Count>&ib
){
    std::vector<Count> rm=im,rb=ib;
    for(int p=p0;p<=p1;++p){std::vector<Count> nm=rm,nb(rb.size(),0);
        for(size_t i=0;i<ms.size();++i){Count c=rm[i];auto z=oneesan::gridfp::include_horizontal_reverse(ms[i],TARGET_W,p);if(!z.valid)continue;if(z.blocked){auto it=di.find(z.mate);if(it==di.end())std::exit(300);nb[it->second]=gdg_add(nb[it->second],c,mod);}else{auto it=mi.find(z.mate);if(it==mi.end())std::exit(301);nm[it->second]=gdg_add(nm[it->second],c,mod);}}
        for(size_t i=0;i<bs.size();++i){MateID z=oneesan::gridfp::blocked_exclude_reverse(bs[i],TARGET_W,p);auto it=mi.find(z);if(it==mi.end())std::exit(302);nm[it->second]=gdg_add(nm[it->second],rb[i],mod);}rm.swap(nm);rb.swap(nb);
    }return {std::move(rm),std::move(rb)};
}

static void bra_run_low(BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,BucketFusedDeviceTables&dt){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[fixed][s];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"bra low H2D");}dt.bind_owner(fixed,phy,d);bucket_launch_reverse_low_atomic(layout,256,4,4);ck(cudaDeviceSynchronize(),"bra low sync");for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[fixed][s];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"bra low D2H");}bkft_free_slots(d);}
}
static void bra_run_high(BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,BucketFusedDeviceTables&dt){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[s][fixed];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"bra high H2D");}dt.bind_owner(fixed,phy,d);bucket_launch_reverse_high_atomic(layout,256,4,4);ck(cudaDeviceSynchronize(),"bra high sync");for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[s][fixed];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"bra high D2H");}bkft_free_slots(d);}
}

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W,L=LOW_LUT_K;static_assert(W<=12,"reverse bucket selftest intentionally uses small width");
    int visible=0;cudaError_t ce=cudaGetDeviceCount(&visible);if(ce!=cudaSuccess||visible<1){std::cout<<"bucket-reverse-atomic-selftest SKIP no CUDA device\n";return 0;}ck(cudaSetDevice(0),"bra set device");
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rho=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rho);
    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);std::mt19937_64 rng(0x1618babeULL);std::vector<Count>im(ms.size()),ib(bs.size());for(auto&x:im)x=Count(rng()%mod);for(auto&x:ib)x=Count(rng()%mod);
    auto[lm,lb]=bra_reference(1,L,mod,ms,bs,mi,di,im,ib);auto[hm,hb]=bra_reference(L+1,W-1,mod,ms,bs,mi,di,im,ib);
    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"bra modulus");BucketFusedDeviceTables dt;dt.install_metadata(layout,borbit,bfused);ReverseBucketAtomicDeviceTables rt;rt.install(rb);
    auto lg=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);bra_run_low(lg,layout,phy,dt);if(!bkft_compare("reverse-bucket-low",lg,ms,bs,lm,lb,storage,layout,owner,phy))return 10;
    auto hg=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy);bra_run_high(hg,layout,phy,dt);if(!bkft_compare("reverse-bucket-high",hg,ms,bs,hm,hb,storage,layout,owner,phy))return 11;
    std::cout<<"bucket-reverse-atomic-selftest OK W="<<W<<" metadata_mib="<<double(rb.bytes())/(1<<20)<<" low_orbit="<<rb.low_orbit.size()<<" high_orbit="<<rb.high_orbit.size()<<" closure_atomic=1\n";rt.release();dt.release();return 0;
}
