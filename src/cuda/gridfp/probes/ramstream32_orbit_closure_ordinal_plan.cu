#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main bkoc_ordinal_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_reverse_fused_validate.hpp"
#include "../ramstream32_bucket_orbit_closure_preflight.cuh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>

struct OrdinalStats{uint32_t max_forward_low=0,max_forward_high=0,max_reverse_low=0,max_reverse_high=0;uint64_t attached_forward=0,attached_reverse=0;};

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);(void)validate_bucket_orbit_closure_fusion(layout,bo,bf,rb,rf);
    auto fa=build_bucket_forward_orbit_closure_attach(layout,bo,bf);auto ra=build_bucket_reverse_orbit_closure_attach(layout,rb,rf);OrdinalStats st;
    auto used_count=[](const std::vector<uint32_t>&v,uint32_t a,uint32_t b){uint32_t z=0;for(uint32_t q=a;q<b;++q)z+=v[q]!=BKOC_NONE;return z;};
    size_t lp=size_t(bo.low_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){uint32_t n=0;uint32_t a=bo.low_nn_off[size_t(pi)*lp+bid],b=bo.low_nn_off[size_t(pi)*lp+bid+1];n+=used_count(fa.low_nn,a,b);a=bo.low_nr_off[size_t(pi)*lp+bid];b=bo.low_nr_off[size_t(pi)*lp+bid+1];n+=used_count(fa.low_nr,a,b);a=bo.low_nl_off[size_t(pi)*lp+bid];b=bo.low_nl_off[size_t(pi)*lp+bid+1];n+=used_count(fa.low_nl,a,b);st.max_forward_low=std::max(st.max_forward_low,n);st.attached_forward+=n;}}
    size_t hp=size_t(bo.high_nblocks)+1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){uint32_t n=0;uint32_t a=bo.high_nn_off[size_t(pi)*hp+bid],b=bo.high_nn_off[size_t(pi)*hp+bid+1];n+=used_count(fa.high_nn,a,b);a=bo.high_nrnl_off[size_t(pi)*hp+bid];b=bo.high_nrnl_off[size_t(pi)*hp+bid+1];n+=used_count(fa.high_nrnl,a,b);st.max_forward_high=std::max(st.max_forward_high,n);st.attached_forward+=n;}}
    size_t rp=size_t(rb.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t a=rb.low_orbit_off[size_t(pi)*rp+bid],b=rb.low_orbit_off[size_t(pi)*rp+bid+1];uint32_t n=used_count(ra.low,a,b);st.max_reverse_low=std::max(st.max_reverse_low,n);st.attached_reverse+=n;}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t a=rb.high_orbit_off[size_t(pi)*rp+bid],b=rb.high_orbit_off[size_t(pi)*rp+bid+1];uint32_t n=used_count(ra.high,a,b);st.max_reverse_high=std::max(st.max_reverse_high,n);st.attached_reverse+=n;}}
    uint32_t mf=std::max(st.max_forward_low,st.max_forward_high),mr=std::max(st.max_reverse_low,st.max_reverse_high);int fbits=0,rbits=0;for(uint32_t x=mf;x;x>>=1)++fbits;for(uint32_t x=mr;x;x>>=1)++rbits;
    std::cout<<"orbit-closure-ordinal-plan OK W="<<TARGET_W<<" max_forward_low="<<st.max_forward_low<<" max_forward_high="<<st.max_forward_high<<" max_reverse_low="<<st.max_reverse_low<<" max_reverse_high="<<st.max_reverse_high<<" forward_bits="<<fbits<<" reverse_bits="<<rbits<<" forward_spare_bits=10 reverse_spare_bits=2 attached_forward="<<st.attached_forward<<" attached_reverse="<<st.attached_reverse<<"\n";
    return 0;
}
