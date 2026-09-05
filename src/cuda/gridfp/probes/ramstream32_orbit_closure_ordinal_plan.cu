#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main bkoc_ordinal_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_reverse_fused_validate.hpp"
#include "../ramstream32_bucket_orbit_closure_preflight.cuh"

#include <algorithm>
#include <cstdint>
#include <iostream>

struct OrdinalStats{uint32_t max_forward_low=0,max_forward_high=0,max_reverse_low=0,max_reverse_high=0;};
static int ordinal_bits(uint32_t n){int b=0;while(n){++b;n>>=1;}return b;}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);(void)validate_bucket_orbit_closure_fusion(layout,bo,bf,rb,rf);
    OrdinalStats st;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p),nt=p==1?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=bf.low_off[size_t(pi)*bf.low_pitch+dbid],b=bf.low_off[size_t(pi)*bf.low_pitch+dbid+1];st.max_forward_low=std::max(st.max_forward_low,b-a);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){uint32_t a=bf.high_off[size_t(pi)*bf.high_pitch+dbid],b=bf.high_off[size_t(pi)*bf.high_pitch+dbid+1];st.max_forward_high=std::max(st.max_forward_high,b-a);}}
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){uint32_t a=rf.low_off[size_t(pi)*rf.low_pitch+dbid],b=rf.low_off[size_t(pi)*rf.low_pitch+dbid+1];st.max_reverse_low=std::max(st.max_reverse_low,b-a);}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1)),nt=p==TARGET_W-1?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=rf.high_off[size_t(pi)*rf.high_pitch+dbid],b=rf.high_off[size_t(pi)*rf.high_pitch+dbid+1];st.max_reverse_high=std::max(st.max_reverse_high,b-a);}}
    uint32_t mf=std::max(st.max_forward_low,st.max_forward_high),mr=std::max(st.max_reverse_low,st.max_reverse_high);
    // One all-ones code point is reserved for NONE. Thus b bits can encode at
    // most 2^b-1 real destination-local ordinals.
    bool forward_fits10=mf<=1023,reverse_fits2=mr<=3;
    bool forward_fits16=mf<=65535,reverse_fits16=mr<=65535;
    std::cout<<"orbit-closure-ordinal-plan OK W="<<TARGET_W<<" max_forward_low="<<st.max_forward_low<<" max_forward_high="<<st.max_forward_high<<" max_reverse_low="<<st.max_reverse_low<<" max_reverse_high="<<st.max_reverse_high<<" forward_ordinal_bits="<<ordinal_bits(mf?mf-1:0)<<" reverse_ordinal_bits="<<ordinal_bits(mr?mr-1:0)<<" forward_spare_bits=10 reverse_spare_bits=2 forward_fits_spare="<<forward_fits10<<" reverse_fits_spare="<<reverse_fits2<<" forward_fits_u16="<<forward_fits16<<" reverse_fits_u16="<<reverse_fits16<<"\n";
    return 0;
}
