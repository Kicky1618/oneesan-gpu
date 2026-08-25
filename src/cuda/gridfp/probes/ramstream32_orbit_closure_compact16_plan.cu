#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main bkoc16_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_reverse_fused_validate.hpp"
#include "../ramstream32_bucket_orbit_closure_preflight.cuh"
#include "../ramstream32_bucket_orbit_closure_compact16.cuh"
#include "../ramstream32_reverse_bucket_derive.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>

static uint32_t rec16(uint16_t o,const std::vector<uint32_t>&off,size_t oi){return o==BKOC16_NONE?BKOC_NONE:off[oi]+uint32_t(o);}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    (void)validate_bucket_orbit_closure_fusion(layout,bo,bf,rb,rf);validate_reverse_bucket_partner_blocks(layout,rb);
    auto f32=build_bucket_forward_orbit_closure_attach(layout,bo,bf);auto r32=build_bucket_reverse_orbit_closure_attach(layout,rb,rf);auto f16=build_bucket_forward_orbit_closure_attach16(layout,bo,bf,f32);auto r16=build_bucket_reverse_orbit_closure_attach16(layout,rb,rf,r32);
    uint64_t checked=0;size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1,rp=size_t(rb.nblocks)+1;
    auto eq=[&](uint32_t a,uint32_t b,const char*w){if(a!=b){std::cerr<<"compact16 reconstruction mismatch "<<w<<" got="<<b<<" expected="<<a<<'\n';std::exit(395);}++checked;};
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){uint32_t dbid=p==1?bid:uint32_t(layout.main_blocks[bid].he);size_t doi=size_t(pi)*bf.low_pitch+dbid;auto ckstream=[&](const auto&s32,const auto&s16,const auto&off){uint32_t a=off[size_t(pi)*lp+bid],b=off[size_t(pi)*lp+bid+1];for(uint32_t q=a;q<b;++q)eq(s32[q],rec16(s16[q],bf.low_off,doi),"forward-low");};ckstream(f32.low_nn,f16.low_nn,bo.low_nn_off);ckstream(f32.low_nr,f16.low_nr,bo.low_nr_off);ckstream(f32.low_nl,f16.low_nl,bo.low_nl_off);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){uint32_t dbid=uint32_t(layout.main_blocks[bid].hs);size_t doi=size_t(pi)*bf.high_pitch+dbid;auto ckstream=[&](const auto&s32,const auto&s16,const auto&off){uint32_t a=off[size_t(pi)*hp+bid],b=off[size_t(pi)*hp+bid+1];for(uint32_t q=a;q<b;++q)eq(s32[q],rec16(s16[q],bf.high_off,doi),"forward-high");};ckstream(f32.high_nn,f16.high_nn,bo.high_nn_off);ckstream(f32.high_nrnl,f16.high_nrnl,bo.high_nrnl_off);}}
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t dbid=uint32_t(layout.main_blocks[bid].he);size_t doi=size_t(pi)*rf.low_pitch+dbid;uint32_t a=rb.low_orbit_off[size_t(pi)*rp+bid],b=rb.low_orbit_off[size_t(pi)*rp+bid+1];for(uint32_t q=a;q<b;++q)eq(r32.low[q],rec16(r16.low[q],rf.low_off,doi),"reverse-low");}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool edge=p==TARGET_W-1;for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t dbid=edge?bid:uint32_t(layout.main_blocks[bid].hs);size_t doi=size_t(pi)*rf.high_pitch+dbid;uint32_t a=rb.high_orbit_off[size_t(pi)*rp+bid],b=rb.high_orbit_off[size_t(pi)*rp+bid+1];for(uint32_t q=a;q<b;++q)eq(r32.high[q],rec16(r16.high[q],rf.high_off,doi),"reverse-high");}}
    double oldm=double(f32.bytes()+r32.bytes())/double(1<<20),newm=double(f16.bytes()+r16.bytes())/double(1<<20);
    std::cout<<"orbit-closure-compact16-plan OK W="<<TARGET_W<<" checked="<<checked<<" old_attach_mib="<<oldm<<" compact_attach_mib="<<newm<<" ratio="<<(oldm?newm/oldm:0.0)<<" jblock_runtime_reads=0\n";return 0;
}
