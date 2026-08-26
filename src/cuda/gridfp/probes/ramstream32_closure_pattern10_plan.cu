#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_plan_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_closure_pattern10.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>

struct P10Hash{uint64_t x=0,s=0,n=0;};
static void p10_mix(P10Hash&h,uint64_t z){z&=BKCP10_BASE_MASK;h.x^=(z+0x9e3779b97f4a7c15ull+(h.n<<6)+(h.n>>2));h.s+=z*0x100000001b3ull+0x517cc1b727220a95ull;++h.n;}
static P10Hash p10_forward_hash(const BucketOrbitStreamsHost&o){P10Hash h;auto add=[&](const auto&v){for(auto z:v)p10_mix(h,z);};add(o.low_nn);add(o.low_nr);add(o.low_nl);add(o.high_nn);add(o.high_nrnl);return h;}
static P10Hash p10_reverse_hash(const ReverseSplit54Host&o){P10Hash h;auto add=[&](const auto&v){for(auto z:v)p10_mix(h,z);};add(o.low.nn);add(o.low.nr);add(o.low.nl);add(o.high.nn);add(o.high.nr);add(o.high.nl);return h;}
static bool p10_same(P10Hash a,P10Hash b){return a.x==b.x&&a.s==b.s&&a.n==b.n;}
static uint16_t p10_max_forward(const BucketOrbitStreamsHost&o,uint64_t&none){uint16_t m=0;none=0;auto add=[&](const auto&v){for(auto z:v){uint16_t id=bkcp10_id(z);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE)++none;else m=std::max(m,id);}};add(o.low_nn);add(o.low_nr);add(o.low_nl);add(o.high_nn);add(o.high_nrnl);return m;}
static uint16_t p10_max_reverse(const ReverseSplit54Host&o,uint64_t&none){uint16_t m=0;none=0;auto add=[&](const auto&v){for(auto z:v){uint16_t id=bkcp10_id(z);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE)++none;else m=std::max(m,id);}};add(o.low.nn);add(o.low.nr);add(o.low.nl);add(o.high.nn);add(o.high.nr);add(o.high.nl);return m;}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseSplit54Host rs=build_reverse_split54(layout,rb,false);
    P10Hash fb=p10_forward_hash(bo),rbh=p10_reverse_hash(rs);build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,rs);P10Hash fa=p10_forward_hash(bo),rah=p10_reverse_hash(rs);
    if(!p10_same(fb,fa)||!p10_same(rbh,rah)){std::cerr<<"pattern10 changed lower54 locator payload\n";return 2;}
    uint64_t fn=0,rn=0;uint16_t fm=p10_max_forward(bo,fn),rm=p10_max_reverse(rs,rn);if(fm>=oneesan::gridfp::CLOSURE_PATTERN10_NONE||rm>=oneesan::gridfp::CLOSURE_PATTERN10_NONE){std::cerr<<"pattern10 id overflow\n";return 3;}
    uint16_t bound_low=0,bound_high=0;for(int p=1;p<LOW_LUT_K+1;++p)bound_low=std::max(bound_low,oneesan::gridfp::closure_pattern10_count(LOW_LUT_K+1,p));for(int p=1;p<HIGH_LUT_K+1;++p)bound_high=std::max(bound_high,oneesan::gridfp::closure_pattern10_count(HIGH_LUT_K+1,p));if(bound_low>=oneesan::gridfp::CLOSURE_PATTERN10_NONE||bound_high>=oneesan::gridfp::CLOSURE_PATTERN10_NONE)return 4;
    std::cout<<"closure-pattern10-plan OK W="<<TARGET_W<<" forward_ops="<<fa.n<<" reverse_ops="<<rah.n<<" forward_none="<<fn<<" reverse_none="<<rn<<" forward_max_id="<<fm<<" reverse_max_id="<<rm<<" low_bound="<<bound_low<<" high_bound="<<bound_high<<" lower54_unchanged=1 extra_bytes=0\n";
    return 0;
}
