#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main cross_height_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main

#include <cstdint>
#include <cstdlib>
#include <iostream>

struct CrossHeightStats{uint64_t n=0;int min_delta=99,max_delta=-99;void add(int d){++n;min_delta=std::min(min_delta,d);max_delta=std::max(max_delta,d);}};

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    CrossHeightStats fl,fh,rl,rh;
    auto low=[&](bool rev){const auto&dst=rev?rf.low_dst:bf.low_dst;const auto&off=rev?rf.low_off:bf.low_off;const auto&op=rev?rf.low_cross_op:bf.low_cross_op;uint32_t pitch=rev?rf.low_pitch:bf.low_pitch;for(int p=rev?1:LOW_LUT_K;rev?p<=LOW_LUT_K:p>=1;rev?++p:--p){uint32_t pi=rev?uint32_t(p-1):uint32_t(LOW_LUT_K-p);bool tm=!rev&&p==1;uint32_t nt=tm?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());for(uint32_t dbid=0;dbid<nt;++dbid){const StorageBlock&db=tm?layout.main_blocks[dbid]:layout.block_blocks[dbid];size_t oi=size_t(pi)*pitch+dbid;for(uint32_t q=off[oi];q<off[oi+1];++q){const auto&r=dst[q];uint32_t cc=r.counts>>16;for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e){uint32_t sbid=bkf_src_block(op[e]);int d=int(layout.main_blocks[sbid].he)-int(db.he);(rev?rl:fl).add(d);}}}}}};
    auto high=[&](bool rev){const auto&dst=rev?rf.high_dst:bf.high_dst;const auto&off=rev?rf.high_off:bf.high_off;const auto&op=rev?rf.high_cross_op:bf.high_cross_op;uint32_t pitch=rev?rf.high_pitch:bf.high_pitch;int p0=LOW_LUT_K+1,p1=TARGET_W-1;for(int p=rev?p0:p1;rev?p<=p1:p>=p0;rev?++p:--p){uint32_t pi=rev?uint32_t(p-p0):uint32_t(p1-p);bool tm=rev&&p==p1;uint32_t nt=tm?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());for(uint32_t dbid=0;dbid<nt;++dbid){const StorageBlock&db=tm?layout.main_blocks[dbid]:layout.block_blocks[dbid];size_t oi=size_t(pi)*pitch+dbid;for(uint32_t q=off[oi];q<off[oi+1];++q){const auto&r=dst[q];uint32_t cc=r.counts>>16;for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e){uint32_t sbid=bkf_src_block(op[e]);int d=int(layout.main_blocks[sbid].hs)-int(db.hs);(rev?rh:fh).add(d);}}}}}};
    low(false);high(false);low(true);high(true);
    auto check=[](const char*tag,const CrossHeightStats&s){if(s.n&&!(s.min_delta==2&&s.max_delta==2)){std::cerr<<"cross height delta mismatch "<<tag<<" n="<<s.n<<" min="<<s.min_delta<<" max="<<s.max_delta<<'\n';std::exit(540);}std::cout<<"cross-height side="<<tag<<" n="<<s.n<<" min_delta="<<(s.n?s.min_delta:0)<<" max_delta="<<(s.n?s.max_delta:0)<<'\n';};
    check("forward-low",fl);check("forward-high",fh);check("reverse-low",rl);check("reverse-high",rh);
    std::cout<<"cross-height-delta-plan OK W="<<TARGET_W<<" source_minus_destination_height=2\n";return 0;
}
