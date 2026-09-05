#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main reverse_closure_inverse_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../../common/gridfp_closure_inverse.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static int rcinv_delta(uint32_t c){return c==uint32_t(::L)?1:(c==uint32_t(R)?-1:0);}
static uint32_t rcinv_low_code_at(const BucketFusedHost&bf,uint32_t loc,int h){uint32_t g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);return bf.low_codes[bf.low_code_off[size_t(g)*bf.code_pitch+h]+r];}
static uint32_t rcinv_high_code_at(const BucketFusedHost&bf,uint32_t loc,int h){uint32_t g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);return bf.high_codes[bf.high_code_off[size_t(g)*bf.code_pitch+h]+r];}

static uint32_t rcinv_low_source(
    MateID partial,int he,const StorageFactorHost&storage,const StorageLayout&layout,const BucketOwnerHost&owner
){
    constexpr int K=LOW_LUT_K;constexpr uint64_t MASK=(uint64_t(1)<<(2*K))-1;
    uint32_t c=uint32_t(mget(partial,K)),lc=uint32_t(partial&MASK);int hs=he+rcinv_delta(c);
    if(he<0||he>HIGH_LUT_K+1||hs<0||hs>LOW_LUT_K+1)return 0xffffffffu;
    uint32_t bid=uint32_t(3*he+int(c));
    if(bid>=layout.main_blocks.size()||!layout.main_blocks[bid].valid||layout.main_blocks[bid].hs!=hs)return 0xffffffffu;
    uint32_t pr=storage.low_packed_rank[lc];if(pr==0xffffffffu)return 0xffffffffu;
    uint32_t ar=pr>>K,n=storage.low_all_off[hs+1]-storage.low_all_off[hs];
    if(ar>=n||storage.low_all_codes[storage.low_all_off[hs]+ar]!=lc)return 0xffffffffu;
    return bkf_src_pack(bid,bucket_low_locator(storage,owner,hs,ar));
}
static uint32_t rcinv_high_source(
    MateID partial,int hs,const StorageFactorHost&storage,const StorageLayout&layout,const BucketOwnerHost&owner
){
    constexpr int K=HIGH_LUT_K;constexpr uint64_t MASK=(uint64_t(1)<<(2*K))-1;
    uint32_t c=uint32_t(mget(partial,0)),hc=uint32_t((partial>>2)&MASK);int he=hs-rcinv_delta(c);
    if(hs<0||hs>LOW_LUT_K+1||he<0||he>HIGH_LUT_K+1)return 0xffffffffu;
    uint32_t bid=uint32_t(3*he+int(c));
    if(bid>=layout.main_blocks.size()||!layout.main_blocks[bid].valid||layout.main_blocks[bid].hs!=hs)return 0xffffffffu;
    uint32_t pr=storage.high_packed_rank[hc];if(pr==0xffffffffu)return 0xffffffffu;
    uint32_t ar=pr>>K,n=storage.high_all_off[he+1]-storage.high_all_off[he];
    if(ar>=n||storage.high_all_codes[storage.high_all_off[he]+ar]!=hc)return 0xffffffffu;
    return bkf_src_pack(bid,bucket_high_locator(storage,owner,he,ar));
}
static void rcinv_compare(std::vector<uint32_t>got,std::vector<uint32_t>want,const char*side,int p,uint32_t dbid,uint32_t q){
    std::sort(got.begin(),got.end());got.erase(std::unique(got.begin(),got.end()),got.end());std::sort(want.begin(),want.end());
    if(got!=want){std::cerr<<"reverse closure inverse mismatch side="<<side<<" p="<<p<<" dbid="<<dbid<<" q="<<q<<" got="<<got.size()<<" want="<<want.size()<<'\n';std::exit(510);}
}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);

    MateID cand[32];uint64_t checked_low=0,checked_high=0,candidates_low=0,candidates_high=0;
    for(int p=1;p<=LOW_LUT_K;++p){
        uint32_t pi=uint32_t(p-1);uint32_t nt=uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){const StorageBlock&db=layout.block_blocks[dbid];size_t oi=size_t(pi)*rf.low_pitch+dbid;
            for(uint32_t q=rf.low_off[oi];q<rf.low_off[oi+1];++q){const auto&r=rf.low_dst[q];uint32_t dc=rcinv_low_code_at(bf,r.dst_locator,db.hs);MateID partial=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p);int n=ordinary_closure_preimages_partial_reverse(partial,LOW_LUT_K+1,p,cand);std::vector<uint32_t>got,want;for(int i=0;i<n;++i){uint32_t x=rcinv_low_source(cand[i],db.he,storage,layout,owner);if(x!=0xffffffffu)got.push_back(x);}uint32_t lc=r.counts&0xffffu;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)want.push_back(rf.low_local_src[e]);rcinv_compare(got,want,"LOW",p,dbid,q);++checked_low;candidates_low+=got.size();}
        }
    }
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
        uint32_t pi=uint32_t(p-(LOW_LUT_K+1)),rel=uint32_t(p-LOW_LUT_K);bool tm=p==TARGET_W-1;uint32_t nt=tm?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){const StorageBlock&db=tm?layout.main_blocks[dbid]:layout.block_blocks[dbid];size_t oi=size_t(pi)*rf.high_pitch+dbid;
            for(uint32_t q=rf.high_off[oi];q<rf.high_off[oi+1];++q){const auto&r=rf.high_dst[q];uint32_t dc=rcinv_high_code_at(bf,r.dst_locator,db.he);MateID partial=tm?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,int(rel));int n=ordinary_closure_preimages_partial_reverse(partial,HIGH_LUT_K+1,int(rel),cand);std::vector<uint32_t>got,want;for(int i=0;i<n;++i){uint32_t x=rcinv_high_source(cand[i],db.hs,storage,layout,owner);if(x!=0xffffffffu)got.push_back(x);}uint32_t lc=r.counts&0xffffu;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)want.push_back(rf.high_local_src[e]);rcinv_compare(got,want,"HIGH",p,dbid,q);++checked_high;candidates_high+=got.size();}
        }
    }
    std::cout<<"reverse-closure-inverse-plan OK W="<<TARGET_W<<" low_records="<<checked_low<<" high_records="<<checked_high<<" low_local_edges="<<candidates_low<<" high_local_edges="<<candidates_high<<" local_source_csr_eliminable=1\n";return 0;
}
