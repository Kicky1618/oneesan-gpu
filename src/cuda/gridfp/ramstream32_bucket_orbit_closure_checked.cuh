#pragma once

#include "ramstream32_bucket_orbit_closure_fused.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

struct BucketOrbitClosureValidation {
    uint64_t forward_low_sources=0,forward_high_sources=0;
    uint64_t reverse_low_sources=0,reverse_high_sources=0;
};

static void bkoc_check_src_disjoint(
    const std::unordered_set<BkocKey>&mut,uint32_t packed,uint64_t&count,const char*what
){
    BkocKey k=bkoc_key(bkf_src_block(packed),bkf_src_locator(packed));
    if(mut.count(k)){
        std::cerr<<"orbit-closure source aliases orbit main write "<<what
                 <<" block="<<bkf_src_block(packed)
                 <<" loc="<<bkf_src_locator(packed)<<'\n';
        std::exit(371);
    }
    ++count;
}

static BucketOrbitClosureValidation validate_bucket_orbit_closure_fusion(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    BucketOrbitClosureValidation st;
    const size_t lpitch=size_t(bo.low_nblocks)+1,hpitch=size_t(bo.high_nblocks)+1,rpitch=size_t(rb.nblocks)+1;

    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);std::unordered_set<BkocKey>mut;
        for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];FBlock fx{};fx.he=xb.he;fx.hs=xb.hs;fx.c=xb.c;
            auto scan=[&](const std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,uint32_t kind){uint32_t a=off[size_t(pi)*lpitch+bid],b=off[size_t(pi)*lpitch+bid+1],jbid=cpu_sparse_jblock(bid,fx,p,kind);for(uint32_t q=a;q<b;++q){mut.insert(bkoc_key(bid,bkf_orbit_src(v[q])));mut.insert(bkoc_key(jbid,bkf_orbit_partner(v[q])));}};
            scan(bo.low_nn,bo.low_nn_off,CPU_ORBIT_NN);scan(bo.low_nr,bo.low_nr_off,CPU_ORBIT_NR);scan(bo.low_nl,bo.low_nl_off,CPU_ORBIT_NL);
        }
        uint32_t nt=p==1?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=bf.low_off[size_t(pi)*bf.low_pitch+dbid],b=bf.low_off[size_t(pi)*bf.low_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=bf.low_dst[q];uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)bkoc_check_src_disjoint(mut,bf.low_local_src[e],st.forward_low_sources,"forward-low-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)bkoc_check_src_disjoint(mut,bf.low_cross_op[e],st.forward_low_sources,"forward-low-cross");}}
    }

    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        uint32_t pi=uint32_t((TARGET_W-1)-p);std::unordered_set<BkocKey>mut;
        for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];FBlock fx{};fx.he=xb.he;fx.hs=xb.hs;fx.c=xb.c;
            auto scan=[&](const std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,bool nn){uint32_t a=off[size_t(pi)*hpitch+bid],b=off[size_t(pi)*hpitch+bid+1],jbid=cpu_high_orbit_partner_block(bid,fx,p,nn);for(uint32_t q=a;q<b;++q){mut.insert(bkoc_key(bid,bkf_orbit_src(v[q])));mut.insert(bkoc_key(jbid,bkf_orbit_partner(v[q])));}};
            scan(bo.high_nn,bo.high_nn_off,true);scan(bo.high_nrnl,bo.high_nrnl_off,false);
        }
        for(uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){uint32_t a=bf.high_off[size_t(pi)*bf.high_pitch+dbid],b=bf.high_off[size_t(pi)*bf.high_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=bf.high_dst[q];uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)bkoc_check_src_disjoint(mut,bf.high_local_src[e],st.forward_high_sources,"forward-high-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)bkoc_check_src_disjoint(mut,bf.high_cross_op[e],st.forward_high_sources,"forward-high-cross");}}
    }

    for(int p=1;p<=LOW_LUT_K;++p){
        uint32_t pi=uint32_t(p-1);std::unordered_set<BkocKey>mut;
        for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t a=rb.low_orbit_off[size_t(pi)*rpitch+bid],b=rb.low_orbit_off[size_t(pi)*rpitch+bid+1];for(uint32_t q=a;q<b;++q){uint64_t w=rb.low_orbit[q];mut.insert(bkoc_key(bid,rb_orbit_src(w)));mut.insert(bkoc_key(rb_orbit_jblock(w),rb_orbit_partner(w)));}}
        for(uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){uint32_t a=rf.low_off[size_t(pi)*rf.low_pitch+dbid],b=rf.low_off[size_t(pi)*rf.low_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=rf.low_dst[q];uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)bkoc_check_src_disjoint(mut,rf.low_local_src[e],st.reverse_low_sources,"reverse-low-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)bkoc_check_src_disjoint(mut,rf.low_cross_op[e],st.reverse_low_sources,"reverse-low-cross");}}
    }

    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
        uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool edge=p==TARGET_W-1;std::unordered_set<BkocKey>mut;
        for(uint32_t bid=0;bid<rb.nblocks;++bid){uint32_t a=rb.high_orbit_off[size_t(pi)*rpitch+bid],b=rb.high_orbit_off[size_t(pi)*rpitch+bid+1];for(uint32_t q=a;q<b;++q){uint64_t w=rb.high_orbit[q];mut.insert(bkoc_key(bid,rb_orbit_src(w)));mut.insert(bkoc_key(rb_orbit_jblock(w),rb_orbit_partner(w)));}}
        uint32_t nt=edge?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=rf.high_off[size_t(pi)*rf.high_pitch+dbid],b=rf.high_off[size_t(pi)*rf.high_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=rf.high_dst[q];uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)bkoc_check_src_disjoint(mut,rf.high_local_src[e],st.reverse_high_sources,"reverse-high-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)bkoc_check_src_disjoint(mut,rf.high_cross_op[e],st.reverse_high_sources,"reverse-high-cross");}}
    }

    std::cerr<<"bucket_orbit_closure_validate forward_low_sources="<<st.forward_low_sources
             <<" forward_high_sources="<<st.forward_high_sources
             <<" reverse_low_sources="<<st.reverse_low_sources
             <<" reverse_high_sources="<<st.reverse_high_sources
             <<" orbit_main_write_overlap=0 OK\n";
    return st;
}

static BucketForwardOrbitClosureAttachHost build_bucket_forward_orbit_closure_attach_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    (void)validate_bucket_orbit_closure_fusion(layout,bo,bf,rb,rf);
    return build_bucket_forward_orbit_closure_attach(layout,bo,bf);
}
