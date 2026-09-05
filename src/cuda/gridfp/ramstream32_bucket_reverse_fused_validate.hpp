#pragma once

#include "ramstream32_bucket_reverse_fused.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

struct ReverseBucketFusedValidation {
    uint32_t low_max_local = 0;
    uint32_t low_max_cross = 0;
    uint32_t high_max_local = 0;
    uint32_t high_max_cross = 0;
};

static ReverseBucketFusedValidation validate_reverse_bucket_fused(
    const StorageLayout& layout,const BucketOwnerHost& owner,
    const ReverseBucketAtomicHost& src,const ReverseBucketFusedHost& f
){
    ReverseBucketFusedValidation v;
    if(f.low_local_src.size()+f.low_cross_op.size()!=src.low_closure.size()){
        std::cerr<<"reverse fused LOW source conservation mismatch got="
                 <<(f.low_local_src.size()+f.low_cross_op.size())
                 <<" expected="<<src.low_closure.size()<<'\n';std::exit(320);
    }
    if(f.high_local_src.size()+f.high_cross_op.size()!=src.high_closure.size()){
        std::cerr<<"reverse fused HIGH source conservation mismatch got="
                 <<(f.high_local_src.size()+f.high_cross_op.size())
                 <<" expected="<<src.high_closure.size()<<'\n';std::exit(321);
    }

    auto check_src_low=[&](uint32_t x,bool cross){
        uint32_t bid=bkf_src_block(x),loc=bkf_src_locator(x),g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);
        if(bid>=layout.main_blocks.size()||g>=BUCKET_NGPU){std::exit(322);}
        const auto& b=layout.main_blocks[bid];
        if(!b.valid||r>=owner.low_count[g][b.hs]){
            std::cerr<<"reverse fused LOW source locator mismatch bid="<<bid<<" owner="<<g<<" rank="<<r<<" hs="<<int(b.hs)<<'\n';std::exit(323);
        }
        if(cross&&!bkf_cross_depth(x)){std::exit(324);}
        if(!cross&&bkf_cross_depth(x)){std::exit(325);}
    };
    auto check_src_high=[&](uint32_t x,bool cross){
        uint32_t bid=bkf_src_block(x),loc=bkf_src_locator(x),g=bkf_loc_owner(loc),r=bkf_loc_rank(loc);
        if(bid>=layout.main_blocks.size()||g>=BUCKET_NGPU){std::exit(326);}
        const auto& b=layout.main_blocks[bid];
        if(!b.valid||r>=owner.high_count[g][b.he]){
            std::cerr<<"reverse fused HIGH source locator mismatch bid="<<bid<<" owner="<<g<<" rank="<<r<<" he="<<int(b.he)<<'\n';std::exit(327);
        }
        if(cross&&!bkf_cross_depth(x)){std::exit(328);}
        if(!cross&&bkf_cross_depth(x)){std::exit(329);}
    };
    for(uint32_t x:f.low_local_src)check_src_low(x,false);
    for(uint32_t x:f.low_cross_op)check_src_low(x,true);
    for(uint32_t x:f.high_local_src)check_src_high(x,false);
    for(uint32_t x:f.high_cross_op)check_src_high(x,true);

    auto check_offsets=[&](const std::vector<uint32_t>& off,uint32_t steps,uint32_t pitch,size_t ndst,int code){
        if(off.size()!=size_t(steps)*pitch)std::exit(code);
        for(uint32_t pi=0;pi<steps;++pi){
            const uint32_t* p=off.data()+size_t(pi)*pitch;
            for(uint32_t i=1;i<pitch;++i)if(p[i]<p[i-1])std::exit(code+1);
            if(p[pitch-1]>ndst)std::exit(code+2);
        }
    };
    check_offsets(f.low_off,LOW_LUT_K,f.low_pitch,f.low_dst.size(),330);
    check_offsets(f.high_off,HIGH_LUT_K,f.high_pitch,f.high_dst.size(),333);

    uint64_t low_refs=0,high_refs=0;
    for(int p=1;p<=LOW_LUT_K;++p){
        uint32_t pi=uint32_t(p-1),nt=uint32_t(layout.block_blocks.size());
        uint32_t begin=f.low_off[size_t(pi)*f.low_pitch],end=f.low_off[size_t(pi)*f.low_pitch+nt];
        for(uint32_t q=begin;q<end;++q){
            const auto& rec=f.low_dst[q];uint32_t g=bkf_loc_owner(rec.dst_locator),r=bkf_loc_rank(rec.dst_locator);
            uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
            v.low_max_local=std::max(v.low_max_local,lc);v.low_max_cross=std::max(v.low_max_cross,cc);low_refs+=uint64_t(lc)+cc;
            // Recover destination block by locating q in the monotone block offsets.
            uint32_t dbid=0;while(dbid+1<=nt&&q>=f.low_off[size_t(pi)*f.low_pitch+dbid+1])++dbid;
            if(dbid>=nt||g>=BUCKET_NGPU)std::exit(336);
            const auto& db=layout.block_blocks[dbid];if(r>=owner.low_count[g][db.hs])std::exit(337);
            if(uint64_t(rec.local_begin)+lc>f.low_local_src.size()||uint64_t(rec.cross_begin)+cc>f.low_cross_op.size())std::exit(338);
        }
    }
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
        uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool tm=p==TARGET_W-1;
        uint32_t nt=tm?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        uint32_t begin=f.high_off[size_t(pi)*f.high_pitch],end=f.high_off[size_t(pi)*f.high_pitch+nt];
        for(uint32_t q=begin;q<end;++q){
            const auto& rec=f.high_dst[q];uint32_t g=bkf_loc_owner(rec.dst_locator),r=bkf_loc_rank(rec.dst_locator);
            uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
            v.high_max_local=std::max(v.high_max_local,lc);v.high_max_cross=std::max(v.high_max_cross,cc);high_refs+=uint64_t(lc)+cc;
            uint32_t dbid=0;while(dbid+1<=nt&&q>=f.high_off[size_t(pi)*f.high_pitch+dbid+1])++dbid;
            if(dbid>=nt||g>=BUCKET_NGPU)std::exit(339);
            if(tm){const auto&db=layout.main_blocks[dbid];if(r>=owner.high_count[g][db.he])std::exit(340);}
            else{const auto&db=layout.block_blocks[dbid];if(r>=owner.high_count[g][db.he])std::exit(341);}
            if(uint64_t(rec.local_begin)+lc>f.high_local_src.size()||uint64_t(rec.cross_begin)+cc>f.high_cross_op.size())std::exit(342);
        }
    }
    if(low_refs!=src.low_closure.size()||high_refs!=src.high_closure.size()){
        std::cerr<<"reverse fused destination reference conservation mismatch low="<<low_refs<<'/'<<src.low_closure.size()
                 <<" high="<<high_refs<<'/'<<src.high_closure.size()<<'\n';std::exit(343);
    }
    std::cerr<<"reverse_bucket_fused_validate"
             <<" low_edges="<<low_refs<<" high_edges="<<high_refs
             <<" low_max_local="<<v.low_max_local<<" low_max_cross="<<v.low_max_cross
             <<" high_max_local="<<v.high_max_local<<" high_max_cross="<<v.high_max_cross
             <<" OK\n";
    return v;
}

static ReverseBucketFusedHost build_reverse_bucket_fused_checked(
    const StorageLayout&layout,const BucketOwnerHost&owner,const ReverseBucketAtomicHost&src
){
    ReverseBucketFusedHost f=build_reverse_bucket_fused(layout,src);
    (void)validate_reverse_bucket_fused(layout,owner,src,f);
    return f;
}
