#pragma once

#include "ramstream32_bucket_reverse_atomic.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>

// The reverse orbit partner block is not independent metadata. Horizontal
// reflection maps the modified pair back to original positions (p,p-1). Away
// from the center boundary the factor block is unchanged. At the boundary only
// the center symbol changes, and the new block follows from source he/hs and
// orbit kind.
static inline uint32_t rb_derived_partner_block(
    uint32_t source_bid,const StorageBlock&source,int p,uint32_t kind,bool active_low
){
    if(active_low){
        if(p!=LOW_LUT_K)return source_bid;
        // mirrored NN -> original LR, mirrored NR/NL -> original NL/NR.
        // Position p is the center here, so its partner symbol is L for NN,
        // otherwise N. HIGH code is unchanged, hence he is unchanged.
        uint32_t center=kind==CPU_ORBIT_NN?uint32_t(::L):uint32_t(N);
        return 3u*uint32_t(source.he)+center;
    }
    if(p!=LOW_LUT_K+1)return source_bid;
    // Position p-1 is the center. The reverse partner center is L only for
    // mirrored NR; NN/NL produce R. Recover partner he from hs=he+Delta(c),
    // with Delta(R)=-1, Delta(N)=0, Delta(L)=+1.
    uint32_t center=kind==CPU_ORBIT_NR?uint32_t(::L):uint32_t(R);
    int he=int(source.hs)+(center==uint32_t(R)?1:(center==uint32_t(::L)?-1:0));
    if(he<0){std::cerr<<"reverse derived partner negative he\n";std::exit(380);}
    return uint32_t(3*he+int(center));
}

static inline void validate_reverse_bucket_partner_blocks(
    const StorageLayout&layout,const ReverseBucketAtomicHost&h
){
    size_t pitch=size_t(h.nblocks)+1;uint64_t checked_low=0,checked_high=0;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<h.nblocks;++bid){uint32_t a=h.low_orbit_off[size_t(pi)*pitch+bid],b=h.low_orbit_off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q){uint64_t w=h.low_orbit[q];uint32_t got=rb_orbit_jblock(w),want=rb_derived_partner_block(bid,layout.main_blocks[bid],p,rb_orbit_kind(w),true);if(got!=want){std::cerr<<"reverse LOW derived partner mismatch p="<<p<<" bid="<<bid<<" kind="<<rb_orbit_kind(w)<<" got="<<got<<" want="<<want<<'\n';std::exit(381);}++checked_low;}}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));for(uint32_t bid=0;bid<h.nblocks;++bid){uint32_t a=h.high_orbit_off[size_t(pi)*pitch+bid],b=h.high_orbit_off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q){uint64_t w=h.high_orbit[q];uint32_t got=rb_orbit_jblock(w),want=rb_derived_partner_block(bid,layout.main_blocks[bid],p,rb_orbit_kind(w),false);if(got!=want){std::cerr<<"reverse HIGH derived partner mismatch p="<<p<<" bid="<<bid<<" kind="<<rb_orbit_kind(w)<<" got="<<got<<" want="<<want<<'\n';std::exit(382);}++checked_high;}}}
    std::cerr<<"reverse_bucket_partner_derive low="<<checked_low<<" high="<<checked_high<<" jblock_redundant_bits=6 OK\n";
}
