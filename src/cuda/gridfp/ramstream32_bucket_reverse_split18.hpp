#pragma once

#include "ramstream32_bucket_orbit_closure_packed18.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Reverse packed18 v2: split NN/NR/NL streams so orbit kind is stream identity.
// The three 18-bit locators occupy bits 0..53; bits 54..63 then hold the low
// 10 bits of the closure ordinal exactly like forward metadata. One byte/op
// stores ordinal bits 10..17. No jblock or kind bits remain in the orbit word.
struct ReverseSplit18SideHost {
    std::vector<BucketOrbitOp> nn,nr,nl;
    std::vector<uint8_t> nn_hi,nr_hi,nl_hi;
    std::vector<uint32_t> nn_off,nr_off,nl_off;
    size_t bytes()const{return (nn.size()+nr.size()+nl.size())*sizeof(BucketOrbitOp)+(nn_hi.size()+nr_hi.size()+nl_hi.size())+(nn_off.size()+nr_off.size()+nl_off.size())*sizeof(uint32_t);}
    size_t ops()const{return nn.size()+nr.size()+nl.size();}
};
struct ReverseSplit18Host {
    ReverseSplit18SideHost low,high;
    uint32_t nblocks=0;
    size_t bytes()const{return low.bytes()+high.bytes();}
};

static ReverseSplit18Host build_reverse_split18(
    const StorageLayout&layout,const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf,
    const BucketReverseOrbitClosureAttachHost&attach
){
    // rb must still contain its original jblock/kind encoding here.
    validate_reverse_bucket_partner_blocks(layout,rb);
    ReverseSplit18Host out;out.nblocks=rb.nblocks;size_t pitch=size_t(rb.nblocks)+1;
    auto build=[&](bool low,ReverseSplit18SideHost&s){
        uint32_t steps=low?LOW_LUT_K:HIGH_LUT_K;s.nn_off.resize(size_t(steps)*pitch);s.nr_off.resize(size_t(steps)*pitch);s.nl_off.resize(size_t(steps)*pitch);
        const auto&ov=low?rb.low_orbit:rb.high_orbit;const auto&oo=low?rb.low_orbit_off:rb.high_orbit_off;const auto&att=low?attach.low:attach.high;
        int p0=low?1:LOW_LUT_K+1;
        for(uint32_t pi=0;pi<steps;++pi){int p=p0+int(pi);bool edge=!low&&p==TARGET_W-1;
            for(uint32_t bid=0;bid<rb.nblocks;++bid){size_t oi=size_t(pi)*pitch+bid;s.nn_off[oi]=uint32_t(s.nn.size());s.nr_off[oi]=uint32_t(s.nr.size());s.nl_off[oi]=uint32_t(s.nl.size());uint32_t dbid=low?uint32_t(layout.main_blocks[bid].he):(edge?bid:uint32_t(layout.main_blocks[bid].hs));const auto&fo=low?rf.low_off:rf.high_off;uint32_t fp=low?rf.low_pitch:rf.high_pitch;size_t doi=size_t(pi)*fp+dbid;uint32_t a=oo[oi],b=oo[oi+1];
                for(uint32_t q=a;q<b;++q){uint64_t w=ov[q];uint32_t kind=rb_orbit_kind(w),z=bkoc18_local_code(att[q],fo,doi,low?"reverse-split-low":"reverse-split-high");BucketOrbitOp nw=BucketOrbitOp(w&BKOC18_FORWARD_BASE_MASK)|(uint64_t(z&0x3ffu)<<54);uint8_t hi=uint8_t(z>>10);if(kind==CPU_ORBIT_NN){s.nn.push_back(nw);s.nn_hi.push_back(hi);}else if(kind==CPU_ORBIT_NR){s.nr.push_back(nw);s.nr_hi.push_back(hi);}else if(kind==CPU_ORBIT_NL){s.nl.push_back(nw);s.nl_hi.push_back(hi);}else{std::cerr<<"reverse split invalid orbit kind="<<kind<<'\n';std::exit(430);}}
            }size_t end=size_t(pi)*pitch+rb.nblocks;s.nn_off[end]=uint32_t(s.nn.size());s.nr_off[end]=uint32_t(s.nr.size());s.nl_off[end]=uint32_t(s.nl.size());
        }
    };
    build(true,out.low);build(false,out.high);
    if(out.low.ops()!=rb.low_orbit.size()||out.high.ops()!=rb.high_orbit.size())std::exit(431);
    std::cerr<<"reverse_split18 low_ops="<<out.low.ops()<<" high_ops="<<out.high.ops()<<" sidecar_bytes_per_orbit=1 kind_bits=0 jblock_bits=0 mib="<<double(out.bytes())/double(1<<20)<<'\n';return out;
}
