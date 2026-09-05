#pragma once

#include "ramstream32_bucket_reverse_atomic.cuh"
#include "ramstream32_reverse_bucket_derive.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Reverse snake orbit metadata with no closure attachment. Stream identity is
// the orbit kind and partner block is reconstructed from (source block,p,kind).
// Each op therefore consists only of the three 18-bit owner-local locators.
struct ReverseSplit54SideHost{
    std::vector<BucketOrbitOp> nn,nr,nl;
    std::vector<uint32_t> nn_off,nr_off,nl_off;
    size_t bytes()const{return (nn.size()+nr.size()+nl.size())*sizeof(BucketOrbitOp)+(nn_off.size()+nr_off.size()+nl_off.size())*sizeof(uint32_t);}
    size_t ops()const{return nn.size()+nr.size()+nl.size();}
};
struct ReverseSplit54Host{
    ReverseSplit54SideHost low,high;uint32_t nblocks=0;
    size_t bytes()const{return low.bytes()+high.bytes();}
};

static ReverseSplit54Host build_reverse_split54(
    const StorageLayout&layout,ReverseBucketAtomicHost&rb,bool release_legacy=false
){
    validate_reverse_bucket_partner_blocks(layout,rb);
    ReverseSplit54Host out;out.nblocks=rb.nblocks;const size_t pitch=size_t(rb.nblocks)+1;
    auto build=[&](bool low,ReverseSplit54SideHost&s){
        auto&ov=low?rb.low_orbit:rb.high_orbit;auto&oo=low?rb.low_orbit_off:rb.high_orbit_off;
        const uint32_t steps=low?LOW_LUT_K:HIGH_LUT_K;s.nn_off.resize(size_t(steps)*pitch);s.nr_off.resize(size_t(steps)*pitch);s.nl_off.resize(size_t(steps)*pitch);
        size_t cnn=0,cnr=0,cnl=0;for(uint64_t w:ov){uint32_t k=rb_orbit_kind(w);if(k==CPU_ORBIT_NN)++cnn;else if(k==CPU_ORBIT_NR)++cnr;else if(k==CPU_ORBIT_NL)++cnl;else std::exit(550);}s.nn.reserve(cnn);s.nr.reserve(cnr);s.nl.reserve(cnl);
        for(uint32_t pi=0;pi<steps;++pi){for(uint32_t bid=0;bid<rb.nblocks;++bid){size_t oi=size_t(pi)*pitch+bid;s.nn_off[oi]=uint32_t(s.nn.size());s.nr_off[oi]=uint32_t(s.nr.size());s.nl_off[oi]=uint32_t(s.nl.size());uint32_t a=oo[oi],b=oo[oi+1];for(uint32_t q=a;q<b;++q){uint64_t w=ov[q];BucketOrbitOp z=BucketOrbitOp(w&((1ull<<54)-1ull));uint32_t k=rb_orbit_kind(w);if(k==CPU_ORBIT_NN)s.nn.push_back(z);else if(k==CPU_ORBIT_NR)s.nr.push_back(z);else s.nl.push_back(z);}}size_t e=size_t(pi)*pitch+rb.nblocks;s.nn_off[e]=uint32_t(s.nn.size());s.nr_off[e]=uint32_t(s.nr.size());s.nl_off[e]=uint32_t(s.nl.size());}
    };
    build(true,out.low);build(false,out.high);
    if(out.low.ops()!=rb.low_orbit.size()||out.high.ops()!=rb.high_orbit.size())std::exit(551);
    size_t old=(rb.low_orbit.size()+rb.high_orbit.size())*sizeof(ReverseBucketOrbitOp)+(rb.low_orbit_off.size()+rb.high_orbit_off.size())*sizeof(uint32_t);
    if(release_legacy){std::vector<ReverseBucketOrbitOp>().swap(rb.low_orbit);std::vector<ReverseBucketOrbitOp>().swap(rb.high_orbit);std::vector<uint32_t>().swap(rb.low_orbit_off);std::vector<uint32_t>().swap(rb.high_orbit_off);std::vector<ReverseBucketClosureOp>().swap(rb.low_closure);std::vector<ReverseBucketClosureOp>().swap(rb.high_closure);std::vector<uint32_t>().swap(rb.low_closure_off);std::vector<uint32_t>().swap(rb.high_closure_off);}
    std::cerr<<"reverse_split54 low_ops="<<out.low.ops()<<" high_ops="<<out.high.ops()<<" old_orbit_mib="<<double(old)/double(1<<20)<<" split54_mib="<<double(out.bytes())/double(1<<20)<<" attach_bytes=0 kind_bits=0 jblock_bits=0\n";return out;
}
