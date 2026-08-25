#pragma once

#include "ramstream32_bucket_reverse_atomic.cuh"

// Hybrid backends only need ReverseBucketAtomicHost after this conversion.
// The generic driver keeps rlow/rhigh/rlo/rhi locals alive until main exits,
// so release their vector payloads as soon as bucket metadata has been built.
// The wrapper keeps the original const signature because bsn_build_reverse()
// receives const references; hybrid callers pass ordinary mutable locals.
static ReverseBucketAtomicHost build_reverse_bucket_atomic_release_inputs(
    const StorageFactorHost&storage,const StorageLayout&layout,const BucketOwnerHost&owner,
    const ReverseLowDescHost&rlow_in,const ReverseHighDescHost&rhigh_in,
    const ReverseOrbitHost&rlo_in,const ReverseOrbitHost&rhi_in
){
    ReverseBucketAtomicHost out=build_reverse_bucket_atomic(
        storage,layout,owner,rlow_in,rhigh_in,rlo_in,rhi_in);

    const size_t desc_bytes=(rlow_in.main_desc.size()+rlow_in.block_desc.size()
        +rhigh_in.main_desc.size()+rhigh_in.block_desc.size())*sizeof(ReverseDesc);
    const size_t orbit_bytes=(rlo_in.rec.size()+rhi_in.rec.size())*sizeof(uint64_t);

    auto&rlow=const_cast<ReverseLowDescHost&>(rlow_in);
    auto&rhigh=const_cast<ReverseHighDescHost&>(rhigh_in);
    auto&rlo=const_cast<ReverseOrbitHost&>(rlo_in);
    auto&rhi=const_cast<ReverseOrbitHost&>(rhi_in);
    std::vector<ReverseDesc>().swap(rlow.main_desc);
    std::vector<ReverseDesc>().swap(rlow.block_desc);
    std::vector<ReverseDesc>().swap(rhigh.main_desc);
    std::vector<ReverseDesc>().swap(rhigh.block_desc);
    std::vector<uint64_t>().swap(rlo.rec);
    std::vector<uint64_t>().swap(rhi.rec);

    std::cerr<<"reverse_build released_source_mib="
             <<double(desc_bytes+orbit_bytes)/double(1<<20)
             <<" descriptor_mib="<<double(desc_bytes)/double(1<<20)
             <<" orbit_source_mib="<<double(orbit_bytes)/double(1<<20)<<'\n';
    return out;
}
