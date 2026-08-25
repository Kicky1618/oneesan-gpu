#pragma once

#include "ramstream32_bucket_reverse_split18.cuh"
#include "ramstream32_bucket_orbit_closure_preflight.cuh"

static ReverseSplit18Host build_reverse_split18_checked_direct(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    auto attach=build_bucket_reverse_orbit_closure_attach_checked(layout,bo,bf,rb,rf);
    auto split=build_reverse_split18(layout,rb,rf,attach);

    // Direct split18 executors install their own orbit streams.  Once the
    // checked split has been built, the legacy reverse orbit words and their
    // offsets are dead host data; retaining them also makes reverse.bytes()
    // overestimate the device-resident metadata used by the admission check.
    const size_t released=reverse_bucket_orbit_bytes(rb);
    std::vector<ReverseBucketOrbitOp>().swap(rb.low_orbit);
    std::vector<ReverseBucketOrbitOp>().swap(rb.high_orbit);
    std::vector<uint32_t>().swap(rb.low_orbit_off);
    std::vector<uint32_t>().swap(rb.high_orbit_off);
    std::cerr<<"reverse_split18 released_legacy_orbit_mib="
             <<double(released)/double(1<<20)<<'\n';
    return split;
}
