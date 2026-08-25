#pragma once

#include "ramstream32_bucket_reverse_split18.cuh"
#include "ramstream32_bucket_orbit_closure_preflight.cuh"

static ReverseSplit18Host build_reverse_split18_checked_direct(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    auto attach=build_bucket_reverse_orbit_closure_attach_checked(layout,bo,bf,rb,rf);
    return build_reverse_split18(layout,rb,rf,attach);
}
