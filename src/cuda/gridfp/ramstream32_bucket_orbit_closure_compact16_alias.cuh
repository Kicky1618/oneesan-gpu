#pragma once

#include "ramstream32_bucket_orbit_closure_compact16.cuh"

static BucketForwardOrbitClosureAttach16Host build_bucket_forward_orbit_closure_attach16_direct(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf
){
    auto full=build_bucket_forward_orbit_closure_attach(layout,bo,bf);
    return build_bucket_forward_orbit_closure_attach16(layout,bo,bf,full);
}
static BucketReverseOrbitClosureAttach16Host build_bucket_reverse_orbit_closure_attach16_direct(
    const StorageLayout&layout,const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    auto full=build_bucket_reverse_orbit_closure_attach(layout,rb,rf);
    return build_bucket_reverse_orbit_closure_attach16(layout,rb,rf,full);
}
