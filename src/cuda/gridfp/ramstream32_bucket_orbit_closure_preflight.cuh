#pragma once
#include "ramstream32_bucket_orbit_closure_checked.cuh"

static BucketReverseOrbitClosureAttachHost build_bucket_reverse_orbit_closure_attach_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    (void)validate_bucket_orbit_closure_fusion(layout,bo,bf,rb,rf);
    return build_bucket_reverse_orbit_closure_attach(layout,rb,rf);
}
