#pragma once
#include "ramstream32_bucket_orbit_closure_checked.cuh"
#include "ramstream32_reverse_bucket_derive.hpp"

static BucketReverseOrbitClosureAttachHost build_bucket_reverse_orbit_closure_attach_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    const ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    validate_reverse_bucket_partner_blocks(layout,rb);
    (void)validate_bucket_orbit_closure_fusion(layout,bo,bf,rb,rf);
    return build_bucket_reverse_orbit_closure_attach(layout,rb,rf);
}
