#pragma once

#include "ramstream32_bucket_reverse_split18.cuh"
#include "ramstream32_bucket_reverse_split18_direct.hpp"

static ReverseSplit18Host build_reverse_split18_checked_direct(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    return build_reverse_split18_direct_checked(layout,bo,bf,rb,rf,true);
}
