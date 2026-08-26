#pragma once

#include "ramstream32_bucket_orbit_closure_pattern10.cuh"
#include "ramstream32_bucket_onepass_zero_alias.cuh"

struct BucketForwardPattern10Host{size_t bytes()const{return 0;}};

static BucketForwardPattern10Host build_bucket_forward_pattern10_zero(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf
){
    build_bucket_forward_pattern10(layout,bo,bf);
    bucket_zero_release_forward_closure(bf);
    return {};
}
static ReverseSplit54Host build_bucket_reverse_pattern10_zero_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    ReverseSplit54Host out=build_reverse_split54(layout,rb,true);
    build_reverse_split54_pattern10(layout,bf,out);
    bucket_zero_release_reverse_closure(rf);
    return out;
}

struct BucketForwardPattern10DeviceTables{void install(const BucketForwardPattern10Host&){}void release(){}};
