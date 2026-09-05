#pragma once

// Compatibility shim.  The canonical depthcode backend now stores fully
// predecoded {left_mask,right_mask,depth,valid} payloads in its context
// codebook, so the former "predecoded" experiment is byte-for-byte the same
// execution path.  Keep the old names for existing B300 A/B wrappers without
// maintaining a second kernel implementation.
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode.cuh"

using BucketForwardPattern10DepthCodePredecodedHost = BucketForwardPattern10DepthCodeHost;
using BucketReversePattern10DepthCodePredecodedHost = BucketReversePattern10DepthCodeHost;
using BucketForwardPattern10DepthCodePredecodedDeviceTables = BucketForwardPattern10DepthCodeDeviceTables;
using BucketReversePattern10DepthCodePredecodedDeviceTables = BucketReversePattern10DepthCodeDeviceTables;

static BucketForwardPattern10DepthCodePredecodedHost
build_bucket_forward_pattern10_depthcode_predecoded_placeholder(
    const StorageLayout& layout, BucketOrbitStreamsHost& bo, BucketFusedHost& bf
) {
    return build_bucket_forward_pattern10_depthcode_placeholder(layout, bo, bf);
}

static BucketReversePattern10DepthCodePredecodedHost
build_bucket_reverse_pattern10_depthcode_predecoded_zero_checked(
    const StorageLayout& layout, BucketOrbitStreamsHost& bo, BucketFusedHost& bf,
    ReverseBucketAtomicHost& rb, ReverseBucketFusedHost& rf
) {
    return build_bucket_reverse_pattern10_depthcode_zero_checked(layout, bo, bf, rb, rf);
}

#define bucket_low_orbit_closure_pattern10_depthcode_predecoded_kernel \
    bucket_low_orbit_closure_pattern10_depthcode_kernel
#define bucket_high_orbit_closure_pattern10_depthcode_predecoded_kernel \
    bucket_high_orbit_closure_pattern10_depthcode_kernel
#define bucket_reverse_low_pattern10_depthcode_predecoded_kernel \
    bucket_reverse_low_pattern10_depthcode_kernel
#define bucket_reverse_high_pattern10_depthcode_predecoded_kernel \
    bucket_reverse_high_pattern10_depthcode_kernel
