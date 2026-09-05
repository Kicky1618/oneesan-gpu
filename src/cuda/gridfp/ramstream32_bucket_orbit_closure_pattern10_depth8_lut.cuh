#pragma once

#include "ramstream32_bucket_closure_pattern10_lut.cuh"

__device__ __forceinline__ void bkcp10_decode_lut(
    uint16_t id, int len, int p, uint16_t& lm, uint16_t& rm
) {
    if (len == LOW_LUT_K + 1) {
        bkcp10_decode_low_lut(id, len, p, lm, rm);
        return;
    }
    if (len == HIGH_LUT_K + 1) {
        bkcp10_decode_high_lut(id, len, p, lm, rm);
        return;
    }
    lm = rm = 0;
}

// Reuse the complete depth8 kernel body, replacing only Fibonacci unrank with
// a direct (p,id)->(left_mask,right_mask) lookup. All orbit algebra, depth8
// metadata and bucket addressing remain the same source path.
#define closure_pattern10_decode bkcp10_decode_lut
#include "ramstream32_bucket_orbit_closure_pattern10_depth8.cuh"
#undef closure_pattern10_decode

struct BucketForwardPattern10Depth8LutHost : BucketForwardPattern10Depth8Host {
    BucketPattern10DecodeLutHost lut;
    size_t bytes() const { return BucketForwardPattern10Depth8Host::bytes() + lut.bytes(); }
};

static BucketForwardPattern10Depth8LutHost build_bucket_forward_pattern10_depth8_lut_zero(
    const StorageLayout& layout, BucketOrbitStreamsHost& bo, BucketFusedHost& bf
) {
    BucketForwardPattern10Depth8LutHost out;
    static_cast<BucketForwardPattern10Depth8Host&>(out) =
        build_bucket_forward_pattern10_depth8_zero(layout, bo, bf);
    out.lut = build_bucket_pattern10_decode_lut();
    return out;
}

struct BucketForwardPattern10Depth8LutDeviceTables : BucketForwardPattern10Depth8DeviceTables {
    BucketPattern10DecodeLutDeviceTables lut;
    void install(const BucketForwardPattern10Depth8LutHost& h) {
        BucketForwardPattern10Depth8DeviceTables::install(
            static_cast<const BucketForwardPattern10Depth8Host&>(h));
        lut.install(h.lut);
    }
    void release() {
        lut.release();
        BucketForwardPattern10Depth8DeviceTables::release();
    }
};
