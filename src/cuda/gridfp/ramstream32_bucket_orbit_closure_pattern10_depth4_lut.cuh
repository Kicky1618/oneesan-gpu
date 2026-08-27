#pragma once

#include "ramstream32_bucket_closure_pattern10_lut.cuh"
#include "ramstream32_bucket_onepass_pattern10_depth4_alias.cuh"

__device__ __forceinline__ void bkcp10_depth4_decode_lut(
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

#define closure_pattern10_decode bkcp10_depth4_decode_lut
#define P10D8_DEPTH_LOAD(ptr,q) uint8_t((((ptr)[uint32_t(q)>>1])>>((uint32_t(q)&1u)*4u))&0xfu)
#include "ramstream32_bucket_orbit_closure_pattern10_depth8.cuh"
#undef P10D8_DEPTH_LOAD
#undef closure_pattern10_decode

struct BucketForwardPattern10Depth4LutHost : BucketForwardPattern10Depth4Host {
    BucketPattern10DecodeLutHost lut;
    size_t bytes() const { return BucketForwardPattern10Depth4Host::bytes() + lut.bytes(); }
};

static BucketForwardPattern10Depth4LutHost build_bucket_forward_pattern10_depth4_lut_zero(
    const StorageLayout& layout, BucketOrbitStreamsHost& bo, BucketFusedHost& bf
) {
    BucketForwardPattern10Depth4LutHost out;
    static_cast<BucketForwardPattern10Depth4Host&>(out) =
        build_bucket_forward_pattern10_depth4_zero(layout, bo, bf);
    out.lut = build_bucket_pattern10_decode_lut();
    return out;
}

struct BucketForwardPattern10Depth4LutDeviceTables : BucketForwardPattern10Depth4DeviceTables {
    BucketPattern10DecodeLutDeviceTables lut;
    void install(const BucketForwardPattern10Depth4LutHost& h) {
        BucketForwardPattern10Depth4DeviceTables::install(
            static_cast<const BucketForwardPattern10Depth4Host&>(h));
        lut.install(h.lut);
    }
    void release() {
        lut.release();
        BucketForwardPattern10Depth4DeviceTables::release();
    }
};
