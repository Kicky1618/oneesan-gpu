#pragma once

#include "ramstream32_bucket_onepass_pattern10_depth4_alias.cuh"

// Two depth descriptors per byte. q is the stream-local orbit index, so the
// packing remains coalesced and requires one byte load plus a shift/mask.
#define P10D8_DEPTH_LOAD(ptr,q) uint8_t((((ptr)[uint32_t(q)>>1])>>((uint32_t(q)&1u)*4u))&0xfu)
#include "ramstream32_bucket_orbit_closure_pattern10_depth8.cuh"
#undef P10D8_DEPTH_LOAD
