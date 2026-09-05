#pragma once

// Include the LUT-substituted depth8 implementation first.  The high-context
// header then reuses bkcpd8_forward_high/reverse_high from that implementation.
#include "ramstream32_bucket_orbit_closure_pattern10_depth8_lut.cuh"
#include "ramstream32_bucket_orbit_closure_pattern10_depth8_highctx.cuh"
