// Reuse the exact same bucket-fused driver and substitute only the transpose
// context.  Include both transpose headers first so the driver's own #include
// is suppressed by #pragma once before the token alias is active.
#include "gridfp_bucket_transpose.cuh"
#include "gridfp_bucket_transpose_events.cuh"

#define BucketTransposeCtx BucketTransposeEventCtx
#include "oneesan_cuda_gridfp_b300_bucket_fused.cu"
#undef BucketTransposeCtx
