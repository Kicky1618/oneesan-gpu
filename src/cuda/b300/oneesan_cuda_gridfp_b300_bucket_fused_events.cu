// Reuse the exact same bucket-fused driver and select the event-driven
// transpose implementation through gridfp_bucket_transpose.cuh's hook.  This
// keeps Count/BucketPhysicalLayoutHost definitions in the driver's normal
// include order and avoids including transpose headers before those types exist.
#define BUCKET_TRANSPOSE_USE_EVENTS 1
#include "oneesan_cuda_gridfp_b300_bucket_fused.cu"
