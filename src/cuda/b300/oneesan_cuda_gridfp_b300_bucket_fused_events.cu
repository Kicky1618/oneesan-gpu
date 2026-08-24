// Reuse the exact same bucket-fused driver and select the event-driven
// transpose implementation through gridfp_bucket_transpose.cuh's hook.  Event
// dependencies remove per-chunk host synchronization, so a 1 GiB staging
// buffer is a safer default than the 4 GiB baseline while still giving each
// peer copy a very large transfer.
#define BUCKET_TRANSPOSE_USE_EVENTS 1
#define BUCKET_TRANSPOSE_DEFAULT_CHUNK_MIB 1024
#include "oneesan_cuda_gridfp_b300_bucket_fused.cu"
