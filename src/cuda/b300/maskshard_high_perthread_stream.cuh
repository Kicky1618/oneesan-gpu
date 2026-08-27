#pragma once

#ifndef MASKSHARD_HIGH_PERTHREAD_STREAM
#error "HIGH per-thread stream header requires MASKSHARD_HIGH_PERTHREAD_STREAM"
#endif
#ifndef MASKSHARD_HIGH_ROW_BATCH_ASYNC
#error "HIGH per-thread stream requires row-batch asynchronous config"
#endif

// v0.70: route every HIGH-worker asynchronous config update and HIGH kernel
// launch through CUDA's built-in per-thread stream. This stream is independent
// of the legacy null stream and, unlike cudaStreamLegacy, can be used as the
// capture target for the next CUDA-Graph experiment. No stream allocation or
// lifetime management is required; one HIGH CPU worker already owns one GPU.
static __host__ __forceinline__ cudaStream_t maskshard_high_execution_stream() {
    return cudaStreamPerThread;
}
