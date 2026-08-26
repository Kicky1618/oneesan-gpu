#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

#ifndef MASKSHARD_HIGH_GROUP_SYNC
#error "HIGH group sync header requires MASKSHARD_HIGH_GROUP_SYNC"
#endif
#ifndef MASKSHARD_LOW_MASKBATCH_CUDA_GRAPH
#error "automatic HIGH sync hook currently requires CUDA-Graph LOW backend"
#endif

// One HIGH job queues:
//   gather, then HIGH_LUT_K * (orbit, closure), then scatter.
// The HIGH worker has one CPU thread per GPU and processes jobs sequentially on
// that GPU's CUDA default stream. Stream order already preserves every data
// dependency.
//
// v0.57 keeps one wait after the final scatter of every job.  v0.61 optionally
// extends the same invariant across the whole HIGH row: constant updates are
// enqueued on the same stream, scratch is reused in stream order, and a
// thread-local destructor performs exactly one device wait when that row's HIGH
// worker exits. std::thread::join() therefore still establishes completion
// before the LOW phase starts.
//
// LOW worker threads use cudaStreamSynchronize(), while reset/setup/cleanup
// device waits run on the main host thread. Therefore explicit
// cudaDeviceSynchronize() calls made by non-main row workers are exactly the
// HIGH job sequence.
static const std::thread::id G_MS_HIGH_GROUP_SYNC_MAIN_THREAD =
    std::this_thread::get_id();
static thread_local std::uint32_t G_MS_HIGH_GROUP_SYNC_PHASE = 0;

#ifdef MASKSHARD_HIGH_ROW_BATCH_ASYNC
struct MaskShardHighRowBatchFinalSync {
    bool touched = false;
    ~MaskShardHighRowBatchFinalSync() {
        if (!touched) return;
        if (G_MS_HIGH_GROUP_SYNC_PHASE != 0) {
            std::cerr << "HIGH row-batch worker exited mid-job phase="
                      << G_MS_HIGH_GROUP_SYNC_PHASE << '\n';
            std::abort();
        }
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::cerr << "HIGH row-batch final sync: "
                      << cudaGetErrorString(e) << '\n';
            std::abort();
        }
    }
};
static thread_local MaskShardHighRowBatchFinalSync G_MS_HIGH_ROW_BATCH_FINAL_SYNC{};
#endif

static cudaError_t maskshard_high_group_device_sync() {
    if (std::this_thread::get_id() == G_MS_HIGH_GROUP_SYNC_MAIN_THREAD)
        return cudaDeviceSynchronize();

    constexpr std::uint32_t EXPECTED =
        2u * std::uint32_t(HIGH_LUT_K) + 2u;
    const std::uint32_t phase = ++G_MS_HIGH_GROUP_SYNC_PHASE;
#ifdef MASKSHARD_HIGH_ROW_BATCH_ASYNC
    G_MS_HIGH_ROW_BATCH_FINAL_SYNC.touched = true;
    if (phase < EXPECTED) return cudaSuccess;
    if (phase == EXPECTED) {
        G_MS_HIGH_GROUP_SYNC_PHASE = 0;
        return cudaSuccess;
    }
#else
    if (phase < EXPECTED) return cudaSuccess;
    if (phase == EXPECTED) {
        G_MS_HIGH_GROUP_SYNC_PHASE = 0;
        return cudaDeviceSynchronize();
    }
#endif

    std::cerr << "HIGH group sync exceeded expected phase count phase="
              << phase << " expected=" << EXPECTED << '\n';
    std::exit(359);
}

// Keep the real runtime call above unexpanded.
#define cudaDeviceSynchronize maskshard_high_group_device_sync
