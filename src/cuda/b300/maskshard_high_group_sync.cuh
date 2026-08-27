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
// that GPU's execution stream. Stream order preserves every data dependency.
//
// v0.57 keeps one wait after the final scatter of every job. v0.61 optionally
// extends the same invariant across the whole HIGH row: constant updates are
// enqueued on the same stream, scratch is reused in stream order, and a
// thread-local destructor performs exactly one device wait when that row's HIGH
// worker exits. std::thread::join() therefore still establishes completion
// before the LOW phase starts.
//
// v0.72 may replay a captured graph instead of passing through the historical
// 2*HIGH_LUT_K+2 synchronize call sites. Such a replay calls
// maskshard_high_row_batch_mark_touched() directly; the phase counter remains
// zero and the same thread-local destructor supplies the row-end device wait.
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

static void maskshard_high_row_batch_mark_touched() {
    if (std::this_thread::get_id() == G_MS_HIGH_GROUP_SYNC_MAIN_THREAD) {
        std::cerr << "HIGH row-batch mark-touched called on main thread\n";
        std::abort();
    }
    if (G_MS_HIGH_GROUP_SYNC_PHASE != 0) {
        std::cerr << "HIGH row-batch mark-touched during phase="
                  << G_MS_HIGH_GROUP_SYNC_PHASE << '\n';
        std::abort();
    }
    G_MS_HIGH_ROW_BATCH_FINAL_SYNC.touched = true;
}
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
