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
// dependency. Only the final scatter wait is required before the next job may
// replace D_F_* constants or reuse the scratch arena.
//
// v0.57 layers on the v0.56 CUDA-Graph LOW backend. LOW worker threads wait
// with cudaStreamSynchronize(), while reset/setup/cleanup device waits run on
// the main host thread. Therefore explicit cudaDeviceSynchronize() calls made
// by non-main row workers are exactly the HIGH job sequence. Keep the hot hook
// counter-free; factor_high_group_sync.cpp pins the structural wait counts.
static const std::thread::id G_MS_HIGH_GROUP_SYNC_MAIN_THREAD =
    std::this_thread::get_id();
static thread_local std::uint32_t G_MS_HIGH_GROUP_SYNC_PHASE = 0;

static cudaError_t maskshard_high_group_device_sync() {
    if (std::this_thread::get_id() == G_MS_HIGH_GROUP_SYNC_MAIN_THREAD)
        return cudaDeviceSynchronize();

    constexpr std::uint32_t EXPECTED =
        2u * std::uint32_t(HIGH_LUT_K) + 2u;
    const std::uint32_t phase = ++G_MS_HIGH_GROUP_SYNC_PHASE;
    if (phase < EXPECTED) return cudaSuccess;
    if (phase == EXPECTED) {
        G_MS_HIGH_GROUP_SYNC_PHASE = 0;
        return cudaDeviceSynchronize();
    }

    std::cerr << "HIGH group sync exceeded expected phase count phase="
              << phase << " expected=" << EXPECTED << '\n';
    std::exit(359);
}

// Keep the real runtime call above unexpanded.
#define cudaDeviceSynchronize maskshard_high_group_device_sync
