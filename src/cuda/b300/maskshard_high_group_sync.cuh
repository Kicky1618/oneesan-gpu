#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifndef MASKSHARD_HIGH_GROUP_SYNC
#error "HIGH group sync header requires MASKSHARD_HIGH_GROUP_SYNC"
#endif

// One HIGH job queues:
//   gather, then HIGH_LUT_K * (orbit, closure), then scatter.
// Every operation uses the same CUDA default stream on the worker's GPU.
// Therefore the intermediate cudaDeviceSynchronize() calls are not needed for
// dependency ordering.  Keep only the final scatter wait; it also guarantees
// that the next job may safely replace D_F_* constants and reuse the scratch.
//
// Activation is explicit and thread-local so reset/LOW/other host threads keep
// normal cudaDeviceSynchronize() semantics.  The HIGH worker has exactly one
// CPU thread per GPU and processes jobs sequentially.
static thread_local bool G_MS_HIGH_GROUP_SYNC_ACTIVE = false;
static thread_local std::uint32_t G_MS_HIGH_GROUP_SYNC_PHASE = 0;
static std::atomic<std::uint64_t> G_MS_HIGH_GROUP_SYNC_GROUPS{0};
static std::atomic<std::uint64_t> G_MS_HIGH_GROUP_SYNC_SKIPPED{0};
static std::atomic<std::uint64_t> G_MS_HIGH_GROUP_SYNC_EXECUTED{0};

static void maskshard_high_group_sync_begin() {
    if (G_MS_HIGH_GROUP_SYNC_ACTIVE) {
        std::cerr << "HIGH group sync nested activation phase="
                  << G_MS_HIGH_GROUP_SYNC_PHASE << '\n';
        std::exit(358);
    }
    G_MS_HIGH_GROUP_SYNC_ACTIVE = true;
    G_MS_HIGH_GROUP_SYNC_PHASE = 0;
    G_MS_HIGH_GROUP_SYNC_GROUPS.fetch_add(1, std::memory_order_relaxed);
}

static cudaError_t maskshard_high_group_device_sync() {
    if (!G_MS_HIGH_GROUP_SYNC_ACTIVE) return cudaDeviceSynchronize();

    constexpr std::uint32_t EXPECTED =
        2u * std::uint32_t(HIGH_LUT_K) + 2u;
    const std::uint32_t phase = ++G_MS_HIGH_GROUP_SYNC_PHASE;
    if (phase < EXPECTED) {
        G_MS_HIGH_GROUP_SYNC_SKIPPED.fetch_add(1, std::memory_order_relaxed);
        return cudaSuccess;
    }
    if (phase == EXPECTED) {
        G_MS_HIGH_GROUP_SYNC_EXECUTED.fetch_add(1, std::memory_order_relaxed);
        G_MS_HIGH_GROUP_SYNC_ACTIVE = false;
        return cudaDeviceSynchronize();
    }

    std::cerr << "HIGH group sync exceeded expected phase count phase="
              << phase << " expected=" << EXPECTED << '\n';
    std::exit(359);
}

// Keep the real runtime call above unexpanded.
#define cudaDeviceSynchronize maskshard_high_group_device_sync
