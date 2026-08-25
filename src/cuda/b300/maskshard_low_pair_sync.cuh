#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifndef MASKSHARD_LOW_PAIR_SYNC
#error "LOW pair sync header requires MASKSHARD_LOW_PAIR_SYNC"
#endif
#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "LOW pair sync currently layers on the compact LOW group host hook"
#endif

// process_fullorbit_batch_low_group() launches, for every LOW position,
//
//   orbit -> cudaDeviceSynchronize -> closure -> cudaDeviceSynchronize.
//
// Orbit and closure are issued by the same host thread to the same CUDA default
// stream, so stream order already enforces orbit-before-closure.  Only the first
// device-wide wait is redundant.  Intercept synchronizations only on LOW worker
// threads after configure_low_group(); HIGH workers, setup/reset, and the main
// thread never enter this thread-local mode and retain the original behavior.
//
// Stage timing caveat: low_orbit_s becomes mostly enqueue time while
// low_closure_s contains the combined orbit+closure device time.  wall_s remains
// directly comparable and is the metric this experiment is intended to test.
static thread_local bool G_MS_LOW_PAIR_SYNC_ACTIVE = false;
static thread_local std::uint32_t G_MS_LOW_PAIR_SYNC_PHASE = 0;
static std::atomic<std::uint64_t> G_MS_LOW_PAIR_SYNC_SKIPPED{0};
static std::atomic<std::uint64_t> G_MS_LOW_PAIR_SYNC_EXECUTED{0};

static cudaError_t maskshard_low_pair_device_sync() {
    if (G_MS_LOW_PAIR_SYNC_ACTIVE) {
        const std::uint32_t phase = G_MS_LOW_PAIR_SYNC_PHASE++;
        if ((phase & 1u) == 0u) {
            G_MS_LOW_PAIR_SYNC_SKIPPED.fetch_add(1, std::memory_order_relaxed);
            return cudaSuccess;
        }
        G_MS_LOW_PAIR_SYNC_EXECUTED.fetch_add(1, std::memory_order_relaxed);
    }
    return cudaDeviceSynchronize();
}

static void maskshard_configure_low_group_pair_sync(std::uint32_t mask) {
    if (G_MS_LOW_PAIR_SYNC_ACTIVE && (G_MS_LOW_PAIR_SYNC_PHASE & 1u)) {
        std::cerr << "LOW pair sync previous group ended on unmatched orbit sync phase="
                  << G_MS_LOW_PAIR_SYNC_PHASE << '\n';
        std::exit(327);
    }
    maskshard_configure_low_group_loworbit_compact(mask);
    G_MS_LOW_PAIR_SYNC_ACTIVE = true;
    G_MS_LOW_PAIR_SYNC_PHASE = 0;
}

#ifdef maskshard_configure_low_group
#undef maskshard_configure_low_group
#endif
#define maskshard_configure_low_group maskshard_configure_low_group_pair_sync

// Define this only after maskshard_low_pair_device_sync() itself has been parsed,
// so its fallback call above still resolves to the CUDA runtime function.
#define cudaDeviceSynchronize maskshard_low_pair_device_sync
