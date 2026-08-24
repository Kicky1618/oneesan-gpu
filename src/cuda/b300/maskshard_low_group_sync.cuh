#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifndef MASKSHARD_LOW_GROUP_SYNC
#error "LOW group sync header requires MASKSHARD_LOW_GROUP_SYNC"
#endif
#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "LOW group sync currently layers on the compact LOW group host hook"
#endif

#ifdef MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS
#ifdef MASKSHARD_LOW_GROUP_PACKED_CONFIG
#include "maskshard_loworbit_warprow_packed.cuh"
#else
#include "maskshard_loworbit_warprow.cuh"
#endif
#endif

// Queue the complete LOW orbit/closure chain for one fixed HIGH-mask group on
// the same CUDA default stream and wait only after the final closure.  Stream
// order preserves every orbit -> closure -> next-orbit dependency.  The final
// wait also guarantees that the next mask may safely replace D_F_* constants.
//
// There are exactly 2*LOW_LUT_K synchronization call sites per processed group:
// one after orbit and one after closure for each p.  Intercept only LOW worker
// threads after configure_low_group(); all other host threads retain the normal
// CUDA runtime synchronization behavior.
//
// Stage timing caveat: per-position low_orbit_s/low_closure_s become enqueue
// timings except the final low_closure_s sample, which contains the whole group
// device time.  wall_s remains the comparable performance metric.
static thread_local bool G_MS_LOW_GROUP_SYNC_ACTIVE = false;
static thread_local std::uint32_t G_MS_LOW_GROUP_SYNC_PHASE = 0;
static std::atomic<std::uint64_t> G_MS_LOW_GROUP_SYNC_SKIPPED{0};
static std::atomic<std::uint64_t> G_MS_LOW_GROUP_SYNC_EXECUTED{0};

static cudaError_t maskshard_low_group_device_sync() {
    if (!G_MS_LOW_GROUP_SYNC_ACTIVE) return cudaDeviceSynchronize();

    constexpr std::uint32_t EXPECTED = 2u * std::uint32_t(LOW_LUT_K);
    const std::uint32_t phase = ++G_MS_LOW_GROUP_SYNC_PHASE;
    if (phase < EXPECTED) {
        G_MS_LOW_GROUP_SYNC_SKIPPED.fetch_add(1, std::memory_order_relaxed);
        return cudaSuccess;
    }
    if (phase == EXPECTED) {
        G_MS_LOW_GROUP_SYNC_EXECUTED.fetch_add(1, std::memory_order_relaxed);
        return cudaDeviceSynchronize();
    }

    std::cerr << "LOW group sync exceeded expected phase count phase="
              << phase << " expected=" << EXPECTED << '\n';
    std::exit(328);
}

static void maskshard_configure_low_group_group_sync(std::uint32_t mask) {
    constexpr std::uint32_t EXPECTED = 2u * std::uint32_t(LOW_LUT_K);
    if (G_MS_LOW_GROUP_SYNC_ACTIVE && G_MS_LOW_GROUP_SYNC_PHASE != EXPECTED) {
        std::cerr << "LOW group sync previous group phase mismatch got="
                  << G_MS_LOW_GROUP_SYNC_PHASE
                  << " expected=" << EXPECTED << '\n';
        std::exit(329);
    }
#ifdef MASKSHARD_LOW_ORBIT_WARP_ROW_TASKS
    maskshard_configure_low_group_warprow(mask);
#else
    maskshard_configure_low_group_loworbit_compact(mask);
#endif
    G_MS_LOW_GROUP_SYNC_ACTIVE = true;
    G_MS_LOW_GROUP_SYNC_PHASE = 0;
}

#ifdef maskshard_configure_low_group
#undef maskshard_configure_low_group
#endif
#define maskshard_configure_low_group maskshard_configure_low_group_group_sync

// Keep the CUDA runtime call in maskshard_low_group_device_sync() above
// unexpanded by defining the interception macro only after that function.
#define cudaDeviceSynchronize maskshard_low_group_device_sync
