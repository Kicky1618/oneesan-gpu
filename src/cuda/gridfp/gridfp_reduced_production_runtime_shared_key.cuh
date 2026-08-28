#pragma once

#include "gridfp_reduced_production_discovery_device.cuh"
#include "gridfp_reduced_production_runtime_label_unrank.cuh"

// Everything including this header is part of the counter-free runtime path.
// Redirect only subsequent runtime call sites; legacy owner-component probes
// were parsed before this macro and keep their original division-based unrank.
#define owner_component_label_unrank_planned_device \
    runtime_owner_component_label_unrank_planned_device

namespace oneesan::gridfp::reducedprod {

#ifndef RP_RUNTIME_PACK_SHARED_KEYS
#define RP_RUNTIME_PACK_SHARED_KEYS 1
#endif
static_assert(RP_RUNTIME_PACK_SHARED_KEYS == 0 || RP_RUNTIME_PACK_SHARED_KEYS == 1,
              "RP_RUNTIME_PACK_SHARED_KEYS must be 0 or 1");

// MateID uses two bits per frontier position. Production is capped at W=28,
// therefore bits 56..63 are unused by every valid mate. Keep blocked in bit 63
// while keys live in shared memory, halving the usual 16-byte DeviceKey storage.
static constexpr std::uint64_t RP_RUNTIME_SHARED_BLOCKED_BIT = 1ULL << 63;
static constexpr std::uint64_t RP_RUNTIME_SHARED_MATE_MASK =
    RP_RUNTIME_SHARED_BLOCKED_BIT - 1ULL;
static_assert(2 * RP_MAX_W < 63,
              "packed shared key requires one free high bit");

#if RP_RUNTIME_PACK_SHARED_KEYS
using RuntimeSharedKey = std::uint64_t;
#else
using RuntimeSharedKey = DeviceKey;
#endif

__device__ __forceinline__ RuntimeSharedKey runtime_shared_key_encode(DeviceKey k) {
#if RP_RUNTIME_PACK_SHARED_KEYS
    return std::uint64_t(k.mate) |
           (k.blocked ? RP_RUNTIME_SHARED_BLOCKED_BIT : 0ULL);
#else
    return k;
#endif
}

__device__ __forceinline__ DeviceKey runtime_shared_key_decode(RuntimeSharedKey k) {
#if RP_RUNTIME_PACK_SHARED_KEYS
    return DeviceKey{
        MateID(std::uint64_t(k) & RP_RUNTIME_SHARED_MATE_MASK),
        std::uint8_t((std::uint64_t(k) & RP_RUNTIME_SHARED_BLOCKED_BIT) != 0)};
#else
    return k;
#endif
}

__device__ __forceinline__ int runtime_find_shared_key(
    const RuntimeSharedKey* a,
    int n,
    DeviceKey k
) {
#if RP_RUNTIME_PACK_SHARED_KEYS
    const RuntimeSharedKey needle = runtime_shared_key_encode(k);
    for (int i = 0; i < n; ++i)
        if (a[i] == needle) return i;
#else
    for (int i = 0; i < n; ++i)
        if (key_equal(a[i], k)) return i;
#endif
    return -1;
}

struct RuntimeSharedKeySetSink {
    RuntimeSharedKey* out = nullptr;
    int* n = nullptr;
    int cap = 0;

    __device__ __forceinline__ bool emit(DeviceKey k) {
        if (runtime_find_shared_key(out, *n, k) >= 0) return true;
        if (*n >= cap) return false;
        out[(*n)++] = runtime_shared_key_encode(k);
        return true;
    }
};

struct RuntimeSharedMirroredKeySetSink {
    RuntimeSharedKeySetSink sink{};
    int W = 0;

    __device__ __forceinline__ bool emit(DeviceKey k) {
        return sink.emit(mirror_key_device(k, W));
    }
};

__device__ __forceinline__ bool runtime_discover_inverse_direction_to_shared(
    DeviceKey dest,
    int W,
    int p,
    bool reverse,
    RuntimeSharedKey* source_set,
    int& source_count,
    int capacity
) {
    RuntimeSharedKeySetSink base{source_set, &source_count, capacity};
    if (!reverse)
        return discover_inverse_reduced_forward(dest, W, p, base);

    RuntimeSharedMirroredKeySetSink mirrored{base, W};
    const DeviceKey md = mirror_key_device(dest, W);
    return discover_inverse_reduced_forward(md, W, W - p, mirrored);
}

static constexpr int RP_RUNTIME_SHARED_KEY_BYTES = sizeof(RuntimeSharedKey);
static constexpr int RP_RUNTIME_DEVICE_KEY_BYTES = sizeof(DeviceKey);

} // namespace oneesan::gridfp::reducedprod
