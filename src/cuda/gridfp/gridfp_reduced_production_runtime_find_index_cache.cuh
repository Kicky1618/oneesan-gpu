#pragma once

#include "gridfp_reduced_production_runtime_shared_key.cuh"

#ifndef RP_RUNTIME_FIND_INDEX_CACHE
#define RP_RUNTIME_FIND_INDEX_CACHE 0
#endif
#ifndef RP_RUNTIME_FIND_INDEX_BUCKETS
#define RP_RUNTIME_FIND_INDEX_BUCKETS 64
#endif
static_assert(RP_RUNTIME_FIND_INDEX_CACHE == 0 || RP_RUNTIME_FIND_INDEX_CACHE == 1,
              "RP_RUNTIME_FIND_INDEX_CACHE must be 0 or 1");
static_assert(RP_RUNTIME_FIND_INDEX_BUCKETS == 16 ||
              RP_RUNTIME_FIND_INDEX_BUCKETS == 32 ||
              RP_RUNTIME_FIND_INDEX_BUCKETS == 64,
              "RP_RUNTIME_FIND_INDEX_BUCKETS must be 16, 32, or 64");

namespace oneesan::gridfp::reducedprod {

// One byte per hash bucket stores latest_index+1 in bits 0..6 and a persistent
// collision flag for the current component in bit 7.  The bytes do not need to
// be cleared between components: a lane-0 register occupancy mask identifies
// buckets written by the current component, so stale shared bytes are ignored.
// Packing collision state into the existing byte avoids the extra source/dest
// collision registers that a separate 64-bit mask would require.
struct RuntimeFindIndexCache {
    std::uint8_t latest_plus_one[RP_RUNTIME_FIND_INDEX_BUCKETS];
};
static_assert(sizeof(RuntimeFindIndexCache) == RP_RUNTIME_FIND_INDEX_BUCKETS,
              "runtime find index cache footprint regression");
static constexpr int RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SET =
    int(sizeof(RuntimeFindIndexCache));
static constexpr int RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SUBGROUP =
    2 * RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SET;
static_assert(RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SUBGROUP ==
              2 * RP_RUNTIME_FIND_INDEX_BUCKETS);
static constexpr std::uint8_t RP_RUNTIME_FIND_INDEX_COLLIDED = 0x80u;
static constexpr std::uint8_t RP_RUNTIME_FIND_INDEX_VALUE_MASK = 0x7fu;

__device__ __forceinline__ int runtime_find_index_bucket(DeviceKey k) {
    std::uint64_t x = std::uint64_t(k.mate) |
        (k.blocked ? RP_RUNTIME_SHARED_BLOCKED_BIT : 0ULL);
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & std::uint64_t(RP_RUNTIME_FIND_INDEX_BUCKETS - 1));
}

__device__ __forceinline__ bool runtime_shared_key_matches(
    RuntimeSharedKey stored, DeviceKey k
) {
#if RP_RUNTIME_PACK_SHARED_KEYS
    return stored == runtime_shared_key_encode(k);
#else
    return key_equal(stored, k);
#endif
}

__device__ __forceinline__ int runtime_find_shared_key_indexed(
    const RuntimeSharedKey* a,
    int n,
    DeviceKey k,
    std::uint64_t occupancy,
    const RuntimeFindIndexCache& cache
) {
    const int bucket = runtime_find_index_bucket(k);
    const std::uint64_t bit = 1ULL << bucket;
    if ((occupancy & bit) == 0) return -1;

    const std::uint8_t packed = cache.latest_plus_one[bucket];
    const int candidate = int(packed & RP_RUNTIME_FIND_INDEX_VALUE_MASK) - 1;
    if (candidate >= 0 && candidate < n) {
        if (runtime_shared_key_matches(a[candidate], k)) return candidate;
        // If this bucket has contained exactly one key in the current component,
        // candidate mismatch proves absence. Only genuinely collided buckets can
        // contain an older matching key and require exact fallback.
        if ((packed & RP_RUNTIME_FIND_INDEX_COLLIDED) == 0) return -1;
#if RP_RUNTIME_FIND_RECENT_FIRST
        for (int i = n - 1; i >= 0; --i) {
            if (i == candidate) continue;
            if (runtime_shared_key_matches(a[i], k)) return i;
        }
        return -1;
#else
        for (int i = 0; i < n; ++i) {
            if (i == candidate) continue;
            if (runtime_shared_key_matches(a[i], k)) return i;
        }
        return -1;
#endif
    }
    return runtime_find_shared_key(a, n, k);
}

__device__ __forceinline__ void runtime_find_index_record(
    RuntimeFindIndexCache& cache,
    std::uint64_t& occupancy,
    DeviceKey k,
    int index
) {
    const int bucket = runtime_find_index_bucket(k);
    const std::uint64_t bit = 1ULL << bucket;
    const bool collided = (occupancy & bit) != 0;
    const std::uint8_t old = cache.latest_plus_one[bucket];
    const std::uint8_t flags = collided
        ? std::uint8_t((old & RP_RUNTIME_FIND_INDEX_COLLIDED) |
                       RP_RUNTIME_FIND_INDEX_COLLIDED)
        : 0u;
    cache.latest_plus_one[bucket] =
        std::uint8_t(flags | std::uint8_t(index + 1));
    occupancy |= bit;
}

struct RuntimeIndexedSharedKeySetSink {
    RuntimeSharedKey* out = nullptr;
    int* n = nullptr;
    int cap = 0;
    std::uint64_t* occupancy = nullptr;
    RuntimeFindIndexCache* cache = nullptr;

    __device__ __forceinline__ bool emit(DeviceKey k) {
        if (occupancy && cache) {
            if (runtime_find_shared_key_indexed(
                    out, *n, k, *occupancy, *cache) >= 0)
                return true;
            if (*n >= cap) return false;
            const int index = (*n)++;
            out[index] = runtime_shared_key_encode(k);
            runtime_find_index_record(*cache, *occupancy, k, index);
            return true;
        }
        RuntimeSharedKeySetSink fallback{out, n, cap, nullptr};
        return fallback.emit(k);
    }
};

struct RuntimeIndexedMirroredKeySetSink {
    RuntimeIndexedSharedKeySetSink sink{};
    int W = 0;
    __device__ __forceinline__ bool emit(DeviceKey k) {
        return sink.emit(mirror_key_device(k, W));
    }
};

__device__ __forceinline__ bool runtime_discover_inverse_direction_to_shared_indexed(
    DeviceKey dest, int W, int p, bool reverse,
    RuntimeSharedKey* source_set, int& source_count, int capacity,
    std::uint64_t& source_occupancy,
    RuntimeFindIndexCache& source_cache
) {
    RuntimeIndexedSharedKeySetSink base{
        source_set, &source_count, capacity, &source_occupancy, &source_cache};
    if (!reverse)
        return runtime_discover_inverse_reduced_forward(dest, W, p, base);

    RuntimeIndexedMirroredKeySetSink mirrored{base, W};
    const DeviceKey md = mirror_key_device(dest, W);
    return runtime_discover_inverse_reduced_forward(md, W, W - p, mirrored);
}

} // namespace oneesan::gridfp::reducedprod
