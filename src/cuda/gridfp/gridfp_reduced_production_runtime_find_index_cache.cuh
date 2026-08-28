#pragma once

#include "gridfp_reduced_production_runtime_shared_key.cuh"

#ifndef RP_RUNTIME_FIND_INDEX_CACHE
#define RP_RUNTIME_FIND_INDEX_CACHE 0
#endif
// Storage bytes per source/destination set.  For WAYS=1 this is also the hash
// bucket count.  Keeping this as the storage-size macro preserves the existing
// subwarp shared-memory accounting when associativity changes.
#ifndef RP_RUNTIME_FIND_INDEX_BUCKETS
#define RP_RUNTIME_FIND_INDEX_BUCKETS 64
#endif
#ifndef RP_RUNTIME_FIND_INDEX_WAYS
#define RP_RUNTIME_FIND_INDEX_WAYS 1
#endif
static_assert(RP_RUNTIME_FIND_INDEX_CACHE == 0 || RP_RUNTIME_FIND_INDEX_CACHE == 1,
              "RP_RUNTIME_FIND_INDEX_CACHE must be 0 or 1");
static_assert(RP_RUNTIME_FIND_INDEX_BUCKETS == 16 ||
              RP_RUNTIME_FIND_INDEX_BUCKETS == 32 ||
              RP_RUNTIME_FIND_INDEX_BUCKETS == 64,
              "RP_RUNTIME_FIND_INDEX_BUCKETS must be 16, 32, or 64");
static_assert(RP_RUNTIME_FIND_INDEX_WAYS == 1 ||
              RP_RUNTIME_FIND_INDEX_WAYS == 2 ||
              RP_RUNTIME_FIND_INDEX_WAYS == 4,
              "RP_RUNTIME_FIND_INDEX_WAYS must be 1, 2, or 4");
static_assert(RP_RUNTIME_FIND_INDEX_BUCKETS % RP_RUNTIME_FIND_INDEX_WAYS == 0,
              "index-cache storage must divide evenly across ways");
static constexpr int RP_RUNTIME_FIND_INDEX_HASH_BUCKETS =
    RP_RUNTIME_FIND_INDEX_BUCKETS / RP_RUNTIME_FIND_INDEX_WAYS;
static_assert(RP_RUNTIME_FIND_INDEX_HASH_BUCKETS == 16 ||
              RP_RUNTIME_FIND_INDEX_HASH_BUCKETS == 32 ||
              RP_RUNTIME_FIND_INDEX_HASH_BUCKETS == 64,
              "index-cache hash buckets must be 16, 32, or 64");

namespace oneesan::gridfp::reducedprod {

// Each bucket keeps WAYS most-recent indices, one byte each. Bits 0..6 store
// index+1 and bit 7 of way 0 records overflow beyond the retained ways. The
// occupancy word makes stale shared bytes harmless across components. First
// insertion into a bucket clears only that bucket's extra ways; no full-table
// clear is needed between components.
struct RuntimeFindIndexCache {
    std::uint8_t slot[RP_RUNTIME_FIND_INDEX_HASH_BUCKETS]
                     [RP_RUNTIME_FIND_INDEX_WAYS];
};
static_assert(sizeof(RuntimeFindIndexCache) == RP_RUNTIME_FIND_INDEX_BUCKETS,
              "runtime find index cache footprint regression");
static constexpr int RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SET =
    int(sizeof(RuntimeFindIndexCache));
static constexpr int RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SUBGROUP =
    2 * RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SET;
static_assert(RP_RUNTIME_FIND_INDEX_CACHE_BYTES_PER_SUBGROUP ==
              2 * RP_RUNTIME_FIND_INDEX_BUCKETS);
static constexpr std::uint8_t RP_RUNTIME_FIND_INDEX_OVERFLOW = 0x80u;
static constexpr std::uint8_t RP_RUNTIME_FIND_INDEX_VALUE_MASK = 0x7fu;

__device__ __forceinline__ int runtime_find_index_bucket(DeviceKey k) {
    std::uint64_t x = std::uint64_t(k.mate) |
        (k.blocked ? RP_RUNTIME_SHARED_BLOCKED_BIT : 0ULL);
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & std::uint64_t(RP_RUNTIME_FIND_INDEX_HASH_BUCKETS - 1));
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

__device__ __forceinline__ bool runtime_find_index_is_cached(
    const RuntimeFindIndexCache& cache,
    int bucket,
    int index
) {
#pragma unroll
    for (int way = 0; way < RP_RUNTIME_FIND_INDEX_WAYS; ++way) {
        const int v = int(cache.slot[bucket][way] & RP_RUNTIME_FIND_INDEX_VALUE_MASK);
        if (!v) break;
        if (v - 1 == index) return true;
    }
    return false;
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

    const bool overflow =
        (cache.slot[bucket][0] & RP_RUNTIME_FIND_INDEX_OVERFLOW) != 0;
#pragma unroll
    for (int way = 0; way < RP_RUNTIME_FIND_INDEX_WAYS; ++way) {
        const int v = int(cache.slot[bucket][way] & RP_RUNTIME_FIND_INDEX_VALUE_MASK);
        if (!v) break;
        const int candidate = v - 1;
        if (candidate < 0 || candidate >= n)
            return runtime_find_shared_key(a, n, k);
        if (runtime_shared_key_matches(a[candidate], k)) return candidate;
    }

    // Every member of a non-overflowed bucket was compared above, so mismatch
    // proves absence. Only buckets that have received more than WAYS distinct
    // keys need an exact linear fallback for an older member.
    if (!overflow) return -1;
#if RP_RUNTIME_FIND_RECENT_FIRST
    for (int i = n - 1; i >= 0; --i) {
        if (runtime_find_index_is_cached(cache, bucket, i)) continue;
        if (runtime_shared_key_matches(a[i], k)) return i;
    }
#else
    for (int i = 0; i < n; ++i) {
        if (runtime_find_index_is_cached(cache, bucket, i)) continue;
        if (runtime_shared_key_matches(a[i], k)) return i;
    }
#endif
    return -1;
}

__device__ __forceinline__ void runtime_find_index_record(
    RuntimeFindIndexCache& cache,
    std::uint64_t& occupancy,
    DeviceKey k,
    int index
) {
    const int bucket = runtime_find_index_bucket(k);
    const std::uint64_t bit = 1ULL << bucket;
    if ((occupancy & bit) == 0) {
        cache.slot[bucket][0] = std::uint8_t(index + 1);
#pragma unroll
        for (int way = 1; way < RP_RUNTIME_FIND_INDEX_WAYS; ++way)
            cache.slot[bucket][way] = 0;
        occupancy |= bit;
        return;
    }

    bool overflow =
        (cache.slot[bucket][0] & RP_RUNTIME_FIND_INDEX_OVERFLOW) != 0;
    overflow |=
        (cache.slot[bucket][RP_RUNTIME_FIND_INDEX_WAYS - 1] &
         RP_RUNTIME_FIND_INDEX_VALUE_MASK) != 0;
#pragma unroll
    for (int way = RP_RUNTIME_FIND_INDEX_WAYS - 1; way > 0; --way) {
        cache.slot[bucket][way] =
            cache.slot[bucket][way - 1] & RP_RUNTIME_FIND_INDEX_VALUE_MASK;
    }
    cache.slot[bucket][0] = std::uint8_t(
        std::uint8_t(index + 1) |
        (overflow ? RP_RUNTIME_FIND_INDEX_OVERFLOW : 0u));
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
