#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define ONEESAN_SHARD_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_SHARD_HD inline
#endif

namespace oneesan {
struct ShardAddress { unsigned owner; uint64_t offset; };

// Precondition: 1 <= devices <= 8, chunk > 0, and g is a valid global
// index with floor(g/chunk) < devices. Binary search of at most eight
// equally sized shards replaces a general 64-bit division. Comparing
// shifted g avoids overflow even when 4*chunk would not fit uint64_t.
ONEESAN_SHARD_HD ShardAddress shard_address(uint64_t g, uint64_t chunk, int devices) {
    if (devices == 1) return {0, g};
    unsigned owner = 0;
    if ((g >> 2) >= chunk) { g -= chunk << 2; owner = 4; }
    if ((g >> 1) >= chunk) { g -= chunk << 1; owner += 2; }
    if (g >= chunk) { g -= chunk; ++owner; }
    return {owner, g};
}
} // namespace oneesan

#undef ONEESAN_SHARD_HD
