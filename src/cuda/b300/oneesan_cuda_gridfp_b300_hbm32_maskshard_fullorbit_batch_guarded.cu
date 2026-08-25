#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>

// Production wrapper for the experimental v0.4 batch solver.  Every cudaMalloc
// must leave a configurable HBM reserve so allocator/context overhead cannot
// turn the final HIGH scratch allocation into a late OOM after ~2 TiB of
// authoritative state has already been allocated across the node.
//
// Default reserve: clamp(total_HBM/32, 256 MiB, 8 GiB).
// Override: GRIDFP_VRAM_RESERVE_MIB=<nonnegative MiB>.
template<class T>
static cudaError_t maskshard_guarded_cuda_malloc(T** out, std::size_t bytes) {
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (e != cudaSuccess) return e;

    std::size_t reserve_mib = std::min<std::size_t>(
        8192, std::max<std::size_t>(256, (total_bytes >> 20) / 32));
    if (const char* s = std::getenv("GRIDFP_VRAM_RESERVE_MIB")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(s, &end, 10);
        if (end && *end == '\0') reserve_mib = std::size_t(v);
    }
    const std::size_t reserve_bytes = reserve_mib << 20;
    if (bytes > free_bytes || free_bytes - bytes < reserve_bytes) {
        int dev = -1;
        cudaGetDevice(&dev);
        std::cerr << "maskshard HBM admission rejected dev=" << dev
                  << " request_mib=" << (bytes >> 20)
                  << " free_mib=" << (free_bytes >> 20)
                  << " total_mib=" << (total_bytes >> 20)
                  << " reserve_mib=" << reserve_mib << '\n';
        return cudaErrorMemoryAllocation;
    }
    return cudaMalloc(reinterpret_cast<void**>(out), bytes);
}

#define cudaMalloc maskshard_guarded_cuda_malloc
#include "oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch.cu"
#undef cudaMalloc
