#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

#ifndef B300_HI32_SEED_MODE
#define B300_HI32_SEED_MODE 0
#endif
static_assert(B300_HI32_SEED_MODE >= 0 && B300_HI32_SEED_MODE <= 2,
              "B300_HI32_SEED_MODE must be 0..2");

namespace {

static constexpr Code MAIN_TOTAL = 385719506620ULL;
static constexpr Code MAIN_CHUNK = 48214938328ULL;

__device__ __forceinline__ Code masked_base(int owner) {
    const Code u = Code(unsigned(owner));
    return ((Code(0) - (u & 1ULL)) & MAIN_CHUNK) +
           ((Code(0) - ((u >> 1) & 1ULL)) & (MAIN_CHUNK << 1)) +
           ((Code(0) - ((u >> 2) & 1ULL)) & (MAIN_CHUNK << 2));
}

__device__ __forceinline__ std::uint32_t seed_mul(Code g) {
    const std::uint32_t h = std::uint32_t(g >> 32);
    return (h * 365u) >> 12;
}

__device__ __forceinline__ std::uint32_t seed_shiftadd(Code g) {
    const std::uint32_t h = std::uint32_t(g >> 32);
    return ((h << 8) + (h << 6) + (h << 5) +
            (h << 3) + (h << 2) + h) >> 12;
}

template<bool SHIFTADD>
__device__ __forceinline__ ShardAddress8 hi32_seed_address(Code g) {
    std::uint32_t owner = SHIFTADD ? seed_shiftadd(g) : seed_mul(g);
    Code local = g - masked_base(int(owner));
    const std::uint32_t correction = std::uint32_t(local >= MAIN_CHUNK);
    owner += correction;
    local -= (Code(0) - Code(correction)) & MAIN_CHUNK;
    return {int(owner), local};
}

__device__ __forceinline__ ShardAddress8 candidate(Code g) {
#if B300_HI32_SEED_MODE == 0
    return shard_address8(g, MAIN_CHUNK);
#elif B300_HI32_SEED_MODE == 1
    return hi32_seed_address<false>(g);
#else
    return hi32_seed_address<true>(g);
#endif
}

__global__ void exact_kernel(const Code* global_index, int* owner, Code* local, int n) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    const ShardAddress8 a = candidate(global_index[tid]);
    owner[tid] = a.owner;
    local[tid] = a.local;
}

__global__ void perf_kernel(Code* out, int n, int iters, Code stride, Code step) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    Code g = Code(tid) * stride;
    if (g >= MAIN_TOTAL) g = MAIN_TOTAL - 1;
    Code acc = 0;
    for (int i = 0; i < iters; ++i) {
        const ShardAddress8 a = candidate(g);
        acc += a.local ^ (Code(a.owner) << 56);
        g += step;
        if (g >= MAIN_TOTAL) g -= MAIN_TOTAL;
    }
    out[tid] = acc;
}

float run_once(Code* out, int n, int blocks, int threads, int iters,
               Code stride, Code step) {
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "event a");
    ck(cudaEventCreate(&b), "event b");
    ck(cudaEventRecord(a), "record a");
    perf_kernel<<<blocks, threads>>>(out, n, iters, stride, step);
    ck(cudaGetLastError(), "perf launch");
    ck(cudaEventRecord(b), "record b");
    ck(cudaEventSynchronize(b), "sync b");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}

float median(std::vector<float> x) {
    std::sort(x.begin(), x.end());
    const size_t n = x.size();
    return n & 1 ? x[n / 2] : 0.5f * (x[n / 2 - 1] + x[n / 2]);
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 256;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 8192;
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 9;
    if (blocks < 1 || threads < 1 || threads > 1024 || iters < 1 || repeats < 1)
        return 2;

    const int n = blocks * threads;
    const Code stride = std::max<Code>(1, MAIN_TOTAL / Code(n));
    const Code step = MAIN_CHUNK / 7 + 1;
    auto h_index = std::vector<Code>(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h_index[static_cast<size_t>(i)] = Code(i) * stride;

    Code *d_index = nullptr, *d_local = nullptr, *d_out = nullptr;
    int* d_owner = nullptr;
    ck(cudaMalloc(&d_index, static_cast<size_t>(n) * sizeof(Code)), "alloc index");
    ck(cudaMalloc(&d_owner, static_cast<size_t>(n) * sizeof(int)), "alloc owner");
    ck(cudaMalloc(&d_local, static_cast<size_t>(n) * sizeof(Code)), "alloc local");
    ck(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(Code)), "alloc out");
    ck(cudaMemcpy(d_index, h_index.data(), static_cast<size_t>(n) * sizeof(Code),
                  cudaMemcpyHostToDevice), "copy index");

    exact_kernel<<<blocks, threads>>>(d_index, d_owner, d_local, n);
    ck(cudaGetLastError(), "exact launch");
    ck(cudaDeviceSynchronize(), "exact sync");
    auto h_owner = std::vector<int>(static_cast<size_t>(n));
    auto h_local = std::vector<Code>(static_cast<size_t>(n));
    ck(cudaMemcpy(h_owner.data(), d_owner, static_cast<size_t>(n) * sizeof(int),
                  cudaMemcpyDeviceToHost), "copy owner");
    ck(cudaMemcpy(h_local.data(), d_local, static_cast<size_t>(n) * sizeof(Code),
                  cudaMemcpyDeviceToHost), "copy local");
    for (int i = 0; i < n; ++i) {
        const Code g = h_index[static_cast<size_t>(i)];
        const int owner = int(g / MAIN_CHUNK);
        const Code local = g - Code(owner) * MAIN_CHUNK;
        if (h_owner[static_cast<size_t>(i)] != owner || h_local[static_cast<size_t>(i)] != local) {
            std::fprintf(stderr,
                "mismatch i=%d g=%llu got=(%d,%llu) exact=(%d,%llu)\n",
                i, (unsigned long long)g, h_owner[static_cast<size_t>(i)],
                (unsigned long long)h_local[static_cast<size_t>(i)], owner,
                (unsigned long long)local);
            return 3;
        }
    }

    run_once(d_out, n, blocks, threads, iters, stride, step);
    std::vector<float> times;
    times.reserve(static_cast<size_t>(repeats));
    for (int r = 0; r < repeats; ++r)
        times.push_back(run_once(d_out, n, blocks, threads, iters, stride, step));
    auto h_out = std::vector<Code>(static_cast<size_t>(n));
    ck(cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(n) * sizeof(Code),
                  cudaMemcpyDeviceToHost), "copy out");
    std::uint64_t checksum = 0;
    for (Code x : h_out)
        checksum ^= x + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2);

    const double ops = double(n) * double(iters);
    const double ms = median(times);
    std::printf(
        "gridfp-b300-shard-owner-hi32-seed-w28-ngpu8-microprobe OK mode=%d "
        "blocks=%d threads=%d iters=%d repeats=%d addresses=%.0f median_ms=%.6f "
        "Gaddr_s=%.6f checksum=%llu exact=OK\n",
        int(B300_HI32_SEED_MODE), blocks, threads, iters, repeats, ops, ms,
        ops / ms / 1.0e6, (unsigned long long)checksum);

    cudaFree(d_out); cudaFree(d_local); cudaFree(d_owner); cudaFree(d_index);
    return 0;
}
