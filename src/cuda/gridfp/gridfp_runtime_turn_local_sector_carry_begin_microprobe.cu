#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef PROBE_TURN_LOCAL_CARRY_BEGIN
#define PROBE_TURN_LOCAL_CARRY_BEGIN 0
#endif
static_assert(PROBE_TURN_LOCAL_CARRY_BEGIN == 0 ||
              PROBE_TURN_LOCAL_CARRY_BEGIN == 1);

namespace {
using Rank64 = std::uint64_t;
__device__ __constant__ std::uint32_t TURN_END[550] = {
#include "gridfp_reduced_production_runtime_turn_local_sector_end_values.inc"
};

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

__device__ __forceinline__ void turn_sector(
    int outer, Rank64 within, int& local_ones, Rank64& local_within
) {
    const int row = 445 + ((outer + 1) >> 1) * 7 + (outer >> 1) * 8;
    const int first = (outer & 1) ? 0 : 1;
    const int count = (outer & 1) ? 8 : 7;
    int lo = 0;
    int hi = count;
#if PROBE_TURN_LOCAL_CARRY_BEGIN
    Rank64 begin = 0;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const Rank64 end = TURN_END[row + mid];
        if (within < end) {
            hi = mid;
        } else {
            lo = mid + 1;
            begin = end;
        }
    }
    local_ones = first + (lo << 1);
    local_within = within - begin;
#else
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (within < TURN_END[row + mid]) hi = mid;
        else lo = mid + 1;
    }
    const Rank64 begin = lo ? TURN_END[row + lo - 1] : 0;
    local_ones = first + (lo << 1);
    local_within = within - begin;
#endif
}

__global__ void probe_kernel(std::uint64_t* output, int iterations) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    int outer = int(tid % 14u);
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        const int row = 445 + ((outer + 1) >> 1) * 7 + (outer >> 1) * 8;
        const int count = (outer & 1) ? 8 : 7;
        const std::uint32_t group = TURN_END[row + count - 1];
        seed = seed * 1664525u + 1013904223u;
        const Rank64 within = (std::uint64_t(seed) * group) >> 32;
        int local_ones = -1;
        Rank64 local_within = 0;
        turn_sector(outer, within, local_ones, local_within);
        const std::uint64_t v =
            (std::uint64_t(std::uint32_t(local_ones)) << 32) ^ local_within;
        acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
        ++outer;
        if (outer == 14) outer = 0;
    }
    output[tid] = acc;
}
}

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 ||
        iterations <= 0 || warmup < 0) return 2;
    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    std::uint64_t* d_output = nullptr;
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc");
    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, threads>>>(d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");
    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "event start");
    cuda_check(cudaEventCreate(&stop), "event stop");
    cuda_check(cudaEventRecord(start), "record start");
    probe_kernel<<<blocks, threads>>>(d_output, iterations);
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "record stop");
    cuda_check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    std::vector<std::uint64_t> output(nthreads);
    cuda_check(cudaMemcpy(output.data(), d_output, output.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost), "copy output");
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (auto v : output) { checksum ^= v; checksum *= 0x100000001b3ULL; }
    const std::uint64_t calls = std::uint64_t(blocks) * threads * iterations;
    std::cout << "gridfp-runtime-turn-local-sector-carry-begin-microprobe"
              << " carry=" << PROBE_TURN_LOCAL_CARRY_BEGIN
              << " W=28 outer_classes=14 calls=" << calls
              << " kernel_ms=" << ms
              << " ns_per_call=" << (double(ms) * 1.0e6 / double(calls))
              << " checksum=" << checksum << '\n';
    cudaEventDestroy(stop); cudaEventDestroy(start); cudaFree(d_output);
    return 0;
}
