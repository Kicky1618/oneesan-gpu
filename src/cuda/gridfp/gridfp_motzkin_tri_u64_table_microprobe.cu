#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_device.cuh"

#ifndef RP_MOTZKIN_TABLE_MODE
#define RP_MOTZKIN_TABLE_MODE 0
#endif
static_assert(RP_MOTZKIN_TABLE_MODE == 0 || RP_MOTZKIN_TABLE_MODE == 1,
              "RP_MOTZKIN_TABLE_MODE must be 0 or 1");

namespace rp = oneesan::gridfp::reducedprod;
namespace {
using MotzkinTable =
    std::array<std::array<rp::Rank64, rp::RP_MAX_W + 2>, rp::RP_MAX_W + 1>;

__device__ __constant__ rp::Rank64 RP_MOTZKIN_TRI_U64_PROBE[435];
static_assert(sizeof(RP_MOTZKIN_TRI_U64_PROBE) == 3480);

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

MotzkinTable make_motzkin() {
    MotzkinTable m{};
    m[0][0] = 1;
    for (int rem = 1; rem <= rp::RP_MAX_W; ++rem) {
        for (int h = 0; h <= rp::RP_MAX_W; ++h) {
            rp::Rank64 z = m[rem - 1][h];
            if (h > 0) z += m[rem - 1][h - 1];
            z += m[rem - 1][h + 1];
            m[rem][h] = z;
        }
    }
    return m;
}

__device__ __forceinline__ rp::Rank64 motzkin_count(int rem, int h) {
#if RP_MOTZKIN_TABLE_MODE == 0
    return rp::RP_MOTZKIN[rem][h];
#else
    if (rem < 0 || rem > rp::RP_MAX_W || h < 0 || h > rem) return 0;
    return RP_MOTZKIN_TRI_U64_PROBE[rem * (rem + 1) / 2 + h];
#endif
}

__global__ void probe_kernel(std::uint64_t* output, int iterations, int pattern) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const int rem = pattern == 0 ? int((seed >> 16) % 29u)
                                     : int((tid + std::uint32_t(i)) % 29u);
        const int h = pattern == 0 ? int((seed >> 8) % std::uint32_t(rem + 1))
                                   : (rem >> 2);
        const rp::Rank64 v = motzkin_count(rem, h);
        acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    }
    output[tid] = acc;
}
} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 512;
    const int pattern = argc > 4 ? std::atoi(argv[4]) : 0;
    const int warmup = argc > 5 ? std::atoi(argv[5]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 || iterations <= 0 ||
        pattern < 0 || pattern > 1 || warmup < 0) return 2;

    const MotzkinTable m = make_motzkin();
    std::array<rp::Rank64, 435> tri{};
    std::size_t ix = 0;
    for (int rem = 0; rem <= rp::RP_MAX_W; ++rem)
        for (int h = 0; h <= rem; ++h) tri[ix++] = m[rem][h];
    if (ix != tri.size()) return 3;
    cuda_check(cudaMemcpyToSymbol(rp::RP_MOTZKIN, m.data(), sizeof(m)),
               "cudaMemcpyToSymbol(RP_MOTZKIN)");
    cuda_check(cudaMemcpyToSymbol(RP_MOTZKIN_TRI_U64_PROBE, tri.data(), sizeof(tri)),
               "cudaMemcpyToSymbol(RP_MOTZKIN_TRI_U64_PROBE)");

    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    std::uint64_t* d_output = nullptr;
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc");
    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, threads>>>(d_output, iterations, pattern);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");
    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    probe_kernel<<<blocks, threads>>>(d_output, iterations, pattern);
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    std::vector<std::uint64_t> output(nthreads);
    cuda_check(cudaMemcpy(output.data(), d_output, output.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (auto v : output) { checksum ^= v; checksum *= 0x100000001b3ULL; }
    const std::uint64_t calls = std::uint64_t(blocks) * std::uint64_t(threads) * std::uint64_t(iterations);
    std::cout << "gridfp-motzkin-tri-u64-table-microprobe"
              << " mode=" << RP_MOTZKIN_TABLE_MODE
              << " pattern=" << pattern
              << " blocks=" << blocks << " threads=" << threads
              << " iterations=" << iterations << " calls=" << calls
              << " kernel_ms=" << elapsed_ms
              << " ns_per_call=" << (double(elapsed_ms) * 1.0e6 / double(calls))
              << " checksum=" << checksum << '\n';
    cudaEventDestroy(stop); cudaEventDestroy(start); cudaFree(d_output);
    return 0;
}
