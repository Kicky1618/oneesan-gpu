#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_device.cuh"

#ifndef RP_PRIMITIVE_SYM_U32_MODE
#define RP_PRIMITIVE_SYM_U32_MODE 0
#endif
static_assert(RP_PRIMITIVE_SYM_U32_MODE == 0 || RP_PRIMITIVE_SYM_U32_MODE == 1,
              "RP_PRIMITIVE_SYM_U32_MODE must be 0 or 1");

namespace rp = oneesan::gridfp::reducedprod;

namespace {

__device__ __constant__ std::uint32_t RP_PRIMITIVE_SYM_U32_PROBE[225] = {
#include "gridfp_reduced_production_primitive_sym_u32_values.inc"
};
static_assert(sizeof(RP_PRIMITIVE_SYM_U32_PROBE) == 900);

using PrimitiveTable =
    std::array<std::array<rp::Rank64, rp::RP_MAX_W + 2>, rp::RP_MAX_W + 1>;

PrimitiveTable make_primitive_table() {
    PrimitiveTable p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= rp::RP_MAX_W; ++rem) {
        for (int h = 0; h <= rp::RP_MAX_W; ++h) {
            rp::Rank64 z = p[rem - 1][h + 1];
            if (h > 0) z += p[rem - 1][h - 1];
            p[rem][h] = z;
        }
    }
    return p;
}

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

__device__ __forceinline__ int row_base(int rem) {
    const int m = rem >> 1;
    return (rem & 1) ? (m + 1) * (m + 1) : m * (m + 1);
}

__device__ __forceinline__ rp::Rank64 primitive_count(int rem, int h) {
#if RP_PRIMITIVE_SYM_U32_MODE == 0
    return rp::RP_PRIMITIVE[rem][h];
#else
    if (rem < 0 || rem > rp::RP_MAX_W || h < 0 || h > rem || ((rem ^ h) & 1))
        return 0;
    return RP_PRIMITIVE_SYM_U32_PROBE[row_base(rem) + (h >> 1)];
#endif
}

__global__ void probe_kernel(std::uint64_t* output, int iterations, int pattern) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        seed = seed * 1664525u + 1013904223u;
        int rem = 0;
        int h = 0;
        if (pattern == 0) {
            rem = int((seed >> 16) % 29u);
            h = int((seed >> 8) % 30u);
        } else if (pattern == 1) {
            const int sector = int((tid + std::uint32_t(i)) % 14u);
            rem = 1 + (sector << 1);
            h = 1;
        } else {
            rem = 1 + int((seed >> 16) % 26u);
            const int count = rem / 2 + 1;
            h = (rem & 1) + 2 * int((seed >> 8) % std::uint32_t(count));
        }
        const rp::Rank64 v = primitive_count(rem, h);
        acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 512;
    const int pattern = argc > 4 ? std::atoi(argv[4]) : 2;
    const int warmup = argc > 5 ? std::atoi(argv[5]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 || iterations <= 0 ||
        pattern < 0 || pattern > 2 || warmup < 0) {
        std::cerr << "usage: probe [blocks] [threads] [iterations] [pattern=0..2] [warmup]\n";
        return 2;
    }

    const PrimitiveTable primitive = make_primitive_table();
    cuda_check(cudaMemcpyToSymbol(
        rp::RP_PRIMITIVE, primitive.data(), sizeof(primitive)),
        "cudaMemcpyToSymbol(RP_PRIMITIVE)");

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
    for (std::uint64_t v : output) {
        checksum ^= v;
        checksum *= 0x100000001b3ULL;
    }

    const std::uint64_t calls = std::uint64_t(blocks) * std::uint64_t(threads) *
                                std::uint64_t(iterations);
    const double ns_per_call = double(elapsed_ms) * 1.0e6 / double(calls);
    std::cout << "gridfp-primitive-sym-u32-table-microprobe"
              << " mode=" << RP_PRIMITIVE_SYM_U32_MODE
              << " pattern=" << pattern
              << " blocks=" << blocks
              << " threads=" << threads
              << " iterations=" << iterations
              << " calls=" << calls
              << " kernel_ms=" << elapsed_ms
              << " ns_per_call=" << ns_per_call
              << " checksum=" << checksum << '\n';

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_output);
    return 0;
}
