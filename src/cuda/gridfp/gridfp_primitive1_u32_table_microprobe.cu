#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_device.cuh"

#ifndef RP_PRIMITIVE1_TABLE_MODE
#define RP_PRIMITIVE1_TABLE_MODE 0
#endif
static_assert(RP_PRIMITIVE1_TABLE_MODE >= 0 && RP_PRIMITIVE1_TABLE_MODE <= 2,
              "RP_PRIMITIVE1_TABLE_MODE must be 0, 1, or 2");

namespace rp = oneesan::gridfp::reducedprod;

namespace {

__device__ __constant__ std::uint32_t RP_PRIMITIVE1_U32_PROBE[14] = {
    1u, 2u, 5u, 14u, 42u, 132u, 429u, 1430u,
    4862u, 16796u, 58786u, 208012u, 742900u, 2674440u};
static_assert(sizeof(RP_PRIMITIVE1_U32_PROBE) == 56);

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

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

__device__ __forceinline__ rp::Rank64 primitive1_count(int occupied) {
#if RP_PRIMITIVE1_TABLE_MODE == 0
    return rp::RP_PRIMITIVE[occupied][1];
#elif RP_PRIMITIVE1_TABLE_MODE == 1
    return rp::RP_SECTOR_PRIMITIVE[occupied >> 1];
#else
    return RP_PRIMITIVE1_U32_PROBE[occupied >> 1];
#endif
}

__global__ void probe_kernel(std::uint64_t* output, int iterations) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sector = int(tid % 14u);
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        const int occupied = 1 + (sector << 1);
        const rp::Rank64 count = primitive1_count(occupied);
        acc ^= count + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
        ++sector;
        if (sector == 14) sector = 0;
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 512;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 ||
        iterations <= 0 || warmup < 0) {
        std::cerr << "usage: probe [blocks>0] [threads=1..1024] "
                     "[iterations>0] [warmup>=0]\n";
        return 2;
    }

    const PrimitiveTable primitive = make_primitive_table();
    std::array<rp::Rank64, rp::RP_MAX_SECTORS> sector_primitive{};
    for (int sector = 0; sector < 14; ++sector)
        sector_primitive[std::size_t(sector)] = primitive[1 + 2 * sector][1];
    cuda_check(cudaMemcpyToSymbol(
        rp::RP_PRIMITIVE, primitive.data(), sizeof(primitive)),
        "cudaMemcpyToSymbol(RP_PRIMITIVE)");
    cuda_check(cudaMemcpyToSymbol(
        rp::RP_SECTOR_PRIMITIVE, sector_primitive.data(), sizeof(sector_primitive)),
        "cudaMemcpyToSymbol(RP_SECTOR_PRIMITIVE)");

    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    std::uint64_t* d_output = nullptr;
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc");

    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, threads>>>(d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    probe_kernel<<<blocks, threads>>>(d_output, iterations);
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    std::vector<std::uint64_t> output(nthreads);
    cuda_check(cudaMemcpy(
        output.data(), d_output, output.size() * sizeof(std::uint64_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (std::uint64_t v : output) {
        checksum ^= v;
        checksum *= 0x100000001b3ULL;
    }

    const std::uint64_t calls =
        std::uint64_t(blocks) * std::uint64_t(threads) *
        std::uint64_t(iterations);
    const double ns_per_call = double(elapsed_ms) * 1.0e6 / double(calls);
    std::cout << "gridfp-primitive1-u32-table-microprobe"
              << " mode=" << RP_PRIMITIVE1_TABLE_MODE
              << " sectors=14 occupied_min=1 occupied_max=27"
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
