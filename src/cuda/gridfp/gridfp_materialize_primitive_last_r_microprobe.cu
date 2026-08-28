#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_codec_device.cuh"

namespace rp = oneesan::gridfp::reducedprod;

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

using PrimitiveTable = std::array<std::array<rp::Rank64, rp::RP_MAX_W + 2>, rp::RP_MAX_W + 1>;

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

__global__ void probe_kernel(std::uint64_t* output, int iterations) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    int sector = int(tid % 14u);

    for (int i = 0; i < iterations; ++i) {
        const int occupied = 1 + (sector << 1);
        const rp::Rank64 count = rp::RP_PRIMITIVE[occupied][1];
        seed = seed * 1664525u + 1013904223u;
        const rp::Rank64 rank =
            (std::uint64_t(seed) * std::uint64_t(count)) >> 32;
        const std::uint32_t support =
            (std::uint32_t(1) << occupied) - 1u;
        const rp::MateID m = rp::materialize_primitive_device(
            support, 28, occupied, rank);
        acc ^= std::uint64_t(m) + 0x9e3779b97f4a7c15ULL +
               (acc << 6) + (acc >> 2);
        ++sector;
        if (sector == 14) sector = 0;
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 ||
        iterations <= 0 || warmup < 0) {
        std::cerr << "usage: probe [blocks>0] [threads=1..1024] [iterations>0] [warmup>=0]\n";
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
        std::uint64_t(blocks) * std::uint64_t(threads) * std::uint64_t(iterations);
    const double ns_per_call = double(elapsed_ms) * 1.0e6 / double(calls);
    std::cout << "gridfp-materialize-primitive-last-r-microprobe"
              << " setbits=" << RP_FAST_MATERIALIZE_PRIMITIVE_SETBITS
              << " last_r=" << RP_FAST_MATERIALIZE_PRIMITIVE_LAST_R
              << " occupied_min=1 occupied_max=27"
              << " blocks=" << blocks << " threads=" << threads
              << " iterations=" << iterations << " calls=" << calls
              << " kernel_ms=" << elapsed_ms
              << " ns_per_call=" << ns_per_call
              << " checksum=" << checksum << '\n';

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_output);
    return 0;
}
