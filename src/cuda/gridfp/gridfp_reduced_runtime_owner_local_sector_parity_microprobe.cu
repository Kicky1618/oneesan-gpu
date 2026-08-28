#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_owner_component_plan_device.cuh"

namespace rp = oneesan::gridfp::reducedprod;

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

__global__ void owner_local_sector_probe_kernel(
    std::uint64_t* output,
    int iterations
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    int outer_ones = int(tid % 14u);
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;

    for (int i = 0; i < iterations; ++i) {
        const int row = 890 + outer_ones * 15;
        const std::uint32_t group = rp::RP_RUNTIME_OWNER_LOCAL_SECTOR_END[row + 14];
        seed = seed * 1664525u + 1013904223u;
        const rp::Rank64 within = (std::uint64_t(seed) * group) >> 32;

        int local_ones = -1;
        rp::Rank64 local_within = 0;
        const bool ok = rp::runtime_owner_local_sector_device(
            28, 15, outer_ones, within, local_ones, local_within);
        if (!ok) {
            acc ^= 0xd1b54a32d192ed03ULL;
        } else {
            const std::uint64_t v =
                (std::uint64_t(std::uint32_t(local_ones)) << 32) ^ local_within;
            acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
        }

        ++outer_ones;
        if (outer_ones == 14) outer_ones = 0;
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 || iterations <= 0 || warmup < 0) {
        std::cerr << "usage: probe [blocks>0] [threads=1..1024] [iterations>0] [warmup>=0]\n";
        return 2;
    }

    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    std::uint64_t* d_output = nullptr;
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc");

    for (int i = 0; i < warmup; ++i) {
        owner_local_sector_probe_kernel<<<blocks, threads>>>(d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    owner_local_sector_probe_kernel<<<blocks, threads>>>(d_output, iterations);
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
    std::cout << "gridfp-runtime-owner-local-sector-parity-microprobe"
              << " parity=" << RP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY
              << " w28_tree=" << RP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE
              << " W=28 L=15 outer_classes=14"
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
