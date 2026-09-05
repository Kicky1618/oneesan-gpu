#include <cuda_runtime.h>

#include <array>
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

__global__ void owner_prefix_probe_kernel(
    const rp::Rank64* __restrict__ prefix,
    const rp::Rank64* __restrict__ ranks,
    std::uint32_t rank_mask,
    std::uint64_t* __restrict__ output,
    int iterations
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t index = (tid * 747796405u + 2891336453u) & rank_mask;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        const rp::Rank64 rank = ranks[index];
        rp::Rank64 begin = 0;
        const int sector = rp::runtime_owner_prefix_sector_begin_device(
            prefix, 13, rank, begin);
        const std::uint64_t v =
            (std::uint64_t(std::uint32_t(sector)) << 56) ^ begin ^ rank;
        acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
        index = (index + 4051u) & rank_mask;
    }
    output[tid] = acc;
}
} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 || iterations <= 0 || warmup < 0)
        return 2;

    // W=28 has O=13 and therefore fourteen outer-popcount sectors. A binomial
    // shape gives a realistic middle-heavy access distribution while keeping
    // the prefix itself tiny, as in production. Multiplication only enlarges
    // the rank range; it does not change the search tree.
    constexpr std::array<std::uint32_t, 14> weights{
        1, 13, 78, 286, 715, 1287, 1716,
        1716, 1287, 715, 286, 78, 13, 1};
    std::array<rp::Rank64, 15> prefix{};
    for (int r = 0; r < 14; ++r)
        prefix[std::size_t(r + 1)] =
            prefix[std::size_t(r)] + rp::Rank64(weights[std::size_t(r)]) * 1000003ULL;

    constexpr std::uint32_t RANK_COUNT = 1u << 16;
    std::vector<rp::Rank64> ranks(RANK_COUNT);
    std::uint64_t x = 0xd1b54a32d192ed03ULL;
    for (std::uint32_t i = 0; i < RANK_COUNT; ++i) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        const std::uint64_t z = x * 2685821657736338717ULL;
        ranks[i] = z % prefix.back();
    }

    rp::Rank64* d_prefix = nullptr;
    rp::Rank64* d_ranks = nullptr;
    std::uint64_t* d_output = nullptr;
    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    cuda_check(cudaMalloc(&d_prefix, prefix.size() * sizeof(rp::Rank64)), "cudaMalloc(prefix)");
    cuda_check(cudaMalloc(&d_ranks, ranks.size() * sizeof(rp::Rank64)), "cudaMalloc(ranks)");
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc(output)");
    cuda_check(cudaMemcpy(d_prefix, prefix.data(), prefix.size() * sizeof(rp::Rank64),
                          cudaMemcpyHostToDevice), "cudaMemcpy(prefix)");
    cuda_check(cudaMemcpy(d_ranks, ranks.data(), ranks.size() * sizeof(rp::Rank64),
                          cudaMemcpyHostToDevice), "cudaMemcpy(ranks)");

    for (int i = 0; i < warmup; ++i) {
        owner_prefix_probe_kernel<<<blocks, threads>>>(
            d_prefix, d_ranks, RANK_COUNT - 1, d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    owner_prefix_probe_kernel<<<blocks, threads>>>(
        d_prefix, d_ranks, RANK_COUNT - 1, d_output, iterations);
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

    const std::uint64_t calls =
        std::uint64_t(blocks) * std::uint64_t(threads) * std::uint64_t(iterations);
    const double ns_per_call = double(elapsed_ms) * 1.0e6 / double(calls);
    std::cout << "gridfp-runtime-owner-prefix-carry-begin-microprobe"
              << " carry=" << RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN
              << " W=28 O=13 sectors=14"
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
    cudaFree(d_ranks);
    cudaFree(d_prefix);
    return 0;
}
