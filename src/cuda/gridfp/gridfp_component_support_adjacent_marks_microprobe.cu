#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_component_codec_device.cuh"

namespace rp = oneesan::gridfp::reducedprod;

namespace {
void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

std::uint64_t choose_host(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    std::uint64_t z = 1;
    for (int i = 1; i <= k; ++i)
        z = z * std::uint64_t(n - k + i) / std::uint64_t(i);
    return z;
}

__global__ void adjacent_mark_probe_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t input_mask,
    std::uint64_t* __restrict__ output,
    int iterations
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t ix = (tid * 747796405u + 2891336453u) & input_mask;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        const std::uint32_t packed = input[ix];
        const int a = int((packed >> 20) & 15u);
        const int ones = int((packed >> 16) & 15u);
        const rp::Rank64 rank = packed & 0xffffu;
        const std::uint32_t support = rp::component_support_unrank_device(
            14, ones, a + 1, a, rank);
        const std::uint64_t v =
            std::uint64_t(support) | (std::uint64_t(std::uint32_t(ones)) << 32) |
            (std::uint64_t(std::uint32_t(a)) << 40);
        acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
        ix = (ix + 4051u) & input_mask;
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

    constexpr std::uint32_t INPUT_COUNT = 1u << 16;
    std::vector<std::uint32_t> input(INPUT_COUNT);
    std::uint64_t x = 0xd1b54a32d192ed03ULL;
    for (std::uint32_t i = 0; i < INPUT_COUNT; ++i) {
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        const std::uint64_t z0 = x * 2685821657736338717ULL;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        const std::uint64_t z1 = x * 2685821657736338717ULL;
        const int a = int(z0 % 13ULL);
        const int ones = 1 + int((z0 >> 8) % 14ULL);
        const std::uint64_t count = choose_host(14, ones) - choose_host(12, ones);
        const std::uint32_t rank = std::uint32_t(z1 % count);
        input[i] = (std::uint32_t(a) << 20) |
                   (std::uint32_t(ones) << 16) | rank;
    }

    std::uint32_t* d_input = nullptr;
    std::uint64_t* d_output = nullptr;
    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    cuda_check(cudaMalloc(&d_input, input.size() * sizeof(std::uint32_t)), "cudaMalloc(input)");
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc(output)");
    cuda_check(cudaMemcpy(d_input, input.data(), input.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "cudaMemcpy(input)");

    for (int i = 0; i < warmup; ++i) {
        adjacent_mark_probe_kernel<<<blocks, threads>>>(
            d_input, INPUT_COUNT - 1, d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    adjacent_mark_probe_kernel<<<blocks, threads>>>(
        d_input, INPUT_COUNT - 1, d_output, iterations);
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
    std::cout << "gridfp-component-support-adjacent-marks-microprobe"
              << " adjacent=" << RP_FAST_COMPONENT_SUPPORT_ADJACENT_MARKS
              << " early_exit=" << RP_FAST_SUPPORT_UNRANK_EARLY_EXIT
              << " len=14 mark_pairs=13"
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
    cudaFree(d_input);
    return 0;
}
