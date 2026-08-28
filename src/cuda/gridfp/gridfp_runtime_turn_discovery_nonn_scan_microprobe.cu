#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/gridfp_transition.hpp"

#ifndef RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
#define RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN 0
#endif
static_assert(RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN == 0 ||
              RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN == 1,
              "RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN must be 0 or 1");

namespace gp = oneesan::gridfp;

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

__device__ __forceinline__ std::uint32_t scan_turn_nn(
    gp::MateID mate,
    int width,
    int& final_balance
) {
    int bal = 0;
    std::uint32_t candidates = 0;
#if RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
    std::uint32_t mask = gp::mate_non_n_mask(mate, width) & ~std::uint32_t(3u);
    while (mask) {
        const int q = gp::mate_lsb_index32(mask);
        const gp::MateValue v = gp::mget(mate, q);
        if (bal == 0 && v == gp::R) candidates |= std::uint32_t(1) << q;
        if (v == gp::R) ++bal;
        else if (v == gp::L) --bal;
        if (bal < 0) break;
        mask &= mask - 1u;
    }
#else
    for (int q = 2; q < width; ++q) {
        const gp::MateValue v = gp::mget(mate, q);
        if (bal == 0 && v == gp::R) candidates |= std::uint32_t(1) << q;
        if (v == gp::R) ++bal;
        else if (v == gp::L) --bal;
        if (bal < 0) break;
    }
#endif
    final_balance = bal;
    return candidates;
}

__device__ __forceinline__ gp::MateID make_mate(
    std::uint32_t seed,
    int nonn_percent
) {
    gp::MateID mate = 0;
    mate = gp::msetpair(mate, 1, gp::NN);
    for (int q = 2; q < 28; ++q) {
        seed = seed * 1664525u + 1013904223u;
        const bool occupied = (seed % 100u) < std::uint32_t(nonn_percent);
        if (!occupied) continue;
        seed = seed * 1664525u + 1013904223u;
        mate = gp::mset(mate, q, (seed & 1u) ? gp::R : gp::L);
    }
    return mate;
}

__global__ void probe_kernel(
    std::uint64_t* output,
    int iterations,
    int nonn_percent
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;

    for (int i = 0; i < iterations; ++i) {
        seed = seed * 747796405u + 2891336453u;
        const gp::MateID mate = make_mate(seed, nonn_percent);
        int balance = 0;
        const std::uint32_t candidates = scan_turn_nn(mate, 28, balance);
        const std::uint64_t value =
            (std::uint64_t(candidates) << 32) ^ std::uint32_t(balance);
        acc ^= value + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const int nonn_percent = argc > 4 ? std::atoi(argv[4]) : 50;
    const int warmup = argc > 5 ? std::atoi(argv[5]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 || iterations <= 0 ||
        nonn_percent < 0 || nonn_percent > 100 || warmup < 0) {
        std::cerr << "usage: probe [blocks>0] [threads=1..1024] [iterations>0] "
                     "[nonn_percent=0..100] [warmup>=0]\n";
        return 2;
    }

    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    std::uint64_t* d_output = nullptr;
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc");

    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, threads>>>(d_output, iterations, nonn_percent);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    probe_kernel<<<blocks, threads>>>(d_output, iterations, nonn_percent);
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
    std::cout << "gridfp-runtime-turn-discovery-nonn-scan-microprobe"
              << " nonn_scan=" << RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
              << " W=28 nonn_percent=" << nonn_percent
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
