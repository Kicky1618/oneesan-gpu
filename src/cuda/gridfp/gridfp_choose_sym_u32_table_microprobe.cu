#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_device.cuh"

#ifndef RP_CHOOSE_TABLE_MODE
#define RP_CHOOSE_TABLE_MODE 0
#endif
static_assert(RP_CHOOSE_TABLE_MODE >= 0 && RP_CHOOSE_TABLE_MODE <= 3,
              "RP_CHOOSE_TABLE_MODE must be 0, 1, 2, or 3");

namespace rp = oneesan::gridfp::reducedprod;
namespace {
__device__ __constant__ std::uint32_t RP_CHOOSE_SYM_U32_PROBE[225] = {
#include "gridfp_reduced_production_choose_sym_u32_values.inc"
};
__device__ __constant__ std::uint32_t RP_CHOOSE_TRI_U32_PROBE[435] = {
#include "gridfp_reduced_production_choose_tri_u32_values.inc"
};
__device__ __constant__ std::uint32_t RP_CHOOSE_FULL_U32_PROBE[29 * 29];
static_assert(sizeof(RP_CHOOSE_SYM_U32_PROBE) == 900);
static_assert(sizeof(RP_CHOOSE_TRI_U32_PROBE) == 1740);
static_assert(sizeof(RP_CHOOSE_FULL_U32_PROBE) == 3364);

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) { std::cerr << what << ": " << cudaGetErrorString(err) << '\n'; std::exit(2); }
}
using ChooseTable = std::array<std::array<rp::Rank64, rp::RP_MAX_W + 1>, rp::RP_MAX_W + 1>;
ChooseTable make_choose_table() {
    ChooseTable c{};
    for (int n = 0; n <= rp::RP_MAX_W; ++n) {
        c[n][0] = c[n][n] = 1;
        for (int k = 1; k < n; ++k) c[n][k] = c[n - 1][k - 1] + c[n - 1][k];
    }
    return c;
}
__device__ __forceinline__ int choose_sym_row_base(int n) { const int m = n >> 1; return (n & 1) ? (m + 1) * (m + 1) : m * (m + 1); }
__device__ __forceinline__ int choose_tri_row_base(int n) { return n * (n + 1) / 2; }
__device__ __forceinline__ rp::Rank64 choose_probe(int n, int k) {
#if RP_CHOOSE_TABLE_MODE == 0
    return rp::RP_CHOOSE[n][k];
#elif RP_CHOOSE_TABLE_MODE == 1
    const int mirror = n - k; if (k > mirror) k = mirror;
    return RP_CHOOSE_SYM_U32_PROBE[choose_sym_row_base(n) + k];
#elif RP_CHOOSE_TABLE_MODE == 2
    return RP_CHOOSE_TRI_U32_PROBE[choose_tri_row_base(n) + k];
#else
    return RP_CHOOSE_FULL_U32_PROBE[n * 29 + k];
#endif
}
__global__ void probe_kernel(std::uint64_t* output, int iterations, int pattern) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = int(tid % 29u); std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        const int k = pattern == 0 ? (n >> 1) : (n - (n >> 2));
        const rp::Rank64 v = choose_probe(n, k);
        acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
        if (++n == 29) n = 0;
    }
    output[tid] = acc;
}
} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 512;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    const int pattern = argc > 5 ? std::atoi(argv[5]) : 0;
    if (blocks <= 0 || threads <= 0 || threads > 1024 || iterations <= 0 || warmup < 0 || pattern < 0 || pattern > 1) return 2;

    const ChooseTable choose = make_choose_table();
    std::array<std::uint32_t, 29 * 29> full{};
    for (int n = 0; n <= rp::RP_MAX_W; ++n)
        for (int k = 0; k <= rp::RP_MAX_W; ++k) {
            if (choose[n][k] > 0xffffffffULL) return 3;
            full[std::size_t(n * 29 + k)] = static_cast<std::uint32_t>(choose[n][k]);
        }
    cuda_check(cudaMemcpyToSymbol(rp::RP_CHOOSE, choose.data(), sizeof(choose)), "cudaMemcpyToSymbol(RP_CHOOSE)");
    cuda_check(cudaMemcpyToSymbol(RP_CHOOSE_FULL_U32_PROBE, full.data(), sizeof(full)), "cudaMemcpyToSymbol(RP_CHOOSE_FULL_U32_PROBE)");

    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads); std::uint64_t* d_output = nullptr;
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc");
    for (int i = 0; i < warmup; ++i) { probe_kernel<<<blocks, threads>>>(d_output, iterations, pattern); cuda_check(cudaGetLastError(), "warmup launch"); }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");
    cudaEvent_t start{}, stop{}; cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)"); cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)"); probe_kernel<<<blocks, threads>>>(d_output, iterations, pattern); cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)"); cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f; cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    std::vector<std::uint64_t> output(nthreads); cuda_check(cudaMemcpy(output.data(), d_output, output.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
    std::uint64_t checksum = 0xcbf29ce484222325ULL; for (auto v : output) { checksum ^= v; checksum *= 0x100000001b3ULL; }
    const std::uint64_t calls = std::uint64_t(blocks) * std::uint64_t(threads) * std::uint64_t(iterations);
    std::cout << "gridfp-choose-sym-u32-table-microprobe mode=" << RP_CHOOSE_TABLE_MODE << " pattern=" << pattern
              << " n_min=0 n_max=28 blocks=" << blocks << " threads=" << threads << " iterations=" << iterations
              << " calls=" << calls << " kernel_ms=" << elapsed_ms << " ns_per_call=" << (double(elapsed_ms) * 1.0e6 / double(calls))
              << " checksum=" << checksum << '\n';
    cudaEventDestroy(stop); cudaEventDestroy(start); cudaFree(d_output); return 0;
}
