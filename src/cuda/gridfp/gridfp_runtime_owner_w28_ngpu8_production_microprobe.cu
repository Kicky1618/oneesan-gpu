#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef RP_RUNTIME_OWNER_U32LIMB
#define RP_RUNTIME_OWNER_U32LIMB 1
#endif
#ifndef RP_RUNTIME_OWNER_W28_NGPU8_DIRECT
#define RP_RUNTIME_OWNER_W28_NGPU8_DIRECT 0
#endif

#include "gridfp_reduced_production_group_context_device.cuh"

namespace {
using namespace oneesan::gridfp::reducedprod;
constexpr Rank64 TOTAL = 473397057701ULL;
constexpr int W = 28;
constexpr int K = 13;
constexpr int NGPU = 8;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

__global__ void owner_exact_kernel(
    const Rank64* midpoint, std::uint32_t* owner, int n,
    int W_arg, int K_arg, int ngpu_arg
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    owner[tid] = static_cast<std::uint32_t>(runtime_owner_from_group_base_device(
        midpoint[tid], 0, W_arg, K_arg, ngpu_arg, nullptr));
}

__global__ void owner_perf_kernel(
    std::uint32_t* out, int n, int iters, Rank64 stride,
    int W_arg, int K_arg, int ngpu_arg
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    Rank64 midpoint = static_cast<Rank64>(tid) * stride;
    if (midpoint >= TOTAL) midpoint = TOTAL - 1;
    std::uint32_t acc = 0;
    for (int i = 0; i < iters; ++i) {
        midpoint += 17;
        if (midpoint >= TOTAL) midpoint -= TOTAL;
        acc += static_cast<std::uint32_t>(runtime_owner_from_group_base_device(
            midpoint, 0, W_arg, K_arg, ngpu_arg, nullptr));
    }
    out[tid] = acc;
}

float run_perf(
    std::uint32_t* d_out, int n, int blocks, int threads, int iters,
    Rank64 stride, int W_arg, int K_arg, int ngpu_arg
) {
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "cudaEventCreate(a)");
    ck(cudaEventCreate(&b), "cudaEventCreate(b)");
    ck(cudaEventRecord(a), "cudaEventRecord(a)");
    owner_perf_kernel<<<blocks, threads>>>(
        d_out, n, iters, stride, W_arg, K_arg, ngpu_arg);
    ck(cudaGetLastError(), "owner_perf_kernel");
    ck(cudaEventRecord(b), "cudaEventRecord(b)");
    ck(cudaEventSynchronize(b), "cudaEventSynchronize(b)");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    ck(cudaEventDestroy(a), "cudaEventDestroy(a)");
    ck(cudaEventDestroy(b), "cudaEventDestroy(b)");
    return ms;
}

float median(std::vector<float> x) {
    std::sort(x.begin(), x.end());
    const std::size_t n = x.size();
    return n & 1 ? x[n / 2] : 0.5f * (x[n / 2 - 1] + x[n / 2]);
}
}  // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 256;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 4096;
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 9;
    if (blocks < 1 || threads < 1 || threads > 1024 || iters < 1 || repeats < 1)
        return 2;

    const int n = blocks * threads;
    const Rank64 stride = std::max<Rank64>(Rank64(1), TOTAL / Rank64(n));
    std::vector<Rank64> h_midpoint(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        Rank64 x = static_cast<Rank64>(i) * stride;
        h_midpoint[static_cast<std::size_t>(i)] = x < TOTAL ? x : TOTAL - 1;
    }

    Rank64* d_midpoint = nullptr;
    std::uint32_t* d_out = nullptr;
    ck(cudaMalloc(&d_midpoint, std::size_t(n) * sizeof(*d_midpoint)), "cudaMalloc(midpoint)");
    ck(cudaMalloc(&d_out, std::size_t(n) * sizeof(*d_out)), "cudaMalloc(out)");
    ck(cudaMemcpy(d_midpoint, h_midpoint.data(), std::size_t(n) * sizeof(*d_midpoint),
                  cudaMemcpyHostToDevice), "cudaMemcpy(midpoint)");

    owner_exact_kernel<<<blocks, threads>>>(d_midpoint, d_out, n, W, K, NGPU);
    ck(cudaGetLastError(), "owner_exact_kernel");
    ck(cudaDeviceSynchronize(), "owner_exact_kernel sync");
    std::vector<std::uint32_t> h_owner(static_cast<std::size_t>(n));
    ck(cudaMemcpy(h_owner.data(), d_out, std::size_t(n) * sizeof(*d_out),
                  cudaMemcpyDeviceToHost), "cudaMemcpy(owner)");
    for (int i = 0; i < n; ++i) {
        const Rank64 exact = (h_midpoint[static_cast<std::size_t>(i)] * NGPU) / TOTAL;
        if (h_owner[static_cast<std::size_t>(i)] != exact) {
            std::fprintf(stderr, "owner mismatch i=%d got=%u expected=%llu\n",
                         i, h_owner[static_cast<std::size_t>(i)],
                         static_cast<unsigned long long>(exact));
            return 3;
        }
    }

    run_perf(d_out, n, blocks, threads, iters, stride, W, K, NGPU);
    std::vector<float> ms;
    ms.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r)
        ms.push_back(run_perf(d_out, n, blocks, threads, iters, stride, W, K, NGPU));

    std::vector<std::uint32_t> h_acc(static_cast<std::size_t>(n));
    ck(cudaMemcpy(h_acc.data(), d_out, std::size_t(n) * sizeof(*d_out),
                  cudaMemcpyDeviceToHost), "cudaMemcpy(acc)");
    std::uint64_t checksum = 0;
    for (std::uint32_t x : h_acc) checksum += x;

    std::printf(
        "gridfp-runtime-owner-w28-ngpu8-production-microprobe OK direct=%d W=28 K=13 ngpu=8 blocks=%d threads=%d iters=%d repeats=%d median_ms=%.6f checksum=%llu exact=OK runtime_W=1 runtime_ngpu=1 u32limb=1\n",
        int(RP_RUNTIME_OWNER_W28_NGPU8_DIRECT), blocks, threads, iters, repeats,
        median(ms), static_cast<unsigned long long>(checksum));

    cudaFree(d_out);
    cudaFree(d_midpoint);
    return 0;
}
