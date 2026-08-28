#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

#ifndef B300_SHARD_ADDRESS_SELECT
#define B300_SHARD_ADDRESS_SELECT 0
#endif
static_assert(B300_SHARD_ADDRESS_SELECT == 0 || B300_SHARD_ADDRESS_SELECT == 1,
              "B300_SHARD_ADDRESS_SELECT must be 0 or 1");

namespace {

__device__ __forceinline__ ShardAddress8 shard_address8_select(Code g, Code chunk) {
    int owner = 0;
    const Code c4 = chunk << 2;
    const bool p4 = g >= c4;
    const Code s4 = g - c4;
    g = p4 ? s4 : g;
    owner |= int(p4) << 2;

    const Code c2 = chunk << 1;
    const bool p2 = g >= c2;
    const Code s2 = g - c2;
    g = p2 ? s2 : g;
    owner |= int(p2) << 1;

    const bool p1 = g >= chunk;
    const Code s1 = g - chunk;
    g = p1 ? s1 : g;
    owner |= int(p1);
    return {owner, g};
}

__device__ __forceinline__ Count shard_load_candidate(Code g) {
#if B300_SHARD_ADDRESS_SELECT
    const ShardAddress8 a = shard_address8_select(g, D_MAIN_CHUNK);
#else
    const ShardAddress8 a = shard_address8(g, D_MAIN_CHUNK);
#endif
    return D_MAIN_PTR[a.owner][a.local];
}

__global__ void shard_load_once_kernel(const Code* global_index, Count* out, int n) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < n) out[tid] = shard_load_candidate(global_index[tid]);
}

__global__ void shard_load_perf_kernel(
    Count* out, int n, int iters, Code total, Code stride, Code step
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    Code g = Code(tid) * stride;
    if (g >= total) g = total - 1;
    Count acc = 0;
    for (int i = 0; i < iters; ++i) {
        acc += shard_load_candidate(g);
        g += step;
        if (g >= total) g -= total;
    }
    out[tid] = acc;
}

float run_once(
    Count* out, int n, int blocks, int threads, int iters,
    Code total, Code stride, Code step
) {
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "event a");
    ck(cudaEventCreate(&b), "event b");
    ck(cudaEventRecord(a), "record a");
    shard_load_perf_kernel<<<blocks, threads>>>(out, n, iters, total, stride, step);
    ck(cudaGetLastError(), "shard perf launch");
    ck(cudaEventRecord(b), "record b");
    ck(cudaEventSynchronize(b), "sync b");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}

float median(std::vector<float> x) {
    std::sort(x.begin(), x.end());
    const size_t n = x.size();
    return n & 1 ? x[n / 2] : 0.5f * (x[n / 2 - 1] + x[n / 2]);
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 256;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 4096;
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 9;
    const Code chunk = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 65536ULL;
    if (blocks < 1 || threads < 1 || threads > 1024 || iters < 1 ||
        repeats < 1 || chunk < 2 || chunk > (~Code(0) >> 3)) return 2;

    constexpr int ngpu = 8;
    const Code total = chunk * ngpu;
    const int n = blocks * threads;
    const Code stride = std::max<Code>(1, total / Code(n));
    const Code step = std::max<Code>(1, chunk / 7 + 1);

    auto h_data = std::vector<Count>(static_cast<size_t>(total));
    for (Code i = 0; i < total; ++i)
        h_data[static_cast<size_t>(i)] = Count((i * 2654435761ULL + 17ULL) & 0xffffffffu);

    Count* d_data = nullptr;
    Count* d_out = nullptr;
    Code* d_index = nullptr;
    ck(cudaMalloc(&d_data, static_cast<size_t>(total) * sizeof(Count)), "alloc data");
    ck(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(Count)), "alloc out");
    ck(cudaMalloc(&d_index, static_cast<size_t>(n) * sizeof(Code)), "alloc index");
    ck(cudaMemcpy(d_data, h_data.data(), static_cast<size_t>(total) * sizeof(Count),
                  cudaMemcpyHostToDevice), "copy data");

    Count* ptrs[MAXGPU]{};
    for (int g = 0; g < ngpu; ++g) ptrs[g] = d_data + Code(g) * chunk;
    ck(cudaMemcpyToSymbol(D_MAIN_PTR, ptrs, sizeof(ptrs)), "main ptrs");
    ck(cudaMemcpyToSymbol(D_MAIN_CHUNK, &chunk, sizeof(chunk)), "main chunk");
    ck(cudaMemcpyToSymbol(D_NGPU, &ngpu, sizeof(ngpu)), "ngpu");

    auto h_index = std::vector<Code>(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h_index[static_cast<size_t>(i)] = (Code(i) * stride) % total;
    ck(cudaMemcpy(d_index, h_index.data(), static_cast<size_t>(n) * sizeof(Code),
                  cudaMemcpyHostToDevice), "copy index");
    shard_load_once_kernel<<<blocks, threads>>>(d_index, d_out, n);
    ck(cudaGetLastError(), "once launch");
    ck(cudaDeviceSynchronize(), "once sync");

    auto h_once = std::vector<Count>(static_cast<size_t>(n));
    ck(cudaMemcpy(h_once.data(), d_out, static_cast<size_t>(n) * sizeof(Count),
                  cudaMemcpyDeviceToHost), "copy once");
    for (int i = 0; i < n; ++i) {
        if (h_once[static_cast<size_t>(i)] != h_data[static_cast<size_t>(h_index[static_cast<size_t>(i)])]) {
            std::fprintf(stderr, "mismatch i=%d\n", i);
            return 3;
        }
    }

    run_once(d_out, n, blocks, threads, iters, total, stride, step);
    std::vector<float> times;
    times.reserve(static_cast<size_t>(repeats));
    for (int r = 0; r < repeats; ++r)
        times.push_back(run_once(d_out, n, blocks, threads, iters, total, stride, step));

    auto h_out = std::vector<Count>(static_cast<size_t>(n));
    ck(cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(n) * sizeof(Count),
                  cudaMemcpyDeviceToHost), "copy out");
    std::uint64_t checksum = 0;
    for (Count x : h_out) checksum += x;

    const double ops = double(n) * double(iters);
    const double ms = median(times);
    std::printf(
        "gridfp-b300-shard-address8-select-microprobe OK select=%d blocks=%d threads=%d "
        "iters=%d repeats=%d chunk=%llu loads=%.0f median_ms=%.6f Gload_s=%.6f "
        "checksum=%llu exact=OK\n",
        int(B300_SHARD_ADDRESS_SELECT), blocks, threads, iters, repeats,
        (unsigned long long)chunk, ops, ms, ops / ms / 1.0e6,
        (unsigned long long)checksum);

    cudaFree(d_index);
    cudaFree(d_out);
    cudaFree(d_data);
    return 0;
}
