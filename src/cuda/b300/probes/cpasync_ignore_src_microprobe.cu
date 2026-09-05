#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(2);
    }
}

__global__ void cpasync_ignore_src_kernel(
    const std::uint32_t* __restrict__ src,
    std::uint32_t* __restrict__ dst,
    int n
) {
    extern __shared__ std::uint32_t smem[];
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= n) return;

    std::uint32_t* sd = smem + threadIdx.x;
#if __CUDA_ARCH__ >= 800
    const std::uint32_t sdst = std::uint32_t(__cvta_generic_to_shared(sd));
    const unsigned long long gsrc = reinterpret_cast<unsigned long long>(src + i);
    const std::uint32_t keep = (i & 1) == 0 ? 1u : 0u;
    asm volatile(
        "{ .reg .pred p; setp.eq.u32 p, %2, 0; "
        "cp.async.ca.shared.global [%0], [%1], 4, p; "
        "cp.async.commit_group; cp.async.wait_group 0; }"
        :: "r"(sdst), "l"(gsrc), "r"(keep));
#else
    *sd = (i & 1) == 0 ? src[i] : 0u;
#endif
    dst[i] = *sd;
}

int main() {
    int dev = 0;
    ck(cudaSetDevice(dev), "set device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, dev), "get properties");

    constexpr int N = 256;
    std::vector<std::uint32_t> hsrc(N), hout(N, 0);
    for (int i = 0; i < N; ++i)
        hsrc[i] = 0x9e3779b9u ^ (std::uint32_t(i) * 0x45d9f3bu);

    std::uint32_t *dsrc = nullptr, *dout = nullptr;
    ck(cudaMalloc(&dsrc, sizeof(std::uint32_t) * N), "alloc src");
    ck(cudaMalloc(&dout, sizeof(std::uint32_t) * N), "alloc dst");
    ck(cudaMemcpy(dsrc, hsrc.data(), sizeof(std::uint32_t) * N,
                  cudaMemcpyHostToDevice), "copy src");
    ck(cudaMemset(dout, 0xa5, sizeof(std::uint32_t) * N), "clear dst");

    constexpr int T = 128;
    cpasync_ignore_src_kernel<<<N / T, T, sizeof(std::uint32_t) * T>>>(dsrc, dout, N);
    ck(cudaGetLastError(), "launch");
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemcpy(hout.data(), dout, sizeof(std::uint32_t) * N,
                  cudaMemcpyDeviceToHost), "copy dst");

    for (int i = 0; i < N; ++i) {
        const std::uint32_t want = (i & 1) == 0 ? hsrc[i] : 0u;
        if (hout[i] != want) {
            std::cerr << "mismatch i=" << i << " got=" << hout[i]
                      << " want=" << want << '\n';
            return 3;
        }
    }

    cudaFree(dout);
    cudaFree(dsrc);
    std::cout << "cpasync-ignore-src"
              << " device=" << prop.name
              << " cc=" << prop.major << '.' << prop.minor
              << " cp_async_ignore_src=OK zero_fill=OK exact=OK\n";
    return 0;
}
