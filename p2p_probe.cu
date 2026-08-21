#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    int n = 0;
    ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    std::printf("gpus=%d\n", n);
    for (int a = 0; a < n; ++a) {
        cudaDeviceProp p{};
        ck(cudaGetDeviceProperties(&p, a), "cudaGetDeviceProperties");
        std::printf("gpu%d=%s cc=%d.%d\n", a, p.name, p.major, p.minor);
    }

    constexpr size_t bytes = 256ull << 20;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, a, b), "cudaDeviceCanAccessPeer");
            std::printf("peer %d->%d can=%d", a, b, can);
            if (!can) {
                std::puts("");
                continue;
            }

            ck(cudaSetDevice(a), "cudaSetDevice a");
            auto e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else ck(e, "cudaDeviceEnablePeerAccess");

            void* dst = nullptr;
            void* src = nullptr;
            ck(cudaMalloc(&dst, bytes), "cudaMalloc dst");
            ck(cudaSetDevice(b), "cudaSetDevice b");
            ck(cudaMalloc(&src, bytes), "cudaMalloc src");
            ck(cudaMemset(src, 0x5a, bytes), "cudaMemset src");

            ck(cudaSetDevice(a), "cudaSetDevice a2");
            cudaEvent_t s{}, t{};
            ck(cudaEventCreate(&s), "event s");
            ck(cudaEventCreate(&t), "event t");
            ck(cudaEventRecord(s), "record s");
            for (int i = 0; i < 8; ++i)
                ck(cudaMemcpyPeerAsync(dst, a, src, b, bytes), "cudaMemcpyPeerAsync");
            ck(cudaEventRecord(t), "record t");
            ck(cudaEventSynchronize(t), "sync t");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, s, t), "elapsed");
            const double gib = double(bytes) * 8.0 / double(1ull << 30);
            std::printf(" bw=%.2f GiB/s\n", gib / (ms / 1000.0));
            cudaEventDestroy(s);
            cudaEventDestroy(t);
            cudaFree(dst);
            ck(cudaSetDevice(b), "cudaSetDevice b3");
            cudaFree(src);
        }
    }
}
