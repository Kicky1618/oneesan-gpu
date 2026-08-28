#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using Rank64 = std::uint64_t;
constexpr std::array<Rank64, 11> TOTAL = {
    632ULL,4451ULL,32427ULL,242413ULL,1849269ULL,14339193ULL,
    112685373ULL,895517316ULL,7184644894ULL,58113695597ULL,
    473397057701ULL
};
constexpr std::array<std::uint32_t, 11> META = {
    1246013u,1245301u,1573381u,1970509u,2032777u,2163287u,
    2631197u,2757423u,2954017u,3150571u,3417385u
};

__device__ __forceinline__ std::uint32_t owner_generic(
    Rank64 midpoint, std::uint32_t meta, int ngpu
) {
    const unsigned shift = meta >> 16;
    const std::uint32_t magic = meta & 0xffffu;
    const std::uint32_t scale = magic * static_cast<std::uint32_t>(ngpu);
    const std::uint32_t lo = static_cast<std::uint32_t>(midpoint);
    const std::uint32_t product_hi = __umulhi(lo, scale);
    if (shift < 32) {
        const std::uint32_t product_lo = lo * scale;
        return (product_lo >> shift) | (product_hi << (32 - shift));
    }
    const std::uint32_t hi = static_cast<std::uint32_t>(midpoint >> 32);
    return (hi * scale + product_hi) >> (shift - 32);
}

__device__ __forceinline__ std::uint32_t owner_ngpu8(
    Rank64 midpoint, std::uint32_t meta
) {
    const unsigned shift = (meta >> 16) - 3;
    const std::uint32_t magic = meta & 0xffffu;
    const std::uint32_t lo = static_cast<std::uint32_t>(midpoint);
    const std::uint32_t product_hi = __umulhi(lo, magic);
    if (shift < 32) {
        const std::uint32_t product_lo = lo * magic;
        return (product_lo >> shift) | (product_hi << (32 - shift));
    }
    const std::uint32_t hi = static_cast<std::uint32_t>(midpoint >> 32);
    return (hi * magic + product_hi) >> (shift - 32);
}

template <bool Specialized>
__global__ void owner_probe_kernel(
    std::uint32_t* out, int n, int iters, Rank64 total, Rank64 stride,
    std::uint32_t meta, int ngpu
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    Rank64 midpoint = static_cast<Rank64>(tid) * stride;
    if (midpoint >= total) midpoint = total - 1;
    std::uint32_t acc = 0;
    for (int i = 0; i < iters; ++i) {
        midpoint += 17;
        if (midpoint >= total) midpoint -= total;
        if constexpr (Specialized)
            acc += owner_ngpu8(midpoint, meta);
        else
            acc += owner_generic(midpoint, meta, ngpu);
    }
    out[tid] = acc;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <bool Specialized>
float run_once(
    std::uint32_t* d_out, int n, int blocks, int threads, int iters,
    Rank64 total, Rank64 stride, std::uint32_t meta, int ngpu
) {
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "cudaEventCreate(a)");
    ck(cudaEventCreate(&b), "cudaEventCreate(b)");
    ck(cudaEventRecord(a), "cudaEventRecord(a)");
    owner_probe_kernel<Specialized><<<blocks, threads>>>(
        d_out, n, iters, total, stride, meta, ngpu);
    ck(cudaGetLastError(), "owner_probe_kernel");
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
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int blocks = argc > 2 ? std::atoi(argv[2]) : 256;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 4096;
    const int repeats = argc > 5 ? std::atoi(argv[5]) : 9;
    if (W < 8 || W > 28 || (W & 1) || blocks < 1 || threads < 1 ||
        threads > 1024 || iters < 1 || repeats < 1) {
        std::fprintf(stderr, "invalid W/blocks/threads/iters/repeats\n");
        return 2;
    }
    const int wi = (W - 8) >> 1;
    const int n = blocks * threads;
    const Rank64 total = TOTAL[wi];
    const Rank64 stride = std::max<Rank64>(Rank64(1), total / Rank64(n));
    const std::uint32_t meta = META[wi];
    constexpr int ngpu = 8;

    std::uint32_t *d0 = nullptr, *d1 = nullptr;
    ck(cudaMalloc(&d0, std::size_t(n) * sizeof(*d0)), "cudaMalloc(d0)");
    ck(cudaMalloc(&d1, std::size_t(n) * sizeof(*d1)), "cudaMalloc(d1)");

    run_once<false>(d0, n, blocks, threads, iters, total, stride, meta, ngpu);
    run_once<true>(d1, n, blocks, threads, iters, total, stride, meta, ngpu);

    std::vector<float> generic, ngpu8;
    generic.reserve(repeats);
    ngpu8.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        if (r & 1) {
            ngpu8.push_back(run_once<true>(d1, n, blocks, threads, iters, total, stride, meta, ngpu));
            generic.push_back(run_once<false>(d0, n, blocks, threads, iters, total, stride, meta, ngpu));
        } else {
            generic.push_back(run_once<false>(d0, n, blocks, threads, iters, total, stride, meta, ngpu));
            ngpu8.push_back(run_once<true>(d1, n, blocks, threads, iters, total, stride, meta, ngpu));
        }
    }

    std::vector<std::uint32_t> h0(n), h1(n);
    ck(cudaMemcpy(h0.data(), d0, std::size_t(n) * sizeof(*d0), cudaMemcpyDeviceToHost),
       "cudaMemcpy(d0)");
    ck(cudaMemcpy(h1.data(), d1, std::size_t(n) * sizeof(*d1), cudaMemcpyDeviceToHost),
       "cudaMemcpy(d1)");
    if (h0 != h1) {
        std::fprintf(stderr, "owner microprobe mismatch\n");
        return 3;
    }

    const float old_ms = median(generic);
    const float new_ms = median(ngpu8);
    std::printf(
        "gridfp-runtime-owner-u32limb-ngpu8-microprobe OK W=%d ngpu=8 blocks=%d threads=%d iters=%d repeats=%d generic_ms=%.6f ngpu8_ms=%.6f speedup=%.6fx delta_pct=%.4f exact=OK ngpu_mul_old=1 ngpu_mul_new=0 shift_bias=3\n",
        W, blocks, threads, iters, repeats, old_ms, new_ms,
        old_ms / new_ms, (new_ms / old_ms - 1.0f) * 100.0f);
    cudaFree(d0);
    cudaFree(d1);
    return 0;
}
