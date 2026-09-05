#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef PROBE_LEN13_TABLE
#define PROBE_LEN13_TABLE 0
#endif
static_assert(PROBE_LEN13_TABLE == 0 || PROBE_LEN13_TABLE == 1);

namespace {
using Rank64 = std::uint64_t;

__device__ __constant__ std::uint16_t PROBE_CHOOSE[14][14];
__device__ __constant__ std::uint16_t PROBE_COUNT[14] = {
    1u, 13u, 78u, 286u, 715u, 1287u, 1716u,
    1716u, 1287u, 715u, 286u, 78u, 13u, 1u
};
__device__ __constant__ std::uint16_t PROBE_BASE[14] = {
    0u, 1u, 14u, 92u, 378u, 1093u, 2380u,
    4096u, 5812u, 7099u, 7814u, 8100u, 8178u, 8191u
};
#if PROBE_LEN13_TABLE
__device__ __constant__ std::uint16_t PROBE_SUPPORT13[8192];
#endif

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

std::array<std::array<std::uint16_t, 14>, 14> make_choose() {
    std::array<std::array<std::uint16_t, 14>, 14> c{};
    c[0][0] = 1;
    for (int n = 1; n <= 13; ++n) {
        c[n][0] = c[n][n] = 1;
        for (int k = 1; k < n; ++k)
            c[n][k] = std::uint16_t(c[n - 1][k - 1] + c[n - 1][k]);
    }
    return c;
}

std::uint16_t unrank13_host(
    const std::array<std::array<std::uint16_t, 14>, 14>& c,
    int ones, Rank64 rank
) {
    std::uint16_t mask = 0;
    int left = ones;
    for (int pos = 0; pos < 13; ++pos) {
        if (!left) break;
        const int remaining = 13 - pos;
        if (left == remaining) {
            mask = std::uint16_t(mask |
                std::uint16_t(((1u << remaining) - 1u) << pos));
            break;
        }
        const Rank64 zero_count = c[remaining - 1][left];
        if (rank < zero_count) continue;
        rank -= zero_count;
        mask = std::uint16_t(mask | (std::uint16_t(1u) << pos));
        --left;
    }
    return mask;
}

__device__ __forceinline__ std::uint16_t unrank13_device(int ones, Rank64 rank) {
    std::uint16_t mask = 0;
    int left = ones;
    for (int pos = 0; pos < 13; ++pos) {
        if (!left) break;
        const int remaining = 13 - pos;
        if (left == remaining) {
            mask = std::uint16_t(mask |
                std::uint16_t(((1u << remaining) - 1u) << pos));
            break;
        }
        const Rank64 zero_count = PROBE_CHOOSE[remaining - 1][left];
        if (rank < zero_count) continue;
        rank -= zero_count;
        mask = std::uint16_t(mask | (std::uint16_t(1u) << pos));
        --left;
    }
    return mask;
}

__device__ __forceinline__ std::uint16_t probe_unrank(int ones, Rank64 rank) {
#if PROBE_LEN13_TABLE
    return PROBE_SUPPORT13[std::uint32_t(PROBE_BASE[ones]) + std::uint32_t(rank)];
#else
    return unrank13_device(ones, rank);
#endif
}

__global__ void probe_kernel(std::uint64_t* output, int iterations) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int sublane = threadIdx.x & 7;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    int ones = int((tid >> 3) % 14u);

    if (sublane == 0) {
        for (int i = 0; i < iterations; ++i) {
            const std::uint32_t count = PROBE_COUNT[ones];
            seed = seed * 1664525u + 1013904223u;
            const Rank64 rank =
                (std::uint64_t(seed) * std::uint64_t(count)) >> 32;
            const std::uint16_t mask = probe_unrank(ones, rank);
            acc ^= std::uint64_t(mask) + 0x9e3779b97f4a7c15ULL +
                   (acc << 6) + (acc >> 2);
            ++ones;
            if (ones == 14) ones = 0;
        }
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 ||
        (threads & 7) || iterations <= 0 || warmup < 0) {
        std::cerr << "usage: probe [blocks>0] [threads multiple-of-8] [iterations>0] [warmup>=0]\n";
        return 2;
    }

    const auto choose = make_choose();
    cuda_check(cudaMemcpyToSymbol(PROBE_CHOOSE, choose.data(), sizeof(choose)),
               "cudaMemcpyToSymbol(PROBE_CHOOSE)");
#if PROBE_LEN13_TABLE
    std::array<std::uint16_t, 8192> table{};
    constexpr std::array<int, 14> base{
        0, 1, 14, 92, 378, 1093, 2380,
        4096, 5812, 7099, 7814, 8100, 8178, 8191};
    for (int ones = 0; ones <= 13; ++ones) {
        const int count = choose[13][ones];
        for (int rank = 0; rank < count; ++rank)
            table[std::size_t(base[std::size_t(ones)] + rank)] =
                unrank13_host(choose, ones, Rank64(rank));
    }
    cuda_check(cudaMemcpyToSymbol(PROBE_SUPPORT13, table.data(), sizeof(table)),
               "cudaMemcpyToSymbol(PROBE_SUPPORT13)");
#endif

    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    std::uint64_t* d_output = nullptr;
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc");
    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, threads>>>(d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    probe_kernel<<<blocks, threads>>>(d_output, iterations);
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

    const std::uint64_t calls = std::uint64_t(blocks) *
        std::uint64_t(threads / 8) * std::uint64_t(iterations);
    const double ns_per_call = double(elapsed_ms) * 1.0e6 / double(calls);
    std::cout << "gridfp-support-unrank-len13-table-microprobe"
              << " table=" << PROBE_LEN13_TABLE
              << " subgroup_width=8 components_per_warp=4"
              << " blocks=" << blocks << " threads=" << threads
              << " iterations=" << iterations << " calls=" << calls
              << " kernel_ms=" << elapsed_ms
              << " ns_per_call=" << ns_per_call
              << " checksum=" << checksum << '\n';

    cudaEventDestroy(stop); cudaEventDestroy(start); cudaFree(d_output);
    return 0;
}
