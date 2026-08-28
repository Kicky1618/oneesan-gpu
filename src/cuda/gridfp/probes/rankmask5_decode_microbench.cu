#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kChunk = 5;
constexpr uint32_t kKeys = 243;
constexpr uint32_t kStates = 26;
constexpr uint32_t kMask = 0x1fu;
constexpr uint32_t kSourceN = 1u << 15;
constexpr int kBlock = 256;
constexpr int kMaxBlocks = 4096;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr uint32_t pow3(uint32_t n) {
    return n == 0 ? 1u : 3u * pow3(n - 1);
}

constexpr uint8_t lmask_host(uint32_t key) {
    uint8_t mask = 0;
    for (uint32_t pos = 0; pos < kChunk; ++pos) {
        const uint32_t v = (key / pow3(pos)) % 3u;
        if (v == 1u) mask = uint8_t(mask | uint8_t(1u << pos));
    }
    return mask;
}

constexpr uint8_t popcount5_host(uint8_t x) {
    uint8_t n = 0;
    for (uint32_t i = 0; i < kChunk; ++i) n = uint8_t(n + ((x >> i) & 1u));
    return n;
}

constexpr uint8_t rankmask_host(uint32_t key, uint32_t input_state) {
    uint32_t s = input_state;
    uint8_t ordinary = 0;
    for (int pos = int(kChunk) - 1; pos >= 0; --pos) {
        const uint32_t v = (key / pow3(uint32_t(pos))) % 3u;
        if (v == 2u) {
            if (s == 1u) break;
            --s;
        } else if (v == 1u) {
            if (s == 1u) ordinary = uint8_t(ordinary | uint8_t(1u << pos));
            ++s;
        }
    }

    const uint8_t lm = lmask_host(key);
    uint8_t rankmask = 0;
    for (uint32_t pos = 0; pos < kChunk; ++pos) {
        if (((ordinary >> pos) & 1u) == 0u) continue;
        const uint8_t lower_or_equal = uint8_t((uint8_t(1u << (pos + 1u))) - 1u);
        const uint8_t higher = uint8_t(lm & uint8_t(~lower_or_equal));
        const uint8_t ordinal = popcount5_host(higher);
        rankmask = uint8_t(rankmask | uint8_t(1u << ordinal));
    }
    return rankmask;
}

__device__ __forceinline__ uint64_t decode_ffs(
    uint8_t rankmask, const uint16_t* rank_row, const uint32_t* source
) {
    uint64_t sum = 0;
    while (rankmask) {
        const int ordinal = __ffs(int(rankmask)) - 1;
        rankmask = uint8_t(rankmask & uint8_t(rankmask - 1));
        const uint16_t rank = rank_row[uint32_t(ordinal)];
        sum += uint64_t(source[uint32_t(rank)]);
    }
    return sum;
}

__device__ __forceinline__ uint64_t decode_unrolled5(
    uint8_t rankmask, const uint16_t* rank_row, const uint32_t* source
) {
    uint64_t sum = 0;
#pragma unroll
    for (uint32_t ordinal = 0; ordinal < kChunk; ++ordinal) {
        if (rankmask & uint8_t(1u << ordinal)) {
            const uint16_t rank = rank_row[ordinal];
            sum += uint64_t(source[uint32_t(rank)]);
        }
    }
    return sum;
}

template<bool UNROLLED>
__global__ void rankmask5_kernel(
    const uint8_t* masks, const uint16_t* ranks, const uint32_t* source,
    size_t n, uint64_t* partial
) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    uint64_t sum = 0;
    for (size_t i = tid; i < n; i += stride) {
        const uint8_t m = masks[i];
        const uint16_t* row = ranks + i * kChunk;
        if constexpr (UNROLLED) sum += decode_unrolled5(m, row, source);
        else sum += decode_ffs(m, row, source);
    }
    partial[tid] = sum;
}

struct Timing {
    float best_ms = std::numeric_limits<float>::infinity();
};

template<bool UNROLLED>
Timing bench(
    const uint8_t* masks, const uint16_t* ranks, const uint32_t* source,
    size_t n, uint64_t* partial, int blocks, int repeats, int trials
) {
    cudaEvent_t start = nullptr, stop = nullptr;
    ck(cudaEventCreate(&start), "event start");
    ck(cudaEventCreate(&stop), "event stop");
    Timing t;
    for (int trial = 0; trial < trials; ++trial) {
        ck(cudaEventRecord(start), "event record start");
        for (int r = 0; r < repeats; ++r) {
            rankmask5_kernel<UNROLLED><<<blocks, kBlock>>>(masks, ranks, source, n, partial);
        }
        ck(cudaEventRecord(stop), "event record stop");
        ck(cudaEventSynchronize(stop), "event sync stop");
        ck(cudaGetLastError(), "rankmask5 kernel");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        t.best_ms = std::min(t.best_ms, ms / float(repeats));
    }
    ck(cudaEventDestroy(start), "event destroy start");
    ck(cudaEventDestroy(stop), "event destroy stop");
    return t;
}

uint64_t checksum(const std::vector<uint64_t>& v) {
    uint64_t s = 0;
    for (uint64_t x : v) s += x;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    cudaError_t dc = cudaGetDeviceCount(&device_count);
    if (dc != cudaSuccess || device_count == 0) {
        std::printf("rankmask5-decode-microbench SKIP no CUDA device\n");
        return 0;
    }

    const size_t n = argc > 1 ? size_t(std::strtoull(argv[1], nullptr, 10)) : (size_t(1) << 22);
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 20;
    const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
    if (!n || repeats <= 0 || trials <= 0) {
        std::fprintf(stderr, "usage: %s [n>0] [repeats>0] [trials>0]\n", argv[0]);
        return 2;
    }

    std::vector<uint8_t> table_masks;
    table_masks.reserve((kStates - 1u) * kKeys);
    std::array<uint64_t, 6> hist{};
    for (uint32_t state = 1; state < kStates; ++state) {
        for (uint32_t key = 0; key < kKeys; ++key) {
            const uint8_t m = rankmask_host(key, state);
            table_masks.push_back(m);
            ++hist[popcount5_host(m)];
        }
    }

    std::vector<uint8_t> h_masks(n);
    std::vector<uint16_t> h_ranks(n * kChunk);
    std::vector<uint32_t> h_source(kSourceN);
    uint32_t rng = 0x12345678u;
    auto next_u32 = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    for (uint32_t i = 0; i < kSourceN; ++i)
        h_source[i] = next_u32() | 1u;
    for (size_t i = 0; i < n; ++i) {
        h_masks[i] = table_masks[i % table_masks.size()];
        for (uint32_t j = 0; j < kChunk; ++j)
            h_ranks[i * kChunk + j] = uint16_t(next_u32() & (kSourceN - 1u));
    }

    uint8_t* d_masks = nullptr;
    uint16_t* d_ranks = nullptr;
    uint32_t* d_source = nullptr;
    uint64_t* d_partial = nullptr;
    const int blocks = std::max(1, std::min(kMaxBlocks, int((n + kBlock - 1) / kBlock)));
    const size_t threads = size_t(blocks) * kBlock;
    ck(cudaMalloc(&d_masks, n * sizeof(uint8_t)), "masks alloc");
    ck(cudaMalloc(&d_ranks, n * kChunk * sizeof(uint16_t)), "ranks alloc");
    ck(cudaMalloc(&d_source, kSourceN * sizeof(uint32_t)), "source alloc");
    ck(cudaMalloc(&d_partial, threads * sizeof(uint64_t)), "partial alloc");
    ck(cudaMemcpy(d_masks, h_masks.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice), "masks H2D");
    ck(cudaMemcpy(d_ranks, h_ranks.data(), n * kChunk * sizeof(uint16_t), cudaMemcpyHostToDevice), "ranks H2D");
    ck(cudaMemcpy(d_source, h_source.data(), kSourceN * sizeof(uint32_t), cudaMemcpyHostToDevice), "source H2D");

    rankmask5_kernel<false><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    rankmask5_kernel<true><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    ck(cudaDeviceSynchronize(), "warmup");

    rankmask5_kernel<false><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    ck(cudaDeviceSynchronize(), "ffs checksum kernel");
    std::vector<uint64_t> h_ffs(threads);
    ck(cudaMemcpy(h_ffs.data(), d_partial, threads * sizeof(uint64_t), cudaMemcpyDeviceToHost), "ffs checksum D2H");
    rankmask5_kernel<true><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    ck(cudaDeviceSynchronize(), "unrolled checksum kernel");
    std::vector<uint64_t> h_unrolled(threads);
    ck(cudaMemcpy(h_unrolled.data(), d_partial, threads * sizeof(uint64_t), cudaMemcpyDeviceToHost), "unrolled checksum D2H");
    const uint64_t sum_ffs = checksum(h_ffs);
    const uint64_t sum_unrolled = checksum(h_unrolled);
    if (sum_ffs != sum_unrolled) {
        std::fprintf(stderr, "checksum mismatch ffs=%llu unrolled=%llu\n",
                     (unsigned long long)sum_ffs, (unsigned long long)sum_unrolled);
        return 3;
    }

    // Alternate benchmark order to reduce first-mode bias; report best launch time.
    Timing ffs_a = bench<false>(d_masks, d_ranks, d_source, n, d_partial, blocks, repeats, trials);
    Timing unr_a = bench<true>(d_masks, d_ranks, d_source, n, d_partial, blocks, repeats, trials);
    Timing unr_b = bench<true>(d_masks, d_ranks, d_source, n, d_partial, blocks, repeats, trials);
    Timing ffs_b = bench<false>(d_masks, d_ranks, d_source, n, d_partial, blocks, repeats, trials);
    const float ffs_ms = std::min(ffs_a.best_ms, ffs_b.best_ms);
    const float unrolled_ms = std::min(unr_a.best_ms, unr_b.best_ms);
    const double speedup = double(ffs_ms) / double(unrolled_ms);

    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "device props");
    std::printf("rankmask5-decode-microbench OK device=%s n=%zu blocks=%d threads=%zu repeats=%d trials=%d\n",
                prop.name, n, blocks, threads, repeats, trials);
    std::printf("table_cases=%zu popcount_hist=%llu,%llu,%llu,%llu,%llu,%llu checksum_exact=1 checksum=%llu\n",
                table_masks.size(),
                (unsigned long long)hist[0], (unsigned long long)hist[1],
                (unsigned long long)hist[2], (unsigned long long)hist[3],
                (unsigned long long)hist[4], (unsigned long long)hist[5],
                (unsigned long long)sum_ffs);
    std::printf("ffs_loop_ms=%.6f unrolled5_ms=%.6f ffs_to_unrolled_speedup=%.6f\n",
                ffs_ms, unrolled_ms, speedup);
    std::printf("decode_model=rank16_then_source32 mask_source=all_state1_25_x_key0_242_repeated\n");

    cudaFree(d_partial);
    cudaFree(d_source);
    cudaFree(d_ranks);
    cudaFree(d_masks);
    return 0;
}
