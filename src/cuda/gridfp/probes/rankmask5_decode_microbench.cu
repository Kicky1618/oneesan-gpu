#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kChunk = 5;
constexpr uint32_t kKeys = 243;
constexpr uint32_t kStates = 26;
constexpr uint32_t kSourceN = 1u << 15;
constexpr int kBlock = 256;
constexpr int kMaxBlocks = 4096;

enum DecodeMode : int {
    kDecodeFfs = 0,
    kDecodeUnrolled5 = 1,
    kDecodeDirect3 = 2,
    kDecodeDirect3Guard = 3,
};

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

static bool parse_weights(const char* text, std::array<uint64_t, 8>& out) {
    if (!text || !*text) return false;
    const char* p = text;
    bool any = false;
    for (size_t i = 0; i < out.size(); ++i) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(p, &end, 10);
        if (end == p) return false;
        out[i] = uint64_t(v);
        any = any || v != 0;
        if (i + 1 < out.size()) {
            if (*end != ',') return false;
            p = end + 1;
        } else if (*end != '\0') {
            return false;
        }
    }
    return any;
}

static void deterministic_shuffle(std::vector<uint8_t>& v, uint64_t seed) {
    uint64_t rng = seed;
    auto next_u64 = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    for (size_t i = v.size(); i > 1; --i) {
        const size_t j = size_t(next_u64() % uint64_t(i));
        std::swap(v[i - 1], v[j]);
    }
}

static std::array<size_t, 8> weighted_counts(
    size_t n, const std::array<uint64_t, 8>& weights
) {
    long double total = 0.0L;
    for (uint64_t x : weights) total += static_cast<long double>(x);
    std::array<size_t, 8> count{};
    std::array<long double, 8> frac{};
    size_t used = 0;
    for (size_t i = 0; i < count.size(); ++i) {
        const long double exact = static_cast<long double>(n) *
            static_cast<long double>(weights[i]) / total;
        const long double base = std::floor(exact);
        count[i] = size_t(base);
        frac[i] = exact - base;
        used += count[i];
    }
    // Largest-remainder apportionment preserves the requested histogram to
    // within one sample per mask, then we shuffle without changing counts.
    std::array<size_t, 8> order{0,1,2,3,4,5,6,7};
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return frac[a] > frac[b];
    });
    if (used <= n) {
        for (size_t k = 0; k < n - used; ++k) ++count[order[k % order.size()]];
    } else {
        size_t excess = used - n;
        for (size_t k = order.size(); excess && k > 0; --k) {
            const size_t i = order[k - 1];
            const size_t take = std::min(excess, count[i]);
            count[i] -= take;
            excess -= take;
        }
        if (excess) std::exit(6);
    }
    return count;
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

__device__ __forceinline__ uint64_t decode_direct3(
    uint8_t rankmask, const uint16_t* rank_row, const uint32_t* source
) {
    uint64_t sum = 0;
    if (rankmask & 0x01u) {
        const uint16_t rank = rank_row[0];
        sum += uint64_t(source[uint32_t(rank)]);
    }
    if (rankmask & 0x02u) {
        const uint16_t rank = rank_row[1];
        sum += uint64_t(source[uint32_t(rank)]);
    }
    if (rankmask & 0x04u) {
        const uint16_t rank = rank_row[2];
        sum += uint64_t(source[uint32_t(rank)]);
    }
    return sum;
}

__device__ __forceinline__ uint64_t decode_direct3_guard(
    uint8_t rankmask, const uint16_t* rank_row, const uint32_t* source
) {
    uint64_t sum = 0;
    if (rankmask != 0u) {
        if (rankmask & 0x01u) {
            const uint16_t rank = rank_row[0];
            sum += uint64_t(source[uint32_t(rank)]);
        }
        if (rankmask & 0x02u) {
            const uint16_t rank = rank_row[1];
            sum += uint64_t(source[uint32_t(rank)]);
        }
        if (rankmask & 0x04u) {
            const uint16_t rank = rank_row[2];
            sum += uint64_t(source[uint32_t(rank)]);
        }
    }
    return sum;
}

template<int MODE>
__global__ void rankmask5_kernel(
    const uint8_t* masks, const uint16_t* ranks, const uint32_t* source,
    size_t n, uint64_t* partial
) {
    static_assert(MODE >= kDecodeFfs && MODE <= kDecodeDirect3Guard);
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    uint64_t sum = 0;
    for (size_t i = tid; i < n; i += stride) {
        const uint8_t m = masks[i];
        const uint16_t* row = ranks + i * kChunk;
        if constexpr (MODE == kDecodeFfs) sum += decode_ffs(m, row, source);
        else if constexpr (MODE == kDecodeUnrolled5) sum += decode_unrolled5(m, row, source);
        else if constexpr (MODE == kDecodeDirect3) sum += decode_direct3(m, row, source);
        else sum += decode_direct3_guard(m, row, source);
    }
    partial[tid] = sum;
}

struct Timing {
    float best_ms = std::numeric_limits<float>::infinity();
};

template<int MODE>
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
            rankmask5_kernel<MODE><<<blocks, kBlock>>>(masks, ranks, source, n, partial);
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

template<int MODE>
uint64_t run_checksum(
    const uint8_t* masks, const uint16_t* ranks, const uint32_t* source,
    size_t n, uint64_t* partial, int blocks, size_t threads
) {
    rankmask5_kernel<MODE><<<blocks, kBlock>>>(masks, ranks, source, n, partial);
    ck(cudaDeviceSynchronize(), "checksum kernel");
    std::vector<uint64_t> host(threads);
    ck(cudaMemcpy(host.data(), partial, threads * sizeof(uint64_t), cudaMemcpyDeviceToHost),
       "checksum D2H");
    uint64_t sum = 0;
    for (uint64_t x : host) sum += x;
    return sum;
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
    const char* mask_order = argc > 4 ? argv[4] : "ordered";
    const bool shuffle_masks = std::strcmp(mask_order, "shuffled") == 0;
    const bool weighted_masks = std::strcmp(mask_order, "weighted") == 0;
    std::array<uint64_t, 8> weights{};
    if (weighted_masks && (argc <= 5 || !parse_weights(argv[5], weights))) {
        std::fprintf(stderr, "weighted mode requires eight comma-separated nonnegative weights\n");
        return 2;
    }
    if (!n || repeats <= 0 || trials <= 0 ||
        (!shuffle_masks && !weighted_masks && std::strcmp(mask_order, "ordered") != 0)) {
        std::fprintf(stderr,
                     "usage: %s [n>0] [repeats>0] [trials>0] [ordered|shuffled|weighted] [w0,...,w7]\n",
                     argv[0]);
        return 2;
    }

    std::vector<uint8_t> table_masks;
    table_masks.reserve((kStates - 1u) * kKeys);
    std::array<uint64_t, 6> hist{};
    uint8_t rankmask_or = 0;
    for (uint32_t state = 1; state < kStates; ++state) {
        for (uint32_t key = 0; key < kKeys; ++key) {
            const uint8_t m = rankmask_host(key, state);
            table_masks.push_back(m);
            ++hist[popcount5_host(m)];
            rankmask_or = uint8_t(rankmask_or | m);
        }
    }
    if ((rankmask_or & 0x18u) != 0u) {
        std::fprintf(stderr, "direct3 invariant failure rankmask_or=0x%02x\n", unsigned(rankmask_or));
        return 4;
    }
    if (shuffle_masks) deterministic_shuffle(table_masks, 0x6d2b79f5u);

    std::vector<uint8_t> h_masks(n);
    if (weighted_masks) {
        const auto count = weighted_counts(n, weights);
        size_t at = 0;
        for (uint32_t m = 0; m < count.size(); ++m)
            for (size_t k = 0; k < count[m]; ++k) h_masks[at++] = uint8_t(m);
        if (at != n) {
            std::fprintf(stderr, "weighted mask apportionment mismatch got=%zu expected=%zu\n", at, n);
            return 6;
        }
        deterministic_shuffle(h_masks, 0xd1b54a32d192ed03ull);
    } else {
        for (size_t i = 0; i < n; ++i) h_masks[i] = table_masks[i % table_masks.size()];
    }

    std::array<uint64_t, 8> input_hist{};
    for (uint8_t m : h_masks) {
        if (m >= input_hist.size()) {
            std::fprintf(stderr, "input rankmask overflow mask=%u\n", unsigned(m));
            return 7;
        }
        ++input_hist[m];
    }

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
    for (size_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < kChunk; ++j)
            h_ranks[i * kChunk + j] = uint16_t(next_u32() & (kSourceN - 1u));

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

    rankmask5_kernel<kDecodeFfs><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    rankmask5_kernel<kDecodeUnrolled5><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    rankmask5_kernel<kDecodeDirect3><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    rankmask5_kernel<kDecodeDirect3Guard><<<blocks, kBlock>>>(d_masks, d_ranks, d_source, n, d_partial);
    ck(cudaDeviceSynchronize(), "warmup");

    const uint64_t sum_ffs = run_checksum<kDecodeFfs>(
        d_masks, d_ranks, d_source, n, d_partial, blocks, threads);
    const uint64_t sum_unrolled = run_checksum<kDecodeUnrolled5>(
        d_masks, d_ranks, d_source, n, d_partial, blocks, threads);
    const uint64_t sum_direct3 = run_checksum<kDecodeDirect3>(
        d_masks, d_ranks, d_source, n, d_partial, blocks, threads);
    const uint64_t sum_direct3_guard = run_checksum<kDecodeDirect3Guard>(
        d_masks, d_ranks, d_source, n, d_partial, blocks, threads);
    if (sum_ffs != sum_unrolled || sum_ffs != sum_direct3 || sum_ffs != sum_direct3_guard) {
        std::fprintf(stderr,
                     "checksum mismatch ffs=%llu unrolled=%llu direct3=%llu direct3_guard=%llu\n",
                     (unsigned long long)sum_ffs, (unsigned long long)sum_unrolled,
                     (unsigned long long)sum_direct3, (unsigned long long)sum_direct3_guard);
        return 3;
    }

    const Timing ffs_a = bench<kDecodeFfs>(d_masks, d_ranks, d_source, n, d_partial,
                                           blocks, repeats, trials);
    const Timing unr_a = bench<kDecodeUnrolled5>(d_masks, d_ranks, d_source, n, d_partial,
                                                 blocks, repeats, trials);
    const Timing dir_a = bench<kDecodeDirect3>(d_masks, d_ranks, d_source, n, d_partial,
                                               blocks, repeats, trials);
    const Timing grd_a = bench<kDecodeDirect3Guard>(d_masks, d_ranks, d_source, n, d_partial,
                                                    blocks, repeats, trials);
    const Timing grd_b = bench<kDecodeDirect3Guard>(d_masks, d_ranks, d_source, n, d_partial,
                                                    blocks, repeats, trials);
    const Timing dir_b = bench<kDecodeDirect3>(d_masks, d_ranks, d_source, n, d_partial,
                                               blocks, repeats, trials);
    const Timing unr_b = bench<kDecodeUnrolled5>(d_masks, d_ranks, d_source, n, d_partial,
                                                 blocks, repeats, trials);
    const Timing ffs_b = bench<kDecodeFfs>(d_masks, d_ranks, d_source, n, d_partial,
                                           blocks, repeats, trials);
    const float ffs_ms = std::min(ffs_a.best_ms, ffs_b.best_ms);
    const float unrolled_ms = std::min(unr_a.best_ms, unr_b.best_ms);
    const float direct3_ms = std::min(dir_a.best_ms, dir_b.best_ms);
    const float direct3_guard_ms = std::min(grd_a.best_ms, grd_b.best_ms);

    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "device props");
    std::printf("rankmask5-decode-microbench OK device=%s n=%zu blocks=%d threads=%zu repeats=%d trials=%d mask_order=%s\n",
                prop.name, n, blocks, threads, repeats, trials, mask_order);
    std::printf("table_cases=%zu popcount_hist=%llu,%llu,%llu,%llu,%llu,%llu rankmask_or=0x%02x upper_bits_zero=1 checksum_exact=1 checksum=%llu\n",
                table_masks.size(),
                (unsigned long long)hist[0], (unsigned long long)hist[1],
                (unsigned long long)hist[2], (unsigned long long)hist[3],
                (unsigned long long)hist[4], (unsigned long long)hist[5],
                unsigned(rankmask_or), (unsigned long long)sum_ffs);
    const double input_zero_frac = n ? double(input_hist[0]) / double(n) : 0.0;
    std::printf("input_mask_hist=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu input_zero_frac=%.9f input_nonzero_frac=%.9f input_source=%s\n",
                (unsigned long long)input_hist[0], (unsigned long long)input_hist[1],
                (unsigned long long)input_hist[2], (unsigned long long)input_hist[3],
                (unsigned long long)input_hist[4], (unsigned long long)input_hist[5],
                (unsigned long long)input_hist[6], (unsigned long long)input_hist[7],
                input_zero_frac, 1.0 - input_zero_frac,
                weighted_masks ? "weighted_histogram_shuffled" :
                (shuffle_masks ? "uniform_table_shuffled" : "uniform_table_ordered"));
    if (weighted_masks) {
        std::printf("requested_mask_weights=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                    (unsigned long long)weights[0], (unsigned long long)weights[1],
                    (unsigned long long)weights[2], (unsigned long long)weights[3],
                    (unsigned long long)weights[4], (unsigned long long)weights[5],
                    (unsigned long long)weights[6], (unsigned long long)weights[7]);
    }
    std::printf("ffs_loop_ms=%.6f unrolled5_ms=%.6f direct3_ms=%.6f direct3_guard_ms=%.6f ffs_to_unrolled5_speedup=%.6f ffs_to_direct3_speedup=%.6f ffs_to_direct3_guard_speedup=%.6f unrolled5_to_direct3_speedup=%.6f direct3_to_guard_speedup=%.6f\n",
                ffs_ms, unrolled_ms, direct3_ms, direct3_guard_ms,
                double(ffs_ms) / double(unrolled_ms),
                double(ffs_ms) / double(direct3_ms),
                double(ffs_ms) / double(direct3_guard_ms),
                double(unrolled_ms) / double(direct3_ms),
                double(direct3_ms) / double(direct3_guard_ms));
    std::printf("decode_model=rank16_then_source32 mask_source=%s direct3_ordinals=0,1,2 direct3_guard=rankmask_nonzero_outer_guard mask_order=%s\n",
                weighted_masks ? "weighted_histogram" : "all_state1_25_x_key0_242_repeated",
                mask_order);

    cudaFree(d_partial);
    cudaFree(d_source);
    cudaFree(d_ranks);
    cudaFree(d_masks);
    return 0;
}
