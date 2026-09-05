#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "gridfp_reduced_production_codec_device.cuh"

namespace rp = oneesan::gridfp::reducedprod;

namespace {

static constexpr int CASES_PER_SECTOR = 64;
static constexpr int SECTORS = 14;
static constexpr int CASES = CASES_PER_SECTOR * SECTORS;

struct ProbeCase {
    rp::MateID mate = 0;
    std::uint32_t support = 0;
    int occupied = 0;
    rp::Rank64 expected = 0;
};

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

using PrimitiveTable =
    std::array<std::array<rp::Rank64, rp::RP_MAX_W + 2>, rp::RP_MAX_W + 1>;

PrimitiveTable make_primitive_table() {
    PrimitiveTable p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= rp::RP_MAX_W; ++rem) {
        for (int h = 0; h <= rp::RP_MAX_W; ++h) {
            rp::Rank64 z = p[rem - 1][h + 1];
            if (h > 0) z += p[rem - 1][h - 1];
            p[rem][h] = z;
        }
    }
    return p;
}

ProbeCase make_case(
    int occupied,
    rp::Rank64 primitive_rank,
    std::mt19937_64& rng,
    const PrimitiveTable& p
) {
    std::array<int, 28> bit{};
    std::iota(bit.begin(), bit.end(), 0);
    std::shuffle(bit.begin(), bit.end(), rng);
    std::sort(bit.begin(), bit.begin() + occupied, std::greater<int>());

    ProbeCase out{};
    out.occupied = occupied;
    out.expected = primitive_rank;
    rp::Rank64 rank = primitive_rank;
    int h = 1;
    for (int seen = 0; seen < occupied; ++seen) {
        const int rem = occupied - seen - 1;
        const rp::Rank64 r_count = h > 0 ? p[rem][h - 1] : 0;
        int value = 1; // R
        if (rank < r_count) {
            --h;
        } else {
            rank -= r_count;
            value = 2; // L
            ++h;
        }
        const int b = bit[std::size_t(seen)];
        out.support |= std::uint32_t(1) << b;
        out.mate |= rp::MateID(value) << (2 * b);
    }
    if (h != 0 || rank != 0) {
        std::cerr << "host primitive construction failed occupied=" << occupied
                  << " rank=" << primitive_rank << '\n';
        std::exit(3);
    }
    return out;
}

std::array<ProbeCase, CASES> make_cases(const PrimitiveTable& p) {
    std::array<ProbeCase, CASES> out{};
    std::mt19937_64 rng(0x72616e6b7061636bULL);
    int ix = 0;
    for (int sector = 0; sector < SECTORS; ++sector) {
        const int occupied = 1 + 2 * sector;
        const rp::Rank64 count = p[occupied][1];
        for (int j = 0; j < CASES_PER_SECTOR; ++j) {
            const rp::Rank64 rank = rng() % count;
            out[std::size_t(ix++)] = make_case(occupied, rank, rng, p);
        }
    }
    return out;
}

__device__ __forceinline__ rp::Rank64 primitive_rank_probe(
    rp::MateID m, std::uint32_t support, int occupied
) {
    int h = 1;
    int seen = 0;
    rp::Rank64 rank = 0;
    std::uint32_t mask = support;
    while (mask) {
        const int bit = 31 - __clz(mask);
        const auto c = oneesan::gridfp::mget(m, bit);
        const int rem = occupied - (++seen);
        if (c == oneesan::gridfp::L) {
            if (h > 0)
                rank += rp::materialize_primitive_r_count_device(rem, h);
            ++h;
        } else {
            --h;
        }
        mask ^= std::uint32_t(1) << bit;
    }
    return rank;
}

__global__ void probe_kernel(
    const ProbeCase* cases,
    std::uint64_t* output,
    int* error,
    int iterations
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const ProbeCase c = cases[tid % CASES];
    rp::MateID mate = c.mate;
    std::uint32_t support = c.support;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        // Keep the rank calculation inside the timed loop without adding an
        // actual instruction to the value path.
        asm volatile("" : "+l"(mate), "+r"(support));
        const rp::Rank64 rank = primitive_rank_probe(mate, support, c.occupied);
        if (rank != c.expected) atomicCAS(error, 0, 1);
        acc ^= rank + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
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
        iterations <= 0 || warmup < 0) {
        std::cerr << "usage: probe [blocks>0] [threads=1..1024] "
                     "[iterations>0] [warmup>=0]\n";
        return 2;
    }

    const PrimitiveTable primitive = make_primitive_table();
    const auto cases = make_cases(primitive);
    cuda_check(cudaMemcpyToSymbol(
        rp::RP_PRIMITIVE, primitive.data(), sizeof(primitive)),
        "cudaMemcpyToSymbol(RP_PRIMITIVE)");

    ProbeCase* d_cases = nullptr;
    std::uint64_t* d_output = nullptr;
    int* d_error = nullptr;
    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    cuda_check(cudaMalloc(&d_cases, sizeof(cases)), "cudaMalloc(cases)");
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)), "cudaMalloc(output)");
    cuda_check(cudaMalloc(&d_error, sizeof(int)), "cudaMalloc(error)");
    cuda_check(cudaMemcpy(
        d_cases, cases.data(), sizeof(cases), cudaMemcpyHostToDevice),
        "cudaMemcpy(cases)");
    cuda_check(cudaMemset(d_error, 0, sizeof(int)), "cudaMemset(error)");

    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, threads>>>(d_cases, d_output, d_error, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");
    cuda_check(cudaMemset(d_error, 0, sizeof(int)), "cudaMemset(error timed)");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    probe_kernel<<<blocks, threads>>>(d_cases, d_output, d_error, iterations);
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    int error = 0;
    cuda_check(cudaMemcpy(&error, d_error, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(error)");
    if (error) {
        std::cerr << "primitive rank mismatch error=" << error << '\n';
        return 4;
    }

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
    std::cout << "gridfp-runtime-primitive-rank-packed-microprobe"
              << " packed=" << RP_FAST_MATERIALIZE_PRIMITIVE_PACKED
              << " shared_table=materialize_primitive_thresholds"
              << " cases=" << CASES
              << " sectors=" << SECTORS
              << " avg_threshold_loads_per_call=6.5"
              << " blocks=" << blocks
              << " threads=" << threads
              << " iterations=" << iterations
              << " calls=" << calls
              << " kernel_ms=" << elapsed_ms
              << " ns_per_call=" << ns_per_call
              << " checksum=" << checksum << '\n';

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_error);
    cudaFree(d_output);
    cudaFree(d_cases);
    return 0;
}
