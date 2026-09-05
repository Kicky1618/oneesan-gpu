#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../common/gridfp_transition.hpp"

#ifndef RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
#define RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN 0
#endif
static_assert(RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN == 0 ||
              RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN == 1,
              "RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN must be 0 or 1");

namespace gp = oneesan::gridfp;

namespace {

constexpr std::uint32_t INPUT_COUNT = 1u << 16;
constexpr std::uint32_t INPUT_MASK = INPUT_COUNT - 1u;
constexpr int WIDTH = 28;
constexpr int FREE_WIDTH = WIDTH - 2;
using Count64 = std::uint64_t;
using MotzkinTable = std::array<std::array<Count64, WIDTH + 2>, WIDTH + 1>;

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

__device__ __forceinline__ std::uint32_t scan_turn_nn(
    gp::MateID mate,
    int width,
    int& final_balance
) {
    int bal = 0;
    std::uint32_t candidates = 0;
#if RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
    std::uint32_t mask = gp::mate_non_n_mask(mate, width) & ~std::uint32_t(3u);
    while (mask) {
        const int q = gp::mate_lsb_index32(mask);
        const gp::MateValue v = gp::mget(mate, q);
        if (bal == 0 && v == gp::R) candidates |= std::uint32_t(1) << q;
        if (v == gp::R) ++bal;
        else if (v == gp::L) --bal;
        if (bal < 0) break;
        mask &= mask - 1u;
    }
#else
    for (int q = 2; q < width; ++q) {
        const gp::MateValue v = gp::mget(mate, q);
        if (bal == 0 && v == gp::R) candidates |= std::uint32_t(1) << q;
        if (v == gp::R) ++bal;
        else if (v == gp::L) --bal;
        if (bal < 0) break;
    }
#endif
    final_balance = bal;
    return candidates;
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

MotzkinTable make_motzkin_table() {
    MotzkinTable table{};
    table[0][0] = 1;
    for (int rem = 1; rem <= WIDTH; ++rem) {
        for (int h = 0; h <= WIDTH; ++h) {
            Count64 count = table[rem - 1][h];
            if (h > 0) count += table[rem - 1][h - 1];
            count += table[rem - 1][h + 1];
            table[rem][h] = count;
        }
    }
    return table;
}

gp::MateID make_motzkin_nn_mate(
    Count64 rank,
    const MotzkinTable& table
) {
    gp::MateID mate = 0;
    int h = 1;
    for (int pos = 0; pos < FREE_WIDTH; ++pos) {
        const int q = WIDTH - 1 - pos;
        const int rem = FREE_WIDTH - pos - 1;

        const Count64 n_count = table[rem][h];
        if (rank < n_count) continue;
        rank -= n_count;

        const Count64 r_count = h > 0 ? table[rem][h - 1] : 0;
        if (rank < r_count) {
            mate = gp::mset(mate, q, gp::R);
            --h;
            continue;
        }
        rank -= r_count;
        mate = gp::mset(mate, q, gp::L);
        ++h;
    }
    return mate;
}

gp::MateID make_bernoulli_mate(std::uint32_t seed, int nonn_percent) {
    gp::MateID mate = 0;
    for (int q = 2; q < WIDTH; ++q) {
        seed = seed * 1664525u + 1013904223u;
        const bool occupied = (seed % 100u) < std::uint32_t(nonn_percent);
        if (!occupied) continue;
        seed = seed * 1664525u + 1013904223u;
        mate = gp::mset(mate, q, (seed & 1u) ? gp::R : gp::L);
    }
    return mate;
}

__global__ void probe_kernel(
    const gp::MateID* __restrict__ inputs,
    std::uint64_t* __restrict__ output,
    int iterations
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t seed = tid * 747796405u + 2891336453u;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ tid;

    for (int i = 0; i < iterations; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const gp::MateID mate = inputs[seed & INPUT_MASK];
        int balance = 0;
        const std::uint32_t candidates = scan_turn_nn(mate, WIDTH, balance);
        const std::uint64_t value =
            (std::uint64_t(candidates) << 32) ^ std::uint32_t(balance);
        acc ^= value + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const std::string input_case = argc > 4 ? argv[4] : "motzkin";
    const int warmup = argc > 5 ? std::atoi(argv[5]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 || iterations <= 0 ||
        warmup < 0) {
        std::cerr << "usage: probe [blocks>0] [threads=1..1024] [iterations>0] "
                     "[motzkin|nonn_percent=0..100] [warmup>=0]\n";
        return 2;
    }

    bool motzkin = input_case == "motzkin";
    int nonn_percent = -1;
    if (!motzkin) {
        char* end = nullptr;
        const long parsed = std::strtol(input_case.c_str(), &end, 10);
        if (!end || *end != '\0' || parsed < 0 || parsed > 100) {
            std::cerr << "input case must be 'motzkin' or an integer 0..100\n";
            return 2;
        }
        nonn_percent = int(parsed);
    }

    const MotzkinTable motzkin_table = make_motzkin_table();
    const Count64 motzkin_count = motzkin_table[FREE_WIDTH][1];
    if (!motzkin_count) {
        std::cerr << "empty Motzkin input space\n";
        return 2;
    }

    std::vector<gp::MateID> inputs(INPUT_COUNT);
    std::uint64_t occupied_total = 0;
    for (std::uint32_t i = 0; i < INPUT_COUNT; ++i) {
        if (motzkin) {
            const Count64 rank = splitmix64(i) % motzkin_count;
            inputs[i] = make_motzkin_nn_mate(rank, motzkin_table);
        } else {
            inputs[i] = make_bernoulli_mate(
                i * 747796405u + 2891336453u, nonn_percent);
        }
        occupied_total += std::uint64_t(__builtin_popcount(
            gp::mate_non_n_mask(inputs[i], WIDTH) & ~std::uint32_t(3u)));
    }
    const double actual_nonn_fraction =
        double(occupied_total) / double(std::uint64_t(INPUT_COUNT) * FREE_WIDTH);

    gp::MateID* d_inputs = nullptr;
    std::uint64_t* d_output = nullptr;
    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    cuda_check(cudaMalloc(&d_inputs, inputs.size() * sizeof(gp::MateID)),
               "cudaMalloc(inputs)");
    cuda_check(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)),
               "cudaMalloc(output)");
    cuda_check(cudaMemcpy(
        d_inputs, inputs.data(), inputs.size() * sizeof(gp::MateID),
        cudaMemcpyHostToDevice), "cudaMemcpy(inputs)");

    for (int i = 0; i < warmup; ++i) {
        probe_kernel<<<blocks, threads>>>(d_inputs, d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    probe_kernel<<<blocks, threads>>>(d_inputs, d_output, iterations);
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
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
    std::cout << "gridfp-runtime-turn-discovery-nonn-scan-microprobe"
              << " nonn_scan=" << RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
              << " W=" << WIDTH
              << " input_case=" << input_case
              << " actual_nonn_fraction=" << actual_nonn_fraction
              << " motzkin_count=" << motzkin_count
              << " input_count=" << INPUT_COUNT
              << " blocks=" << blocks
              << " threads=" << threads
              << " iterations=" << iterations
              << " calls=" << calls
              << " kernel_ms=" << elapsed_ms
              << " ns_per_call=" << ns_per_call
              << " checksum=" << checksum << '\n';

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_output);
    cudaFree(d_inputs);
    return 0;
}
