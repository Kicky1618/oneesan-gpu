#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridfp_reduced_production_owner_component_plan_device.cuh"

#ifndef RP_PROBE_OWNER_PLAN_SHARED_CACHE
#define RP_PROBE_OWNER_PLAN_SHARED_CACHE 0
#endif
static_assert(RP_PROBE_OWNER_PLAN_SHARED_CACHE == 0 ||
              RP_PROBE_OWNER_PLAN_SHARED_CACHE == 1,
              "RP_PROBE_OWNER_PLAN_SHARED_CACHE must be 0 or 1");

namespace rp = oneesan::gridfp::reducedprod;

namespace {
void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(2);
    }
}

__global__ void owner_plan_probe_kernel(
    const rp::Rank64* __restrict__ prefix,
    const rp::Rank64* __restrict__ sr_begin,
    const rp::Rank64* __restrict__ group,
    const rp::Rank64* __restrict__ ranks,
    std::uint32_t rank_mask,
    std::uint64_t* __restrict__ output,
    int iterations
) {
#if RP_PROBE_OWNER_PLAN_SHARED_CACHE
    // W=28: prefix has O+2=15 entries, sr/group have O+1=14 each.
    __shared__ rp::Rank64 sh_plan[43];
    if (threadIdx.x < 43) {
        if (threadIdx.x < 15) {
            sh_plan[threadIdx.x] = prefix[threadIdx.x];
        } else if (threadIdx.x < 29) {
            sh_plan[threadIdx.x] = sr_begin[threadIdx.x - 15];
        } else {
            sh_plan[threadIdx.x] = group[threadIdx.x - 29];
        }
    }
    __syncthreads();
    const rp::Rank64* pfx = sh_plan;
    const rp::Rank64* sr = sh_plan + 15;
    const rp::Rank64* cg = sh_plan + 29;
#else
    const rp::Rank64* pfx = prefix;
    const rp::Rank64* sr = sr_begin;
    const rp::Rank64* cg = group;
#endif

    // Production label reconstruction is done by sublane 0 of each 8-lane
    // subgroup: 32 active label-unrank lanes per 256-thread block.
    if ((threadIdx.x & 7) != 0) return;
    const std::uint32_t subgroup_global =
        blockIdx.x * (blockDim.x >> 3) + (threadIdx.x >> 3);
    std::uint32_t ix = (subgroup_global * 747796405u + 2891336453u) & rank_mask;
    std::uint64_t acc = 0x9e3779b97f4a7c15ULL ^ subgroup_global;
    for (int i = 0; i < iterations; ++i) {
        const rp::Rank64 rank = ranks[ix];
        rp::Rank64 begin = 0;
        const int outer = rp::runtime_owner_prefix_sector_begin_device(
            pfx, 13, rank, begin);
        if (outer < 0) {
            acc ^= 0xd1b54a32d192ed03ULL;
        } else {
            const rp::Rank64 local = rank - begin;
            const rp::Rank64 s = sr[outer];
            const rp::Rank64 g = cg[outer];
            const std::uint64_t v =
                local ^ (s * 0x9e3779b97f4a7c15ULL) ^
                (g * 0xbf58476d1ce4e5b9ULL) ^ std::uint64_t(outer);
            acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
        }
        ix = (ix + 4051u) & rank_mask;
    }
    output[subgroup_global] = acc;
}
} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 128;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 2;
    if (blocks <= 0 || threads <= 0 || threads > 1024 ||
        (threads & 7) || iterations <= 0 || warmup < 0)
        return 2;

    constexpr std::array<std::uint32_t, 14> support_weights{
        1, 13, 78, 286, 715, 1287, 1716,
        1716, 1287, 715, 286, 78, 13, 1};
    // Use production-scale group values rather than unit widths. Exact values
    // are not material to the cache test; variation keeps dependent arithmetic
    // alive and makes all three plan arrays observable in the checksum.
    constexpr std::array<std::uint32_t, 14> group_values{
        170614, 300482, 534796, 960260, 1737232, 3163388, 5793184,
        10662569, 19712662, 36590252, 6912959, 23790549, 83022878, 510468519};

    std::array<rp::Rank64, 15> prefix{};
    std::array<rp::Rank64, 14> sr{};
    std::array<rp::Rank64, 14> group{};
    for (int r = 0; r < 14; ++r) {
        group[std::size_t(r)] = group_values[std::size_t(r)];
        sr[std::size_t(r)] = rp::Rank64(support_weights[std::size_t(r)] / 3u);
        const rp::Rank64 owned = 1u + support_weights[std::size_t(r)] / 4u;
        prefix[std::size_t(r + 1)] =
            prefix[std::size_t(r)] + owned * group[std::size_t(r)];
    }

    constexpr std::uint32_t RANK_COUNT = 1u << 16;
    std::vector<rp::Rank64> ranks(RANK_COUNT);
    std::uint64_t x = 0xd1b54a32d192ed03ULL;
    for (std::uint32_t i = 0; i < RANK_COUNT; ++i) {
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        ranks[i] = (x * 2685821657736338717ULL) % prefix.back();
    }

    rp::Rank64 *d_prefix = nullptr, *d_sr = nullptr, *d_group = nullptr, *d_ranks = nullptr;
    std::uint64_t* d_output = nullptr;
    const std::size_t active_per_block = std::size_t(threads >> 3);
    const std::size_t active = std::size_t(blocks) * active_per_block;
    cuda_check(cudaMalloc(&d_prefix, prefix.size() * sizeof(rp::Rank64)), "cudaMalloc(prefix)");
    cuda_check(cudaMalloc(&d_sr, sr.size() * sizeof(rp::Rank64)), "cudaMalloc(sr)");
    cuda_check(cudaMalloc(&d_group, group.size() * sizeof(rp::Rank64)), "cudaMalloc(group)");
    cuda_check(cudaMalloc(&d_ranks, ranks.size() * sizeof(rp::Rank64)), "cudaMalloc(ranks)");
    cuda_check(cudaMalloc(&d_output, active * sizeof(std::uint64_t)), "cudaMalloc(output)");
    cuda_check(cudaMemcpy(d_prefix, prefix.data(), prefix.size() * sizeof(rp::Rank64), cudaMemcpyHostToDevice), "copy prefix");
    cuda_check(cudaMemcpy(d_sr, sr.data(), sr.size() * sizeof(rp::Rank64), cudaMemcpyHostToDevice), "copy sr");
    cuda_check(cudaMemcpy(d_group, group.data(), group.size() * sizeof(rp::Rank64), cudaMemcpyHostToDevice), "copy group");
    cuda_check(cudaMemcpy(d_ranks, ranks.data(), ranks.size() * sizeof(rp::Rank64), cudaMemcpyHostToDevice), "copy ranks");

    for (int i = 0; i < warmup; ++i) {
        owner_plan_probe_kernel<<<blocks, threads>>>(
            d_prefix, d_sr, d_group, d_ranks, RANK_COUNT - 1, d_output, iterations);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "event start");
    cuda_check(cudaEventCreate(&stop), "event stop");
    cuda_check(cudaEventRecord(start), "record start");
    owner_plan_probe_kernel<<<blocks, threads>>>(
        d_prefix, d_sr, d_group, d_ranks, RANK_COUNT - 1, d_output, iterations);
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "record stop");
    cuda_check(cudaEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");

    std::vector<std::uint64_t> output(active);
    cuda_check(cudaMemcpy(output.data(), d_output, output.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost), "copy output");
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (std::uint64_t v : output) {
        checksum ^= v;
        checksum *= 0x100000001b3ULL;
    }

    const std::uint64_t calls = std::uint64_t(active) * std::uint64_t(iterations);
    const double ns_per_call = double(elapsed_ms) * 1.0e6 / double(calls);
    std::cout << "gridfp-runtime-owner-plan-shared-cache-microprobe"
              << " shared=" << RP_PROBE_OWNER_PLAN_SHARED_CACHE
              << " prefix_carry=" << RP_RUNTIME_OWNER_PREFIX_CARRY_BEGIN
              << " W=28 O=13 plan_entries=43 plan_bytes=344"
              << " blocks=" << blocks << " threads=" << threads
              << " iterations=" << iterations << " calls=" << calls
              << " kernel_ms=" << elapsed_ms << " ns_per_call=" << ns_per_call
              << " checksum=" << checksum << '\n';

    cudaEventDestroy(stop); cudaEventDestroy(start);
    cudaFree(d_output); cudaFree(d_ranks); cudaFree(d_group); cudaFree(d_sr); cudaFree(d_prefix);
    return 0;
}
