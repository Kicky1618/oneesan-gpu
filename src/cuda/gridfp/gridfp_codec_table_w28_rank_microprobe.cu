#include <cuda_runtime.h>

#pragma push_macro("main")
#undef main
#define main gridfp_owner_subwarp_main_unused_for_codec_table_rank_probe
#include "gridfp_reduced_production_owner_subwarp_microprobe.cu"
#pragma pop_macro("main")

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef RP_RUNTIME_CODEC_CHOOSE_U32_MODE
#define RP_RUNTIME_CODEC_CHOOSE_U32_MODE 0
#endif
#ifndef RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE
#define RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE 0
#endif

namespace {

__device__ __forceinline__ std::uint64_t rank_probe_mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void codec_table_rank_sample_kernel(
    std::uint64_t* __restrict__ output,
    Rank64 local_components,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int gpu_id,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ component_prefix,
    const Rank64* __restrict__ component_sr_begin,
    const Rank64* __restrict__ component_group,
    int iterations,
    int* error
) {
    const std::uint64_t tid =
        std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const OwnerComponentPlanDevice plan{
        component_prefix, component_sr_begin, component_group};
    std::uint64_t acc = 0x6a09e667f3bcc909ULL ^ tid;
    for (int i = 0; i < iterations; ++i) {
        const std::uint64_t x = rank_probe_mix64(
            tid ^ (std::uint64_t(i + 1) * 0xd1b54a32d192ed03ULL));
        const Rank64 local_rank = __umul64hi(x, local_components);
        const MateID label = owner_component_label_unrank_planned_device(
            W, q, reverse, tile_start, K, plan, local_rank);
        bool eligible = false;
        const DeviceKey seed = component_seed_direction(
            label, W, q, reverse, eligible);
        if (!eligible) {
            atomicCAS(error, 0, 1);
            continue;
        }
        const GroupedComponentContextDevice ctx = grouped_component_context_device(
            seed, W, q, reverse, tile_start, K, ngpu, owner_begin);
        const GroupedDeviceRank gr = grouped_rank_in_component_device(
            seed, W, q, reverse, ctx);
        if (ctx.owner != gpu_id || gr.owner != gpu_id ||
            gr.local >= local_components * Rank64(64)) {
            atomicCAS(error, 0, 2);
        }
        acc ^= rank_probe_mix64(
            std::uint64_t(label) ^ (std::uint64_t(gr.owner + 1) << 56) ^
            std::uint64_t(gr.local) ^ std::uint64_t(ctx.local_group_base));
    }
    output[tid] = acc;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int K = argc > 2 ? std::atoi(argv[2]) : 13;
    const int logical_ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int owner = argc > 4 ? std::atoi(argv[4]) : 0;
    const int reverse_i = argc > 5 ? std::atoi(argv[5]) : 0;
    const int blocks = argc > 6 ? std::atoi(argv[6]) : 2048;
    const int threads = argc > 7 ? std::atoi(argv[7]) : 256;
    const int iterations = argc > 8 ? std::atoi(argv[8]) : 64;
    const int warmup = argc > 9 ? std::atoi(argv[9]) : 2;
    if (W != 28 || K != 13 || logical_ngpu < 2 || logical_ngpu > 16 ||
        owner < 0 || owner >= logical_ngpu || (reverse_i != 0 && reverse_i != 1) ||
        blocks < 1 || threads < 1 || threads > 1024 || iterations < 1 || warmup < 0) {
        std::cerr << "usage: probe [W=28] [K=13] [logical_ngpu=8] [owner=0..n-1] "
                     "[reverse=0|1] [blocks] [threads] [iterations] [warmup]\n";
        return 2;
    }
    const bool reverse = reverse_i != 0;
    const int q = reverse ? 1 : W - 1;
    const int tile_start = reverse ? 1 : W - 1;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "codec rank device count");
    if (visible < 1) return 3;
    ck(cudaSetDevice(0), "codec rank set device0");

    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan tile = make_host_tile_plan(tables, K, logical_ngpu);
    const HostOwnerComponentPlan hp =
        make_host_owner_component_plan(tables, K, owner, logical_ngpu);
    const Rank64 local_components = hp.prefix.back();
    if (!local_components || tile.owner_begin.size() != std::size_t(logical_ngpu))
        return 4;

    Rank64 *d_owner_begin = nullptr, *d_prefix = nullptr;
    Rank64 *d_sr_begin = nullptr, *d_component_group = nullptr;
    std::uint64_t* d_output = nullptr;
    int* d_error = nullptr;
    const std::size_t nthreads = std::size_t(blocks) * std::size_t(threads);
    ck(cudaMalloc(&d_owner_begin, tile.owner_begin.size() * sizeof(Rank64)),
       "codec rank alloc owner begin");
    ck(cudaMalloc(&d_prefix, hp.prefix.size() * sizeof(Rank64)),
       "codec rank alloc prefix");
    ck(cudaMalloc(&d_sr_begin, hp.sr_begin.size() * sizeof(Rank64)),
       "codec rank alloc sr begin");
    ck(cudaMalloc(&d_component_group, hp.component_group.size() * sizeof(Rank64)),
       "codec rank alloc component group");
    ck(cudaMalloc(&d_output, nthreads * sizeof(std::uint64_t)),
       "codec rank alloc output");
    ck(cudaMalloc(&d_error, sizeof(int)), "codec rank alloc error");
    ck(cudaMemcpy(d_owner_begin, tile.owner_begin.data(),
                  tile.owner_begin.size() * sizeof(Rank64), cudaMemcpyHostToDevice),
       "codec rank copy owner begin");
    ck(cudaMemcpy(d_prefix, hp.prefix.data(),
                  hp.prefix.size() * sizeof(Rank64), cudaMemcpyHostToDevice),
       "codec rank copy prefix");
    ck(cudaMemcpy(d_sr_begin, hp.sr_begin.data(),
                  hp.sr_begin.size() * sizeof(Rank64), cudaMemcpyHostToDevice),
       "codec rank copy sr begin");
    ck(cudaMemcpy(d_component_group, hp.component_group.data(),
                  hp.component_group.size() * sizeof(Rank64), cudaMemcpyHostToDevice),
       "codec rank copy component group");

    auto launch = [&] {
        ck(cudaMemset(d_error, 0, sizeof(int)), "codec rank zero error");
        codec_table_rank_sample_kernel<<<blocks, threads>>>(
            d_output, local_components, W, q, reverse, tile_start, K,
            owner, logical_ngpu, d_owner_begin, d_prefix, d_sr_begin,
            d_component_group, iterations, d_error);
        ck(cudaGetLastError(), "codec rank launch");
    };
    for (int i = 0; i < warmup; ++i) launch();
    ck(cudaDeviceSynchronize(), "codec rank warmup sync");

    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "codec rank event a");
    ck(cudaEventCreate(&b), "codec rank event b");
    ck(cudaEventRecord(a), "codec rank record a");
    launch();
    ck(cudaEventRecord(b), "codec rank record b");
    ck(cudaEventSynchronize(b), "codec rank timed sync");
    float kernel_ms = 0.0f;
    ck(cudaEventElapsedTime(&kernel_ms, a, b), "codec rank elapsed");

    int error = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "codec rank copy error");
    if (error) {
        std::cerr << "codec rank device error=" << error << '\n';
        return 5;
    }
    std::vector<std::uint64_t> output(nthreads);
    ck(cudaMemcpy(output.data(), d_output, output.size() * sizeof(std::uint64_t),
                  cudaMemcpyDeviceToHost), "codec rank copy output");
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (std::uint64_t v : output) {
        checksum ^= v;
        checksum *= 0x100000001b3ULL;
    }
    const std::uint64_t samples =
        std::uint64_t(blocks) * std::uint64_t(threads) *
        std::uint64_t(iterations);
    const double ns_per_sample =
        double(kernel_ms) * 1.0e6 / double(samples);
    std::cout << "gridfp-codec-table-w28-rank-microprobe"
              << " W=" << W << " K=" << K
              << " logical_ngpu=" << logical_ngpu
              << " owner=" << owner
              << " direction=" << (reverse ? "reverse" : "forward")
              << " choose_mode=" << RP_RUNTIME_CODEC_CHOOSE_U32_MODE
              << " primitive_mode=" << RP_RUNTIME_CODEC_PRIMITIVE_U32_MODE
              << " local_components=" << local_components
              << " blocks=" << blocks << " threads=" << threads
              << " iterations=" << iterations
              << " samples=" << samples
              << " kernel_ms=" << kernel_ms
              << " ns_per_sample=" << ns_per_sample
              << " checksum=" << checksum
              << " error=0\n";

    cudaEventDestroy(b); cudaEventDestroy(a);
    cudaFree(d_error); cudaFree(d_output); cudaFree(d_component_group);
    cudaFree(d_sr_begin); cudaFree(d_prefix); cudaFree(d_owner_begin);
    return 0;
}
