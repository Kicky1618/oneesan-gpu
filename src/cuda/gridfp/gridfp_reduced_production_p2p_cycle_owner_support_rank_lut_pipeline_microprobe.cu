#include "gridfp_reduced_production_grouped_support_device.cuh"

namespace oneesan::gridfp::reducedprod {

static constexpr int SUPPORT_RANK_LUT_MAX_W = 11;
static constexpr int SUPPORT_RANK_LUT_SIZE = 1 << SUPPORT_RANK_LUT_MAX_W;

__device__ std::uint8_t SUPPORT_RANK_LUT_OWNER[2][SUPPORT_RANK_LUT_SIZE];
__device__ Rank64 SUPPORT_RANK_LUT_LOCAL[2][SUPPORT_RANK_LUT_SIZE];

__device__ __forceinline__ GroupedDeviceRank support_rank_lut_lookup_device(
    std::uint32_t full_support,
    bool blocked,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int ngpu,
    const Rank64* owner_begin
) {
    (void)W;
    (void)q;
    (void)reverse;
    (void)tile_start;
    (void)K;
    (void)ngpu;
    (void)owner_begin;
    if (full_support >= SUPPORT_RANK_LUT_SIZE)
        return GroupedDeviceRank{-1, 0};
    const int b = blocked ? 1 : 0;
    const int owner = int(__ldg(&SUPPORT_RANK_LUT_OWNER[b][full_support]));
    if (owner == 0xff) return GroupedDeviceRank{-1, 0};
    const Rank64 local = __ldg(&SUPPORT_RANK_LUT_LOCAL[b][full_support]);
    return GroupedDeviceRank{owner, local};
}

} // namespace oneesan::gridfp::reducedprod

#define grouped_support_slab_rank_device support_rank_lut_lookup_device
#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_precomputed_dst_route_pipeline_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_precomputed_dst_route_pipeline_microprobe.cu"
#pragma pop_macro("main")
#undef grouped_support_slab_rank_device

namespace {

using oneesan::gridfp::reducedprod::SUPPORT_RANK_LUT_LOCAL;
using oneesan::gridfp::reducedprod::SUPPORT_RANK_LUT_MAX_W;
using oneesan::gridfp::reducedprod::SUPPORT_RANK_LUT_OWNER;

__global__ void build_support_rank_lut_kernel(
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int ngpu,
    const Rank64* __restrict__ owner_begin
) {
    const unsigned limit = 1u << W;
    const unsigned linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= 2u * limit) return;
    const bool blocked = linear >= limit;
    const std::uint32_t support = linear & (limit - 1u);

    const GroupedDeviceRank gr = grouped_support_slab_rank_device(
        support, blocked, W, q, reverse, tile_start,
        Kwin, ngpu, owner_begin);
    if (gr.owner < 0 || gr.owner >= ngpu) {
        SUPPORT_RANK_LUT_OWNER[blocked ? 1 : 0][support] = 0xff;
        SUPPORT_RANK_LUT_LOCAL[blocked ? 1 : 0][support] = 0;
        return;
    }
    SUPPORT_RANK_LUT_OWNER[blocked ? 1 : 0][support] =
        static_cast<std::uint8_t>(gr.owner);
    SUPPORT_RANK_LUT_LOCAL[blocked ? 1 : 0][support] = gr.local;
}

double install_support_rank_lut(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu
) {
    if (W < 1 || W > SUPPORT_RANK_LUT_MAX_W)
        fail("support rank LUT width");

    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);
    const unsigned count = 2u * (1u << W);
    const unsigned threads = 256;
    const unsigned blocks = (count + threads - 1) / threads;

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "support rank LUT set device");
        install_tables(tables);
        Rank64* d_owner_begin = nullptr;
        ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)),
           "support rank LUT alloc owner begin");
        ck(cudaMemcpy(
               d_owner_begin, plan.owner_begin.data(),
               ngpu * sizeof(Rank64), cudaMemcpyHostToDevice),
           "support rank LUT copy owner begin");
        build_support_rank_lut_kernel<<<blocks, threads>>>(
            W, q, reverse, tile_start, Kwin, ngpu, d_owner_begin);
        ck(cudaGetLastError(), "support rank LUT build launch");
        ck(cudaDeviceSynchronize(), "support rank LUT build sync");
        cudaFree(d_owner_begin);
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 4;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 16;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 256u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;

    if (W < 7 || W > SUPPORT_RANK_LUT_MAX_W || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W ||
        batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) {
        return 2;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible),
       "support rank LUT cycle-owner pipeline device count");
    if (visible < ngpu) return 3;

    const double forward_lut_ms =
        install_support_rank_lut(W, Kwin, S, false, ngpu);
    std::cout << "gridfp-support-rank-lut"
              << " direction=forward"
              << " entries=" << (2u * (1u << W))
              << " bytes_per_gpu="
              << (2ULL * (1ULL << SUPPORT_RANK_LUT_MAX_W) *
                  (sizeof(std::uint8_t) + sizeof(Rank64)))
              << " build_ms=" << forward_lut_ms
              << " runtime_support_rank_combinadics=0"
              << " support_rank_lut=1\n";
    run_precomputed_dst_route_cycle_owner_pipeline(
        W, Kwin, S, false, ngpu, batches, blocks);

    const double reverse_lut_ms =
        install_support_rank_lut(W, Kwin, S, true, ngpu);
    std::cout << "gridfp-support-rank-lut"
              << " direction=reverse"
              << " entries=" << (2u * (1u << W))
              << " bytes_per_gpu="
              << (2ULL * (1ULL << SUPPORT_RANK_LUT_MAX_W) *
                  (sizeof(std::uint8_t) + sizeof(Rank64)))
              << " build_ms=" << reverse_lut_ms
              << " runtime_support_rank_combinadics=0"
              << " support_rank_lut=1\n";
    run_precomputed_dst_route_cycle_owner_pipeline(
        W, Kwin, S, true, ngpu, batches, blocks);

    std::cout
        << "ALL_OK gridfp_p2p_cycle_owner_support_rank_lut_pipeline=1"
        << " phase_a_support_rank_lut=1"
        << " phase_b_support_rank_lut=1"
        << " runtime_support_rank_combinadics=0"
        << " destination_support_precomputed=1"
        << " runtime_destination_support_shift_steps=0\n";
    return 0;
}
