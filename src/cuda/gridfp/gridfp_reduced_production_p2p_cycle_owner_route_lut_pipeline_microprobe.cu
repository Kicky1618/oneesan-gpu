#include "gridfp_reduced_production_shift_cycle_device.cuh"

namespace oneesan::gridfp::reducedprod {

static constexpr int NEXT_SUPPORT_LUT_MAX_W = 11;
static constexpr int NEXT_SUPPORT_LUT_SIZE = 1 << NEXT_SUPPORT_LUT_MAX_W;

__device__ std::uint32_t NEXT_SUPPORT_LUT[2][NEXT_SUPPORT_LUT_SIZE];

__device__ __forceinline__ std::uint32_t next_support_lut_lookup_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    (void)W;
    (void)q;
    (void)Kwin;
    (void)S;
    (void)reverse;
    if (support >= NEXT_SUPPORT_LUT_SIZE) return 0xffffffffu;
    return __ldg(&NEXT_SUPPORT_LUT[blocked ? 1 : 0][support]);
}

} // namespace oneesan::gridfp::reducedprod

#define shift_next_support_device next_support_lut_lookup_device
#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_support_rank_lut_pipeline_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_support_rank_lut_pipeline_microprobe.cu"
#pragma pop_macro("main")
#undef shift_next_support_device

namespace {

using oneesan::gridfp::reducedprod::NEXT_SUPPORT_LUT;
using oneesan::gridfp::reducedprod::NEXT_SUPPORT_LUT_MAX_W;
using oneesan::gridfp::reducedprod::NEXT_SUPPORT_LUT_SIZE;

__global__ void build_next_support_lut_kernel(
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const unsigned limit = 1u << W;
    const unsigned linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= 2u * limit) return;
    const bool blocked = linear >= limit;
    const std::uint32_t support = linear & (limit - 1u);
    NEXT_SUPPORT_LUT[blocked ? 1 : 0][support] =
        shift_next_support_device(
            support, blocked, W, q, Kwin, S, reverse);
}

double install_next_support_lut(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu
) {
    if (W < 1 || W > NEXT_SUPPORT_LUT_MAX_W)
        fail("next support LUT width");

    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);
    const unsigned count = 2u * (1u << W);
    const unsigned threads = 256;
    const unsigned blocks = (count + threads - 1) / threads;

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "next support LUT set device");
        build_next_support_lut_kernel<<<blocks, threads>>>(
            W, q, Kwin, S, reverse);
        ck(cudaGetLastError(), "next support LUT build launch");
        ck(cudaDeviceSynchronize(), "next support LUT build sync");
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

void print_route_lut_stats(
    int W,
    const char* direction,
    double rank_ms,
    double next_ms
) {
    const unsigned long long rank_bytes =
        2ULL * (1ULL << SUPPORT_RANK_LUT_MAX_W) *
        (sizeof(std::uint8_t) + sizeof(Rank64));
    const unsigned long long next_bytes =
        2ULL * NEXT_SUPPORT_LUT_SIZE * sizeof(std::uint32_t);
    std::cout << "gridfp-route-lut"
              << " direction=" << direction
              << " rank_entries=" << (2u * (1u << W))
              << " next_entries=" << (2u * (1u << W))
              << " rank_bytes_per_gpu=" << rank_bytes
              << " next_bytes_per_gpu=" << next_bytes
              << " total_lut_bytes_per_gpu=" << (rank_bytes + next_bytes)
              << " rank_build_ms=" << rank_ms
              << " next_build_ms=" << next_ms
              << " runtime_support_rank_combinadics=0"
              << " cross_route_shift_next_bitops=0"
              << " local_cycle_legacy_prev_length=1"
              << " support_rank_lut=1"
              << " next_support_lut=1\n";
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

    if (W < 7 || W > NEXT_SUPPORT_LUT_MAX_W || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W ||
        batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) {
        return 2;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible),
       "route LUT cycle-owner pipeline device count");
    if (visible < ngpu) return 3;

    const double forward_rank_ms =
        install_support_rank_lut(W, Kwin, S, false, ngpu);
    const double forward_next_ms =
        install_next_support_lut(W, Kwin, S, false, ngpu);
    print_route_lut_stats(
        W, "forward", forward_rank_ms, forward_next_ms);
    run_precomputed_dst_route_cycle_owner_pipeline(
        W, Kwin, S, false, ngpu, batches, blocks);

    const double reverse_rank_ms =
        install_support_rank_lut(W, Kwin, S, true, ngpu);
    const double reverse_next_ms =
        install_next_support_lut(W, Kwin, S, true, ngpu);
    print_route_lut_stats(
        W, "reverse", reverse_rank_ms, reverse_next_ms);
    run_precomputed_dst_route_cycle_owner_pipeline(
        W, Kwin, S, true, ngpu, batches, blocks);

    std::cout
        << "ALL_OK gridfp_p2p_cycle_owner_route_lut_pipeline=1"
        << " phase_a_support_rank_lut=1"
        << " phase_b_support_rank_lut=1"
        << " runtime_support_rank_combinadics=0"
        << " next_support_lut=1"
        << " cross_route_shift_next_bitops=0"
        << " local_cycle_legacy_prev_length=1"
        << " destination_support_precomputed=1"
        << " runtime_destination_support_shift_steps=0\n";
    return 0;
}
