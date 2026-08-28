#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_microprobe_main_unused
#include "gridfp_reduced_production_shift_cycle_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_grouped_support_device.cuh"

namespace {

__global__ void grouped_support_rank_equivalence_kernel(
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* __restrict__ checked,
    int* error
) {
    unsigned long long local_checked = 0;
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const DeviceKey key = equal_run_key0_device(
                run.support, blocked, W, q, reverse);
            const GroupedDeviceRank reference = grouped_rank_device(
                key, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
            const GroupedDeviceRank support_only = grouped_support_slab_rank_device(
                run.support, blocked, W, q, reverse, old_start, Kwin,
                ngpu, owner_begin);
            if (reference.owner != support_only.owner ||
                reference.local != support_only.local) {
                set_error(error, 201);
            }
            ++local_checked;
        }
    }
    if (local_checked) atomicAdd(checked, local_checked);
}

void run_grouped_support_rank_equivalence(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    ck(cudaSetDevice(0), "support rank set device");
    install_tables(tables);

    Rank64* d_owner_begin = nullptr;
    unsigned long long* d_checked = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)),
       "support rank alloc owner begin");
    ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                  cudaMemcpyHostToDevice), "support rank copy owner begin");
    ck(cudaMalloc(&d_checked, sizeof(unsigned long long)),
       "support rank alloc checked");
    ck(cudaMalloc(&d_error, sizeof(int)), "support rank alloc error");
    ck(cudaMemset(d_checked, 0, sizeof(unsigned long long)),
       "support rank zero checked");
    ck(cudaMemset(d_error, 0, sizeof(int)), "support rank zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 needed_blocks =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed_blocks)));

    grouped_support_rank_equivalence_kernel<<<blocks, THREADS>>>(
        base_supports, W, Kwin, S, reverse, ngpu, d_owner_begin,
        d_checked, d_error);
    ck(cudaGetLastError(), "support rank launch");
    ck(cudaDeviceSynchronize(), "support rank sync");

    int error = 0;
    unsigned long long checked = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "support rank copy error");
    ck(cudaMemcpy(&checked, d_checked, sizeof(checked), cudaMemcpyDeviceToHost),
       "support rank copy checked");
    if (error) fail("support-only grouped slab rank mismatch");

    std::cout << "gridfp-grouped-support-rank-proof"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " checked_seeds=" << checked
              << " key0_rank_equivalent=1"
              << " mate_materialization_elidable=1 exact=OK\n";

    cudaFree(d_error);
    cudaFree(d_checked);
    cudaFree(d_owner_begin);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 11;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const unsigned blocks = argc > 3
        ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 6 || W > 16 || Kwin < 1 || !blocks ||
        ngpu < 2 || ngpu > P2P_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "support rank device count");
    if (visible < 1) return 3;

    int cases = 0;
    for (int S = 1; S <= Kwin && Kwin + S + 2 <= W; ++S) {
        run_grouped_support_rank_equivalence(
            W, Kwin, S, false, ngpu, blocks);
        run_grouped_support_rank_equivalence(
            W, Kwin, S, true, ngpu, blocks);
        cases += 2;
    }
    if (!cases) return 4;
    std::cout << "ALL_OK grouped_support_slab_rank=1 cases=" << cases << '\n';
    return 0;
}
