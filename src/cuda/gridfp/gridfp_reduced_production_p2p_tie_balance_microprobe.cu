#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_ownerfirst_ab_microprobe_main_unused
#include "gridfp_reduced_production_p2p_ownerfirst_ab_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void p2p_tie_balance_stats_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    unsigned long long* __restrict__ low_cycles,
    unsigned long long* __restrict__ hash_cycles,
    unsigned long long* __restrict__ low_work,
    unsigned long long* __restrict__ hash_work,
    unsigned long long* __restrict__ tied_cycles,
    unsigned long long* __restrict__ remote_ops,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) {
                atomicCAS(error, 0, 371);
                continue;
            }
            if (len <= 1) continue;

            int counts[P2P_MAX_GPU]{};
            std::uint32_t cur = run.support;
            for (int h = 0; h < len; ++h) {
                const int owner = p2p_support_owner_device(
                    cur, W, old_start, K, reverse, ngpu);
                if (owner < 0 || owner >= ngpu) {
                    atomicCAS(error, 0, 372);
                    break;
                }
                ++counts[owner];
                cur = shift_next_support_device(
                    cur, blocked, W, q, K, K, reverse);
            }

            const int low = p2p_pick_modal_owner_device(
                counts, ngpu, run.support, blocked, reverse, false);
            const int hash = p2p_pick_modal_owner_device(
                counts, ngpu, run.support, blocked, reverse, true);
            if (low < 0 || hash < 0 || low >= ngpu || hash >= ngpu ||
                counts[low] != counts[hash]) {
                atomicCAS(error, 0, 373);
                continue;
            }
            int max_count = counts[low];
            int nties = 0;
            for (int g = 0; g < ngpu; ++g)
                nties += counts[g] == max_count;
            if (nties > 1) atomicAdd(tied_cycles, 1ULL);

            const Rank64 pc = RP_PRIMITIVE[__popc(run.support)][1];
            const unsigned long long work =
                static_cast<unsigned long long>(len) *
                static_cast<unsigned long long>(pc);
            const unsigned long long remote =
                2ULL * static_cast<unsigned long long>(len - max_count) *
                static_cast<unsigned long long>(pc);
            atomicAdd(low_cycles + low, 1ULL);
            atomicAdd(hash_cycles + hash, 1ULL);
            atomicAdd(low_work + low, work);
            atomicAdd(hash_work + hash, work);
            atomicAdd(remote_ops, remote);
        }
    }
}

void run_tie_balance_device_probe(
    int W,
    bool reverse,
    int ngpu,
    unsigned blocks
) {
    const int K = (W - 2) / 2;
    ProductionFactorTables tables(W);
    install_tables(tables);

    unsigned long long *d_low_cycles = nullptr, *d_hash_cycles = nullptr;
    unsigned long long *d_low_work = nullptr, *d_hash_work = nullptr;
    unsigned long long *d_tied = nullptr, *d_remote = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_low_cycles, ngpu * sizeof(unsigned long long)), "tie alloc low cycles");
    ck(cudaMalloc(&d_hash_cycles, ngpu * sizeof(unsigned long long)), "tie alloc hash cycles");
    ck(cudaMalloc(&d_low_work, ngpu * sizeof(unsigned long long)), "tie alloc low work");
    ck(cudaMalloc(&d_hash_work, ngpu * sizeof(unsigned long long)), "tie alloc hash work");
    ck(cudaMalloc(&d_tied, sizeof(unsigned long long)), "tie alloc tied");
    ck(cudaMalloc(&d_remote, sizeof(unsigned long long)), "tie alloc remote");
    ck(cudaMalloc(&d_error, sizeof(int)), "tie alloc error");
    ck(cudaMemset(d_low_cycles, 0, ngpu * sizeof(unsigned long long)), "tie zero low cycles");
    ck(cudaMemset(d_hash_cycles, 0, ngpu * sizeof(unsigned long long)), "tie zero hash cycles");
    ck(cudaMemset(d_low_work, 0, ngpu * sizeof(unsigned long long)), "tie zero low work");
    ck(cudaMemset(d_hash_work, 0, ngpu * sizeof(unsigned long long)), "tie zero hash work");
    ck(cudaMemset(d_tied, 0, sizeof(unsigned long long)), "tie zero tied");
    ck(cudaMemset(d_remote, 0, sizeof(unsigned long long)), "tie zero remote");
    ck(cudaMemset(d_error, 0, sizeof(int)), "tie zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const unsigned threads = 256;
    const Rank64 one_pass = (base_supports + threads - 1) / threads;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
    const auto t0 = std::chrono::steady_clock::now();
    p2p_tie_balance_stats_kernel<<<launch_blocks, threads>>>(
        base_supports, W, K, reverse, ngpu,
        d_low_cycles, d_hash_cycles, d_low_work, d_hash_work,
        d_tied, d_remote, d_error);
    ck(cudaGetLastError(), "tie stats launch");
    ck(cudaDeviceSynchronize(), "tie stats sync");
    const double ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<unsigned long long> low_cycles(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> hash_cycles(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> low_work(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> hash_work(static_cast<std::size_t>(ngpu));
    unsigned long long tied = 0, remote = 0;
    int error = 0;
    ck(cudaMemcpy(low_cycles.data(), d_low_cycles,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "tie copy low cycles");
    ck(cudaMemcpy(hash_cycles.data(), d_hash_cycles,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "tie copy hash cycles");
    ck(cudaMemcpy(low_work.data(), d_low_work,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "tie copy low work");
    ck(cudaMemcpy(hash_work.data(), d_hash_work,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "tie copy hash work");
    ck(cudaMemcpy(&tied, d_tied, sizeof(tied), cudaMemcpyDeviceToHost), "tie copy tied");
    ck(cudaMemcpy(&remote, d_remote, sizeof(remote), cudaMemcpyDeviceToHost), "tie copy remote");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "tie copy error");
    if (error) fail("tie-balance device error=" + std::to_string(error));

    unsigned long long low_total_cycles = 0, hash_total_cycles = 0;
    unsigned long long low_total_work = 0, hash_total_work = 0;
    unsigned long long low_min = ~0ULL, low_max = 0;
    unsigned long long hash_min = ~0ULL, hash_max = 0;
    for (int g = 0; g < ngpu; ++g) {
        low_total_cycles += low_cycles[static_cast<std::size_t>(g)];
        hash_total_cycles += hash_cycles[static_cast<std::size_t>(g)];
        low_total_work += low_work[static_cast<std::size_t>(g)];
        hash_total_work += hash_work[static_cast<std::size_t>(g)];
        low_min = std::min(low_min, low_work[static_cast<std::size_t>(g)]);
        low_max = std::max(low_max, low_work[static_cast<std::size_t>(g)]);
        hash_min = std::min(hash_min, hash_work[static_cast<std::size_t>(g)]);
        hash_max = std::max(hash_max, hash_work[static_cast<std::size_t>(g)]);
    }
    if (low_total_cycles != hash_total_cycles || low_total_work != hash_total_work)
        fail("tie-balance totals changed");

    std::cout << "gridfp-reduced-production-p2p-tie-balance"
              << " W=" << W << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " cycles=" << low_total_cycles
              << " tied_cycles=" << tied
              << " tied_fraction="
              << (low_total_cycles ? double(tied) / double(low_total_cycles) : 0.0)
              << " low_work_spread=" << (low_max - low_min)
              << " hash_work_spread=" << (hash_max - hash_min)
              << " low_max_work=" << low_max
              << " hash_max_work=" << hash_max
              << " remote_u32_ops=" << remote
              << " remote_traffic_identical_by_modal_tie=1"
              << " wall_ms=" << ms
              << " exact_modal_constraint=OK\n";

    cudaFree(d_error); cudaFree(d_remote); cudaFree(d_tied);
    cudaFree(d_hash_work); cudaFree(d_low_work);
    cudaFree(d_hash_cycles); cudaFree(d_low_cycles);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    const unsigned blocks = argc > 3
        ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    if (W < 8 || W > 16 || (W & 1) || ngpu < 2 || ngpu > P2P_MAX_GPU || !blocks)
        return 2;
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "tie device count");
    if (visible < 1) return 3;
    ck(cudaSetDevice(0), "tie set device");
    run_tie_balance_device_probe(W, false, ngpu, blocks);
    run_tie_balance_device_probe(W, true, ngpu, blocks);
    std::cout << "ALL_OK production_p2p_modal_tie_balance_cuda=1\n";
    return 0;
}
