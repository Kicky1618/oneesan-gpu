#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_grouped_support_device.cuh"

namespace {

__global__ void p2p_cycle_traffic_kernel(
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned long long* __restrict__ cycles,
    unsigned long long* __restrict__ rotated_values,
    unsigned long long* __restrict__ cross_values,
    unsigned long long* __restrict__ remote_position_values,
    int* error
) {
    unsigned long long local_cycles = 0;
    unsigned long long local_rotated = 0;
    unsigned long long local_cross = 0;
    unsigned long long local_remote = 0;

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
            const int cycle_len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            if (cycle_len < 0) {
                set_error(error, 191);
                continue;
            }
            if (cycle_len <= 1) continue;

            // Traffic depends only on the physical support mask.  In
            // particular, no MateID materialization, primitive rank, or local
            // grouped offset is needed here.
            const int leader_owner = grouped_support_owner_device(
                run.support, W, reverse, old_start, Kwin, ngpu);
            if (leader_owner < 0 || leader_owner >= ngpu) {
                set_error(error, 192);
                continue;
            }

            int prev_owner = leader_owner;
            unsigned cross_edges = 0;
            unsigned remote_positions = 0;
            std::uint32_t cur_support = shift_next_support_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            int hops = 1;
            while (cur_support != run.support && hops < cycle_len) {
                const int owner = grouped_support_owner_device(
                    cur_support, W, reverse, old_start, Kwin, ngpu);
                if (owner < 0 || owner >= ngpu) {
                    set_error(error, 193);
                    break;
                }
                cross_edges += owner != prev_owner;
                remote_positions += owner != leader_owner;
                prev_owner = owner;
                cur_support = shift_next_support_device(
                    cur_support, blocked, W, q, Kwin, S, reverse);
                ++hops;
            }
            if (hops != cycle_len || cur_support != run.support) {
                set_error(error, 194);
                continue;
            }
            cross_edges += prev_owner != leader_owner;

            const int occupied = __popc(run.support);
            const unsigned long long pc = RP_PRIMITIVE[occupied][1];
            ++local_cycles;
            local_rotated += pc * static_cast<unsigned long long>(cycle_len);
            local_cross += pc * static_cast<unsigned long long>(cross_edges);
            local_remote += pc * static_cast<unsigned long long>(remote_positions);
        }
    }

    if (local_cycles) atomicAdd(cycles, local_cycles);
    if (local_rotated) atomicAdd(rotated_values, local_rotated);
    if (local_cross) atomicAdd(cross_values, local_cross);
    if (local_remote) atomicAdd(remote_position_values, local_remote);
}

void run_p2p_traffic_probe(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);

    ck(cudaSetDevice(0), "traffic set device");
    install_tables(tables);

    unsigned long long* d_cycles = nullptr;
    unsigned long long* d_rotated = nullptr;
    unsigned long long* d_cross = nullptr;
    unsigned long long* d_remote = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_cycles, sizeof(unsigned long long)), "traffic alloc cycles");
    ck(cudaMalloc(&d_rotated, sizeof(unsigned long long)), "traffic alloc rotated");
    ck(cudaMalloc(&d_cross, sizeof(unsigned long long)), "traffic alloc cross");
    ck(cudaMalloc(&d_remote, sizeof(unsigned long long)), "traffic alloc remote");
    ck(cudaMalloc(&d_error, sizeof(int)), "traffic alloc error");
    ck(cudaMemset(d_cycles, 0, sizeof(unsigned long long)), "traffic zero cycles");
    ck(cudaMemset(d_rotated, 0, sizeof(unsigned long long)), "traffic zero rotated");
    ck(cudaMemset(d_cross, 0, sizeof(unsigned long long)), "traffic zero cross");
    ck(cudaMemset(d_remote, 0, sizeof(unsigned long long)), "traffic zero remote");
    ck(cudaMemset(d_error, 0, sizeof(int)), "traffic zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 needed_blocks =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed_blocks)));

    const auto t0 = std::chrono::steady_clock::now();
    p2p_cycle_traffic_kernel<<<blocks, THREADS>>>(
        base_supports, W, Kwin, S, reverse, ngpu,
        d_cycles, d_rotated, d_cross, d_remote, d_error);
    ck(cudaGetLastError(), "traffic launch");
    ck(cudaDeviceSynchronize(), "traffic sync");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int error = 0;
    unsigned long long cycles = 0;
    unsigned long long rotated = 0;
    unsigned long long cross = 0;
    unsigned long long remote = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "traffic copy error");
    ck(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost),
       "traffic copy cycles");
    ck(cudaMemcpy(&rotated, d_rotated, sizeof(rotated), cudaMemcpyDeviceToHost),
       "traffic copy rotated");
    ck(cudaMemcpy(&cross, d_cross, sizeof(cross), cudaMemcpyDeviceToHost),
       "traffic copy cross");
    ck(cudaMemcpy(&remote, d_remote, sizeof(remote), cudaMemcpyDeviceToHost),
       "traffic copy remote");
    if (error) fail("p2p traffic device error=" + std::to_string(error));

    const unsigned long long direct_remote_ops = 2ULL * remote;
    const double logical_gib = double(cross) * 4.0 / double(1ULL << 30);
    const double direct_gib = double(direct_remote_ops) * 4.0 / double(1ULL << 30);
    const double overhead = cross ? double(direct_remote_ops) / double(cross) : 0.0;
    const double support_gs = ms > 0.0
        ? double(base_supports) / (ms * 1.0e6) : 0.0;

    std::cout << "gridfp-reduced-production-p2p-traffic"
              << " W=" << W << " Kwin=" << Kwin << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " base_supports=" << base_supports
              << " cycles=" << cycles
              << " rotated_values=" << rotated
              << " logical_peer_values=" << cross
              << " logical_peer_GiB=" << logical_gib
              << " leader_remote_position_values=" << remote
              << " direct_remote_memory_ops=" << direct_remote_ops
              << " direct_remote_GiB=" << direct_gib
              << " direct_over_logical=" << overhead
              << " blocks=" << blocks
              << " wall_ms=" << ms
              << " support_Gscan_s=" << support_gs
              << " owner_from_support_only=1"
              << " mate_materializations=0 primitive_rank_calls=0"
              << " state_allocation_bytes=0"
              << " support_scan_once=1 exact_traffic=OK\n";

    cudaFree(d_error);
    cudaFree(d_remote);
    cudaFree(d_cross);
    cudaFree(d_rotated);
    cudaFree(d_cycles);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const int S = argc > 3 ? std::atoi(argv[3]) : Kwin;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 1024u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 8;
    if (W < 6 || W > RP_MAX_W || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || !blocks || ngpu < 2 || ngpu > P2P_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "traffic device count");
    if (visible < 1) return 3;

    run_p2p_traffic_probe(W, Kwin, S, false, ngpu, blocks);
    run_p2p_traffic_probe(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK gridfp_reduced_production_p2p_traffic=1\n";
    return 0;
}
