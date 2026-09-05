#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_microprobe_main_unused
#include "gridfp_reduced_production_shift_cycle_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int P2P_MAX_GPU = 8;

__global__ void shifted_tile_p2p_cycle_kernel(
    std::uint32_t** __restrict__ peer_state,
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int gpu_id,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* __restrict__ local_cycles,
    unsigned long long* __restrict__ local_cross_values,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
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
                if (lane == 0) set_error(error, 181);
                continue;
            }
            if (cycle_len <= 1) continue;

            const DeviceKey leader = equal_run_key0_device(
                run.support, blocked, W, q, reverse);
            const GroupedDeviceRank lr = grouped_rank_device(
                leader, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
            if (lr.owner != gpu_id) continue;

            const int occupied = __popc(run.support);
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp = peer_state[lr.owner][lr.local + i];
                int prev_owner = lr.owner;
                unsigned cross = 0;
                std::uint32_t cur_support = shift_next_support_device(
                    run.support, blocked, W, q, Kwin, S, reverse);
                int hops = 1;
                while (cur_support != run.support) {
                    const DeviceKey cur = equal_run_key0_device(
                        cur_support, blocked, W, q, reverse);
                    const GroupedDeviceRank cr = grouped_rank_device(
                        cur, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
                    const std::uint32_t next_value = peer_state[cr.owner][cr.local + i];
                    peer_state[cr.owner][cr.local + i] = temp;
                    temp = next_value;
                    cross += cr.owner != prev_owner;
                    prev_owner = cr.owner;
                    cur_support = shift_next_support_device(
                        cur_support, blocked, W, q, Kwin, S, reverse);
                    ++hops;
                    if (hops > cycle_len) {
                        set_error(error, 182);
                        break;
                    }
                }
                peer_state[lr.owner][lr.local + i] = temp;
                cross += lr.owner != prev_owner;
                if (cross) atomicAdd(
                    local_cross_values,
                    static_cast<unsigned long long>(cross));
            }
            __syncwarp();
            if (lane == 0) atomicAdd(local_cycles, 1ULL);
        }
    }
}

struct DevicePeerCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t** peer = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* cycles = nullptr;
    unsigned long long* cross_values = nullptr;
    int* error = nullptr;
};

void enable_all_peer_access(int ngpu) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "p2p set peer source");
        for (int h = 0; h < ngpu; ++h) {
            if (g == h) continue;
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, g, h), "p2p can access peer");
            if (!can) fail("required CUDA peer access is unavailable");
            const cudaError_t e = cudaDeviceEnablePeerAccess(h, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "p2p enable peer");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

void run_p2p_shift_probe(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned blocks
) {
    if (ngpu < 2 || ngpu > P2P_MAX_GPU) fail("p2p gpu count");
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    const OwnerPlan owner_plan{plan.owner_begin, plan.owner_size};

    std::vector<std::uint32_t> flat_input, flat_expected;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, flat_input, flat_expected);

    enable_all_peer_access(ngpu);
    std::vector<DevicePeerCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "p2p alloc set device");
        install_tables(tables);
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].state, n * sizeof(std::uint32_t)), "p2p alloc local state");
        peer_ptr[static_cast<std::size_t>(g)] = ctx[static_cast<std::size_t>(g)].state;
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].state,
                      flat_input.data() + base,
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "p2p copy local state");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].owner_begin, ngpu * sizeof(Rank64)), "p2p alloc owner begin");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].owner_begin,
                      plan.owner_begin.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "p2p copy owner begin");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].cycles, sizeof(unsigned long long)), "p2p alloc cycles");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].cross_values, sizeof(unsigned long long)), "p2p alloc cross");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].error, sizeof(int)), "p2p alloc error");
        ck(cudaMemset(ctx[static_cast<std::size_t>(g)].cycles, 0, sizeof(unsigned long long)), "p2p zero cycles");
        ck(cudaMemset(ctx[static_cast<std::size_t>(g)].cross_values, 0, sizeof(unsigned long long)), "p2p zero cross");
        ck(cudaMemset(ctx[static_cast<std::size_t>(g)].error, 0, sizeof(int)), "p2p zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "p2p peer table set device");
        ck(cudaMalloc(&ctx[static_cast<std::size_t>(g)].peer, ngpu * sizeof(std::uint32_t*)), "p2p alloc peer table");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer,
                      peer_ptr.data(), ngpu * sizeof(std::uint32_t*), cudaMemcpyHostToDevice), "p2p copy peer table");
    }

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 one_pass_blocks = (base_supports + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass_blocks)));

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "p2p launch set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        shifted_tile_p2p_cycle_kernel<<<launch_blocks, THREADS>>>(
            c.peer, base_supports, W, Kwin, S, reverse, ngpu, g,
            c.owner_begin, c.cycles, c.cross_values, c.error);
        ck(cudaGetLastError(), "p2p cycle launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "p2p sync set device");
        ck(cudaDeviceSynchronize(), "p2p cycle sync");
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    unsigned long long cycles = 0, cross_values = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "p2p gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        unsigned long long gc = 0, gx = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost), "p2p copy error");
        ck(cudaMemcpy(&gc, c.cycles, sizeof(gc), cudaMemcpyDeviceToHost), "p2p copy cycles");
        ck(cudaMemcpy(&gx, c.cross_values, sizeof(gx), cudaMemcpyDeviceToHost), "p2p copy cross");
        if (error) fail("p2p cycle device error=" + std::to_string(error));
        cycles += gc;
        cross_values += gx;
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, c.state,
                      n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "p2p gather local state");
    }
    if (flat_output != flat_expected) fail("p2p cycle redistribution mismatch");

    std::cout << "gridfp-reduced-production-p2p-cycle"
              << " W=" << W << " Kwin=" << Kwin << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " cycles=" << cycles
              << " peer_boundary_crossings=" << cross_values
              << " blocks_per_gpu=" << launch_blocks
              << " wall_ms=" << ms
              << " direct_peer_load_store=1 staging_bytes=0"
              << " run_table_bytes=0 visited_bytes=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "p2p free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.error); cudaFree(c.cross_values); cudaFree(c.cycles);
        cudaFree(c.peer); cudaFree(c.owner_begin); cudaFree(c.state);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const int S = argc > 3 ? std::atoi(argv[3]) : Kwin;
    const unsigned blocks = argc > 4 ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 2;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > RP_MAX_W || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || !blocks || ngpu < 2 || ngpu > P2P_MAX_GPU) return 2;

    ProductionFactorTables tables(W);
    if (plan_only) {
        const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
        Rank64 max_local = 0;
        for (Rank64 z : plan.owner_size) max_local = std::max(max_local, z);
        std::cout << "gridfp-reduced-production-p2p-cycle-plan"
                  << " W=" << W << " Kwin=" << Kwin << " shift=" << S
                  << " ngpu=" << ngpu
                  << " states=" << tables.size()
                  << " max_local_u32_GiB=" << double(max_local) * 4.0 / double(1ULL<<30)
                  << " main_cycle_order=" << (Kwin + S + 2) / std::gcd(Kwin + S + 2, S)
                  << " blocked_cycle_order=" << (Kwin + S) / std::gcd(Kwin + S, S)
                  << " peer_pointer_table_bytes=" << ngpu * sizeof(std::uint32_t*)
                  << " staging_bytes=0 run_table_bytes=0 visited_bytes=0\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "p2p device count");
    if (visible < ngpu) return 4;
    run_p2p_shift_probe(W, Kwin, S, false, ngpu, blocks);
    run_p2p_shift_probe(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK gridfp_reduced_production_p2p_cycle=1\n";
    return 0;
}
