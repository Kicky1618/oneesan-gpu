#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void shifted_tile_p2p_shared_modal_kernel(
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
    unsigned long long* __restrict__ remote_value_ops,
    int* error
) {
    __shared__ Rank64 sh_local[WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_owner[WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_len[WARPS_PER_BLOCK];
    __shared__ int sh_exec[WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        for (int ri = 0; ri < 3; ++ri) {
            if (lane == 0) {
                sh_len[warp] = 0;
                sh_exec[warp] = -1;
                sh_pc[warp] = 0;

                EqualTileRunSeed seeds[3]{};
                const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
                if (ri < nr) {
                    const EqualTileRunSeed run = seeds[ri];
                    const bool blocked = run.blocked != 0;
                    const int cycle_len = shift_cycle_leader_length_device(
                        run.support, blocked, W, q, Kwin, S, reverse);
                    if (cycle_len < 0 || cycle_len > RP_MAX_W) {
                        set_error(error, 301);
                    } else if (cycle_len > 1) {
                        int counts[P2P_MAX_GPU]{};
                        std::uint32_t cur = run.support;
                        for (int h = 0; h < cycle_len; ++h) {
                            const DeviceKey key = equal_run_key0_device(
                                cur, blocked, W, q, reverse);
                            const GroupedDeviceRank gr = grouped_rank_device(
                                key, W, q, reverse, old_start, Kwin, ngpu, owner_begin);
                            if (gr.owner < 0 || gr.owner >= ngpu) {
                                set_error(error, 302);
                                break;
                            }
                            sh_owner[warp][h] = gr.owner;
                            sh_local[warp][h] = gr.local;
                            ++counts[gr.owner];
                            cur = shift_next_support_device(
                                cur, blocked, W, q, Kwin, S, reverse);
                        }
                        int exec = 0;
                        for (int g = 1; g < ngpu; ++g)
                            if (counts[g] > counts[exec]) exec = g;
                        sh_exec[warp] = exec;
                        sh_len[warp] = cycle_len;
                        sh_pc[warp] = RP_PRIMITIVE[__popc(run.support)][1];
                    }
                }
            }
            __syncwarp();

            const int len = sh_len[warp];
            if (len > 1 && sh_exec[warp] == gpu_id) {
                const Rank64 pc = sh_pc[warp];
                for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                    std::uint32_t temp =
                        peer_state[sh_owner[warp][0]][sh_local[warp][0] + i];
                    for (int h = 1; h < len; ++h) {
                        std::uint32_t* ptr =
                            peer_state[sh_owner[warp][h]] + sh_local[warp][h] + i;
                        const std::uint32_t next = *ptr;
                        *ptr = temp;
                        temp = next;
                    }
                    peer_state[sh_owner[warp][0]][sh_local[warp][0] + i] = temp;
                }
                __syncwarp();
                if (lane == 0) {
                    unsigned remote_runs = 0;
                    for (int h = 0; h < len; ++h)
                        remote_runs += sh_owner[warp][h] != gpu_id;
                    atomicAdd(local_cycles, 1ULL);
                    atomicAdd(remote_value_ops,
                              2ULL * static_cast<unsigned long long>(remote_runs) *
                              static_cast<unsigned long long>(pc));
                }
            }
            __syncwarp();
        }
    }
}

void run_p2p_shared_modal_probe(
    int W, int Kwin, int S, bool reverse, int ngpu, unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    std::vector<std::uint32_t> flat_input, flat_expected;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, flat_input, flat_expected);

    enable_all_peer_access(ngpu);
    std::vector<DevicePeerCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "modal p2p alloc set device");
        install_tables(tables);
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)), "modal p2p alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = c.state;
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(c.state, flat_input.data() + base, n * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "modal p2p copy state");
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)), "modal p2p alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "modal p2p copy owner begin");
        ck(cudaMalloc(&c.cycles, sizeof(unsigned long long)), "modal p2p alloc cycles");
        ck(cudaMalloc(&c.cross_values, sizeof(unsigned long long)), "modal p2p alloc remote ops");
        ck(cudaMalloc(&c.error, sizeof(int)), "modal p2p alloc error");
        ck(cudaMemset(c.cycles, 0, sizeof(unsigned long long)), "modal p2p zero cycles");
        ck(cudaMemset(c.cross_values, 0, sizeof(unsigned long long)), "modal p2p zero remote ops");
        ck(cudaMemset(c.error, 0, sizeof(int)), "modal p2p zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "modal p2p peer table set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.peer, ngpu * sizeof(std::uint32_t*)), "modal p2p alloc peer table");
        ck(cudaMemcpy(c.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "modal p2p copy peer table");
    }

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 one_pass = (base_supports + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "modal p2p launch set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        shifted_tile_p2p_shared_modal_kernel<<<launch_blocks, THREADS>>>(
            c.peer, base_supports, W, Kwin, S, reverse, ngpu, g,
            c.owner_begin, c.cycles, c.cross_values, c.error);
        ck(cudaGetLastError(), "modal p2p launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "modal p2p sync set device");
        ck(cudaDeviceSynchronize(), "modal p2p sync");
    }
    const double ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    unsigned long long cycles = 0, remote_ops = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "modal p2p gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0; unsigned long long gc = 0, gr = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost), "modal p2p copy error");
        ck(cudaMemcpy(&gc, c.cycles, sizeof(gc), cudaMemcpyDeviceToHost), "modal p2p copy cycles");
        ck(cudaMemcpy(&gr, c.cross_values, sizeof(gr), cudaMemcpyDeviceToHost), "modal p2p copy remote ops");
        if (error) fail("modal p2p device error=" + std::to_string(error));
        cycles += gc; remote_ops += gr;
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, c.state, n * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost), "modal p2p gather state");
    }
    if (flat_output != flat_expected) fail("modal p2p redistribution mismatch");

    std::cout << "gridfp-reduced-production-p2p-shared-modal"
              << " W=" << W << " Kwin=" << Kwin << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu << " states=" << tables.size()
              << " cycles=" << cycles
              << " remote_u32_load_store_ops=" << remote_ops
              << " remote_GiB=" << double(remote_ops) * 4.0 / double(1ULL << 30)
              << " blocks_per_gpu=" << launch_blocks << " wall_ms=" << ms
              << " cycle_rank_once_per_warp=1 modal_executor=1"
              << " shared_cycle_metadata_bytes_per_block="
              << (WARPS_PER_BLOCK * RP_MAX_W * (sizeof(Rank64) + sizeof(int)) +
                  WARPS_PER_BLOCK * (2 * sizeof(int) + sizeof(Rank64)))
              << " staging_bytes=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "modal p2p free set device");
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
    if (W < 6 || W > 11 || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || !blocks || ngpu < 2 || ngpu > P2P_MAX_GPU) return 2;
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "modal p2p device count");
    if (visible < ngpu) return 3;

    // Baseline first, optimized kernel second, with a fresh state allocation for
    // each run.  The two lines can be compared directly on the same machine.
    run_p2p_shift_probe(W, Kwin, S, false, ngpu, blocks);
    run_p2p_shared_modal_probe(W, Kwin, S, false, ngpu, blocks);
    run_p2p_shift_probe(W, Kwin, S, true, ngpu, blocks);
    run_p2p_shared_modal_probe(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK production_p2p_shared_modal_ab=1\n";
    return 0;
}
